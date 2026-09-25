"""Phase 3 data: drum audio with sample-aligned MIDI, plus a synthetic fallback.

Two real corpora are supported, both Magenta, both the same Roland TD-11 setup:

  * **GMD** (``groove-v1.0.0``, 5.1 GB) -- audio *and* aligned MIDI, with style
    labels in ``info.csv``. This is what actually trains here.
  * **E-GMD** (``e-gmd-v1.0.0``) -- the expanded set PLAN.md names. Its audio
    archive is 96 GB; the MIDI-only archive (107 MB) carries the same style
    vocabulary and is used to build the style list even when its audio is
    absent.

``SyntheticDrums`` generates click-track patterns with known onsets. It is not
a substitute for real data -- it exists so the pipeline, the ablation harness
and the tests can run end to end on a machine with no corpus at all, and so a
failure in the model is distinguishable from a failure in the data path.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "egmd"

#: Velocity-collapsed classes, ordered as in decoder.KIT_TIERS['8piece'].
#: General MIDI percussion notes are many-to-one onto these -- a drummer hits
#: three different pads that all mean "snare" to a listener.
MIDI_TO_CLASS: dict[int, str] = {
    35: "kick", 36: "kick",
    38: "snare", 40: "snare", 37: "snare",
    42: "hat_closed", 44: "hat_closed", 22: "hat_closed",
    46: "hat_open", 26: "hat_open",
    41: "tom_low", 43: "tom_low", 45: "tom_low", 58: "tom_low",
    47: "tom_mid", 48: "tom_mid",
    50: "tom_high",
    49: "crash", 52: "crash", 55: "crash", 57: "crash",
    51: "ride", 59: "ride", 53: "ride_bell",
    37: "sidestick", 44: "hat_pedal",
}

#: Where a class goes when the kit in use does not have it. The 8-piece kit has
#: no ride, so ride notes must land on crash rather than be dropped -- but the
#: articulated tier does have one, and sending its rides to crash would be
#: wrong. Resolved per kit at load time instead of baked into the note map.
CLASS_FALLBACK: dict[str, str] = {
    "ride": "crash", "ride_bell": "crash", "cowbell": "crash",
    "sidestick": "snare", "clap": "snare",
    "hat_pedal": "hat_closed",
}


def resolve_class(name: str, classes: set[str]) -> str | None:
    """Map a MIDI-derived class onto the kit actually in use."""
    seen = set()
    while name is not None and name not in classes:
        if name in seen:
            return None
        seen.add(name)
        name = CLASS_FALLBACK.get(name)
    return name


@dataclass
class Clip:
    audio: np.ndarray        # (samples,) float32 mono
    onsets: np.ndarray       # (steps, n_classes) float32, Gaussian-smoothed
    velocity: np.ndarray     # (steps, n_classes) float32 in 0..1, held over each onset
    style: int
    tempo: float


def smooth_targets(
    times: "list[tuple[float, str] | tuple[float, str, float]]",
    classes: list[str], n_steps: int, step_ms: float, sigma_ms: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Paired targets: a Gaussian-smoothed onset plane and a velocity plane.

    Smoothing is what makes BCE trainable here: an unsmoothed target is a
    single 1 in 1,600 steps per class, and the loss would just learn silence.

    The velocity plane holds each hit's velocity flat across that hit's own
    kernel support, rather than scaled by the kernel. Velocity is a property of
    the hit, not of how close a step sits to its centre -- scaling it by the
    bell would teach the head to regress the detection envelope a second time,
    which is exactly the confusion between "how sure" and "how hard" that
    having two heads is meant to end. Where two hits of one class overlap, the
    step takes the velocity of whichever hit's kernel is taller there, so the
    two planes always agree about which hit a step belongs to.

    ``times`` entries are ``(t, cls)`` or ``(t, cls, velocity)`` with velocity
    in 0..1; a 2-tuple means "unknown", recorded as 1.0.
    """
    idx = {c: i for i, c in enumerate(classes)}
    available = set(classes)
    y = np.zeros((n_steps, len(classes)), dtype=np.float32)
    vel = np.zeros((n_steps, len(classes)), dtype=np.float32)
    sigma = max(sigma_ms / step_ms, 1e-3)
    half = int(math.ceil(3 * sigma))
    kern = np.exp(-0.5 * (np.arange(-half, half + 1) / sigma) ** 2).astype(np.float32)

    for event in times:
        t, cls = event[0], event[1]
        v = float(event[2]) if len(event) > 2 else 1.0
        resolved = resolve_class(cls, available)
        j = idx.get(resolved) if resolved else None
        if j is None:
            continue
        c = int(round(t * 1000.0 / step_ms))
        lo, hi = max(0, c - half), min(n_steps, c + half + 1)
        if lo >= hi:
            continue
        k = kern[lo - (c - half): hi - (c - half)]
        taller = k > y[lo:hi, j]
        vel[lo:hi, j] = np.where(taller, np.float32(np.clip(v, 0.0, 1.0)), vel[lo:hi, j])
        y[lo:hi, j] = np.maximum(y[lo:hi, j], k)
    return y, vel


def smooth_onsets(
    times: list[tuple[float, str]], classes: list[str], n_steps: int,
    step_ms: float, sigma_ms: float = 20.0,
) -> np.ndarray:
    """The onset plane alone. See :func:`smooth_targets`."""
    return smooth_targets(times, classes, n_steps, step_ms, sigma_ms)[0]


class SyntheticDrums(Dataset):
    """Click patterns with exactly known onsets. Sanity floor, not a corpus."""

    def __init__(
        self, classes: list[str], n_clips: int = 256, seconds: float = 4.0,
        sample_rate: int = 22_050, step_ms: float = 5.0, n_styles: int = 4, seed: int = 0,
    ):
        self.classes, self.sample_rate, self.step_ms = classes, sample_rate, step_ms
        self.seconds, self.n_styles = seconds, n_styles
        self.rng = np.random.default_rng(seed)
        self.items = [self._make(i) for i in range(n_clips)]

    # one timbre per class so the encoder has something to separate
    _TIMBRE = {"kick": (60.0, 0.10), "snare": (200.0, 0.06), "hat_closed": (6000.0, 0.02),
               "hat_open": (5000.0, 0.12), "tom_low": (110.0, 0.14),
               "tom_mid": (160.0, 0.12), "tom_high": (220.0, 0.10), "crash": (3500.0, 0.5)}

    def _make(self, i: int) -> Clip:
        sr, n = self.sample_rate, int(self.seconds * self.sample_rate)
        rng = np.random.default_rng(1000 + i)
        style = int(rng.integers(self.n_styles))
        tempo = float(rng.uniform(80, 160))
        beat = 60.0 / tempo
        step = beat / 4.0

        audio = np.zeros(n, dtype=np.float32)
        events: list[tuple[float, str]] = []
        n_steps16 = int(self.seconds / step)
        for s in range(n_steps16):
            t = s * step
            if t >= self.seconds:
                break
            fire = []
            if s % 8 == 0 or (style == 1 and s % 8 == 6):
                fire.append("kick")
            if s % 8 == 4:
                fire.append("snare")
            if s % 2 == 0 and "hat_closed" in self.classes:
                fire.append("hat_closed")
            if style >= 2 and rng.random() < 0.12:
                fire.append(str(rng.choice([c for c in self.classes if c.startswith("tom")] or ["snare"])))
            for c in fire:
                if c not in self.classes:
                    continue
                f, dur = self._TIMBRE.get(c, (300.0, 0.05))
                k = int(dur * sr)
                tt = np.arange(k) / sr
                env = np.exp(-tt / (dur / 3.0)).astype(np.float32)
                if c.startswith("hat") or c == "crash":
                    sig = rng.standard_normal(k).astype(np.float32) * env
                else:
                    sig = np.sin(2 * np.pi * f * tt).astype(np.float32) * env
                a = int(t * sr)
                b = min(n, a + k)
                # The amplitude this synth already randomises per hit is the
                # velocity, so the click track carries a real dynamics signal
                # for the velocity head to find rather than a constant one.
                # Anything learnable here is learnable from the audio alone.
                amp = float(rng.uniform(0.35, 1.0))
                audio[a:b] += sig[: b - a] * amp
                events.append((t, c, amp))

        audio += rng.standard_normal(n).astype(np.float32) * 0.005
        peak = np.abs(audio).max()
        if peak > 0:
            audio /= peak
        n_steps = int(round(self.seconds * 1000.0 / self.step_ms))
        y, vel = smooth_targets(events, self.classes, n_steps, self.step_ms)
        return Clip(audio, y, vel, style, tempo)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        c = self.items[i]
        return (torch.from_numpy(c.audio), torch.from_numpy(c.onsets),
                torch.from_numpy(c.velocity), torch.tensor(c.style),
                torch.tensor(c.tempo))


#: How ``max_files`` picks its rows. See :func:`subset_rows`.
SUBSETS = ("first", "stratified")


def _allocate(sizes: dict, n: int) -> dict:
    """Split ``n`` slots across groups in proportion to their sizes.

    Every group gets one slot first when ``n`` allows it, so no group is
    dropped just for being small; the rest go out by largest remainder. No
    group is given more slots than it has rows. Ties break on the key, so the
    allocation itself is deterministic and only the draw inside each group
    depends on the seed.
    """
    total = sum(sizes.values())
    if n >= total:
        return dict(sizes)
    floor = 1 if n >= len(sizes) else 0
    spare = {k: v - floor for k, v in sizes.items()}
    left, pool = n - floor * len(sizes), sum(spare.values())
    share = {k: divmod(left * v, pool) for k, v in spare.items()}   # exact, no floats
    out = {k: floor + q for k, (q, _) in share.items()}
    short = n - sum(out.values())
    for k in sorted(share, key=lambda k: (-share[k][1], str(k)))[:short]:
        out[k] += 1
    return out


def subset_rows(rows: list[dict], n: int | None, how: str = "first",
                seed: int = 0) -> list[dict]:
    """The ``n`` rows a capped run trains on.

    ``first`` is the historical behaviour: the first ``n`` in info.csv order.
    GMD's info.csv is ordered by drummer, so on GMD's training split that is
    one drummer -- ``max_files: 256`` is 256 drummer1 clips, 227 of them
    fills. It stays the default so existing configs and checkpoints reproduce.

    ``stratified`` draws a seeded sample spread across drummers, and within
    each drummer across beat and fill, in proportion to the full list -- so the
    subset is a scaled-down copy of the split rather than one corner of it.
    Every drummer gets at least one clip when ``n`` is at least the number of
    drummers. The chosen rows keep their info.csv order, so ``n`` at or above
    the row count returns exactly what ``first`` does.
    """
    if how not in SUBSETS:
        raise ValueError(f"unknown data.subset {how!r}; have {', '.join(SUBSETS)}")
    if not n or n >= len(rows):
        return list(rows)
    if how == "first":
        return rows[:n]

    groups: dict[str, dict[str, list[int]]] = {}
    for i, r in enumerate(rows):
        groups.setdefault(r.get("drummer", ""), {}).setdefault(r.get("beat_type", ""), []).append(i)
    rng = np.random.default_rng(seed)
    keep: list[int] = []
    per_drummer = _allocate({d: sum(map(len, g.values())) for d, g in groups.items()}, n)
    for d in sorted(groups):
        per_type = _allocate({t: len(ix) for t, ix in groups[d].items()}, per_drummer[d])
        for t in sorted(groups[d]):
            ix = groups[d][t]
            keep.extend(rng.choice(ix, size=per_type[t], replace=False).tolist())
    return [rows[i] for i in sorted(keep)]


class GrooveDataset(Dataset):
    """Magenta GMD / E-GMD: real drum audio with sample-aligned MIDI.

    Clips are cut to a fixed window so they batch; the window is chosen by
    ``seconds``. On the training split the offset is random, for augmentation.
    On any other split it is **fixed per clip** -- see ``random_windows``.
    """

    def __init__(
        self, root: Path, classes: list[str], split: str = "train",
        seconds: float = 4.0, sample_rate: int = 22_050, step_ms: float = 5.0,
        sigma_ms: float = 20.0, max_files: int | None = None,
        random_windows: bool = True, window_seed: int = 0,
        subset: str = "first", subset_seed: int = 0,
    ):
        import soundfile  # noqa: F401  (fail early with a clear message)

        self.root, self.classes = Path(root), classes
        self.seconds, self.sample_rate, self.step_ms = seconds, sample_rate, step_ms
        self.sigma_ms = sigma_ms
        self.n_steps = int(round(seconds * 1000.0 / step_ms))
        self.random_windows, self.window_seed = random_windows, window_seed

        info = self.root / "info.csv"
        if not info.exists():
            raise FileNotFoundError(
                f"{info} not found. Fetch a corpus first:\n"
                f"  python scripts/fetch_egmd.py"
            )
        all_rows = list(csv.DictReader(info.open()))

        # The style vocabulary is built from the WHOLE corpus, never from one
        # split. Per-split vocabularies give the same integer different meanings
        # in train and validation (id 2 is 'jazz' in one and 'latin' in the
        # other), so the genre embedding is conditioned on noise -- and when a
        # split contains a style the training split lacked, the embedding lookup
        # goes out of range outright. Both happened here before this was fixed.
        self.styles = sorted({r["style"].split("/")[0] for r in all_rows})
        self.style_id = {s: i for i, s in enumerate(self.styles)}

        rows = [r for r in all_rows if r.get("split") == split]
        rows = [r for r in rows
                if r.get("audio_filename") and (self.root / r["audio_filename"]).exists()]
        rows = subset_rows(rows, max_files, subset, subset_seed)
        if not rows:
            raise RuntimeError(f"no usable {split} rows under {self.root}")
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def n_styles(self) -> int:
        return len(self.styles)

    def _window_start(self, i: int, span: int) -> int:
        """Where in the clip the window begins.

        Two jobs, and conflating them quietly costs a comparison. Training
        wants a different window each epoch, as augmentation. Validation wants
        the *same* window every epoch and in every ablation arm -- otherwise
        onset F moves because the audio moved, and a gap between two arms is
        partly a gap between two random slices of the corpus.

        This was one unseeded ``np.random.randint`` for both splits. Random
        windows now come from torch's per-worker generator, which the
        DataLoader seeds deterministically from ``train.seed`` (see
        ``train.build_loaders``) -- reproducible without being frozen, and
        without numpy's global RNG, which is not reseeded per worker and so
        handed every worker the same offsets. Fixed windows are derived from
        the row index, so clip *i* always yields the same audio.
        """
        if span <= 1:
            return 0
        if self.random_windows:
            return int(torch.randint(0, span, (1,)).item())
        return int(np.random.default_rng((self.window_seed, i)).integers(0, span))

    def __getitem__(self, i: int):
        import soundfile as sf
        import pretty_midi

        row = self.rows[i]
        wav_path = self.root / row["audio_filename"]
        mid_path = self.root / row["midi_filename"]

        audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != self.sample_rate:
            n_out = int(len(audio) * self.sample_rate / sr)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio
            ).astype(np.float32)

        want = int(self.seconds * self.sample_rate)
        if len(audio) < want:
            audio = np.pad(audio, (0, want - len(audio)))
            offset_s = 0.0
        else:
            start = self._window_start(i, len(audio) - want + 1)
            audio = audio[start: start + want]
            offset_s = start / self.sample_rate

        events: list[tuple[float, str, float]] = []
        try:
            pm = pretty_midi.PrettyMIDI(str(mid_path))
            for inst in pm.instruments:
                for note in inst.notes:
                    cls = MIDI_TO_CLASS.get(note.pitch)
                    if cls is None:
                        continue
                    t = note.start - offset_s
                    if 0.0 <= t < self.seconds:
                        # A drummer's dynamics, straight off the TD-11 pads.
                        # MIDI velocity is 1..127; 0 would be a note-off.
                        events.append((t, cls, note.velocity / 127.0))
        except Exception:
            pass  # a handful of GMD files have malformed MIDI; an empty target
                  # is honest about that rather than dropping the clip silently

        y, vel = smooth_targets(events, self.classes, self.n_steps, self.step_ms,
                                self.sigma_ms)
        peak = float(np.abs(audio).max())
        if peak > 0:
            audio = audio / peak
        style = self.style_id.get(row["style"].split("/")[0], 0)
        tempo = float(row.get("bpm") or 0.0)
        return (torch.from_numpy(audio.astype(np.float32)), torch.from_numpy(y),
                torch.from_numpy(vel), torch.tensor(style), torch.tensor(tempo))


def build_dataset(cfg: dict, classes: list[str], split: str = "train"):
    """Build the configured dataset.

    Falling back to synthetic data when the real corpus is missing is NOT the
    behaviour here, and used to be: a run would quietly train on click tracks
    while its config said ``data/egmd/groove``, and nothing in the logs or the
    metrics would distinguish that from a real run. Synthetic data is used only
    when the config explicitly asks for it.
    """
    root = Path(cfg.get("root") or DATA / "groove")
    if not root.is_absolute():
        # relative to the repo root, not to whatever directory the process was
        # started from -- running from src/ and from the root must agree
        root = ROOT / root
    if not cfg.get("synthetic"):
        if not (root / "info.csv").exists():
            raise FileNotFoundError(
                f"no corpus at {root} (expected info.csv there).\n"
                f"Fetch one:  python scripts/fetch_egmd.py\n"
                f"or set data.synthetic: true in the config to use the click-track "
                f"fallback deliberately."
            )
    else:
        return SyntheticDrums(
            classes,
            n_clips=cfg.get("n_clips", 256 if split == "train" else 64),
            seconds=cfg.get("seconds", 4.0),
            sample_rate=cfg.get("sample_rate", 22_050),
            step_ms=cfg.get("step_ms", 5.0),
            n_styles=cfg.get("n_styles", 4),
            seed=0 if split == "train" else 1,
        )
    return GrooveDataset(
        root, classes, split=split,
        seconds=cfg.get("seconds", 4.0),
        sample_rate=cfg.get("sample_rate", 22_050),
        step_ms=cfg.get("step_ms", 5.0),
        sigma_ms=cfg.get("sigma_ms", 20.0),
        max_files=cfg.get("max_files"),
        # Which max_files rows: `first` (default) or `stratified`. The seed is
        # its own key, not train.seed -- a seed run must not also change which
        # clips it trains on.
        subset=cfg.get("subset", "first"),
        subset_seed=int(cfg.get("subset_seed", 0)),
        # Augment the training split; hold every other split still, so a
        # metric read twice on one checkpoint reads the same both times.
        random_windows=(split == "train"),
        window_seed=int(cfg.get("window_seed", 0)),
    )
