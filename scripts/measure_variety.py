"""How varied is the fly's drumming, and what actually changes it?

Nothing in training or evaluation measures this. Onset F, beat alignment and
groove similarity all score how closely the fly copies the drummer it hears.
This script measures two other things: how much its playing changes when you
turn its dials, and how much it varies from bar to bar within one song.

Renders go through the real playback path, ``StreamingDrummer`` in 20 ms
blocks. Hits are picked at the threshold the checkpoint was scored at, which
is what an exported bundle plays at, so the numbers describe what a user would
hear. (``realtime.py --checkpoint`` uses the config's fixed 0.3 instead.)

Songs are GMD test clips by default: 4/4 grooves, one per style where the split
allows, so each one comes with the human drummer's MIDI as a yardstick. Test
clips were never trained on. ``--audio`` adds songs of your own; ``--bpm`` gives
their tempo for the bar-level numbers.

For each song, at its own style with the sliders at rest:

* **busyness** -- hits per second, and each drum's share of them;
* **bar-to-bar variety** -- the output folded onto a 16th-note grid, one
  pattern per 4/4 bar: the share of bars that are distinct, and how different
  two bars are on average (Jaccard distance between their hit sets);
* **closeness to the drummer it hears** -- groove similarity (the construction
  in ``metrics.groove_similarity``, applied to hits) against the clip's MIDI,
  and the same busyness and variety numbers for the human, as a yardstick.

Across renders of the same song, **change** is the share of hits that differ
(1 - F1, matching within the eval tolerance of 50 ms per drum):

* **style dial** -- every style against the song's own. Styles the checkpoint
  had no training files for are reported apart, because their dial settings
  were never trained;
* **sliders** -- each setting against the default;
* **repeat** -- the default rendered twice. Playback has no randomness, and the
  report says so if that ever stops being true.

A dial can move the output without changing a hit, when the drums it moves sit
below the threshold. So each setting also reports **signal shift**: the largest
movement of any drum's output curve (0..1, the value hits are picked from). A
shift of 0.0003 against a 0.5 threshold is a dial that does nothing yet. A shift
of 0.16 with no hits changed is a dial that moves drums that never play.

CPU only, one thread by default so it can run beside training. A 20-second
song takes 26 renders (the default, a repeat, the 18 styles apart from its
own, and 6 slider settings).

    python scripts/measure_variety.py --checkpoint runs/velocity_probe_cpu/best.pt
    python scripts/measure_variety.py --compare runs/variety/a/variety.json runs/variety/b/variety.json
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DEFAULT_SLIDERS = "drive=0.5,drive=1.5,tightness=0.5,tightness=1.5,pocket=0.8,pocket=1.25"
TOLERANCE_S = 0.05
SUBDIVISIONS = 16

Hit = tuple[str, float, float]          # (drum, seconds, loudness 0..1)


# ------------------------------------------------------------------ measures


def change(a: list[Hit], b: list[Hit], tol: float = TOLERANCE_S) -> float:
    """Share of hits that differ between two renders: 1 - F1, per drum.

    Greedy one-to-one matching in time order within ``tol``, the way onset F
    matches. Two silent renders are identical, not undefined.
    """
    n = len(a) + len(b)
    if n == 0:
        return 0.0
    matched = 0
    for drum in {h[0] for h in a} | {h[0] for h in b}:
        ta = sorted(h[1] for h in a if h[0] == drum)
        tb = sorted(h[1] for h in b if h[0] == drum)
        j = 0
        for t in ta:
            while j < len(tb) and tb[j] < t - tol:
                j += 1
            if j < len(tb) and abs(tb[j] - t) <= tol:
                matched += 1
                j += 1
    return 1.0 - 2.0 * matched / n


def bar_patterns(hits: list[Hit], bpm: float, seconds: float, drums: list[str],
                 sub: int = SUBDIVISIONS) -> np.ndarray:
    """One row per complete 4/4 bar: which drum hit on which 16th."""
    bar = 4 * 60.0 / bpm
    n_bars = int(seconds // bar)
    grid = np.zeros((n_bars, len(drums) * sub), dtype=bool)
    col = {d: i for i, d in enumerate(drums)}
    for drum, t, _ in hits:
        b = int(t // bar)
        if b < n_bars and drum in col:
            slot = int(round((t % bar) / bar * sub)) % sub
            if slot == 0 and (t % bar) / bar * sub > sub / 2:
                b += 1                      # rounds up onto the next downbeat
                if b >= n_bars:
                    continue
            grid[b, col[drum] * sub + slot] = True
    return grid


def bar_variety(grid: np.ndarray) -> dict:
    """Distinct bars, and the mean Jaccard distance between pairs of bars."""
    n = len(grid)
    if n == 0:
        return {"bars": 0, "distinct_share": float("nan"), "mean_bar_difference": float("nan")}
    distinct = len({row.tobytes() for row in grid})
    dists = []
    for i, j in combinations(range(n), 2):
        union = np.logical_or(grid[i], grid[j]).sum()
        inter = np.logical_and(grid[i], grid[j]).sum()
        dists.append(0.0 if union == 0 else 1.0 - inter / union)
    return {"bars": n, "distinct_share": distinct / n,
            "mean_bar_difference": float(np.mean(dists)) if dists else 0.0}


def groove_similarity(a: list[Hit], b: list[Hit], bpm: float, drums: list[str],
                      sub: int = SUBDIVISIONS) -> float:
    """``metrics.groove_similarity``'s construction, on hits instead of curves:
    both rhythms folded onto one bar per drum, compared by cosine."""
    bar = 4 * 60.0 / bpm
    col = {d: i for i, d in enumerate(drums)}

    def fold(hits):
        v = np.zeros((sub, len(drums)))
        for drum, t, _ in hits:
            if drum in col:
                v[int((t % bar) / bar * sub) % sub, col[drum]] += 1.0
        return v.ravel()

    x, y = fold(a), fold(b)
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    return 0.0 if nx == 0 or ny == 0 else float(x @ y / (nx * ny))


def busyness(hits: list[Hit], seconds: float, drums: list[str]) -> dict:
    counts = Counter(h[0] for h in hits)
    total = sum(counts.values())
    return {"hits_per_s": total / seconds,
            "share": {d: (counts[d] / total if total else 0.0) for d in drums}}


# --------------------------------------------------------------------- songs


def pick_songs(root: Path, n: int, seconds: float, seed: int = 0) -> list[dict]:
    """GMD test clips: 4/4 grooves long enough, one per style in turn."""
    rows = [r for r in csv.DictReader((root / "info.csv").open(newline=""))
            if r["split"] == "test" and r.get("beat_type") == "beat"
            and r.get("time_signature") == "4-4" and float(r["duration"]) >= seconds
            and r.get("audio_filename") and (root / r["audio_filename"]).exists()]
    rng = random.Random(seed)
    by_style: dict[str, list[dict]] = {}
    for r in sorted(rows, key=lambda r: r["id"]):
        by_style.setdefault(r["style"].split("/")[0], []).append(r)
    for v in by_style.values():
        rng.shuffle(v)
    styles = sorted(by_style)
    rng.shuffle(styles)
    picked = []
    while len(picked) < n and any(by_style.values()):
        for s in styles:
            if by_style[s] and len(picked) < n:
                picked.append(by_style[s].pop())
    return picked


def load_audio(path: Path, sr: int, seconds: float) -> np.ndarray:
    """``render_file``'s loading, verbatim, then the first ``seconds``.

    Normalised by the whole file's peak before cropping, as ``render_file``
    does. Playback is causal, so this equals rendering the whole song and
    keeping its opening.
    """
    import soundfile as sf

    audio, in_sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if in_sr != sr:
        audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * sr / in_sr)),
                          np.arange(len(audio)), audio).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak
    return audio[: int(seconds * sr)]


def human_hits(midi_path: Path, seconds: float, drums: list[str]) -> list[Hit]:
    import pretty_midi
    from dataset import MIDI_TO_CLASS, resolve_class

    have = set(drums)
    out = []
    for inst in pretty_midi.PrettyMIDI(str(midi_path)).instruments:
        for note in inst.notes:
            drum = resolve_class(MIDI_TO_CLASS.get(note.pitch, ""), have) \
                if note.pitch in MIDI_TO_CLASS else None
            if drum and 0.0 <= note.start < seconds:
                out.append((drum, float(note.start), note.velocity / 127.0))
    return sorted(out, key=lambda h: h[1])


# -------------------------------------------------------------------- render


def render(model, kit, cfg, audio: np.ndarray, style: int | None, threshold: float,
           block_ms: float = 20.0) -> tuple[list[Hit], np.ndarray]:
    """Hits, plus the per-drum output curve they were picked from.

    The curve comes from a hook on the decoder, so it is the very tensor
    ``StreamingDrummer.push`` thresholds. A dial can move it without changing a
    single hit, and the size of that movement says how far the dial is from
    mattering.
    """
    import torch
    from realtime import drummer_for

    curves: list[np.ndarray] = []
    hook = model.decoder.register_forward_hook(
        lambda m, i, o: curves.append(torch.sigmoid(o).squeeze(0).cpu().numpy()))
    try:
        drummer = drummer_for(model, kit, cfg, threshold)
        drummer.set_style(style)
        sr = cfg["audio"]["sample_rate"]
        block = int(sr * block_ms / 1000.0)
        hits = []
        for a in range(0, len(audio) - block + 1, block):
            hits += [(h.cls, float(h.t), float(h.velocity))
                     for h in drummer.push(audio[a:a + block])]
    finally:
        hook.remove()
    curve = np.concatenate(curves) if curves else np.zeros((0, kit.n), np.float32)
    return hits, curve


def shift(curve: np.ndarray, base: np.ndarray) -> float:
    """Largest movement of any drum's output, on its 0..1 scale."""
    n = min(len(curve), len(base))
    return float(np.abs(curve[:n] - base[:n]).max()) if n else 0.0


def trained_style_counts(cfg: dict, classes: list[str]) -> tuple[list[str], Counter]:
    """The checkpoint's style vocabulary, and its training files per style.

    Built with the loader the checkpoint was trained with, from its own config,
    so a ``max_files`` or ``subset`` setting is honoured.
    """
    from dataset import build_dataset

    ds = build_dataset(cfg["data"], classes, split="train")
    return list(ds.styles), Counter(r["style"].split("/")[0] for r in ds.rows)


def measure(a) -> dict:
    import torch

    torch.set_num_threads(a.threads)
    from build import build_model, get_subgraph, role_index
    from decoder import DrumKit

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sg = get_subgraph(cfg)
    model, kit = build_model(cfg, sg, n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    model.load_state_dict(ck["model"])
    model = model.eval()
    kit = DrumKit(ck["kit"])
    roles = role_index(sg)
    drums = list(kit.classes)
    threshold = float(a.threshold if a.threshold is not None
                      else ck.get("best_threshold") or cfg["eval"].get("threshold", 0.3))
    sr = cfg["audio"]["sample_rate"]

    styles, trained = trained_style_counts(cfg, drums)
    n_styles = model.genre.embed.num_embeddings if model.genre is not None else 0
    sliders = [(s.split("=")[0], float(s.split("=")[1])) for s in a.sliders.split(",") if s]

    root = ROOT / cfg["data"].get("root", "data/egmd/groove")
    songs = [{"name": Path(r["audio_filename"]).stem, "audio": root / r["audio_filename"],
              "midi": root / r["midi_filename"], "bpm": float(r["bpm"]),
              "style": r["style"].split("/")[0], "drummer": r["drummer"]}
             for r in pick_songs(root, a.songs, a.seconds, a.seed)]
    songs += [{"name": p.stem, "audio": p, "midi": None, "bpm": a.bpm, "style": None,
               "drummer": None} for p in a.audio]

    print(f"{a.checkpoint}: threshold {threshold}, {len(songs)} songs x {a.seconds:.0f} s, "
          f"{n_styles} styles, {len(sliders)} slider settings, {a.threads} thread(s)")
    t0 = time.time()
    per_song = []
    for si, song in enumerate(songs):
        audio = load_audio(song["audio"], sr, a.seconds)
        secs = len(audio) / sr
        own = styles.index(song["style"]) if song["style"] in styles else None
        base, base_curve = render(model, kit, cfg, audio, own, threshold)
        repeat, repeat_curve = render(model, kit, cfg, audio, own, threshold)
        rec = {"song": song["name"], "style": song["style"], "drummer": song["drummer"],
               "bpm": song["bpm"], "seconds": secs,
               "repeat_identical": base == repeat and shift(repeat_curve, base_curve) == 0.0,
               "fly": busyness(base, secs, drums), "styles": {}, "sliders": {}}
        rec["fly"]["max_output"] = {d: float(base_curve[:, i].max()) for i, d in enumerate(drums)}
        rec["fly"]["share_over_threshold"] = {
            d: float((base_curve[:, i] >= threshold).mean()) for i, d in enumerate(drums)}
        if song["bpm"]:
            rec["fly"].update(bar_variety(bar_patterns(base, song["bpm"], secs, drums)))
        if song["midi"] is not None:
            human = human_hits(song["midi"], secs, drums)
            rec["human"] = {**busyness(human, secs, drums),
                            **bar_variety(bar_patterns(human, song["bpm"], secs, drums))}
            rec["groove_similarity_to_human"] = groove_similarity(base, human, song["bpm"], drums)
            rec["change_vs_human"] = change(base, human)

        by_style, curves = {}, {}
        for sid in range(n_styles):
            by_style[sid], curves[sid] = ((base, base_curve) if sid == own
                                          else render(model, kit, cfg, audio, sid, threshold))
        none = render(model, kit, cfg, audio, None, threshold)[0] if n_styles else base
        for sid, hits in by_style.items():
            name = styles[sid] if sid < len(styles) else f"style{sid}"
            rec["styles"][name] = {
                "trained_files": trained.get(name, 0),
                "change_vs_own_style": change(hits, base),
                "change_vs_no_style": change(hits, none),
                "max_output_shift": shift(curves[sid], base_curve),
                "hits_per_s": len(hits) / secs,
            }
        pairs = [change(by_style[i], by_style[j]) for i, j in combinations(sorted(by_style), 2)]
        rec["style_pairwise_change"] = float(np.mean(pairs)) if pairs else 0.0

        for name, value in sliders:
            model.set_slider(name, value, roles)
            try:
                hits, curve = render(model, kit, cfg, audio, own, threshold)
            finally:
                model.reset_sliders()
            rec["sliders"][f"{name}={value:g}"] = {
                "change_vs_default": change(hits, base),
                "max_output_shift": shift(curve, base_curve),
                "hits_per_s": len(hits) / secs,
            }
        per_song.append(rec)
        el = time.time() - t0
        print(f"  [{si + 1}/{len(songs)}] {song['name'][:40]:<40} "
              f"{rec['fly']['hits_per_s']:.1f} hits/s, style change "
              f"{rec['style_pairwise_change']:.0%}  ({el / 60:.1f} min, "
              f"~{el / (si + 1) * (len(songs) - si - 1) / 60:.0f} min left)", flush=True)

    return {"checkpoint": str(a.checkpoint), "threshold": threshold, "seconds": a.seconds,
            "seed": a.seed, "styles_vocab": styles, "trained_files_per_style": dict(trained),
            "songs": per_song, "summary": summarise(per_song, drums),
            "measured": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "wall_minutes": round((time.time() - t0) / 60, 1)}


# ------------------------------------------------------------------- summary


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(xs)) if xs else float("nan")


def summarise(songs: list[dict], drums: list[str]) -> dict:
    trained, untrained = [], []
    for s in songs:
        for name, st in s["styles"].items():
            if name == s["style"]:
                continue
            (trained if st["trained_files"] else untrained).append(st["change_vs_own_style"])
    human = [s for s in songs if "human" in s]
    out = {
        "repeat_identical": all(s["repeat_identical"] for s in songs),
        "fly_hits_per_s": _mean([s["fly"]["hits_per_s"] for s in songs]),
        "human_hits_per_s": _mean([s["human"]["hits_per_s"] for s in human]),
        "fly_share": {d: _mean([s["fly"]["share"][d] for s in songs]) for d in drums},
        "human_share": {d: _mean([s["human"]["share"][d] for s in human]) for d in drums},
        "fly_distinct_bar_share": _mean([s["fly"].get("distinct_share") for s in songs]),
        "human_distinct_bar_share": _mean([s["human"]["distinct_share"] for s in human]),
        "fly_mean_bar_difference": _mean([s["fly"].get("mean_bar_difference") for s in songs]),
        "human_mean_bar_difference": _mean([s["human"]["mean_bar_difference"] for s in human]),
        "groove_similarity_to_human": _mean([s["groove_similarity_to_human"] for s in human]),
        "change_vs_human": _mean([s["change_vs_human"] for s in human]),
        "style_pairwise_change": _mean([s["style_pairwise_change"] for s in songs]),
        "style_change_trained": _mean(trained),
        "style_change_untrained": _mean(untrained),
        "style_max_output_shift": max((st["max_output_shift"] for s in songs
                                       for st in s["styles"].values()), default=0.0),
        "drums_that_play": [d for d in drums
                            if any(s["fly"]["share"][d] > 0 for s in songs)],
        "max_output": {d: max((s["fly"]["max_output"][d] for s in songs), default=0.0)
                       for d in drums},
        "sliders": {},
    }
    for key in (songs[0]["sliders"] if songs else {}):
        out["sliders"][key] = {
            "max_output_shift": max(s["sliders"][key]["max_output_shift"] for s in songs),
            "change_vs_default": _mean([s["sliders"][key]["change_vs_default"] for s in songs]),
            "hits_per_s_ratio": _mean([s["sliders"][key]["hits_per_s"] / s["fly"]["hits_per_s"]
                                       if s["fly"]["hits_per_s"] else None for s in songs]),
        }
    return out


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.0%}"


def report(res: dict) -> str:
    """Plain language first, numbers in brackets."""
    s = res["summary"]
    vocab_trained = sum(1 for v in res["styles_vocab"] if res["trained_files_per_style"].get(v))
    top = lambda share: ", ".join(f"{d} {v:.0%}" for d, v in
                                  sorted(share.items(), key=lambda kv: -kv[1])[:3])
    lines = [
        f"How varied is the fly's drumming? ({Path(res['checkpoint']).parent.name})",
        f"{len(res['songs'])} test songs the fly never trained on, first {res['seconds']:.0f} s "
        f"of each, played the way a finished model file would play them.",
        "",
        "SAME INPUT, SAME OUTPUT",
        "  Playing a song twice gives " + ("exactly the same drums." if s["repeat_identical"]
                                           else "DIFFERENT drums -- something random crept in."),
        "",
        "WHICH DRUMS IT PLAYS",
        f"  A drum plays when its signal crosses {res['threshold']}. "
        f"Drums that ever did: {', '.join(s['drums_that_play']) or 'none'}.",
        "  Highest signal each drum reached: " + ", ".join(
            f"{d} {v:.2f}" for d, v in s["max_output"].items()) + ".",
        "",
        "HOW BUSY",
        f"  The fly plays {s['fly_hits_per_s']:.1f} hits a second; the human drummer on the same "
        f"songs played {s['human_hits_per_s']:.1f}.",
        f"  Fly's most-used drums: {top(s['fly_share'])}.",
        f"  Human's most-used drums: {top(s['human_share'])}.",
        "",
        "VARIETY WITHIN A SONG",
        f"  Of the fly's bars, {pct(s['fly_distinct_bar_share'])} are different from each other "
        f"(human: {pct(s['human_distinct_bar_share'])}).",
        f"  Two bars picked at random differ in {pct(s['fly_mean_bar_difference'])} of their hits "
        f"(human: {pct(s['human_mean_bar_difference'])}).",
        "",
        "HOW CLOSE TO THE DRUMMER IT IS HEARING",
        f"  Groove match with the human part: {s['groove_similarity_to_human']:.2f} on a 0-1 scale "
        f"(1 = same groove). Hit by hit, {pct(s['change_vs_human'])} of hits differ.",
        "",
        "THE STYLE DIAL",
        f"  This model was trained on {vocab_trained} of the {len(res['styles_vocab'])} styles.",
        f"  Switching style changes {pct(s['style_pairwise_change'])} of the hits on average.",
        f"  To a style it trained on: {pct(s['style_change_trained'])} of hits change. "
        f"To one it never trained on: {pct(s['style_change_untrained'])}.",
        f"  The most any style moved any drum's signal: {s['style_max_output_shift']:.4f} "
        f"(on a 0-1 scale, where a hit needs the signal to cross {res['threshold']}).",
        "",
        "THE SLIDERS (each against the default)",
    ]
    for key, v in s["sliders"].items():
        lines.append(f"  {key:<16} changes {pct(v['change_vs_default'])} of hits, "
                     f"plays {v['hits_per_s_ratio']:.2f}x as many, and moves the signal "
                     f"by up to {v['max_output_shift']:.3f}.")
    lines += ["", f"(threshold {res['threshold']}, {res['wall_minutes']} min to measure, "
                  f"{res['measured']})"]
    return "\n".join(lines) + "\n"


KEYS = [
    ("repeat_identical", "same output twice"),
    ("fly_hits_per_s", "fly hits/s"), ("human_hits_per_s", "human hits/s"),
    ("fly_distinct_bar_share", "fly distinct bars"),
    ("fly_mean_bar_difference", "fly bar-to-bar difference"),
    ("groove_similarity_to_human", "groove match to human"),
    ("change_vs_human", "hits differing from human"),
    ("style_pairwise_change", "style dial: hits changed"),
    ("style_change_trained", "  ...to a trained style"),
    ("style_change_untrained", "  ...to an untrained style"),
    ("style_max_output_shift", "style dial: max signal shift"),
]


def compare(paths: list[Path]) -> str:
    res = [json.loads(p.read_text()) for p in paths]
    names = [Path(r["checkpoint"]).parent.name[:22] for r in res]
    lines = [f"{'':<30}" + "".join(f"{n:>24}" for n in names)]
    fmt = lambda v: (str(v) if isinstance(v, bool) else "n/a" if v is None or np.isnan(v)
                     else f"{v:.4f}" if 0 < abs(v) < 0.01 else f"{v:.2f}")
    for key, label in KEYS:
        lines.append(f"{label:<30}" + "".join(f"{fmt(r['summary'][key]):>24}" for r in res))
    for key in res[0]["summary"]["sliders"]:
        lines.append(f"{'slider ' + key + ' change':<30}" + "".join(
            f"{fmt(r['summary']['sliders'].get(key, {}).get('change_vs_default', float('nan'))):>24}"
            for r in res))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--songs", type=int, default=6, help="GMD test clips to use")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--audio", type=Path, nargs="*", default=[],
                    help="your own songs, measured alongside (no human yardstick)")
    ap.add_argument("--bpm", type=float, default=0.0, help="tempo of --audio songs")
    ap.add_argument("--sliders", default=DEFAULT_SLIDERS)
    ap.add_argument("--threshold", type=float, default=None,
                    help="default: the checkpoint's own best_threshold, as a bundle plays")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: runs/variety/<run name>/")
    ap.add_argument("--compare", type=Path, nargs="+", default=None,
                    help="print two or more variety.json files side by side")
    a = ap.parse_args(argv)

    if a.compare:
        print(compare(a.compare))
        return 0
    if a.checkpoint is None:
        ap.error("--checkpoint is required unless --compare is given")
    res = measure(a)
    out = a.out or ROOT / "runs" / "variety" / a.checkpoint.parent.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "variety.json").write_text(json.dumps(res, indent=2))
    text = report(res)
    (out / "report.txt").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"wrote {out / 'variety.json'} and {out / 'report.txt'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
