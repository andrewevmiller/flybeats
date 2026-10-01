"""Would more test material per song, or more controls, give the main win rule more power? CPU only; no network is
trained, and nothing is locked or changed.

1 Material per song: the feasibility model (reports/feasibility/final.pt) scored on beat alignment on the full set's
  validation songs, hearing its own song and another song's band, with 1, 2 or 4 of the fixed validation clips per
  song, and with the whole song as consecutive 16 s windows (each scored after its first 2 s, as every clip is). For
  each: the headroom (mean gap) and the song-sampling noise of one network's score on 151 test songs
  (sd of the per-song gaps / sqrt(2) / sqrt(151)).
2 Number of controls: the locked rule (real > highest control + sample sd of the controls), simulated with 5 and with
  10 controls, for each material option: the chance of a false win with no true effect, and the smallest true
  advantage with an 80% chance of winning, at three control spreads (0, and 0.115 and 0.23 of the headroom, as in
  scripts/power_estimate.py).
The locked rule names 5 controls and 4 clips per song; the other options are only simulated here.
Writes reports/power_options.json.
    .venv\\Scripts\\python.exe scripts\\power_options.py [--trial]
--trial: 12 songs, few simulations; writes nothing.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler, load_validation_clips  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model import ear as model_ear  # noqa: E402
from flybeats.scoring.scorecard import Song, check_scorecard, hits_from_probs  # noqa: E402

HIDDEN, LAYERS = 128, 2                                   # as in feasibility.py
TEST_SONGS = 151
SPREADS = [0.0, 0.115, 0.23]                              # control spread, as shares of the headroom


class Ceiling(torch.nn.Module):                           # feasibility.py's model, to load its weights
    def __init__(self, n_drums, mean, std):
        super().__init__()
        self.ear = model_ear.Bands(mean, std)
        self.gru = torch.nn.GRU(model_ear.N_BANDS, HIDDEN, LAYERS, batch_first=True)
        self.hit = torch.nn.Linear(HIDDEN, n_drums)
        self.vel = torch.nn.Linear(HIDDEN, n_drums)

    def probs(self, bands):
        h, _ = self.gru(bands)
        return torch.sigmoid(self.hit(h))


def win_chance(delta, sigma_c, sigma_m, n_controls, sims, rng):
    ctrl = rng.normal(0, sigma_c, (sims, n_controls)) + rng.normal(0, sigma_m, (sims, n_controls))
    real = delta + rng.normal(0, sigma_c, sims) + rng.normal(0, sigma_m, sims)
    return float(np.mean(real > ctrl.max(1) + ctrl.std(1, ddof=1)))


def needed(headroom, sigma_c, sigma_m, n_controls, sims, rng):
    """Smallest true advantage with an 80% chance of winning (searched up to 3 x the headroom)."""
    for d in np.linspace(0, 3 * headroom, 301):
        if win_chance(d, sigma_c, sigma_m, n_controls, sims, rng) >= 0.8:
            return float(d)
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--trial", action="store_true", help="12 songs, few simulations; writes nothing")
args = ap.parse_args()
torch.set_num_threads(2)                                  # leave the CPU to the GPU run going alongside
cfg, paths = load_locked(), load_paths()
check_scorecard(cfg)
td = cfg["training_data"]
pieces = cfg["kit"]["pieces"]
first = round(td["warmup_ignored_seconds"] / (td["frame_ms"] / 1000))
clip_s = td["clip_seconds"]
manifest, events, beats = load_manifest(paths["work_dir"], "full")
val = ClipSampler(unpacked_root(paths["work_dir"], "full"), manifest, events, beats, cfg, "validation")
mean, std = model_ear.load_band_stats(set_dir(paths["work_dir"], "full") / "band_stats.json")
model = Ceiling(len(pieces), mean, std)
model.load_state_dict(torch.load(reports_dir() / "feasibility" / "final.pt", map_location="cpu")["model"])
model.eval()

listed = load_validation_clips(set_dir(paths["work_dir"], "full") / "validation_clips.json")
ev = events[events.piece.isin(pieces)]
ev_t = {s: g.time.to_numpy() for s, g in ev.groupby("song")}
ev_p = {s: g.piece.map({q: j for j, q in enumerate(pieces)}).to_numpy() for s, g in ev.groupby("song")}
bt = {s: g.sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
dbt = {s: g[g.downbeat].sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
dur = manifest.set_index("song").duration
all_songs = sorted({c["song"] for c in listed})
songs = [s for s in all_songs if s in ev_t and len(bt.get(s, [])) >= 2]
if args.trial:
    songs = songs[:12]
starts = {"fixed": {s: [c["start"] for c in listed if c["song"] == s] for s in songs},
          "whole": {s: list(np.arange(0, float(dur[s]) - clip_s + 1e-9, clip_s)) for s in songs}}


@torch.no_grad()
def hits_for(windows):
    """windows: list of (song, start) -> per window, hits when hearing its own band."""
    out = []
    for i in range(0, len(windows), 16):
        audio = [val.clip(s, st)["audio"] for s, st in windows[i:i + 16]]
        probs = model.probs(model.ear(torch.from_numpy(np.stack(audio))))
        out += [hits_from_probs(p, cfg["training"]["hit_threshold"], first) for p in probs.numpy()]
    return out


# per song and window: hits hearing its own band; the "other band" is the next song's window at the same index
# (wrapping over its windows), as in feasibility.py where each clip heard the next song's clip
own = {}
for kind in ("fixed", "whole"):
    windows = [(s, st) for s in songs for st in starts[kind][s]]
    h = hits_for(windows) if not args.trial else hits_for(windows)
    own[kind] = {}
    k = 0
    for s in songs:
        own[kind][s] = h[k:k + len(starts[kind][s])]
        k += len(starts[kind][s])
    print(f"{kind}: {len(windows)} windows run", flush=True)

next_song = {s: songs[(i + 1) % len(songs)] for i, s in enumerate(songs)}
gaps = {}
options = {"1 clip": ("fixed", 1), "2 clips": ("fixed", 2), "4 clips (locked)": ("fixed", 4), "whole song": ("whole", None)}
for name, (kind, k) in options.items():
    per_song = []
    for i, s in enumerate(tqdm(songs, desc=f"scoring ({name})", unit="song")):
        st = starts[kind][s][:k] if k else starts[kind][s]
        fs = all_songs[(all_songs.index(s) + 1) % len(all_songs)]           # the floor song, as locked
        song = Song(st, bt[s], dbt.get(s, []), (ev_t[s], ev_p[s]),
                    (ev_t.get(fs, np.zeros(0)), ev_p.get(fs, np.zeros(0, int))), bt[fs], float(dur[fs]), cfg)
        if not song.kept("beat_alignment"):
            continue
        mine = own[kind][s][:len(st)]
        theirs = own[kind][next_song[s]]
        other = [theirs[j % len(theirs)] for j in range(len(st))]
        a = song.scores([(st[j], *mine[j]) for j in range(len(st))])["beat_alignment"]
        b = song.scores([(st[j], *other[j]) for j in range(len(st))])["beat_alignment"]
        per_song.append(a - b)
    g = np.array(per_song)
    gaps[name] = {"songs": len(g), "headroom": float(g.mean()),
                  "sigma_m": float(g.std(ddof=1) / np.sqrt(2) / np.sqrt(TEST_SONGS))}

rng = np.random.default_rng(0)
sims = 2_000 if args.trial else 20_000
report = {"options": {}}
print(f"\nbeat alignment, feasibility model, validation songs; noise is one network's sd on {TEST_SONGS} test songs")
print(f"{'material':<18}{'controls':>9}{'headroom':>10}{'noise':>7}{'ratio':>7}{'false wins':>12}"
      + "".join(f"{'80% needs, spread ' + str(sp):>26}" for sp in SPREADS))
for name, g in gaps.items():
    for n_c in (5, 10):
        H, sm = g["headroom"], g["sigma_m"]
        fw = win_chance(0.0, 0.0, sm, n_c, sims * 5, rng)
        need = [needed(H, sp * H, sm, n_c, sims, rng) for sp in SPREADS]
        report["options"][f"{name}, {n_c} controls"] = {**g, "controls": n_c, "false_wins": fw,
                                                        "needed_for_80": dict(zip(map(str, SPREADS), need))}
        cells = "".join(f"{(f'{n:.2f} ({n / H:.0%})' if n is not None else '> 3x headroom'):>26}" for n in need)
        print(f"{name:<18}{n_c:>9}{H:>10.2f}{sm:>7.2f}{H / sm:>7.1f}{fw:>12.1%}{cells}")

if args.trial:
    sys.exit("trial run: nothing written")
(reports_dir() / "power_options.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(f"wrote {reports_dir() / 'power_options.json'}")
