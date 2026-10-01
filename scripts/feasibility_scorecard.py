"""The feasibility model (scripts/feasibility.py; reports/feasibility/final.pt) scored with the locked scorecard
(prereg-v5.2) on the full set's validation clips, hearing each clip's own band and, as before, another song's band
(clip i hears clip i + 4's: the next song's). The gap between the two is the room the main test has on each facet:
how much hearing the right song helps an unrestricted model. No network is trained; CPU only.

The gap's 95% interval comes from resampling validation songs (2,000 resamples, seed 0), as in feasibility.py.
Writes reports/feasibility_scorecard.json (scripts/power_estimate.py --metric beat_alignment reads it).
    .venv\\Scripts\\python.exe scripts\\feasibility_scorecard.py
"""
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
from flybeats.scoring.scorecard import COUNTED, FACETS, Song, aggregate, check_scorecard, hits_from_probs  # noqa: E402

HIDDEN, LAYERS, SHIFT = 128, 2, 4                         # as in feasibility.py


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


torch.set_num_threads(4)
cfg, paths = load_locked(), load_paths()
check_scorecard(cfg)
sc, td = cfg["scorecard"], cfg["training_data"]
pieces = cfg["kit"]["pieces"]
first = round(td["warmup_ignored_seconds"] / (td["frame_ms"] / 1000))
manifest, events, beats = load_manifest(paths["work_dir"], "full")
val = ClipSampler(unpacked_root(paths["work_dir"], "full"), manifest, events, beats, cfg, "validation")
mean, std = model_ear.load_band_stats(set_dir(paths["work_dir"], "full") / "band_stats.json")
model = Ceiling(len(pieces), mean, std)
model.load_state_dict(torch.load(reports_dir() / "feasibility" / "final.pt", map_location="cpu")["model"])
model.eval()

listed = load_validation_clips(set_dir(paths["work_dir"], "full") / "validation_clips.json")
bands = []
with torch.no_grad():
    for c in tqdm(listed, desc="reading validation clips", unit="clip"):
        bands.append(model.ear(torch.from_numpy(val.clip(c["song"], c["start"])["audio"])[None])[0])
songs_of = [c["song"] for c in listed]
assert all(songs_of[i] != songs_of[(i + SHIFT) % len(listed)] for i in range(len(listed)))
hits = {"matched": [], "other_band": []}
with torch.no_grad():
    for name, src in (("matched", bands), ("other_band", [bands[(i + SHIFT) % len(bands)] for i in range(len(bands))])):
        for i in tqdm(range(0, len(src), 32), desc=f"running the model ({name})", unit="batch"):
            for p in model.probs(torch.stack(src[i:i + 32])).numpy():
                hits[name].append(hits_from_probs(p, cfg["training"]["hit_threshold"], first))

ev = events[events.piece.isin(pieces)]
ev_t = {s: g.time.to_numpy() for s, g in ev.groupby("song")}
ev_p = {s: g.piece.map({q: j for j, q in enumerate(pieces)}).to_numpy() for s, g in ev.groupby("song")}
bt = {s: g.sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
dbt = {s: g[g.downbeat].sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
dur = manifest.set_index("song").duration
songs = sorted(set(songs_of))
per_song = {}
for i, s in enumerate(tqdm(songs, desc="scoring songs", unit="song")):
    if s not in ev_t or len(bt.get(s, [])) < 2:
        continue
    fs = songs[(i + 1) % len(songs)]
    idx = [k for k, c in enumerate(listed) if c["song"] == s]
    song = Song([listed[k]["start"] for k in idx], bt[s], dbt.get(s, []), (ev_t[s], ev_p[s]),
                (ev_t.get(fs, np.zeros(0)), ev_p.get(fs, np.zeros(0, int))), bt[fs], float(dur[fs]), cfg)
    per_song[s] = {name: song.scores([(listed[k]["start"], *h[k]) for k in idx]) for name, h in hits.items()}

report = {"songs": len(per_song), "facets": {}}
agg = {name: aggregate([r[name] for r in per_song.values()], sc["musicianship_weights"], sc["failure_floor"])
       for name in hits}
rng = np.random.default_rng(0)
names = list(per_song)
print(f"\n{len(per_song)} validation songs; rescaled 0-100 (each facet over its kept songs)")
print(f"{'facet':<17}{'hears its song':>15}{'another band':>14}{'gap':>8}   95% interval     share under 20")
for f in FACETS:
    keep = [s for s in names if per_song[s]["matched"][f] is not None]
    a = np.array([per_song[s]["matched"][f] for s in keep])
    b = np.array([per_song[s]["other_band"][f] for s in keep])
    boots = [(a[j] - b[j]).mean() for j in (rng.integers(0, len(keep), len(keep)) for _ in range(2000))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    report["facets"][f] = {"matched": float(a.mean()), "other_band": float(b.mean()), "gap": float((a - b).mean()),
                           "interval_95": [float(lo), float(hi)], "songs": len(keep),
                           "failed_share_matched": agg["matched"][f]["failed_share"],
                           "rule": "main" if f == "beat_alignment" else "supporting" if f in COUNTED else "reported only"}
    print(f"{f:<17}{a.mean():>15.1f}{b.mean():>14.1f}{(a - b).mean():>+8.1f}   {lo:+5.1f} to {hi:+5.1f}"
          f"{agg['matched'][f]['failed_share']:>14.0%}")
report["musicianship"] = {name: agg[name]["musicianship"] for name in hits}
print(f"musicianship score (reported only): {agg['matched']['musicianship']:.1f} hearing its song, "
      f"{agg['other_band']['musicianship']:.1f} hearing another band")
report["per_song"] = per_song
(reports_dir() / "feasibility_scorecard.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(f"wrote {reports_dir() / 'feasibility_scorecard.json'}")
