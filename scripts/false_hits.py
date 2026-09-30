"""What a memorisation run's false hits are, from its final weights (reports/memorise/<name>/final.pt). CPU, read-only.
Rebuilds memorise.py's model and 8 clips, loads the final weights (including the readout scaling), rescores exactly as
the check does, and sorts every unmatched prediction:
  duplicate   within tolerance of a real hit of that drum that another prediction already took
  near miss   no real hit of that drum within tolerance, but one within 2x tolerance (timing slightly off)
  other drum  none of that drum within 2x tolerance, but a real hit of another drum within tolerance
  stray       nothing real nearby

    .venv\\Scripts\\python.exe scripts\\false_hits.py [real | control-N]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)                                   # leave most of the CPU to any run that is going
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, reports_dir  # noqa: E402
from flybeats.model.loss import peaks, tolerance_frames  # noqa: E402
from flybeats.model.training import setup, to_tensors  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name", nargs="?", default="real", help="the run: real or control-N")
args = ap.parse_args()
control = None if args.name == "real" else int(args.name.split("-")[1])
t0 = time.time()
cfg = load_locked()
mc = cfg["memorisation"]
torch.manual_seed(cfg["model"]["init_seed"])
s = setup("baby", control, "cpu", cfg)
model, train = s["model"], s["train"]
ck = torch.load(reports_dir() / "memorise" / args.name / "final.pt", map_location="cpu")
model.load_state_dict(ck["model"])
model.eval()
rng = np.random.default_rng(mc["clip_seed"])
songs = sorted(train.songs.index)[:mc["songs"]]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s)))
         for song in songs * mc["clips_per_song"]]
b = to_tensors(clips, "cpu")
print(f"set up in {time.time() - t0:.0f} s; final weights from step {ck['step']}; running the 8 clips...")
with torch.no_grad():
    out, _ = model(b["audio"])
prob = torch.sigmoid(out["hit_logits"]).numpy()
hits, mask = b["hits"].numpy(), b["mask"].numpy()
frame_s = cfg["training_data"]["frame_ms"] / 1000
frac = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
thr = cfg["training"]["hit_threshold"]
pieces = model.pieces
kinds = ["duplicate", "near miss", "other drum", "stray"]
tot = {p: {"tp": 0, "fp": 0, "fn": 0, **{k: 0 for k in kinds}, "peaks_per_found_hit": []} for p in pieces}
for i in range(len(clips)):
    live = np.flatnonzero(mask[i])
    true_all = {d: np.intersect1d(np.flatnonzero(hits[i, :, d]), live) for d in range(len(pieces))}
    for d, p in enumerate(pieces):
        pred = np.intersect1d(peaks(prob[i, :, d], thr), live)
        true = true_all[d]
        tol = tolerance_frames(true, clips[i]["beats"], frame_s, frac)
        used = np.zeros(len(true), bool)
        taken_by = [[] for _ in true]
        unmatched = []
        for q in pred:                                     # the check's greedy one-to-one matching
            if len(true):
                dist = np.abs(true - q).astype(float)
                dist[used | (dist > tol)] = np.inf
                k = int(np.argmin(dist))
                if np.isfinite(dist[k]):
                    used[k] = True
                    taken_by[k].append(q)
                    continue
            unmatched.append(q)
        t = tot[p]
        t["tp"] += int(used.sum()); t["fp"] += len(unmatched); t["fn"] += int((~used).sum())
        for q in unmatched:
            dist = np.abs(true - q) if len(true) else np.array([np.inf])
            near_tol = tol if len(true) else np.array([np.inf])
            if len(true) and np.any((dist <= near_tol) & used):
                t["duplicate"] += 1
            elif len(true) and np.any(dist <= 2 * near_tol):
                t["near miss"] += 1
            elif any(len(true_all[e]) and np.any(np.abs(true_all[e] - q) <=
                     tolerance_frames(true_all[e], clips[i]["beats"], frame_s, frac)) for e in range(len(pieces)) if e != d):
                t["other drum"] += 1
            else:
                t["stray"] += 1
        for k, q in enumerate(true):                       # how many peaks sit within tolerance of each found hit
            if used[k]:
                t["peaks_per_found_hit"].append(int(np.sum(np.abs(pred - q) <= tol[k])))

print(f"\n{'drum':<13} {'F1':>5} {'found':>5} {'false':>5} {'missed':>6}   false hits by kind: "
      + "  ".join(f"{k}" for k in kinds) + "   peaks per found hit")
all_k = {k: 0 for k in kinds}
for p, t in tot.items():
    f1 = 2 * t["tp"] / max(2 * t["tp"] + t["fp"] + t["fn"], 1)
    share = "  ".join(f"{t[k] / max(t['fp'], 1):>{len(k)}.0%}" for k in kinds)
    ppf = np.mean(t["peaks_per_found_hit"]) if t["peaks_per_found_hit"] else float("nan")
    print(f"{p:<13} {f1:5.3f} {t['tp']:>5} {t['fp']:>5} {t['fn']:>6}   {share}   {ppf:.2f}")
    for k in kinds:
        all_k[k] += t[k]
n = sum(all_k.values())
print(f"\nall drums: {n} false hits = " + ", ".join(f"{k} {v} ({v / n:.0%})" for k, v in all_k.items()))
print(f"(took {time.time() - t0:.0f} s)")
