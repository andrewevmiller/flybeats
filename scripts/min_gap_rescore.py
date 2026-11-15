"""Spec 5 (fix 05), first step: a minimum gap between peaks, rescored on saved weights. CPU, read-only.

    .venv\\Scripts\\python.exe scripts\\min_gap_rescore.py [real | control-N] [--gap 10]

Rebuilds the memorisation check's model and 8 clips, loads the run's final weights (reports/memorise/<name>/final.pt),
and scores them twice: with the locked peak rule (gap 0, which must reproduce the run's memorise.json), and with the
minimum-gap rule: peaks above the threshold are taken highest first, and any peak within --gap frames of a kept peak
is dropped. Matching, tolerance and F1 are the locked ones. Nothing is written except reports/min_gap_rescore.json.

This changes the scoring rule. It is a measurement, not an adoption: the scorecard uses the locked rule, so a gap in
use needs its own tag and disclosure (see reports/fix-specs-proposal.md, spec 5).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(4)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, reports_dir  # noqa: E402
from flybeats.model.loss import match_counts, peaks, tolerance_frames  # noqa: E402
from flybeats.model.training import setup, to_tensors  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("name", nargs="?", default="real", help="the run: real or control-N")
ap.add_argument("--gap", type=int, default=10, help="minimum gap in frames (10 = 50 ms, fixed in the spec)")
args = ap.parse_args()
control = None if args.name == "real" else int(args.name.split("-")[1])
t0 = time.time()
cfg = load_locked()
mc = cfg["memorisation"]
torch.manual_seed(cfg["model"]["init_seed"])
s = setup("baby", control, "cpu", cfg)
model, train = s["model"], s["train"]
ck = torch.load(reports_dir() / "memorise" / args.name / "final.pt", map_location="cpu", weights_only=False)
model.load_state_dict(ck["model"])
model.eval()
rng = np.random.default_rng(mc["clip_seed"])
songs = sorted(train.songs.index)[:mc["songs"]]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s)))
         for song in songs * mc["clips_per_song"]]
b = to_tensors(clips, "cpu")
with torch.no_grad():
    out, _ = model(b["audio"])
prob = torch.sigmoid(out["hit_logits"]).numpy()
hits, mask = b["hits"].numpy(), b["mask"].numpy()
frame_s = cfg["training_data"]["frame_ms"] / 1000
frac = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
thr = cfg["training"]["hit_threshold"]
pieces = model.pieces


def min_gap_peaks(p, threshold, gap):
    """Locked peaks, highest first; a peak within `gap` frames of a kept one is dropped. gap 0 = the locked rule."""
    pk = peaks(p, threshold)
    if gap <= 0 or len(pk) < 2:
        return pk
    kept = []
    for q in pk[np.argsort(-p[pk], kind="stable")]:
        if all(abs(int(q) - k) > gap for k in kept):
            kept.append(int(q))
    return np.sort(np.array(kept, dtype=pk.dtype))


def f1_by_drum(gap):
    tp = np.zeros(len(pieces)); fp = np.zeros(len(pieces)); fn = np.zeros(len(pieces))
    for i in range(len(clips)):
        live = np.flatnonzero(mask[i])
        for d in range(len(pieces)):
            pred = np.intersect1d(min_gap_peaks(prob[i, :, d], thr, gap), live)
            true = np.intersect1d(np.flatnonzero(hits[i, :, d]), live)
            tol = tolerance_frames(true, clips[i]["beats"], frame_s, frac)
            a, f, m = match_counts(pred, true, tol)
            tp[d] += a; fp[d] += f; fn[d] += m
    return {p: round(float(2 * tp[d] / max(2 * tp[d] + fp[d] + fn[d], 1)), 3) for d, p in enumerate(pieces)}, \
           {p: {"found": int(tp[d]), "extra": int(fp[d]), "missed": int(fn[d])} for d, p in enumerate(pieces)}


locked_f1, _ = f1_by_drum(0)
gap_f1, gap_counts = f1_by_drum(args.gap)
saved = json.loads((reports_dir() / "memorise" / args.name / "memorise.json").read_text(encoding="utf-8"))["f1"]
check = {p: abs(locked_f1[p] - saved[p]) < 0.002 for p in ["kick", "snare", "hihat_closed"]}

rows = []
for p in ["kick", "snare", "hihat_closed"]:
    rows.append(f"{p:<13} locked {locked_f1[p]:.3f} (saved {saved[p]:.3f})   gap {args.gap}: {gap_f1[p]:.3f}   "
                f"change {gap_f1[p] - locked_f1[p]:+.3f}")
print(f"{args.name}, final weights from step {ck['step']}, set up in {time.time() - t0:.0f} s")
print("\n".join(rows))
print(f"gap-0 reproduces the saved F1: {all(check.values())}")
result = {"run": args.name, "gap_frames": args.gap, "weights_step": ck["step"],
          "locked_f1": locked_f1, "min_gap_f1": gap_f1, "min_gap_counts": gap_counts,
          "saved_f1": saved, "gap0_reproduces_saved": all(check.values())}
out = reports_dir() / "min_gap_rescore.json"
existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
existing[args.name] = result
out.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
