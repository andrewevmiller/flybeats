"""Phase 3, step 9: the memorisation check. Train on fixed clips from 4 BabySlakh training songs until
kick, snare and closed hi-hat reach hit F1 >= 0.9 on those same clips. If it cannot, something is broken.
Writes reports/memorise.json.

    .venv\\Scripts\\python.exe scripts\\memorise.py [--steps 3000] [--control 1]
Uses the GPU.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import reports_dir  # noqa: E402
from flybeats.model.training import evaluate, setup, to_tensors, train_step  # noqa: E402
from tqdm import tqdm  # noqa: E402

MUST_PASS = ["kick", "snare", "hihat_closed"]

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--control", type=int, default=None)
ap.add_argument("--songs", type=int, default=4)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
s = setup("baby", args.control, args.device)
model, weights, train = s["model"], s["weights"], s["train"]
frame_s = s["cfg"]["training_data"]["frame_ms"] / 1000

rng = np.random.default_rng(7)
songs = sorted(train.songs.index)[:args.songs]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s))) for song in songs * 2]
batch = to_tensors(clips, args.device)                      # 8 fixed clips, 2 per song, no stem dropout
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

result = None
bar = tqdm(range(1, args.steps + 1), unit="step", desc="memorising")
for step in bar:
    stats = train_step(model, opt, batch, weights)
    bar.set_postfix(loss=f"{stats['loss']:.4f}")
    if step % 250 == 0 or step == args.steps:
        model.eval()
        result = evaluate(model, clips, weights, frame_s)
        model.train()
        bar.write(f"step {step}: loss {stats['loss']:.4f}  F1 {result['f1']}")
        if all(result["f1"][p] >= 0.9 for p in MUST_PASS):
            break
bar.close()
passed = all(result["f1"][p] >= 0.9 for p in MUST_PASS)
report = {"songs": songs, "steps": step, "f1": result["f1"], "passes": passed, "control": args.control}
(reports_dir() / "memorise.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print("PASS" if passed else "FAIL: the model cannot memorise; something is broken")
sys.exit(0 if passed else 1)
