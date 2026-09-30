"""Phase 3, step 7: stability on real audio. 20 random training clips, each followed by 4 s of silence,
through the untrained network at overall gains from 0.5x to 2.0x. Writes reports/stability.json.

    .venv\\Scripts\\python.exe scripts\\stability.py --set baby [--control 1] [--device cuda]
Uses the GPU unless --device cpu (much slower).
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import reports_dir  # noqa: E402
from flybeats.model.training import setup, silence_test  # noqa: E402
from tqdm import tqdm  # noqa: E402

GAINS = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], default="baby")
ap.add_argument("--control", type=int, default=None)
ap.add_argument("--device", default="cuda")
ap.add_argument("--clips", type=int, default=20)
args = ap.parse_args()
s = setup(args.set, args.control, args.device)
model, cfg = s["model"], s["cfg"]
frame_s = cfg["training_data"]["frame_ms"] / 1000
clips = []
while len(clips) < args.clips:
    clips += s["train"].batch(min(8, len(s["train"].songs)))
clips = clips[:args.clips]

results = {}
bar = tqdm(total=len(GAINS) * len(clips), unit="clip", desc="stability")
for gain in GAINS:
    model.gain_scale = gain
    bar.set_postfix_str(f"gain {gain}x")
    r = silence_test(model, clips, frame_s, batch_size=4, progress=bar.update)
    results[str(gain)] = r
    bar.write(f"gain {gain:>4}x  {'PASS' if r['passes'] else 'fail'}  bounded={r['bounded']}  "
          f"settled={r['settled_ratio']:.3f}  responding={r['responding_share']:.2f}  per drum={r['responding_per_drum']}")

bar.close()
initial = cfg["model"]["initial_gain"]
passing = [g for g in GAINS if results[str(g)]["passes"]]
if results.get(str(1.0), {}).get("passes"):
    advice = f"1.0x passes: keep initial_gain: {initial}."
elif [g for g in passing if g < 1.0]:
    best = max(g for g in passing if g < 1.0)
    advice = f"1.0x fails; the largest passing gain below it is {best}x. Set initial_gain: {initial * best} and log it in config/CHANGELOG.md."
else:
    advice = "No gain below 1.0x passes. Stop and investigate before training."
print(advice)
name = "stability.json" if args.control is None else f"stability_control{args.control}.json"
(reports_dir() / name).write_text(json.dumps({"set": args.set, "control": args.control, "clips": len(clips),
                                              "gains": results, "advice": advice}, indent=2) + "\n", encoding="utf-8")
