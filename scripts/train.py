"""Phase 3, step 6: train the fly network (or a random control) on clips from the sampler.

    .venv\\Scripts\\python.exe scripts\\train.py --set baby --steps 1000 --run baby_real
    .venv\\Scripts\\python.exe scripts\\train.py --set full --steps 20000 --control 1 --run full_control1

Adam at 1e-3 with cosine decay, gradient norm clipped at 1.0, batches of 8 clips from 8 songs.
Every 500 steps: loss and per-drum hit F1 on the fixed validation clips, plus the silence test; the run
stops if activity no longer settles. Keeps the checkpoint with the best mean hit F1.
Writes runs/<run>/log.jsonl, best.pt and last.pt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import ROOT  # noqa: E402
from flybeats.data.sampler import load_validation_clips  # noqa: E402
from flybeats.data.manifest import set_dir  # noqa: E402
from flybeats.model.training import evaluate, setup, silence_test, to_tensors, train_step  # noqa: E402
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--control", type=int, default=None)
ap.add_argument("--segment", type=int, default=0, help="frames per backprop segment; 0 = the whole clip")
ap.add_argument("--val-every", type=int, default=500)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

torch.manual_seed(0)
run_dir = ROOT / "runs" / args.run
run_dir.mkdir(parents=True, exist_ok=False)
s = setup(args.set, args.control, args.device)
model, cfg, weights = s["model"], s["cfg"], s["weights"]
frame_s = cfg["training_data"]["frame_ms"] / 1000
val_clips = [s["val"].clip(c["song"], c["start"]) for c in
             load_validation_clips(set_dir(s["paths"]["work_dir"], args.set) / "validation_clips.json")]
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
(run_dir / "args.json").write_text(json.dumps(vars(args) | {"eigenvalue": model.eigenvalue, "g": model.g}, indent=1))

best = -1.0
t0 = time.time()
bar = tqdm(range(1, args.steps + 1), unit="step", desc="training")
for step in bar:
    stats = train_step(model, opt, to_tensors(s["train"].batch(8), args.device), weights, args.segment)
    sched.step()
    rec = {"step": step, "lr": sched.get_last_lr()[0], **stats, "elapsed_s": round(time.time() - t0, 1)}
    if step % args.val_every == 0 or step == args.steps:
        model.eval()
        with tqdm(total=len(val_clips) + 4, unit="clip", desc="validating", leave=False) as vbar:
            rec["val"] = evaluate(model, val_clips, weights, frame_s, progress=vbar.update)
            rec["silence"] = silence_test(model, val_clips[:4], frame_s, progress=vbar.update)
        model.train()
        if rec["val"]["mean_f1"] > best:
            best = rec["val"]["mean_f1"]
            torch.save({"model": model.state_dict(), "step": step, "val": rec["val"]}, run_dir / "best.pt")
        bar.write(f"step {step}: loss {stats['loss']:.4f}  val mean F1 {rec['val']['mean_f1']:.3f}  "
              f"settled {rec['silence']['settled_ratio']:.3f}  responding {rec['silence']['responding_share']:.2f}")
    bar.set_postfix(loss=f"{stats['loss']:.4f}", best_f1=f"{max(best, 0):.3f}")
    log.write(json.dumps(rec) + "\n")
    log.flush()
    if "silence" in rec and not (rec["silence"]["bounded"] and rec["silence"]["settled_ratio"] < 0.05):
        bar.write("activity no longer settles after music: stopping the run")
        break
bar.close()
torch.save({"model": model.state_dict(), "step": step, "opt": opt.state_dict()}, run_dir / "last.pt")
print(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h, best mean F1 {best:.3f}; {run_dir}")
