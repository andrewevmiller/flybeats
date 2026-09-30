"""Phase 3, step 6: train the fly network (or a random control) on clips from the sampler.

    .venv\\Scripts\\python.exe scripts\\train.py --set baby --run baby_real
    .venv\\Scripts\\python.exe scripts\\train.py --set full --control 1 --run full_control1

Every training setting (optimiser, learning rate and schedule, batch size, gradient clipping, steps,
validation interval, stop rule) comes from the training section of config/locked.yaml, so the real network
and every control train identically; there are no command-line overrides. At each validation: loss and
per-drum hit F1 on the fixed validation clips, plus the silence test; the run stops if activity is unbounded
or no longer settles. Keeps the checkpoint with the best mean hit F1.
Writes runs/<run>/log.jsonl, best.pt and last.pt.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import ROOT, load_locked  # noqa: E402
from flybeats.data.sampler import load_validation_clips  # noqa: E402
from flybeats.data.manifest import set_dir  # noqa: E402
from flybeats.model.training import evaluate, make_optimiser, setup, silence_test, to_tensors, train_step  # noqa: E402
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--control", type=int, default=None)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()

cfg = load_locked()
tc = cfg["training"]
if tc["schedule"] != "cosine_to_zero" or tc["backprop"] != "whole_clip":
    sys.exit(f"train.py implements cosine_to_zero and whole_clip only, not {tc['schedule']} / {tc['backprop']}")
steps = tc["steps"][args.set]
if steps is None:
    sys.exit(f"training.steps.{args.set} is not set in config/locked.yaml; set it (new tag + CHANGELOG line) first")
torch.manual_seed(cfg["model"]["init_seed"])
run_dir = ROOT / "runs" / args.run
run_dir.mkdir(parents=True, exist_ok=False)
s = setup(args.set, args.control, args.device, cfg)
model, loss_of = s["model"], s["loss"]
val_clips = [s["val"].clip(c["song"], c["start"]) for c in
             load_validation_clips(set_dir(s["paths"]["work_dir"], args.set) / "validation_clips.json")]
opt = make_optimiser(model, cfg)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
(run_dir / "args.json").write_text(json.dumps(vars(args) | {"version": cfg["version"], "steps": steps, "training": tc,
                                                    "eigenvalue": model.eigenvalue, "g": model.g}, indent=1))

best = -1.0
t0 = time.time()
bar = tqdm(range(1, steps + 1), unit="step", desc="training")
for step in bar:
    stats = train_step(model, opt, to_tensors(s["train"].batch(tc["batch_clips"]), args.device), loss_of,
                       tc["grad_clip_norm"])
    sched.step()
    rec = {"step": step, "lr": sched.get_last_lr()[0], **stats, "elapsed_s": round(time.time() - t0, 1),
           "t": round(time.time(), 1)}
    if step % tc["validate_every"] == 0 or step == steps:
        model.eval()
        with tqdm(total=len(val_clips) + 4, unit="clip", desc="validating", leave=False) as vbar:
            rec["val"] = evaluate(model, val_clips, loss_of, cfg, progress=vbar.update)
            rec["silence"] = silence_test(model, val_clips[:4], cfg, progress=vbar.update)
        model.train()
        if rec["val"]["mean_f1"] > best:
            best = rec["val"]["mean_f1"]
            torch.save({"model": model.state_dict(), "step": step, "val": rec["val"]}, run_dir / "best.pt")
        bar.write(f"step {step}: loss {stats['loss']:.4f}  val mean F1 {rec['val']['mean_f1']:.3f}  "
              f"settled {rec['silence']['settled_ratio']:.3f}  responding {rec['silence']['responding_share']:.2f}")
    bar.set_postfix(loss=f"{stats['loss']:.4f}", best_f1=f"{max(best, 0):.3f}")
    log.write(json.dumps(rec) + "\n")
    log.flush()
    if "silence" in rec and not (rec["silence"]["bounded"]
                                 and rec["silence"]["settled_ratio"] < cfg["silence_test"]["settle_fraction"]):
        bar.write("activity no longer settles after music: stopping the run")
        break
bar.close()
torch.save({"model": model.state_dict(), "step": step, "opt": opt.state_dict()}, run_dir / "last.pt")
print(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h, best mean F1 {best:.3f}; {run_dir}")
