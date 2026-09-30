"""Phase 3, step 6: train the fly network (or a random control) on clips from the sampler.

    .venv\\Scripts\\python.exe scripts\\train.py --set baby --run baby_real
    .venv\\Scripts\\python.exe scripts\\train.py --set full --control 1 --run full_control1

Every training setting (optimiser, learning rate and schedule, batch size, gradient clipping, steps,
validation interval, stop rule) comes from the training section of config/locked.yaml, so the real network
and every control train identically; there are no command-line overrides. At each validation: loss and
per-drum hit F1 on the fixed validation clips, plus the silence test; the run stops if activity is unbounded
or no longer settles. Keeps the checkpoint with the best mean hit F1.
Writes runs/<run>/log.jsonl, best.pt and last.pt.
Each step is the recorded one (a CUDA graph, training.GraphedStep, given each new batch through load()) once
scripts/check_cuda_graph.py has shown on this exact code that it trains like the ordinary one, including with a
new batch every step; until then, or with --no-graph, the ordinary step. args.json and the start of the run say which.
The batches are the same, in the same order, either way: the first is drawn once and used for recording and step 1.
While the GPU runs a step, a background thread already makes the next batches (reading and mixing the audio): one
thread, drawing from the sampler in order, so the batches and their order are exactly those without it. The log's
data_s is how long each step still waited for its batch.
"""
import argparse
import json
import queue
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import ROOT, load_locked, reports_dir  # noqa: E402
from flybeats.data.sampler import load_validation_clips  # noqa: E402
from flybeats.data.manifest import set_dir  # noqa: E402
from flybeats.model.training import (GraphedStep, evaluate, graph_checked, make_optimiser, make_schedule, setup,  # noqa: E402
                                     silence_test, to_tensors, train_step)
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], required=True)
ap.add_argument("--run", required=True)
ap.add_argument("--control", type=int, default=None)
ap.add_argument("--device", default="cuda")
ap.add_argument("--no-graph", action="store_true", help="the ordinary training step, not the recorded CUDA graph")
args = ap.parse_args()


class Prefetch(threading.Thread):
    """Makes the sampler's next batches (as numpy clips) ahead of time, in order, keeping at most `ahead` waiting.
    Only this thread draws from the sampler once it starts, so the sequence is the one the sampler would give
    anyway. A failure while making a batch is raised in the training loop when that batch is asked for."""

    def __init__(self, sampler, size, ahead=2):
        super().__init__(daemon=True)
        self.sampler, self.size = sampler, size
        self.ready, self.stop = queue.Queue(maxsize=ahead), threading.Event()

    def run(self):
        while not self.stop.is_set():
            try:
                item = self.sampler.batch(self.size)
            except BaseException as e:                # handed to the loop, which raises it
                item = e
            while not self.stop.is_set():
                try:
                    self.ready.put(item, timeout=1)
                    break
                except queue.Full:
                    continue
            if isinstance(item, BaseException):
                return

    def next(self):
        item = self.ready.get()
        if isinstance(item, BaseException):
            raise RuntimeError("making a training batch failed") from item
        return item


def graph_for_training():
    """(use the recorded step?, why): memorisation's check, plus its part with a new batch every step."""
    if args.no_graph:
        return False, "turned off (--no-graph)"
    if args.device != "cuda":
        return False, "not on the GPU"
    ok, why = graph_checked()
    if not ok:
        return ok, why
    new = json.loads((reports_dir() / "cuda_graph_check.json").read_text(encoding="utf-8")).get("new_batches")
    if not new:
        return False, "checked only on a fixed batch; run scripts/check_cuda_graph.py again to check new batches"
    return (True, why) if new["passes"] else (False, "its check with a new batch every step failed")

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
sched = make_schedule(opt, steps, cfg)
first = to_tensors(s["train"].batch(tc["batch_clips"]), args.device)   # step 1's batch, also the one recorded with
use_graph, why = graph_for_training()
graphed = GraphedStep(model, opt, first, loss_of, tc["grad_clip_norm"]) if use_graph else None
print(f"training step: {'recorded (CUDA graph)' if use_graph else 'ordinary'}, {why}")
log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
(run_dir / "args.json").write_text(json.dumps(vars(args) | {"version": cfg["version"], "steps": steps, "training": tc,
                                                    "eigenvalue": model.eigenvalue, "g": model.g,
                                                    "cuda_graph": use_graph}, indent=1))

prefetch = Prefetch(s["train"], tc["batch_clips"])    # draws batch 2 onwards, after `first`
prefetch.start()
best = -1.0
t0 = time.time()
bar = tqdm(range(1, steps + 1), unit="step", desc="training")
for step in bar:
    t_data = time.time()
    batch = first if step == 1 else to_tensors(prefetch.next(), args.device)
    data_s = time.time() - t_data                    # how long this step waited for its batch
    if graphed:
        graphed.load(batch)
        stats = graphed()
    else:
        stats = train_step(model, opt, batch, loss_of, tc["grad_clip_norm"])
    sched.step()
    rec = {"step": step, "lr": sched.get_last_lr()[0], **stats, "data_s": round(data_s, 2),
           "elapsed_s": round(time.time() - t0, 1), "t": round(time.time(), 1)}
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
prefetch.stop.set()
torch.save({"model": model.state_dict(), "step": step, "opt": opt.state_dict()}, run_dir / "last.pt")
print(f"done: {step} steps in {(time.time() - t0) / 3600:.2f} h, best mean F1 {best:.3f}; {run_dir}")
