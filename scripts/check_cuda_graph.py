"""Check that the CUDA-graph training step (GraphedStep) trains exactly like the ordinary one (train_step), before
memorisation or training switch to it, and measure how much faster it is.

From the same start (seeded model, the same 8 fixed clips as memorise.py) it trains --steps steps three ways:
  A  ordinary step
  B  ordinary step again   how far two ordinary runs already differ: some GPU sums add in a varying order
  C  recorded step         GraphedStep
and compares the loss at every step and every learned weight at the end. Passes if C differs from A by no more
than 10 x the A-B difference, or by a relative 1e-5 where A and B agree exactly (a recorded kernel may round its
last bit differently). Writes reports/cuda_graph_check.json with a fingerprint of the model and training code:
memorise.py uses the recorded step only when this check passed on exactly that code.

    .venv\\Scripts\\python.exe scripts\\check_cuda_graph.py [--steps 20]
Uses the GPU: about 2 x steps x 8.5 s for A and B, then C, plus a minute or two of setup.
"""
import argparse
import copy
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, reports_dir  # noqa: E402
from flybeats.model.training import (GraphedStep, make_optimiser, setup, step_code_fingerprint, to_tensors,  # noqa: E402
                                     train_step)
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=20)
args = ap.parse_args()

cfg = load_locked()
torch.manual_seed(cfg["model"]["init_seed"])
s = setup("baby", None, "cuda", cfg)
model, loss_of, train = s["model"], s["loss"], s["train"]
clip_norm = cfg["training"]["grad_clip_norm"]

rng = np.random.default_rng(7)                              # the same clips as memorise.py
songs = sorted(train.songs.index)[:4]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s))) for song in songs * 2]
batch = to_tensors(clips, "cuda")
start = copy.deepcopy(model.state_dict())


def run(label, graphed):
    model.load_state_dict(start)                            # copies into the same tensors: nothing else moves
    opt = make_optimiser(model, cfg)
    record_s, step = None, None
    if graphed:
        torch.cuda.empty_cache()
        t0 = time.time()
        step = GraphedStep(model, opt, batch, loss_of, clip_norm)
        torch.cuda.synchronize()
        record_s = time.time() - t0
        print(f"recorded the step in {record_s:.0f} s")
    losses, times = [], []
    for _ in tqdm(range(args.steps), desc=label, unit="step"):
        t = time.time()
        stats = step() if graphed else train_step(model, opt, batch, loss_of, clip_norm)
        times.append(time.time() - t)                       # float() in both waits for the GPU to finish
        losses.append(stats["loss"])
    weights = {k: p.detach().clone() for k, p in model.named_parameters()}
    return {"losses": losses, "weights": weights, "times": times, "record_s": record_s}


def loss_diff(a, b):
    return float(np.max(np.abs(np.array(a["losses"]) - np.array(b["losses"])) / np.abs(np.array(a["losses"]))))


def weight_diff(a, b):
    """Largest difference in any learned tensor, relative to that tensor's largest value."""
    return max(float((a["weights"][k] - b["weights"][k]).abs().max() / a["weights"][k].abs().max().clamp_min(1e-12))
               for k in a["weights"])


A = run("A ordinary", False)
B = run("B ordinary again", False)
try:
    C = run("C recorded", True)
except Exception as e:                                       # recording can fail on an operation it cannot capture
    print(f"FAIL: could not record the step: {e!r}")
    sys.exit(1)

noise = {"loss": loss_diff(A, B), "weights": weight_diff(A, B)}
graph = {"loss": loss_diff(A, C), "weights": weight_diff(A, C)}
allowed = {k: max(10 * noise[k], 1e-5) for k in noise}
passes = all(graph[k] <= allowed[k] for k in graph)
eager_s = statistics.median(A["times"][1:] + B["times"][1:])
graph_s = statistics.median(C["times"][1:])
report = {"steps": args.steps, "passes": passes, "ordinary_vs_ordinary": noise, "recorded_vs_ordinary": graph,
          "allowed": allowed, "ordinary_s_per_step": eager_s, "recorded_s_per_step": graph_s,
          "speedup": eager_s / graph_s, "record_s": C["record_s"],
          "peak_gpu_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
          "losses": {"A": A["losses"], "B": B["losses"], "C": C["losses"]}, "version": cfg["version"],
          "code": step_code_fingerprint()}        # memorise.py and train.py use the recorded step only for this code
(reports_dir() / "cuda_graph_check.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

print(f"\nordinary step {eager_s:.2f} s, recorded step {graph_s:.2f} s: {eager_s / graph_s:.2f}x as fast "
      f"(recording took {C['record_s']:.0f} s, once per run)")
for k, name in (("loss", "loss, every step"), ("weights", "weights at the end")):
    print(f"{name}: largest relative difference  ordinary vs ordinary {noise[k]:.1e}, "
          f"recorded vs ordinary {graph[k]:.1e}  (allowed {allowed[k]:.1e})")
print(f"peak GPU memory {report['peak_gpu_memory_gb']:.2f} GB")
print("PASS: the recorded step trains like the ordinary one" if passes else
      "FAIL: the recorded step does not match the ordinary one; do not use it")
sys.exit(0 if passes else 1)
