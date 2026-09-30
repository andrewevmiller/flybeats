"""Phase 3, step 8: time 50 training batches on 16 s clips and write reports/cost.json.

    .venv\\Scripts\\python.exe scripts\\measure_cost.py --set baby [--segment 800] [--batches 50]
Hours per run = steps x seconds per batch / 3600. Uses the GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import reports_dir  # noqa: E402
from flybeats.model.training import setup, to_tensors, train_step  # noqa: E402
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], default="baby")
ap.add_argument("--batches", type=int, default=50)
ap.add_argument("--segment", type=int, default=0)
ap.add_argument("--device", default="cuda")
args = ap.parse_args()
s = setup(args.set, None, args.device)
model, weights = s["model"], s["weights"]
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
cuda = args.device.startswith("cuda")
sync = torch.cuda.synchronize if cuda else (lambda: None)

bar = tqdm(total=3 + args.batches, unit="batch", desc="warm-up")
for _ in range(3):                                            # warm-up, not timed
    bar.update(1)
    train_step(model, opt, to_tensors(s["train"].batch(8), args.device), weights, args.segment)
if cuda:
    torch.cuda.reset_peak_memory_stats()
data_s = compute_s = 0.0
bar.set_description("timing")
for _ in range(args.batches):
    t = time.perf_counter()
    batch = to_tensors(s["train"].batch(8), args.device)
    sync()
    data_s += time.perf_counter() - t
    t = time.perf_counter()
    train_step(model, opt, batch, weights, args.segment)
    sync()
    compute_s += time.perf_counter() - t
    bar.update(1)
bar.close()

per_batch = (data_s + compute_s) / args.batches
cost = {"device": torch.cuda.get_device_name(0) if cuda else "cpu", "batches_timed": args.batches, "batch_size": 8,
        "clip_seconds": s["cfg"]["training_data"]["clip_seconds"], "segment_frames": args.segment or "whole clip",
        "seconds_per_batch": round(per_batch, 3), "data_seconds_per_batch": round(data_s / args.batches, 3),
        "compute_seconds_per_batch": round(compute_s / args.batches, 3),
        "hours_per_1000_batches": round(per_batch * 1000 / 3600, 2),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if cuda else None}
(reports_dir() / "cost.json").write_text(json.dumps(cost, indent=2) + "\n", encoding="utf-8")
print(json.dumps(cost, indent=2))
