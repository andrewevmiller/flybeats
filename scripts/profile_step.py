"""Where one training step's GPU time goes, to choose what to speed up. Changes nothing and trains nothing that is
kept: the model is set up as memorise.py sets it up (real network, the locked memorisation clips), a few steps are
taken and timed, and the result is printed and written to reports/profile_step.json.

  1. The recorded step (CUDA graph), timed as it runs in memorise.py and train.py.
  2. One ordinary step under the PyTorch profiler: GPU time per operation, grouped (the network's big multiply,
     the per-frame neuron update, the readout, and so on). The recorded step runs the same GPU work without Python
     in between, so the gap between (1) and the GPU total in (2) is time the GPU sat waiting.
  3. The big multiply on its own, one frame's worth, three ways: dense (as now), and sparse (only the real
     connections, CSR and COO), for the forward pass. A guide to how much a sparse network could save.

    .venv\\Scripts\\python.exe scripts\\profile_step.py
Uses the GPU for about 2 minutes, plus a minute or two of setup.
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, reports_dir  # noqa: E402
from flybeats.model.training import GraphedStep, make_optimiser, setup, to_tensors, train_step  # noqa: E402

GROUPS = [   # (group, words in the operation's name), first match wins
    ("big multiply (network connections)", ["mm", "matmul", "addmm", "bmm", "gemm"]),
    ("indexing and copying", ["index", "copy", "gather", "scatter", "cat", "stack", "zeros", "fill", "clone"]),
    ("per-frame neuron update (add, multiply, relu, ...)", ["add", "sub", "mul", "div", "relu", "threshold", "where",
                                                             "neg", "sigmoid", "exp", "clamp", "softplus"]),
    ("sums and means", ["sum", "mean", "norm", "max", "min"]),
]


def group_of(name):
    n = name.lower()
    for g, words in GROUPS:
        if any(w in n for w in words):
            return g
    return "other"


def gpu_ms(fn, reps):
    """Median GPU time of fn() in ms, from CUDA events."""
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return float(np.median(times))


cfg = load_locked()
mc = cfg["memorisation"]
torch.manual_seed(cfg["model"]["init_seed"])
s = setup("baby", None, "cuda", cfg)
model, loss_of, train = s["model"], s["loss"], s["train"]
rng = np.random.default_rng(mc["clip_seed"])
songs = sorted(train.songs.index)[:mc["songs"]]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s)))
         for song in songs * mc["clips_per_song"]]
batch = to_tensors(clips, "cuda")
clip_norm = cfg["training"]["grad_clip_norm"]
frames = batch["hits"].shape[1]
report = {"frames_per_clip": frames, "clips": len(clips), "neurons": model.n,
          "connections": int(model.pre.numel()), "gpu": torch.cuda.get_device_name()}

# 1. The recorded step, as the runs use it
opt = make_optimiser(model, cfg)
graphed = GraphedStep(model, opt, batch, loss_of, clip_norm)
graphed()
torch.cuda.synchronize()
t = time.time()
for _ in range(3):
    graphed()
report["recorded_step_s"] = (time.time() - t) / 3
print(f"recorded step: {report['recorded_step_s']:.2f} s")
del graphed

# 2. One ordinary step, profiled
opt = make_optimiser(model, cfg)
train_step(model, opt, batch, loss_of, clip_norm)                       # warm up outside the profile
torch.cuda.synchronize()
print("profiling one ordinary step (the profiler slows it; its GPU times are what count)...")
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    train_step(model, opt, batch, loss_of, clip_norm)
    torch.cuda.synchronize()
rows = []
for e in prof.key_averages():
    us = getattr(e, "self_device_time_total", None)
    if us is None:
        us = getattr(e, "self_cuda_time_total", 0)
    if us > 0 and e.key.startswith("aten::"):
        rows.append((e.key, us / 1000, e.count))
total_ms = sum(r[1] for r in rows)
groups = defaultdict(float)
for name, ms, _ in rows:
    groups[group_of(name)] += ms
report["profiled_gpu_ms"] = total_ms
report["groups_ms"] = dict(sorted(groups.items(), key=lambda kv: -kv[1]))
report["top_operations"] = [{"op": n, "gpu_ms": round(ms, 1), "calls": c}
                            for n, ms, c in sorted(rows, key=lambda r: -r[1])[:15]]
print(f"\nGPU time in one ordinary step: {total_ms / 1000:.2f} s "
      f"(the recorded step takes {report['recorded_step_s']:.2f} s in all)")
for g, ms in report["groups_ms"].items():
    print(f"  {ms / total_ms:6.1%}  {ms / 1000:6.2f} s  {g}")
print("\nlargest operations (GPU time, calls):")
for r in report["top_operations"]:
    print(f"  {r['gpu_ms'] / 1000:6.2f} s  {r['calls']:>7}  {r['op']}")

# 3. The big multiply alone, one frame, forward: dense (as now) against sparse
with torch.no_grad():
    w = model.weight_matrix()                                   # (post, pre), dense, as the network builds it
    r = torch.relu(torch.randn(len(clips), model.n, device="cuda"))
    w_t = w.t().contiguous()
    w_csr, w_coo = w.to_sparse_csr(), w.to_sparse()
    r_t = r.t().contiguous()
    dense = gpu_ms(lambda: r @ w_t, 200)
    csr = gpu_ms(lambda: torch.sparse.mm(w_csr, r_t), 200)
    coo = gpu_ms(lambda: torch.sparse.mm(w_coo, r_t), 200)
    same = torch.allclose(torch.sparse.mm(w_csr, r_t).t(), r @ w_t, rtol=1e-4, atol=1e-6)
report["one_frame_forward_multiply_ms"] = {"dense": dense, "sparse_csr": csr, "sparse_coo": coo,
                                           "sparse_matches_dense": bool(same)}
print(f"\none frame's forward multiply ({len(clips)} clips x {model.n} neurons, "
      f"{report['connections'] / model.n ** 2:.1%} of the grid are connections):")
print(f"  dense {dense * 1000:.0f} us, sparse CSR {csr * 1000:.0f} us, sparse COO {coo * 1000:.0f} us "
      f"(sparse gives the same numbers: {same})")
print(f"  x {frames} frames: dense {dense * frames / 1000:.2f} s, sparse CSR {csr * frames / 1000:.2f} s per clip "
      f"batch, forward only (the backward pass does about twice this)")

out = reports_dir() / "profile_step.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {out}")
