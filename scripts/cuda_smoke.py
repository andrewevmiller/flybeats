"""Does this project's CUDA path actually work? Ask before booking GPU hours.

Nothing in this repository has ever executed on a GPU. Every number in it came
off 4 CPU cores, and three pieces of the training path are only *reached* on
CUDA:

1. **The hand-written sparse backward.** ``SparseSpMM`` exists because torch's
   own CSR autograd returns a gradient sized to the *deduplicated* values when
   the edge list repeats a (row, col) pair -- which the Phase 4 rewiring arm
   produces. The replacement is exact algebra, and it has only ever been
   checked against a dense reference on CPU.
2. **bfloat16 autocast** (``train.bf16``), which ``run_epoch`` honours on CUDA
   and ignores everywhere else. It has therefore never been on.
3. **Gradient checkpointing** on device, where the recomputed forward has to
   reproduce the first one closely enough that the gradient is unchanged.

Each would fail *quietly*: a mis-sized or mis-scaled gradient still trains,
still evaluates, and still prints a plausible onset F. A 40-hour ablation
campaign is a bad place to find out. This script is ten minutes.

    python scripts/cuda_smoke.py                 # the checks, on CUDA
    python scripts/cuda_smoke.py --device cpu    # same checks, as a self-test
    python scripts/cuda_smoke.py --config configs/v1_8piece.yaml   # real tier

With no ``--config`` it uses the committed 2,000-neuron fixture and synthetic
audio, so it runs on a fresh clone with no connectome and no corpus. Point it
at a real config once the caches are built: that is the run that measures
per-epoch time and peak VRAM at the tier you intend to train.

Exit status is 0 only if every check passed.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import build_model, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit  # noqa: E402
from model import ConnectomeRNN, ModelConfig, SparseSpMM  # noqa: E402

FIXTURE = ROOT / "tests" / "fixtures" / "subgraph_2k.npz"

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []


def record(name: str, ok: bool | None, detail: str = "") -> bool:
    status = SKIP if ok is None else (PASS if ok else FAIL)
    results.append((status, name, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    return bool(ok)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title), flush=True)


# --------------------------------------------------------------- environment --

def report_environment(device: torch.device) -> None:
    section("Environment")
    print(f"  torch {torch.__version__}, CUDA build {torch.version.cuda}")
    if device.type != "cuda":
        print(f"  device: {device} (not CUDA -- running as a self-test)")
        return
    i = device.index or 0
    p = torch.cuda.get_device_properties(i)
    print(f"  device: {p.name}, capability {p.major}.{p.minor}, "
          f"{p.total_memory / 1e9:.1f} GB, {p.multi_processor_count} SMs")
    bf16 = torch.cuda.is_bf16_supported()
    print(f"  bfloat16 supported: {bf16}")
    if not bf16:
        print("  NOTE: set train.bf16 false in your config; autocast would fall "
              "back or error on this card.")


# ------------------------------------------------------------ the three risks --

def _dense_reference(edge_index, values, n, r):
    """The same reference tests/test_model.py uses: edge_index is [post, pre]."""
    w = torch.zeros(n, n, dtype=values.dtype, device=values.device)
    w.index_put_((edge_index[0].to(values.device), edge_index[1].to(values.device)),
                 values, accumulate=True)
    return r @ w.t()


def check_sparse_backward(device: torch.device) -> bool:
    """The custom backward, against a dense reference, *with duplicate edges*.

    Duplicates are the whole reason this code exists, so the graph here has
    them on purpose. float64 on both sides: at this size any real discrepancy
    is orders of magnitude above float64 noise, and a tolerance loose enough to
    hide a wrong gradient would defeat the check.
    """
    torch.manual_seed(0)
    n, b = 64, 3
    rng = np.random.default_rng(0)
    ei = np.stack([rng.integers(0, n, 400), rng.integers(0, n, 400)]).astype(np.int32)
    ei[:, :40] = ei[:, :1]                      # force repeated (post, pre) pairs

    rnn = ConnectomeRNN(
        edge_index=ei,
        edge_sign=rng.choice([-1.0, 1.0], ei.shape[1]).astype(np.float32),
        weight=rng.integers(1, 50, ei.shape[1]).astype(np.float32),
        n_nodes=n, sensory_idx=np.arange(5), motor_idx=np.arange(5, 10),
        cfg=ModelConfig(),
    ).to(device)

    r = torch.randn(b, n, dtype=torch.float64, device=device)
    vals = rnn.edge_weight().double().detach().requires_grad_(True)

    out = SparseSpMM.apply(vals, r.clone().requires_grad_(True), rnn.crow,
                           rnn.edge_col, rnn.crow_t, rnn.col_t, rnn.perm_t,
                           rnn.edge_row, n)
    want = _dense_reference(rnn.edge_index, vals, n, r)
    fwd = float((out - want).abs().max().detach())

    # gradients: ours vs the dense reference's autograd
    r_ref = r.clone().requires_grad_(True)
    _dense_reference(rnn.edge_index, vals, n, r_ref).sum().backward()
    g_v_ref, g_r_ref = vals.grad.clone(), r_ref.grad.clone()

    vals.grad = None
    r_ours = r.clone().requires_grad_(True)
    SparseSpMM.apply(vals, r_ours, rnn.crow, rnn.edge_col, rnn.crow_t,
                     rnn.col_t, rnn.perm_t, rnn.edge_row, n).sum().backward()
    d_v = float((vals.grad - g_v_ref).abs().max())
    d_r = float((r_ours.grad - g_r_ref).abs().max())

    ok = fwd < 1e-9 and d_v < 1e-8 and d_r < 1e-8
    return record("sparse spmm + custom backward vs dense (duplicate edges)", ok,
                  f"forward {fwd:.2e}, grad_values {d_v:.2e}, grad_r {d_r:.2e}")


def check_bf16(model, batch, cfg, device) -> bool | None:
    """One autocast step: finite, non-zero, and close to the fp32 gradient.

    bf16 has ~3 decimal digits of mantissa, so this is a sanity band, not an
    equality. What it catches is the real failure -- NaN, a dead gradient, or a
    silent dtype error inside the sparse op -- not the expected small drift.
    """
    import train as T

    if device.type != "cuda":
        return record("bf16 autocast step", None, "CUDA only")
    if not torch.cuda.is_bf16_supported():
        return record("bf16 autocast step", None, "not supported by this card")

    def one_step(bf16: bool) -> dict:
        c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        c["train"]["bf16"] = bf16
        c["train"]["grad_checkpoint"] = False
        opt = torch.optim.SGD(model.parameters(), lr=0.0)   # grads only, no step
        torch.manual_seed(0)
        T.run_epoch(model, batch, opt, c, device, train=True)
        return {n: p.grad.detach().float().clone()
                for n, p in model.named_parameters() if p.grad is not None}

    g32, g16 = one_step(False), one_step(True)
    shared = sorted(set(g32) & set(g16))
    if not shared:
        return record("bf16 autocast step", False, "no gradients at all")

    finite = all(torch.isfinite(g16[k]).all() for k in shared)
    alive = any(float(g16[k].abs().max()) > 0 for k in shared)
    rels = {k: float((g16[k] - g32[k]).abs().max())
               / max(float(g32[k].abs().max()), 1e-12) for k in shared}
    worst_k = max(rels, key=rels.get)
    worst = rels[worst_k]
    ok = finite and alive and worst < 0.25
    return record("bf16 autocast step", ok,
                  f"finite={finite}, non-zero={alive}, worst relative drift "
                  f"{worst:.3f} on {worst_k}")


def check_grad_checkpoint(model, batch, cfg, device) -> bool:
    """Checkpointing is a memory optimisation; it must not move the gradient."""
    import train as T

    def grads(ckpt: bool) -> dict:
        c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
        c["train"]["grad_checkpoint"] = ckpt
        c["train"]["bf16"] = False
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        torch.manual_seed(0)
        T.reseed_loader(batch, 0)
        T.run_epoch(model, batch, opt, c, device, train=True)
        return {n: p.grad.detach().clone()
                for n, p in model.named_parameters() if p.grad is not None}

    plain, ckpt = grads(False), grads(True)
    shared = sorted(set(plain) & set(ckpt))
    if not shared:
        return record("gradient checkpointing is transparent", False,
                      "no gradients to compare")
    diffs = {k: float((plain[k] - ckpt[k]).abs().max()) for k in shared}
    worst_k = max(diffs, key=diffs.get)
    worst = diffs[worst_k]
    ok = worst < 1e-5
    return record("gradient checkpointing is transparent", ok,
                  f"{len(shared)} tensors, worst difference {worst:.2e} ({worst_k})")


# ------------------------------------------------------------- end to end -----

def check_training_step(model, batch, cfg, device) -> bool:
    """A real epoch of the real loop, timed, with peak memory."""
    import train as T

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["train"].get("lr", 3e-3))
    t0 = time.perf_counter()
    out = T.run_epoch(model, batch, opt, cfg, device, train=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 1e9) if device.type == "cuda" else 0.0
    ok = np.isfinite(out["loss"]) and out["loss"] > 0
    detail = f"loss {out['loss']:.4f}, {dt:.1f}s for {len(batch.dataset)} clips"
    if device.type == "cuda":
        detail += f", peak VRAM {peak:.2f} GB"
    return record("one training epoch end to end", ok, detail)


def check_inference(model, kit, cfg, device) -> bool:
    """The streaming path -- and how many CPU threads it should get.

    Thread scaling here is a property of the machine, not of the code. The same
    10k model on one cloud container: 12.2 ms per 20 ms block on one thread and
    160.9 ms on four while the box was loaded; 9.0 / 6.6 / 4.5 ms on 1 / 2 / 4
    threads once it was idle. Contention inverts the scaling completely, so no
    number from another machine is worth copying onto yours.

    Hence the retry: a missed budget is worth one measurement at a single
    thread before it is believed, because the conclusion "this machine cannot
    play live" and the conclusion "this machine was busy" look identical from
    one reading.
    """
    from realtime import benchmark

    def run(n_threads: int | None = None):
        before = torch.get_num_threads()
        if n_threads:
            torch.set_num_threads(n_threads)
        try:
            return benchmark(model, kit, cfg, block_ms=20.0, n_blocks=20)
        finally:
            torch.set_num_threads(before)

    threads = torch.get_num_threads()
    try:
        stats = run()
    except Exception as exc:                                    # noqa: BLE001
        return record("streaming inference benchmark", False, f"{type(exc).__name__}: {exc}")

    detail = (f"{stats['inference_ms_mean']:.1f} ms mean / "
              f"{stats['inference_ms_p95']:.1f} ms p95 per 20 ms block "
              f"({stats['realtime_factor']:.1f}x realtime) on {threads} threads")

    if not stats["meets_budget"] and threads > 1:
        try:
            single = run(1)
        except Exception:                                       # noqa: BLE001
            single = None
        if single and single["inference_ms_mean"] < stats["inference_ms_mean"] * 0.8:
            detail += (f"; {single['inference_ms_mean']:.1f} ms on 1 thread "
                       f"({single['realtime_factor']:.1f}x) -- set OMP_NUM_THREADS=1 "
                       f"for the live path")
            stats = single

    if not stats["meets_budget"]:
        detail += " -- misses the 20 ms budget; offline --render is unaffected"
    return record("streaming inference benchmark", stats["inference_ms_mean"] > 0, detail)


# ------------------------------------------------------------------- driver ---

def build(a, device):
    """Model, loader and config -- from a real config, or from the fixture."""
    import train as T

    if a.config:
        cfg = load_config(a.config)
    else:
        cfg = load_config(ROOT / "configs" / "sanity_3piece.yaml")
        if not FIXTURE.exists():
            raise SystemExit(f"no fixture at {FIXTURE}; pass --config instead")
        cfg["subgraph"].update(max_nodes=2000, min_weight=5, cache=str(FIXTURE))
        cfg["data"].update(synthetic=True, n_clips=a.clips, seconds=a.seconds)
        cfg["train"].update(batch_size=2, tbptt_steps=50, workers=0)

    cfg["train"]["device"] = str(device)
    sg = get_subgraph(cfg)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    loader, _val, n_styles = T.build_loaders(cfg, kit)
    model, kit = build_model(cfg, sg, n_styles=max(int(n_styles or 1), 1))
    model = model.to(device)
    T.calibrate_encoder(model, loader.dataset, cfg, device)
    print(f"  graph: {sg.n_nodes:,} neurons, {sg.n_edges:,} edges | "
          f"kit: {len(kit.classes)} classes")
    return model, loader, kit, cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda",
                    help="cuda (default), cpu to self-test the checks, or cuda:1")
    ap.add_argument("--config", type=Path, default=None,
                    help="a real config, to measure the tier you will train")
    ap.add_argument("--clips", type=int, default=4, help="synthetic clips (no --config)")
    ap.add_argument("--seconds", type=float, default=1.0, help="clip length (no --config)")
    a = ap.parse_args(argv)

    if a.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA is not available to torch here.\n"
              "  - `nvidia-smi` should list the card; if it does not, the driver is the problem\n"
              "  - a CPU-only wheel is the usual cause: reinstall from the CUDA\n"
              "    index for a build your driver supports (pytorch.org's selector\n"
              "    names it), e.g. --index-url https://download.pytorch.org/whl/cu124\n"
              "  - run with --device cpu to exercise these checks anyway")
        return 2
    device = torch.device(a.device)

    report_environment(device)

    section("Building")
    model, loader, kit, cfg = build(a, device)

    section("The three things only CUDA reaches")
    check_sparse_backward(device)
    check_bf16(model, loader, cfg, device)
    check_grad_checkpoint(model, loader, cfg, device)

    section("End to end")
    check_training_step(model, loader, cfg, device)
    check_inference(model, kit, cfg, device)

    section("Summary")
    failed = [r for r in results if r[0] == FAIL]
    skipped = [r for r in results if r[0] == SKIP]
    print(f"  {len(results) - len(failed) - len(skipped)} passed, "
          f"{len(failed)} failed, {len(skipped)} skipped")
    for _s, name, detail in failed:
        print(f"    FAILED: {name} -- {detail}")
    if failed:
        print("\n  Do not start a long run on this machine until these pass: each\n"
              "  one is a wrong number that still looks like a right number.")
    elif skipped:
        print(f"\n  Nothing failed, but {len(skipped)} check(s) never ran -- including the\n"
              "  CUDA-only ones if this was a CPU self-test. This is not yet a verdict\n"
              "  on the GPU path; re-run it on the machine that will do the training.")
    else:
        print("\n  The CUDA path is sound. RUNBOOK.md has what to run next.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
