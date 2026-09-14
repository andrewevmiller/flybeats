"""The tests only a GPU can run.

CI is CPU-only and always will be, so these skip there and everywhere else
without a card. They exist because three pieces of the training path are
*reached* only on CUDA, and each fails by producing a plausible number rather
than an error:

- ``SparseSpMM``'s hand-written backward, which exists because torch's CSR
  autograd returns a gradient sized to the *deduplicated* values when an edge
  list repeats a (row, col) pair -- exactly what the Phase 4 rewiring arm
  emits. Checked against a dense reference, with duplicates present on purpose.
- ``train.bf16``, honoured on CUDA and ignored everywhere else, so it has never
  been switched on by any run in this repository.
- gradient checkpointing on device, where the recomputed forward has to
  reproduce the first one closely enough to leave the gradient untouched.

``scripts/cuda_smoke.py`` runs the same checks as a one-shot with timings and
peak VRAM, for the first five minutes on a new machine. These are the version
that runs on every push from the box that has the card:

    pytest -m gpu -v
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import build_model, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit  # noqa: E402
from model import ConnectomeRNN, ModelConfig, SparseSpMM  # noqa: E402

from conftest import use_small_graph  # noqa: E402

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(),
                       reason="no CUDA device (expected on CI and in cloud sessions)"),
]


@pytest.fixture(scope="module")
def device():
    return torch.device("cuda")


@pytest.fixture(scope="module")
def trained_bits(device):
    """A small real model on the GPU, plus a loader of synthetic audio."""
    import train as T

    cfg = use_small_graph(load_config(ROOT / "configs" / "sanity_3piece.yaml"))
    cfg["data"].update(synthetic=True, n_clips=4, seconds=0.5)
    cfg["train"].update(batch_size=2, tbptt_steps=25, workers=0, seed=0)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    loader, _val, n_styles = T.build_loaders(cfg, kit)
    model, kit = build_model(cfg, get_subgraph(cfg), n_styles=max(int(n_styles or 1), 1))
    model = model.to(device)
    T.calibrate_encoder(model, loader.dataset, cfg, device)
    return model, loader, cfg


def _grads(model, loader, cfg, device, **overrides):
    import train as T

    c = {k: (dict(v) if isinstance(v, dict) else v) for k, v in cfg.items()}
    c["train"].update(overrides)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)     # measure grads, don't move
    torch.manual_seed(0)
    T.reseed_loader(loader, 0)
    T.run_epoch(model, loader, opt, c, device, train=True)
    return {n: p.grad.detach().float().clone()
            for n, p in model.named_parameters() if p.grad is not None}


def test_sparse_backward_matches_dense_with_duplicate_edges(device):
    """The reason SparseSpMM exists, checked on the device that has never run it."""
    rng = np.random.default_rng(0)
    n, b = 64, 3
    ei = np.stack([rng.integers(0, n, 400), rng.integers(0, n, 400)]).astype(np.int32)
    ei[:, :40] = ei[:, :1]                                  # repeated (post, pre) pairs

    rnn = ConnectomeRNN(
        edge_index=ei,
        edge_sign=rng.choice([-1.0, 1.0], ei.shape[1]).astype(np.float32),
        weight=rng.integers(1, 50, ei.shape[1]).astype(np.float32),
        n_nodes=n, sensory_idx=np.arange(5), motor_idx=np.arange(5, 10),
        cfg=ModelConfig(),
    ).to(device)

    def dense(values, r):
        w = torch.zeros(n, n, dtype=values.dtype, device=values.device)
        w.index_put_((rnn.edge_index[0].to(values.device),
                      rnn.edge_index[1].to(values.device)), values, accumulate=True)
        return r @ w.t()

    r = torch.randn(b, n, dtype=torch.float64, device=device)
    vals = rnn.edge_weight().double().detach().requires_grad_(True)

    out = SparseSpMM.apply(vals, r.clone(), rnn.crow, rnn.edge_col, rnn.crow_t,
                           rnn.col_t, rnn.perm_t, rnn.edge_row, n)
    assert torch.allclose(out, dense(vals, r), atol=1e-9), \
        f"forward differs from dense by {(out - dense(vals, r)).abs().max():.3e}"

    r_ref = r.clone().requires_grad_(True)
    dense(vals, r_ref).sum().backward()
    g_v_ref, g_r_ref = vals.grad.clone(), r_ref.grad.clone()

    vals.grad = None
    r_ours = r.clone().requires_grad_(True)
    SparseSpMM.apply(vals, r_ours, rnn.crow, rnn.edge_col, rnn.crow_t, rnn.col_t,
                     rnn.perm_t, rnn.edge_row, n).sum().backward()

    assert vals.grad.shape == g_v_ref.shape, \
        f"gradient is sized {tuple(vals.grad.shape)} for {ei.shape[1]} edges -- " \
        "this is the deduplication bug the custom backward exists to avoid"
    assert torch.allclose(vals.grad, g_v_ref, atol=1e-8), \
        f"value gradient differs by {(vals.grad - g_v_ref).abs().max():.3e}"
    assert torch.allclose(r_ours.grad, g_r_ref, atol=1e-8), \
        f"input gradient differs by {(r_ours.grad - g_r_ref).abs().max():.3e}"


@pytest.mark.skipif(not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
                    reason="card does not support bfloat16")
def test_bf16_autocast_gradients_are_finite_and_track_fp32(trained_bits, device):
    """bf16 has ~3 decimal digits, so this is a sanity band, not an equality.

    What it catches is the failure that matters: NaN, a dead gradient, or a
    dtype error inside the sparse op. Expected rounding drift is not a bug, and
    a tolerance tight enough to call it one would make this test useless.
    """
    model, loader, cfg = trained_bits
    g32 = _grads(model, loader, cfg, device, bf16=False, grad_checkpoint=False)
    g16 = _grads(model, loader, cfg, device, bf16=True, grad_checkpoint=False)

    shared = sorted(set(g32) & set(g16))
    assert shared, "no gradients to compare"
    for k in shared:
        assert torch.isfinite(g16[k]).all(), f"{k} has non-finite gradients under bf16"
    assert any(float(g16[k].abs().max()) > 0 for k in shared), \
        "every bf16 gradient is zero"

    worst = max(float((g16[k] - g32[k]).abs().max()) / max(float(g32[k].abs().max()), 1e-12)
                for k in shared)
    assert worst < 0.25, f"bf16 gradient is {worst:.2f} off fp32 -- too far for rounding"


def test_grad_checkpointing_is_transparent_on_device(trained_bits, device):
    """Memory optimisation, not a numerical one. It must change nothing."""
    model, loader, cfg = trained_bits
    plain = _grads(model, loader, cfg, device, grad_checkpoint=False, bf16=False)
    ckpt = _grads(model, loader, cfg, device, grad_checkpoint=True, bf16=False)

    assert set(plain) == set(ckpt)
    for k in plain:
        assert torch.allclose(plain[k], ckpt[k], atol=1e-5), \
            f"checkpointing changed the gradient for {k}: " \
            f"{(plain[k] - ckpt[k]).abs().max():.3e}"


def test_model_runs_on_device_and_agrees_with_cpu(trained_bits, device):
    """A forward pass, and the same one on CPU.

    Not bitwise -- different kernels, different reduction orders -- but a
    disagreement past float32 noise means the device path is computing
    something else, which no metric downstream would reveal.
    """
    model, _loader, _cfg = trained_bits
    wav = torch.randn(1, 11025, generator=torch.Generator().manual_seed(3)) * 0.1

    model.eval()
    with torch.no_grad():
        on_gpu, _ = model(wav.to(device))
        on_cpu, _ = model.cpu()(wav)
        model.to(device)                                     # leave the fixture as found

    assert on_gpu.shape == on_cpu.shape
    diff = float((on_gpu.cpu() - on_cpu).abs().max())
    assert diff < 1e-3, f"GPU and CPU forward passes differ by {diff:.3e}"
