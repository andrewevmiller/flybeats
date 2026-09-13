"""Training-loop invariants.

Gradient checkpointing and truncated BPTT are both memory optimisations that
are supposed to be numerically invisible. If either silently changes the
gradient, every result downstream is quietly wrong and nothing in the loss
curve would say so.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as T  # noqa: E402
from build import build_model, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit  # noqa: E402

#: The checkpointing test needs a real subgraph, and building one falls back to
#: the connectome download when no cache is on disk. On a fresh checkout that
#: raised FileNotFoundError from deep inside pandas instead of skipping, which
#: reads as a broken repo rather than as missing data.
TEST_CACHE = ROOT / "data" / "cache" / "subgraph_test2k.npz"
ANNOTATIONS = ROOT / "data" / "raw" / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
needs_graph = pytest.mark.skipif(
    not TEST_CACHE.exists() and not ANNOTATIONS.exists(),
    reason="no cached subgraph and no connectome download in this checkout",
)


def _tiny_cfg():
    cfg = load_config(ROOT / "configs" / "sanity_3piece.yaml")
    cfg["data"].update(synthetic=True, n_clips=4, seconds=0.5)
    cfg["train"].update(batch_size=2, tbptt_steps=25, workers=0)
    cfg["subgraph"]["max_nodes"] = 2000
    cfg["subgraph"]["cache"] = str(ROOT / "data" / "cache" / "subgraph_test2k.npz")
    return cfg


def _grads(cfg, sg, loader, n_styles, **overrides):
    torch.manual_seed(0)
    np.random.seed(0)
    model, _ = build_model(cfg, sg, n_styles=max(n_styles, 1))
    c = copy.deepcopy(cfg)
    c["train"].update(overrides)
    opt = torch.optim.SGD(model.parameters(), lr=0.0)     # measure grads, don't move
    T.run_epoch(model, loader, opt, c, torch.device("cpu"), train=True)
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


@needs_graph
def test_gradient_checkpointing_is_numerically_transparent():
    cfg = _tiny_cfg()
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    tr, _, n_styles = T.build_loaders(cfg, kit)
    sg = get_subgraph(cfg)

    plain = _grads(cfg, sg, tr, n_styles, grad_checkpoint=False)
    ckpt = _grads(cfg, sg, tr, n_styles, grad_checkpoint=True)

    assert set(plain) == set(ckpt)
    for k in plain:
        assert torch.allclose(plain[k], ckpt[k], atol=1e-6), \
            f"checkpointing changed the gradient for {k}: " \
            f"{(plain[k] - ckpt[k]).abs().max():.3e}"


def test_rate_penalty_pushes_toward_the_target():
    step_ms, target_hz = 5.0, 5.0
    on_target = torch.full((2, 10, 50), target_hz * step_ms / 1000.0)
    assert float(T.rate_penalty(on_target, target_hz, step_ms)) < 1e-9
    saturated = torch.full((2, 10, 50), 5.0)
    assert float(T.rate_penalty(saturated, target_hz, step_ms)) > 1.0


def test_onset_loss_prefers_the_right_answer():
    y = torch.zeros(1, 40, 3)
    y[0, 10, 0] = 1.0
    right = torch.full_like(y, -6.0)
    right[0, 10, 0] = 6.0
    wrong = torch.full_like(y, -6.0)
    wrong[0, 30, 2] = 6.0
    assert float(T.onset_loss(right, y, 8.0)) < float(T.onset_loss(wrong, y, 8.0))
