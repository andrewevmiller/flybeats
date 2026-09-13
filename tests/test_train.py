"""Training-loop invariants.

Gradient checkpointing and truncated BPTT are both memory optimisations that
are supposed to be numerically invisible. If either silently changes the
gradient, every result downstream is quietly wrong and nothing in the loss
curve would say so.
"""
import copy
import sys
import types
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


def test_rate_penalty_is_one_sided():
    """It exists to keep the units off their saturation ceiling. Penalising
    quiet as well is what taught the encoder to silence its own input."""
    ceiling = 3.0
    quiet = torch.full((2, 10, 50), 0.01)
    assert float(T.rate_penalty(quiet, ceiling)) == 0.0
    at_ceiling = torch.full((2, 10, 50), ceiling)
    assert float(T.rate_penalty(at_ceiling, ceiling)) == 0.0
    saturated = torch.full((2, 10, 50), 20.0)
    assert float(T.rate_penalty(saturated, ceiling)) > 1.0


def test_rate_ceiling_is_measured_from_the_models_own_activity():
    """'auto' means "several times as loud as this network starts", in the
    activation units the network actually has -- not an invented Hz constant."""
    tb = {"rate_ceiling": "auto", "rate_headroom": 4.0}
    model = types.SimpleNamespace()
    rates = torch.full((2, 10, 50), 0.5)
    assert T.resolve_rate_ceiling(model, tb, rates) == 2.0
    # cached on the model, so every later chunk uses the initial operating point
    assert T.resolve_rate_ceiling(model, tb, torch.full((2, 10, 50), 9.0)) == 2.0
    # and a config that names a number is taken at its word
    assert T.resolve_rate_ceiling(types.SimpleNamespace(),
                                  {"rate_ceiling": 1.25}, rates) == 1.25


def test_stale_target_rate_hz_is_refused():
    """The old key silently meant something else. A config carrying it must
    fail loudly rather than train against a penalty on all activity."""
    cfg = _tiny_cfg()
    cfg["train"]["target_rate_hz"] = 5.0
    model = types.SimpleNamespace(train=lambda *_: None)
    with pytest.raises(SystemExit, match="target_rate_hz"):
        T.run_epoch(model, [], None, cfg, torch.device("cpu"), train=True)


def test_onset_loss_prefers_the_right_answer():
    y = torch.zeros(1, 40, 3)
    y[0, 10, 0] = 1.0
    right = torch.full_like(y, -6.0)
    right[0, 10, 0] = 6.0
    wrong = torch.full_like(y, -6.0)
    wrong[0, 30, 2] = 6.0
    assert float(T.onset_loss(right, y, 8.0)) < float(T.onset_loss(wrong, y, 8.0))
