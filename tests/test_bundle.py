"""A bundle has to be the same model, not a similar one.

The bundle drops the derivable buffers and rebuilds them through
``ConnectomeRNN.__init__``. If that reconstruction ever put the edges in a
different order, the trained ``log_gain`` would line up against the wrong
connections -- and the model would still load, still run, and still emit
plausible drums. So the test is exact equality of the output, not closeness.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import build_model, get_subgraph, load_config  # noqa: E402
from bundle import DERIVED_BUFFERS, build_bundle, load_bundle  # noqa: E402


def _tiny_model():
    cfg = load_config(ROOT / "configs" / "sanity_3piece.yaml")
    cfg["subgraph"]["max_nodes"] = 2000
    cfg["subgraph"]["cache"] = str(ROOT / "data" / "cache" / "subgraph_test2k.npz")
    if not Path(cfg["subgraph"]["cache"]).exists():
        pytest.skip("no cached subgraph in this checkout")
    sg = get_subgraph(cfg)
    torch.manual_seed(0)
    model, _ = build_model(cfg, sg, n_styles=3)
    # move off the initialisation so a silently-wrong permutation cannot pass
    with torch.no_grad():
        model.rnn.log_gain.add_(torch.randn_like(model.rnn.log_gain) * 0.1)
        model.encoder.to_jo.weight.add_(torch.rand_like(model.encoder.to_jo.weight) * 0.01)
    model.eval()
    ck = {"model": model.state_dict(), "config": cfg, "kit": ["kick", "snare", "hat_closed"],
          "n_styles": 3, "best_threshold": 0.45}
    return cfg, sg, model, ck


def test_bundle_reproduces_the_model_exactly(tmp_path):
    cfg, sg, model, ck = _tiny_model()
    path = tmp_path / "m.fb"
    torch.save(build_bundle(ck, sg, model), path)

    bundled, kit, bcfg, roles = load_bundle(path)
    wav = torch.randn(1, 22050, generator=torch.Generator().manual_seed(1)) * 0.1
    with torch.no_grad():
        a, _ = model(wav)
        b, _ = bundled(wav)
    assert torch.equal(a, b), f"bundle changed the output: {(a - b).abs().max():.3e}"
    assert kit.classes == ck["kit"]
    assert bcfg["eval"]["threshold"] == 0.45, "the scored threshold must survive export"
    assert {"pC1", "pIP10", "inhibitory", "sensory", "motor"} <= set(roles)


def test_bundle_is_smaller_and_drops_only_derivable_buffers(tmp_path):
    cfg, sg, model, ck = _tiny_model()
    path, ckpt = tmp_path / "m.fb", tmp_path / "m.pt"
    torch.save(build_bundle(ck, sg, model), path)
    torch.save(ck, ckpt)
    assert path.stat().st_size < ckpt.stat().st_size

    stored = set(torch.load(path, map_location="cpu", weights_only=False)["state"])
    assert stored == set(model.state_dict()) - set(DERIVED_BUFFERS)


def test_a_bundle_from_the_future_is_refused(tmp_path):
    cfg, sg, model, ck = _tiny_model()
    payload = build_bundle(ck, sg, model)
    payload["format"] = 999
    path = tmp_path / "m.fb"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format"):
        load_bundle(path)


def test_loading_a_bundle_never_touches_the_dataset(tmp_path, monkeypatch):
    """The whole point: a laptop with no connectome and no corpus can run it."""
    cfg, sg, model, ck = _tiny_model()
    path = tmp_path / "m.fb"
    torch.save(build_bundle(ck, sg, model), path)

    import build as build_mod

    def explode(*_a, **_k):
        raise AssertionError("load_bundle reached for the subgraph")

    monkeypatch.setattr(build_mod, "get_subgraph", explode)
    monkeypatch.setattr(build_mod, "build_neuron_graph", explode)
    bundled, _, _, _ = load_bundle(path)
    assert bundled.rnn.n_nodes == model.rnn.n_nodes
