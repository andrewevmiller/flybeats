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

from conftest import use_small_graph  # noqa: E402


def _tiny_model():
    cfg = use_small_graph(load_config(ROOT / "configs" / "sanity_3piece.yaml"))
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


def test_a_v1_bundle_without_a_velocity_head_still_loads(tmp_path):
    """A bundle is the only durable form a trained model has here -- runs/ is
    gitignored. Refusing one because this build grew a head since it was
    exported would strand hours of training, so the loader takes its shape from
    what the bundle contains rather than from what this build would construct.
    """
    from bundle import FORMAT_VERSION, READABLE_VERSIONS

    cfg, sg, model, ck = _tiny_model()
    b = build_bundle(ck, sg, model)
    assert b["format"] == FORMAT_VERSION == 2

    # age it: strip the head and stamp it v1, exactly as an older export looks
    b["state"] = {k: v for k, v in b["state"].items()
                  if not k.startswith("decoder.vel_readout")}
    b["format"] = 1
    assert 1 in READABLE_VERSIONS
    path = tmp_path / "old.fb"
    torch.save(b, path)

    old, kit, _, _ = load_bundle(path)
    assert not old.decoder.has_velocity
    wav = torch.randn(1, 22050, generator=torch.Generator().manual_seed(3)) * 0.1
    with torch.no_grad():
        logits, _ = old(wav)
        assert old.decoder.velocity(logits.new_zeros(1, 4, len(sg.role("motor")))) is None
    assert logits.shape[-1] == kit.n


def test_an_unknown_future_bundle_version_is_still_refused(tmp_path):
    cfg, sg, model, ck = _tiny_model()
    b = build_bundle(ck, sg, model)
    b["format"] = 99
    path = tmp_path / "future.fb"
    torch.save(b, path)
    with pytest.raises(ValueError, match="bundle format"):
        load_bundle(path)


def test_a_checkpoint_plays_at_its_scored_threshold_like_its_bundle(tmp_path):
    """``realtime.py --checkpoint`` and a bundle of the same checkpoint must pick
    hits at the same threshold. The checkpoint path used to hand back the
    config's fixed eval.threshold, so the same model rendered a wall of notes
    from one entry point and a handful from the other.
    """
    from realtime import drummer_for, load_checkpoint

    cfg, sg, model, ck = _tiny_model()
    ck["best_threshold"] = 0.5
    assert cfg.get("eval", {}).get("threshold", 0.3) != 0.5, "test needs a different default"
    ckpt, fb = tmp_path / "best.pt", tmp_path / "m.fb"
    torch.save(ck, ckpt)
    torch.save(build_bundle(ck, sg, model), fb)

    m, kit, ccfg = load_checkpoint(ckpt, torch.device("cpu"))
    assert drummer_for(m, kit, ccfg).threshold == 0.5
    b, bkit, bcfg, _ = load_bundle(fb)
    assert drummer_for(b, bkit, bcfg).threshold == 0.5
    # an explicit --threshold still wins over the scored one
    assert drummer_for(m, kit, ccfg, threshold=0.2).threshold == 0.2
