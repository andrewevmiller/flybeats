"""The exporter: rebuilds what the checkpoint lacks, checkpoint round trip, cliff calibration, provenance."""
import json

import numpy as np
import pytest
import torch

from flybrain.brainfile import read_brain
from flybrain.exporter import export, load_checkpoint, measure_cliff, realised_weights, runaway_limits
from flybrain.parity import compare, make_model, torch_outputs
from flybrain.reference import run_clip
from flybrain.synthetic import noise_clips


def test_g_and_gain_scale_are_not_in_the_state_dict(brain_small):
    _, model, _, _ = brain_small
    keys = set(model.state_dict().keys())
    assert "g" not in keys and "gain_scale" not in keys
    assert "mn_mean" in keys and "mn_std" in keys and "alive" in keys     # buffers that are carried


def test_checkpoint_round_trip_exports_identically(brain_small, cfg, tmp_path):
    b, model, neurons, edges = brain_small
    ck = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "step": 7, "val": {"mean_f1": 0.1}}, ck)
    fresh = make_model(cfg, neurons, edges, seed=99, band_stats=(model.bands.mean.tolist(), model.bands.std.tolist()),
                       lively=True)                        # different parameters until loaded
    extra = load_checkpoint(fresh, ck)
    assert extra["step"] == 7
    out = tmp_path / "again.flybrain"
    export(fresh, neurons, edges, cfg, out, name="again")
    c = read_brain(out)
    for k in b.arrays:
        assert np.array_equal(c[k], b[k]), k
    assert c.meta["g"] == b.meta["g"] and c.meta["eigenvalue"] == b.meta["eigenvalue"]


def test_realised_weights_match_weight_matrix(brain_small):
    b, model, _, edges = brain_small
    vals, unsigned = realised_weights(model, edges)
    w = model.weight_matrix().detach()
    assert np.array_equal(w[model.post, model.pre].numpy(), vals)
    assert np.all(unsigned >= 0)
    # CSR by post gives the same dense matrix
    n = b.n
    dense = np.zeros((n, n), np.float32)
    for row in range(n):
        s, e = b["w_indptr"][row], b["w_indptr"][row + 1]
        dense[row, b["w_indices"][s:e]] = b["w_values"][s:e]
    assert np.array_equal(dense, w.numpy())


def test_alpha_and_bias_match_the_model(brain_small):
    b, model, _, _ = brain_small
    alpha = (model.dt_ms / model.tau_ms())[model.type_idx].detach().numpy()
    assert np.array_equal(b["alpha"], alpha)
    assert np.array_equal(b["bias"], model.bias[model.type_idx].detach().numpy())
    assert np.array_equal(b["alive"], model.alive.numpy())


def test_display_metadata(brain_small):
    b, model, neurons, _ = brain_small
    assert np.array_equal(b["body_id"], neurons.bodyId.to_numpy())
    assert set(b["hop"][b["ear_idx"]]) == {0}
    assert (b["hop"][b["mn_idx"]] >= 2).all()                # no direct ear-to-motor connection
    assert (b["drum_of_neuron"][b["mn_idx"]] == b["mn_drum"]).all()
    assert len(b.meta["type_names"]) == len(model.type_names)
    assert b.meta["band_range"]["low"][0] < b.meta["band_range"]["high"][0]


def test_cliff_calibration_and_runaway_limits(cfg, slice_small):
    neurons, edges = slice_small
    model = make_model(cfg, neurons, edges, seed=0, lively=False)   # untrained: settles after music at 1x
    clips = noise_clips(n=2, seconds=3.0)
    cliff = measure_cliff(model, clips, cfg, grid=(1.0, 2.0, 4.0, 8.0, 16.0), resolution=0.5)
    assert cliff["measured"] and cliff["music_level_at_1x"] > 0
    assert cliff["last_pass_gain"] >= 1.0, cliff["results"]
    assert cliff["first_fail_gain"] is not None, "no gain up to 16x fails: the test network has no cliff"
    assert cliff["last_pass_gain"] < cliff["first_fail_gain"] <= cliff["last_pass_gain"] + 0.5   # bisected
    fail = cliff["results"][f"{cliff['first_fail_gain']:.4f}"]
    assert not (fail["bounded"] and fail["settled_ratio"] < cfg["silence_test"]["settle_fraction"])
    ok = cliff["results"][f"{cliff['last_pass_gain']:.4f}"]
    assert ok["bounded"] and ok["settled_ratio"] < cfg["silence_test"]["settle_fraction"]
    assert model.gain_scale == 1.0                        # restored
    limits = runaway_limits(cliff, cfg)
    assert limits["mean_rate_limit"] == cfg["silence_test"]["bound_factor"] * cliff["music_level_at_1x"]
    assert limits["rate_ceiling"] > limits["mean_rate_limit"]


def test_history_readout_exports_and_matches(cfg, slice_small, tmp_path):
    import copy
    neurons, edges = slice_small
    on_cfg = copy.deepcopy(cfg)
    on_cfg["model"]["readout_history"] = {"present": True, "blocks_ms": [50, 50, 50, 50], "profile": "per_drum",
                                          "init": "present_only", "velocity": "same"}
    model = make_model(on_cfg, neurons, edges, seed=3)
    with torch.no_grad():
        model.p_hit[:, 1:] = 0.4                            # weigh the past
        model.p_vel[:, 1:] = -0.2
    out = tmp_path / "hist.flybrain"
    export(model, neurons, edges, on_cfg, out, name="history")
    b = read_brain(out)
    assert list(b["history_blocks"]) == [10, 10, 10, 10] and b.meta["readout_history"]
    a = (np.random.default_rng(8).standard_normal(16000) * 0.1).astype(np.float32)
    rep = compare(run_clip(b, a), torch_outputs(model, a))
    assert rep["passes"], rep


def test_cli_help_imports(capsys):
    from flybrain import cli
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    assert "observation_only" in capsys.readouterr().out
