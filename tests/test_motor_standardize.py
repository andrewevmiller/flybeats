"""The decoder's optional motor standardisation.

A fixed affine in front of both readouts, calibrated once on the untrained
network. The readouts it feeds were found sitting at their initialisation
after training, because the raw motor rates are a large shared level with the
dynamics in a small wobble on top. These pin that the affine is inert until
asked for, exact once calibrated, and carried by the state dict.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decoder import DrumKit, MotorToDrums  # noqa: E402

KIT = DrumKit(["kick", "snare", "hat_closed"])


def _rates(seed=0):
    g = torch.Generator().manual_seed(seed)
    return 3.0 + 0.05 * torch.randn(2, 50, 16, generator=g)


def test_off_by_default_and_no_new_state():
    """Every checkpoint trained so far has none of these buffers."""
    dec = MotorToDrums(16, KIT)
    assert not dec.standardize_motor
    assert not any(k.startswith("motor_") for k in dec.state_dict())
    assert dec.calibrate(_rates()) == {"calibrated": False}


def test_uncalibrated_standardisation_is_the_identity():
    torch.manual_seed(0)
    plain = MotorToDrums(16, KIT)
    torch.manual_seed(0)
    std = MotorToDrums(16, KIT, standardize_motor=True)
    r = _rates()
    assert torch.allclose(plain(r), std(r))
    assert torch.allclose(plain.velocity(r), std.velocity(r))


def test_calibration_centres_and_scales_each_unit():
    dec = MotorToDrums(16, KIT, standardize_motor=True)
    r = _rates()
    stats = dec.calibrate([r[:1], r[1:]])
    z = dec._inputs(r).reshape(-1, 16)
    assert stats["calibrated"] and stats["frames"] == 100
    assert stats["level_to_sd"] > 100          # the shape that stalled the heads
    assert z.mean(0).abs().max() < 1e-4
    assert (z.std(0) - 1).abs().max() < 1e-3


def test_both_readouts_see_the_standardised_rates():
    dec = MotorToDrums(16, KIT, standardize_motor=True)
    r = _rates()
    dec.calibrate(r)
    z = (r - dec.motor_mean) / dec.motor_scale
    w = dec.readout.weight * dec.mask
    assert torch.allclose(dec(r), torch.nn.functional.linear(z, w, dec.readout.bias), atol=1e-5)
    wv = dec.vel_readout.weight * dec.mask
    assert torch.allclose(dec.velocity(r),
                          torch.sigmoid(torch.nn.functional.linear(z, wv, dec.vel_readout.bias)),
                          atol=1e-5)


def test_a_silent_unit_is_not_divided_by_zero():
    dec = MotorToDrums(16, KIT, standardize_motor=True)
    r = _rates()
    r[..., 5] = 2.0
    stats = dec.calibrate(r)
    assert stats["flat_units"] == 1
    assert dec.motor_scale[5] == 1.0
    assert torch.isfinite(dec(r)).all()


def test_calibration_round_trips_through_the_state_dict():
    dec = MotorToDrums(16, KIT, standardize_motor=True)
    dec.calibrate(_rates())
    fresh = MotorToDrums(16, KIT, standardize_motor=True)
    fresh.load_state_dict(dec.state_dict())
    assert bool(fresh.motor_calibrated)
    r = _rates(seed=1)
    assert torch.allclose(fresh(r), dec(r))
    assert np.allclose(fresh.velocity(r).detach().numpy(), dec.velocity(r).detach().numpy())
