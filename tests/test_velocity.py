"""The velocity head: how hard, kept separate from whether.

Before this head, the "velocity" a drum machine received was the height of a
detection peak above the threshold. A model that was merely confident played
loudly. These tests pin the three places that could quietly put the old
behaviour back: the targets carrying the drummer's real MIDI velocity, the loss
scoring it only where a hit is, and the streaming path reading the head rather
than the peak.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as T  # noqa: E402
from dataset import SyntheticDrums, smooth_onsets, smooth_targets  # noqa: E402
from decoder import DrumKit, MotorToDrums  # noqa: E402


# --- targets ---------------------------------------------------------------

def test_velocity_is_held_flat_across_the_hit_not_shaped_like_the_kernel():
    """Velocity is a property of the hit, not of nearness to its centre.

    Scaling it by the Gaussian would make the head regress the detection
    envelope a second time -- the exact confusion two heads exist to end.
    """
    y, vel = smooth_targets([(0.5, "kick", 0.4)], ["kick"], 400, step_ms=5.0)
    support = y[:, 0] > 0
    assert support.sum() > 1, "the onset kernel should cover several steps"
    assert np.allclose(vel[support, 0], 0.4), "velocity must be flat over the hit"
    assert np.all(vel[~support, 0] == 0.0), "silence carries no velocity"
    # and the onset plane is untouched by any of this
    assert np.array_equal(y, smooth_onsets([(0.5, "kick")], ["kick"], 400, 5.0))


def test_overlapping_hits_take_the_velocity_of_the_nearer_one():
    """Where two hits of one class overlap, the two planes must agree about
    which hit a step belongs to -- the taller kernel wins in both."""
    y, vel = smooth_targets([(0.50, "snare", 0.2), (0.56, "snare", 0.9)],
                            ["snare"], 400, step_ms=5.0)
    near_first = int(round(0.50 * 1000 / 5.0))
    near_second = int(round(0.56 * 1000 / 5.0))
    assert vel[near_first, 0] == pytest.approx(0.2)
    assert vel[near_second, 0] == pytest.approx(0.9)
    assert y[near_first, 0] > 0 and y[near_second, 0] > 0


def test_a_two_tuple_event_means_unknown_velocity_not_silence():
    _, vel = smooth_targets([(0.5, "kick")], ["kick"], 400, step_ms=5.0)
    assert vel.max() == pytest.approx(1.0)


def test_the_synthetic_click_track_carries_real_dynamics():
    """The synthetic corpus randomises each hit's amplitude, and that amplitude
    is now the velocity target -- so the sanity run can actually learn one."""
    ds = SyntheticDrums(["kick", "snare", "hat_closed"], n_clips=2, seconds=2.0,
                        sample_rate=22050, step_ms=5.0)
    _, y, vel, _, _ = ds[0]
    hit = y.numpy() > 0.5
    v = vel.numpy()[hit]
    assert v.size > 0
    assert v.min() < v.max() - 0.05, "velocity targets must vary between hits"
    assert 0.0 < v.min() and v.max() <= 1.0


# --- loss ------------------------------------------------------------------

def test_velocity_loss_ignores_everything_that_is_not_a_hit():
    """98% of steps are silence. Scored unweighted, the head would learn the
    mean of nothing, so silence must contribute exactly zero."""
    onsets = torch.zeros(1, 8, 1)
    onsets[0, 3, 0] = 1.0
    target = torch.zeros(1, 8, 1)
    target[0, 3, 0] = 0.7

    exact = target.clone()
    wrong_only_in_silence = target.clone()
    wrong_only_in_silence[0, 5, 0] = 1.0        # nonsense where there is no hit

    assert float(T.velocity_loss(exact, target, onsets)) == pytest.approx(0.0)
    assert float(T.velocity_loss(wrong_only_in_silence, target, onsets)) == pytest.approx(0.0)


def test_velocity_loss_is_weighted_by_how_much_of_a_hit_is_there():
    onsets = torch.tensor([[[1.0], [0.5]]])
    target = torch.zeros(1, 2, 1)
    pred = torch.tensor([[[0.2], [0.2]]])
    # (0.04*1 + 0.04*0.5) / 1.5
    assert float(T.velocity_loss(pred, target, onsets)) == pytest.approx(0.04)


def test_velocity_loss_is_zero_when_a_chunk_holds_no_onsets():
    """Quiet intros are real. Dividing by an empty weight sum is not."""
    onsets = torch.zeros(1, 4, 2)
    loss = T.velocity_loss(torch.rand(1, 4, 2), torch.rand(1, 4, 2), onsets)
    assert float(loss) == 0.0 and torch.isfinite(loss)


# --- the head --------------------------------------------------------------

def test_the_head_is_bounded_and_optional():
    kit = DrumKit(["kick", "snare"])
    rates = torch.randn(2, 5, 16) * 10

    with_head = MotorToDrums(n_motor=16, kit=kit)
    with torch.no_grad():
        v = with_head.velocity(rates)
    assert with_head.has_velocity and v.shape == (2, 5, 2)
    assert float(v.min()) >= 0.0 and float(v.max()) <= 1.0

    without = MotorToDrums(n_motor=16, kit=kit, velocity_head=False)
    assert not without.has_velocity and without.velocity(rates) is None
    # and the detection head is unaffected either way
    assert with_head(rates).shape == without(rates).shape


def test_the_velocity_head_wears_the_same_hemisphere_mask():
    """A hit's strength has to come from the hemisphere that produced the hit,
    or bilateral independence holds for onsets and leaks through velocity."""
    kit = DrumKit(["snare", "hat_closed"])          # left hand, right hand
    side = np.array(["L"] * 8 + ["R"] * 8)
    dec = MotorToDrums(n_motor=16, kit=kit, motor_side=side, bilateral=True)
    assert dec.has_velocity

    masked = dec.vel_readout.weight * dec.mask
    for c, name in enumerate(kit.classes):
        live = dec.mask[c] > 0
        assert float(masked[c][~live].abs().sum()) == 0.0, f"{name} reads the wrong side"
        assert live.sum() == 8


def test_velocity_rides_along_with_one_pass_of_the_model():
    """``with_velocity`` must not change what the other outputs are."""
    from build import build_model, get_subgraph, load_config

    from conftest import use_small_graph

    cfg = use_small_graph(load_config(ROOT / "configs" / "sanity_3piece.yaml"))

    torch.manual_seed(0)
    model, _ = build_model(cfg, get_subgraph(cfg), n_styles=1)
    model.eval()
    wav = torch.randn(1, 11025, generator=torch.Generator().manual_seed(2)) * 0.1
    with torch.no_grad():
        a, _ = model(wav)
        b, _, vel = model(wav, with_velocity=True)
    assert torch.equal(a, b), "asking for velocity changed the onset logits"
    assert vel.shape == a.shape
    assert float(vel.min()) >= 0.0 and float(vel.max()) <= 1.0


# --- inference -------------------------------------------------------------

def _drummer(velocity_of_step):
    """A streaming drummer over a fake model with a fixed detection ramp.

    ``velocity_of_step`` is None for a model with no head, or a per-step
    velocity the head reports.
    """
    from realtime import StreamingDrummer

    ramp = np.array([0.05, 0.9, 0.05, 0.05], dtype=np.float32)

    class FakeEncoder:
        hop = 110
        context_samples = 256

        def forward_window(self, wav, start, k):
            return torch.zeros(1, k, 1)

    class FakeRNN:
        def initial_state(self, b, device=None, dtype=None):
            return torch.zeros(b, 1)

        def __call__(self, drive, state=None, tonic=None, substeps=1):
            return torch.zeros(1, drive.shape[1], 1), state

    class FakeDecoder:
        def __call__(self, rates):
            n = rates.shape[1]
            return torch.tensor([[[float(np.log(v / (1 - v)))] for v in ramp[:n]]])

        def velocity(self, rates):
            if velocity_of_step is None:
                return None
            n = rates.shape[1]
            return torch.tensor([[[velocity_of_step] for _ in range(n)]])

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.rnn = FakeEncoder(), FakeRNN()
            self.decoder, self.genre = FakeDecoder(), None

    return StreamingDrummer(FakeModel(), DrumKit(["kick"]), 22_050, 5.0,
                            threshold=0.3, refractory_ms=1.0)


def test_streaming_reads_velocity_from_the_head_not_the_peak():
    hits = _drummer(0.25).push(np.zeros(110 * 4, dtype=np.float32))
    assert len(hits) == 1
    # peak height would give (0.9 - 0.3) / 0.7 = 0.857; the head says 0.25
    assert hits[0].velocity == pytest.approx(0.25)


def test_a_model_without_a_head_still_plays_with_peak_height():
    """Bundles exported before the head exists must keep working."""
    hits = _drummer(None).push(np.zeros(110 * 4, dtype=np.float32))
    assert len(hits) == 1
    assert hits[0].velocity == pytest.approx((0.9 - 0.3) / 0.7, abs=1e-3)


# --- the Phase A' knobs ------------------------------------------------------

def test_a_linear_head_is_allowed_to_leave_the_unit_interval():
    """The sigmoid's gradient is flattest exactly where GMD's velocities sit,
    so the bounded head has least reason to move where it matters most. A
    linear head can overshoot; the streaming path clips it, and being visibly
    out of range beats being squashed into range."""
    kit = DrumKit(["kick", "snare"])
    rates = torch.randn(1, 6, 16, generator=torch.Generator().manual_seed(3)) * 8

    with torch.no_grad():
        bounded = MotorToDrums(16, kit).velocity(rates)
        linear = MotorToDrums(16, kit, velocity_activation="linear").velocity(rates)

    assert 0.0 <= float(bounded.min()) and float(bounded.max()) <= 1.0
    assert float(linear.min()) < 0.0 or float(linear.max()) > 1.0


def test_an_unknown_velocity_activation_is_refused():
    with pytest.raises(ValueError, match="sigmoid or linear"):
        MotorToDrums(16, DrumKit(["kick"]), velocity_activation="softmax")


def test_peak_only_scores_the_step_the_drummer_is_read_at():
    """The streaming path reads velocity at the picked peak and nowhere else,
    so an error out on the kernel's skirt must stop counting."""
    onsets = torch.tensor([[[1.0], [0.6]]])       # one peak step, one skirt step
    target = torch.zeros(1, 2, 1)
    pred = torch.tensor([[[0.0], [1.0]]])         # right at the peak, wrong off it

    full = float(T.velocity_loss(pred, target, onsets))
    peak = float(T.velocity_loss(pred, target, onsets, peak_only=0.95))
    assert full > 0.3, "the off-peak error should dominate the unrestricted loss"
    assert peak == pytest.approx(0.0), "peak-only must ignore the skirt entirely"


def test_peak_only_off_by_default_keeps_the_old_weighting():
    onsets = torch.tensor([[[1.0], [0.6]]])
    target = torch.zeros(1, 2, 1)
    pred = torch.tensor([[[0.0], [1.0]]])
    assert float(T.velocity_loss(pred, target, onsets)) == pytest.approx(
        float(T.velocity_loss(pred, target, onsets, peak_only=0.0)))
