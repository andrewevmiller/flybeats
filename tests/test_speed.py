"""The speed control: k core updates per encoder frame, holding the drive.

The invariant that matters is that speed changes *when* the network's trajectory
happens, not *what* it is. alpha = step / tau, and going k times faster divides
both by k -- so the discrete update must stay bit-identical and only its mapping
onto wall-clock time may move. A later "simplification" into a tau rescale would
break that silently: it would still run, still sound like drums, and no longer
be the model that was trained. Hence the exactness here.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import ConnectomeRNN, ModelConfig  # noqa: E402


def _rnn(n=48, e=240, seed=0):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    return ConnectomeRNN(
        edge_index=rng.integers(0, n, (2, e)).astype(np.int64),
        edge_sign=rng.choice([-1.0, 1.0], e).astype(np.float32),
        weight=(rng.random(e) * 5 + 1).astype(np.float32),
        n_nodes=n, sensory_idx=np.arange(6), motor_idx=np.arange(6, 12),
        cfg=ModelConfig(gain_scale=1.0),
    )


@pytest.mark.parametrize("k", [1, 2, 3, 4, 8])
def test_substeps_equal_holding_the_drive(k):
    """Sub-stepping IS repeat_interleave on the drive. Exactly, not nearly."""
    rnn = _rnn()
    drive = torch.randn(2, 9, 6, generator=torch.Generator().manual_seed(1))
    fast, vf = rnn(drive, substeps=k)
    held, vh = rnn(torch.repeat_interleave(drive, k, dim=1), substeps=1)
    assert torch.equal(fast, held), (fast - held).abs().max()
    assert torch.equal(vf, vh), "the carried state must match too"


def test_speed_one_is_the_old_behaviour():
    rnn = _rnn()
    drive = torch.randn(1, 12, 6, generator=torch.Generator().manual_seed(2))
    a, va = rnn(drive)
    b, vb = rnn(drive, substeps=1)
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_output_grid_scales_with_speed():
    rnn = _rnn()
    drive = torch.randn(1, 10, 6, generator=torch.Generator().manual_seed(3))
    for k in (1, 2, 5):
        out, _ = rnn(drive, substeps=k)
        assert out.shape[1] == 10 * k


def test_alpha_does_not_move_with_speed():
    """The whole argument for this design. If speed ever starts changing alpha,
    it has stopped being the trained network."""
    rnn = _rnn()
    before = (1.0 / rnn.tau_steps).clamp(max=1.0).clone()
    drive = torch.randn(1, 6, 6, generator=torch.Generator().manual_seed(4))
    rnn(drive, substeps=8)
    assert torch.equal((1.0 / rnn.tau_steps).clamp(max=1.0), before)


def test_state_stays_bounded_at_high_speed():
    rnn = _rnn()
    drive = torch.randn(1, 20, 6, generator=torch.Generator().manual_seed(5)) * 3
    out, v = rnn(drive, substeps=16)
    assert torch.isfinite(out).all() and torch.isfinite(v).all()
    assert float(v.abs().max()) <= rnn.cfg.state_clip + 1e-6


def test_substeps_must_be_at_least_one():
    rnn = _rnn()
    drive = torch.randn(1, 4, 6)
    for bad in (0, -3):
        with pytest.raises(ValueError, match="substeps"):
            rnn(drive, substeps=bad)


def test_return_all_also_sub_steps():
    rnn = _rnn()
    drive = torch.randn(1, 5, 6, generator=torch.Generator().manual_seed(6))
    out, _ = rnn(drive, return_all=True, substeps=3)
    assert out.shape == (1, 15, rnn.n_nodes)


def test_streaming_timestamps_land_on_the_finer_grid():
    """Speed resolves the same frame into more of them, so event times must
    move onto the finer grid rather than staying on the encoder's."""
    from decoder import DrumKit
    from realtime import StreamingDrummer

    class FakeEncoder:
        hop = 110
        context_samples = 256

        def forward_window(self, wav, start, n):
            return torch.zeros(1, n, 1)

    class FakeRNN:
        def initial_state(self, b, device=None, dtype=None):
            return torch.zeros(b, 1)

        def __call__(self, drive, state=None, tonic=None, substeps=1):
            reps = (torch.tensor(substeps) if not isinstance(substeps, int)
                    else substeps)
            return torch.repeat_interleave(drive, reps, dim=1), state

    class FakeDecoder:
        def __init__(self, pattern):
            self.pattern, self.i = pattern, 0

        def velocity(self, rates):
            # No velocity head: this test is about where hits land in time,
            # so the drummer should use the peak-height fallback.
            return None

        def __call__(self, rates):
            n = rates.shape[1]
            vals = self.pattern[self.i: self.i + n]
            self.i += n
            return torch.tensor([[[float(np.log(v / (1 - v)))] for v in vals]])

    pattern = np.array([0.05, 0.9, 0.05, 0.9, 0.05, 0.9, 0.05, 0.05] * 4, dtype=np.float32)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.rnn, self.decoder, self.genre = \
                FakeEncoder(), FakeRNN(), FakeDecoder(pattern), None

    d = StreamingDrummer(FakeModel(), DrumKit(["kick"]), 22_050, 5.0,
                         threshold=0.3, refractory_ms=1.0, speed=4)
    assert d.substeps == 4
    hits = d.push(np.zeros(110 * 4, dtype=np.float32))
    assert hits, "the fine grid must still produce hits"
    fine = 110 / (22_050 * 4)
    for h in hits:
        steps = h.t / fine
        assert abs(steps - round(steps)) < 1e-9, h.t
    # and inside the audio it came from: 4 frames is ~20 ms
    assert max(h.t for h in hits) < 4 * 110 / 22_050


# --- fractional speeds and ramping (SPEED_PLAN steps 5 and 6) --------------

def _sched_drummer(speed, classes=("kick",)):
    """A drummer over a fake core that just repeats each frame's drive, so the
    schedule the drummer chose is readable straight off the output length."""
    from decoder import DrumKit
    from realtime import StreamingDrummer

    class FakeEncoder:
        hop = 110
        context_samples = 256

        def forward_window(self, wav, start, n):
            return torch.zeros(1, n, 1)

    class FakeRNN:
        def initial_state(self, b, device=None, dtype=None):
            return torch.zeros(b, 1)

        def __call__(self, drive, state=None, tonic=None, substeps=1):
            reps = (torch.tensor(substeps) if not isinstance(substeps, int)
                    else substeps)
            return torch.repeat_interleave(drive, reps, dim=1), state

    class FakeDecoder:
        def __call__(self, rates):
            n = rates.shape[1]
            return torch.full((1, n, len(classes)), -4.0)   # never fires

        def velocity(self, rates):
            return None

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.rnn = FakeEncoder(), FakeRNN()
            self.decoder, self.genre = FakeDecoder(), None

    return StreamingDrummer(FakeModel(), DrumKit(list(classes)), 22_050, 5.0,
                            threshold=0.3, speed=speed)


def test_a_fractional_speed_alternates_rather_than_rounding():
    """2.5 is not 2.5 updates on any one frame -- there is no such thing. It is
    alternating 2 and 3, and the average is what the dial promises."""
    d = _sched_drummer(2.5)
    sched = d._schedule(8)
    assert set(sched) == {2, 3}, sched
    assert sum(sched) == 20, "eight frames at 2.5 is twenty updates"


def test_the_phase_carries_across_blocks():
    """Reset the phase per block and 2.5 with four frames per block becomes
    3,2,3,2 every time -- an average of 2.5 by luck of the block size, drifting
    the moment the block changes. The accumulator has to survive the boundary."""
    d = _sched_drummer(2.5)
    blocks = [d._schedule(3) for _ in range(4)]
    flat = [k for b in blocks for k in b]
    assert sum(flat) == 30, f"twelve frames at 2.5 is thirty updates, got {sum(flat)}"
    assert blocks[0] != blocks[1] or blocks[1] != blocks[2], \
        "identical blocks means the phase reset at the boundary"


def test_an_integer_speed_still_gives_every_frame_the_same_k():
    for speed in (1, 2, 4, 8):
        d = _sched_drummer(float(speed))
        assert d._schedule(6) == [speed] * 6


def test_timestamps_stay_on_the_frame_grid_when_k_varies():
    """With k varying there is no uniform grid, but every sub-step must still
    sit inside its own frame's span, in order, and the frame boundaries must
    land exactly where the encoder's hop puts them."""
    d = _sched_drummer(2.5)
    frame_s = 110 / 22_050
    sched = d._schedule(4)
    times = d._step_times(sched)

    assert times == sorted(times), "sub-step times must be monotonic"
    assert len(times) == sum(sched)
    i = 0
    for f, k in enumerate(sched):
        assert times[i] == pytest.approx(f * frame_s), "frame must start on the hop grid"
        for j in range(k):
            lo, hi = f * frame_s, (f + 1) * frame_s
            assert lo <= times[i] < hi, f"sub-step {j} escaped frame {f}"
            i += 1


def test_a_uniform_speed_reproduces_the_old_uniform_grid_exactly():
    """The regression that matters: switching to per-frame timing must not move
    a single event at an integer speed."""
    for speed in (1, 2, 4):
        d = _sched_drummer(float(speed))
        sched = d._schedule(5)
        times = d._step_times(sched)
        span = 110 / (22_050 * speed)
        assert times == pytest.approx([i * span for i in range(len(times))])


def test_speed_can_change_between_blocks_without_a_discontinuity():
    """Ramping: k may change block to block -- the state carries and no
    parameter jumps -- but time must not go backwards at the seam."""
    d = _sched_drummer(1.0)
    t1 = d._step_times(d._schedule(4))
    d.frame += 4
    d.speed = 4.0
    t2 = d._step_times(d._schedule(4))

    assert t2[0] > t1[-1], "time went backwards when the dial moved"
    frame_s = 110 / 22_050
    assert t2[0] == pytest.approx(4 * frame_s), "the seam must land on a frame boundary"
    assert len(t2) == 16 and t2 == sorted(t2)


def test_a_uniform_schedule_is_exactly_the_scalar_it_spells_out():
    """The generalisation must not perturb the scalar path by a single bit."""
    rnn = _rnn()
    drive = torch.randn(1, 5, 6, generator=torch.Generator().manual_seed(11))
    a, va = rnn(drive, substeps=3)
    b, vb = rnn(drive, substeps=[3] * 5)
    assert torch.equal(a, b) and torch.equal(va, vb)


def test_a_schedule_that_does_not_match_the_frames_is_refused():
    rnn = _rnn()
    drive = torch.randn(1, 4, 6, generator=torch.Generator().manual_seed(12))
    with pytest.raises(ValueError, match="schedule"):
        rnn(drive, substeps=[2, 2, 2])
    with pytest.raises(ValueError, match="substeps"):
        rnn(drive, substeps=[2, 0, 2, 2])
