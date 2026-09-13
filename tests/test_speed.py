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
            return torch.repeat_interleave(drive, substeps, dim=1), state

    class FakeDecoder:
        def __init__(self, pattern):
            self.pattern, self.i = pattern, 0

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
