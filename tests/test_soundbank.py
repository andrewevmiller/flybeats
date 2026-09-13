"""The sound layer is what makes the model audible without other software,
so its failure modes are the ones a user meets first: a kit missing a class,
two hi-hats sounding at once, the same snare sample twice in a row, or a mixer
that allocates inside what will become an audio callback.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from decoder import DrumKit  # noqa: E402
from soundbank import MidiBank, SampleBank, make_bank  # noqa: E402
from voice import CHOKE_MS, RoundRobin, VoicePool, layer_gains  # noqa: E402

KIT = ROOT / "kits" / "synth"
needs_kit = pytest.mark.skipif(
    not (KIT / "manifest.yaml").exists(),
    reason="run scripts/make_synth_kit.py to generate the starter kit")


def _bank():
    b = SampleBank(sample_rate=22_050)
    b.load({"dir": str(KIT)})
    return b


def test_voices_of_different_lengths_can_be_removed():
    """A dataclass __eq__ over numpy buffers made list.remove compare arrays
    elementwise, which raised as soon as two voices had different lengths."""
    pool = VoicePool(sample_rate=22_050)
    pool.start(np.ones(10, dtype=np.float32))
    pool.start(np.ones(4000, dtype=np.float32))
    pool.mix(64)                      # the short one finishes and is removed
    assert pool.active == 1


def test_choke_group_silences_the_previous_voice():
    pool = VoicePool(sample_rate=22_050)
    pool.start(np.ones(22_050, dtype=np.float32), group="hihat", cls="hat_open")
    pool.start(np.ones(22_050, dtype=np.float32), group="hihat", cls="hat_closed")
    # after the fade window only the new voice is left
    pool.mix(int(22_050 * CHOKE_MS / 1000.0) + 8)
    assert pool.active == 1
    assert pool.voices[0].cls == "hat_closed"


def test_ungrouped_voices_do_not_choke_each_other():
    pool = VoicePool(sample_rate=22_050)
    pool.start(np.ones(2205, dtype=np.float32), cls="kick")
    pool.start(np.ones(2205, dtype=np.float32), cls="snare")
    pool.mix(64)
    assert pool.active == 2


def test_voice_pool_is_bounded():
    pool = VoicePool(max_voices=4, sample_rate=22_050)
    for _ in range(20):
        pool.start(np.ones(22_050, dtype=np.float32))
    assert pool.active <= 4


def test_fade_is_monotonic_and_reaches_zero():
    pool = VoicePool(sample_rate=22_050)
    pool.start(np.ones(22_050, dtype=np.float32), group="g")
    pool.start(np.zeros(22_050, dtype=np.float32), group="g")   # choke, silent
    out = pool.mix(int(22_050 * CHOKE_MS / 1000.0))
    assert out[0] > out[-1] and out[-1] < 0.05, "choke must ramp down, not click"


def test_layer_crossfade_is_continuous_and_sums_to_one():
    for v in np.linspace(0, 1, 21):
        gains = layer_gains(float(v), 3)
        assert abs(sum(g for _, g in gains) - 1.0) < 1e-6
        assert all(0 <= i < 3 for i, _ in gains)
    assert layer_gains(0.0, 3)[0][0] == 0
    assert layer_gains(1.0, 3)[0][0] == 2


def test_round_robin_never_repeats_immediately():
    for seed in range(5):
        r = RoundRobin(3, seed=seed)
        seq = [r.next() for _ in range(60)]
        assert all(a != b for a, b in zip(seq, seq[1:])), seq
        assert set(seq) == {0, 1, 2}


def test_single_variant_round_robin_is_stable():
    r = RoundRobin(1, seed=0)
    assert [r.next() for _ in range(3)] == [0, 0, 0]


def test_midi_bank_matches_the_original_velocity_mapping():
    """The refactor must not change what a sampler receives."""
    kit = DrumKit.from_tier("8piece")
    bank = MidiBank(kit)
    assert bank.to_midi_velocity(0.0) == 40
    assert bank.to_midi_velocity(1.0) == 127
    bank.trigger("snare", 0.5, 1.25)
    t, note, vel = bank.events[0]
    assert (t, note) == (1.25, kit.notes[kit.classes.index("snare")])
    assert 80 <= vel <= 90


def test_unknown_class_is_silent_not_fatal():
    bank = MidiBank(DrumKit.from_tier("3piece"))
    bank.trigger("crash", 1.0, 0.0)         # not in a 3-piece kit
    assert bank.events == []


@needs_kit
def test_sample_bank_loads_and_sounds():
    bank = _bank()
    assert bank.validate(DrumKit.from_tier("8piece").classes) == []
    assert bank.mix(128).max() == 0.0, "silent before anything is triggered"
    bank.trigger("kick", 0.8, 0.0)
    assert np.abs(bank.mix(2048)).max() > 0.0


@needs_kit
def test_sample_bank_reports_classes_it_cannot_sound(tmp_path):
    (tmp_path / "kick").mkdir()
    import soundfile as sf
    sf.write(tmp_path / "kick" / "0_0.wav", np.zeros(64, dtype=np.float32), 22_050)
    bank = SampleBank()
    bank.load({"dir": str(tmp_path)})
    missing = bank.validate(["kick", "snare", "crash"])
    assert missing == ["snare", "crash"]
    bank.trigger("snare", 1.0, 0.0)          # silence, not an exception
    assert bank.mix(64).max() == 0.0


@needs_kit
def test_hot_swap_keeps_sounding_voices_alive(tmp_path):
    bank = _bank()
    bank.trigger("crash", 1.0, 0.0)
    bank.mix(64)
    before = bank.pool.active
    bank.hot_swap({"dir": str(KIT), "name": "again"})
    assert bank.pool.active == before, "a swap must not cut the tails"
    assert np.abs(bank.mix(2048)).max() > 0.0


@needs_kit
def test_make_bank_selects_the_backend():
    kit = DrumKit.from_tier("8piece")
    assert isinstance(make_bank("midi", kit, 22_050), MidiBank)
    assert isinstance(make_bank("samples", kit, 22_050, KIT), SampleBank)
    with pytest.raises(ValueError):
        make_bank("gramophone", kit, 22_050)


def _ramp_drummer(classes, **kw):
    """A drummer over a fake model whose output is a fixed per-class ramp."""
    import torch
    from decoder import DrumKit
    from realtime import StreamingDrummer

    n = len(classes)
    # every class peaks on the same steps, so only the refractory can separate
    # what fires from what does not
    ramp = np.array([0.05, 0.9, 0.05, 0.9, 0.05, 0.9, 0.05, 0.9,
                     0.05, 0.9, 0.05, 0.9, 0.05, 0.9, 0.05, 0.05], dtype=np.float32)

    class FakeEncoder:
        hop = 110
        context_samples = 256

        def forward_window(self, wav, start, k):
            return torch.zeros(1, k, 1)

    class FakeRNN:
        def initial_state(self, b, device=None, dtype=None):
            return torch.zeros(b, 1)

        def __call__(self, drive, state=None, tonic=None, substeps=1):
            # the real core holds the drive across sub-steps, which is exactly
            # repeat_interleave -- see tests/test_speed.py
            reps = (torch.tensor(substeps) if not isinstance(substeps, int)
                    else substeps)
            return torch.repeat_interleave(drive, reps, dim=1), state

    class FakeDecoder:
        def __init__(self):
            self.i = 0

        def velocity(self, rates):
            # No velocity head, as a model trained before it existed. The
            # drummer must fall back to peak height, which is what these
            # tests measure.
            return None

        def __call__(self, rates):
            k = rates.shape[1]
            vals = ramp[self.i: self.i + k]
            self.i += k
            logits = torch.tensor([[float(np.log(v / (1 - v)))] * n for v in vals])
            return logits.unsqueeze(0)

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.rnn, self.decoder, self.genre = \
                FakeEncoder(), FakeRNN(), FakeDecoder(), None

    return StreamingDrummer(FakeModel(), DrumKit(list(classes)), 22_050, 5.0,
                            threshold=0.3, **kw)


def test_per_class_refractory_lets_one_class_flutter():
    """'Hats flutter while the kick stays human' is this, and nothing else:
    the dynamics offer the peaks, the refractory decides who may use them."""
    d = _ramp_drummer(["kick", "hat_closed"],
                      class_refractory_ms={"kick": 60.0, "hat_closed": 5.0})
    hits = d.push(np.zeros(110 * 16, dtype=np.float32))
    hats = [h for h in hits if h.cls == "hat_closed"]
    kicks = [h for h in hits if h.cls == "kick"]
    assert len(hats) > len(kicks), f"hats {len(hats)} vs kicks {len(kicks)}"
    assert len(kicks) >= 1, "the slow class must still play"


def test_refractory_is_honoured_in_milliseconds():
    d = _ramp_drummer(["kick"], class_refractory_ms={"kick": 60.0})
    hits = d.push(np.zeros(110 * 16, dtype=np.float32))
    gaps = [(b.t - a.t) * 1000.0 for a, b in zip(hits, hits[1:])]
    assert all(g >= 60.0 - 1e-9 for g in gaps), gaps


def test_a_class_without_an_entry_gets_the_default():
    d = _ramp_drummer(["kick", "snare"], refractory_ms=45.0,
                      class_refractory_ms={"kick": 5.0})
    assert d.refractory_ms == {"kick": 5.0, "snare": 45.0}
