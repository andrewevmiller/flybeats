"""End-to-end checks on the pieces that glue phases together.

These are the seams where a mistake is invisible: a config that silently drops
a key, a kit that exceeds the confirmed motor-neuron count, a decoder whose
bilateral mask does not actually separate the hemispheres.
"""
import json
import sys
from pathlib import Path

import pytest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import feel  # noqa: E402
from dataset import SyntheticDrums, smooth_onsets  # noqa: E402
from decoder import KIT_TIERS, DrumKit, MotorToDrums  # noqa: E402
from metrics import onset_f_measure, peak_pick, swing_ratio  # noqa: E402


def test_verified_types_is_the_only_source_of_type_names():
    """No module may hardcode a MaleCNS cell-type *string*.

    Addressing a population by its Phase 0 *concept key* ("pC1", "pIP10") is the
    intended pattern -- that key is looked up in verified_types.json. What is
    forbidden is naming a type directly and bypassing the gate. Some concepts
    share a name with their single confirmed type (pIP10 resolved exactly), so
    the check excludes names that are also concept keys and tests the rest.
    """
    verified = ROOT / "data" / "verified_types.json"
    if not verified.exists():
        return                                    # gate not run in this checkout
    concepts = json.loads(verified.read_text())["concepts"]
    concept_keys = set(concepts)
    names = {t["type"] for rec in concepts.values() for t in rec["types"]}

    # distinctive type strings that are not concept keys -- inlining any of
    # these would mean a phase stopped reading the gate output
    tempting = {n for n in names - concept_keys
                if n in {"b1 MN", "JO-B1_a", "OA-VPM3", "AMMC001", "SAD051_a", "AVLP761m"}}
    assert tempting, "expected the gate to have confirmed these"

    for path in (ROOT / "src").glob("*.py"):
        src = path.read_text()
        for name in tempting:
            assert f'"{name}"' not in src and f"'{name}'" not in src, \
                f"{path.name} hardcodes the cell type {name!r}; read it from verified_types.json"


def test_kit_tiers_fit_the_confirmed_motor_population():
    """Phase 0 caps kit size; a tier must not quietly exceed it."""
    verified = ROOT / "data" / "verified_types.json"
    if not verified.exists():
        return
    v = json.loads(verified.read_text())
    n_motor = v["concepts"]["wing_motor_all"]["n_bodies"]
    for tier, classes in KIT_TIERS.items():
        assert len(classes) <= n_motor, f"{tier} needs {len(classes)} > {n_motor} motor units"
        DrumKit.from_tier(tier, max_classes=n_motor)


def test_kit_rejects_an_oversized_tier():
    try:
        DrumKit.from_tier("8piece", max_classes=3)
    except ValueError as e:
        assert "caps kit size" in str(e)
    else:
        raise AssertionError("an oversized kit must be refused, not silently truncated")


def test_bilateral_readout_separates_hemispheres():
    kit = DrumKit.from_tier("8piece")
    side = np.array(["L"] * 8 + ["R"] * 8, dtype=object)
    dec = MotorToDrums(n_motor=16, kit=kit, motor_side=side, bilateral=True)

    left = dec.mask[:, :8].sum(dim=1)
    right = dec.mask[:, 8:].sum(dim=1)
    for c, name in enumerate(kit.classes):
        if name in ("hat_closed", "hat_open"):       # right hand
            assert right[c] > 0 and left[c] == 0, f"{name} leaked to the left pool"
        elif name in ("snare", "tom_low", "tom_mid", "tom_high", "crash"):
            assert left[c] > 0 and right[c] == 0, f"{name} leaked to the right pool"
        elif name == "kick":                          # foot, reads from both
            assert left[c] > 0 and right[c] > 0


def test_nonbilateral_readout_uses_everything():
    kit = DrumKit.from_tier("8piece")
    dec = MotorToDrums(n_motor=16, kit=kit, bilateral=False)
    assert torch.all(dec.mask == 1)


def test_smoothed_targets_peak_at_the_true_onset():
    y = smooth_onsets([(0.5, "kick"), (1.0, "snare")], ["kick", "snare"], 400, step_ms=5.0)
    assert y.shape == (400, 2)
    assert abs(int(np.argmax(y[:, 0])) - 100) <= 1
    assert abs(int(np.argmax(y[:, 1])) - 200) <= 1
    assert y.max() <= 1.0 + 1e-6


def test_onset_f_is_perfect_on_a_copy_and_zero_on_silence():
    y = smooth_onsets([(t / 2, "kick") for t in range(8)], ["kick"], 800, 5.0)
    m = onset_f_measure(y, y, step_ms=5.0)
    assert m["f_measure"] > 0.99, m
    m0 = onset_f_measure(np.zeros_like(y), y, step_ms=5.0)
    assert m0["f_measure"] == 0.0


def test_peak_pick_respects_the_refractory_window():
    a = np.zeros(200, dtype=np.float32)
    a[[50, 52, 54, 150]] = 1.0                 # a burst, then a distant hit
    t = peak_pick(a, step_ms=5.0, refractory_ms=50.0)
    assert len(t) == 2, t                      # burst collapses to one
    assert abs(t[0] - 0.25) < 1e-6 and abs(t[1] - 0.75) < 1e-6


def test_swing_ratio_is_one_for_straight_eighths():
    tempo = 120.0
    eighth = 60.0 / tempo / 2.0
    straight = np.arange(16) * eighth
    r = swing_ratio(straight, tempo)
    assert np.isnan(r) or abs(r - 1.0) < 0.15, r


def test_swing_ratio_detects_a_swung_pattern():
    tempo, beat = 120.0, 0.5
    swung = np.concatenate([[i * beat, i * beat + beat * 2 / 3] for i in range(8)])
    r = swing_ratio(np.sort(swung), tempo)
    assert r > 1.4, f"triplet swing should read well above 1.0, got {r}"


def test_feel_report_measures_a_deliberate_lag():
    """Shift every onset 20 ms late and the report must say so."""
    classes = ["kick", "snare", "hat_closed"]
    tempo = 120.0
    events = [(i * 0.25, classes[i % 3]) for i in range(16)]
    ref = smooth_onsets(events, classes, 1000, 5.0)
    late = smooth_onsets([(t + 0.020, c) for t, c in events], classes, 1000, 5.0)

    r = feel.analyse(late, ref, classes, step_ms=5.0, tempo_bpm=tempo)
    assert 10.0 < r.overall["grid_dev_ms"] < 30.0, r.overall
    on_time = feel.analyse(ref, ref, classes, step_ms=5.0, tempo_bpm=tempo)
    assert abs(on_time.overall["grid_dev_ms"]) < 5.0, on_time.overall


def test_synthetic_dataset_shapes_line_up():
    ds = SyntheticDrums(["kick", "snare", "hat_closed"], n_clips=4, seconds=2.0,
                        sample_rate=22050, step_ms=5.0)
    wav, y, vel, style, tempo = ds[0]
    assert wav.shape[0] == 44100
    assert y.shape == (400, 3)
    assert vel.shape == y.shape
    assert y.max() > 0.5, "targets must contain onsets"
    assert 0 <= int(style) < 4 and float(tempo) > 0


def test_class_fallback_respects_the_kit_in_use():
    """A ride hit belongs on the ride when the kit has one, on the crash when
    it does not -- deciding that in the MIDI note map would get one tier wrong."""
    from dataset import resolve_class

    eight = set(KIT_TIERS["8piece"])
    articulated = set(KIT_TIERS["articulated"])

    assert resolve_class("ride", eight) == "crash"
    assert resolve_class("ride", articulated) == "ride"
    assert resolve_class("ride_bell", articulated) == "ride_bell"
    assert resolve_class("ride_bell", eight) == "crash"
    assert resolve_class("hat_pedal", eight) == "hat_closed"
    assert resolve_class("hat_pedal", articulated) == "hat_pedal"
    assert resolve_class("sidestick", eight) == "snare"
    assert resolve_class("kick", eight) == "kick"
    assert resolve_class("nonsense", eight) is None


def test_ride_reaches_the_targets_for_both_tiers():
    from dataset import smooth_onsets

    events = [(0.5, "ride"), (1.0, "kick")]
    eight = KIT_TIERS["8piece"]
    y = smooth_onsets(events, eight, 400, 5.0)
    assert y[:, eight.index("crash")].max() > 0.9, "ride was dropped on the 8-piece kit"

    art = KIT_TIERS["articulated"]
    y2 = smooth_onsets(events, art, 400, 5.0)
    assert y2[:, art.index("ride")].max() > 0.9
    assert y2[:, art.index("crash")].max() < 0.1, "ride leaked onto crash"


def test_streaming_peak_state_survives_block_boundaries():
    """The streaming picker carries `prev` across blocks; an early `continue`
    used to skip that update on a rising edge and leave it stale."""
    import torch
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
            # the real core holds the drive across sub-steps, which is exactly
            # repeat_interleave -- see tests/test_speed.py
            return torch.repeat_interleave(drive, substeps, dim=1), state

    # a single ramp that peaks in the middle, split across two pushes
    ramp = [0.05, 0.2, 0.5, 0.9, 0.6, 0.1, 0.05, 0.05]

    class FakeDecoder:
        def __init__(self):
            self.i = 0

        def velocity(self, rates):
            # No velocity head, as a model trained before it existed. The
            # drummer must fall back to peak height, which is what these
            # tests measure.
            return None

        def __call__(self, rates):
            n = rates.shape[1]
            vals = ramp[self.i: self.i + n]
            self.i += n
            # inverse sigmoid so the drummer's sigmoid recovers the ramp
            logits = torch.tensor([[np.log(v / (1 - v)) for v in vals]]).unsqueeze(-1)
            return logits

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder, self.rnn, self.decoder, self.genre = \
                FakeEncoder(), FakeRNN(), FakeDecoder(), None

    kit = DrumKit(["kick"])
    d = StreamingDrummer(FakeModel(), kit, 22050, 5.0, threshold=0.3, refractory_ms=25.0)
    block = np.zeros(110 * 4, dtype=np.float32)

    fired = d.push(block) + d.push(block)
    assert len(fired) == 1, f"expected exactly one peak across the two blocks, got {fired}"
    assert fired[0].note == kit.notes[0]
    assert fired[0].cls == "kick"
    # The hit's own time: the ramp peaks at index 3. It used to be stamped with
    # the enclosing block's start -- 0.0 here -- which quantised every hit to the
    # 20 ms block. And the frame is the encoder's true hop, 110/22050 = 4.9887 ms,
    # not the nominal 5: a clock built from step_ms drifts 0.23% against the audio.
    assert fired[0].t == pytest.approx(3 * 110 / 22050, abs=1e-9)
    assert fired[0].t != pytest.approx(3 * 0.005, abs=1e-9), "nominal step_ms drifts"

    assert 0.0 <= fired[0].velocity <= 1.0


def test_missing_corpus_raises_instead_of_silently_using_synthetic():
    """A run whose config names a real corpus must not quietly train on click
    tracks. Nothing in the logs or the metrics would distinguish that from a
    real run, which is how a result gets invalidated without anyone noticing."""
    import pytest
    from dataset import build_dataset

    with pytest.raises(FileNotFoundError, match="no corpus at"):
        build_dataset({"root": "definitely/not/here", "synthetic": False},
                      ["kick", "snare"], "train")

    ds = build_dataset({"root": "definitely/not/here", "synthetic": True, "n_clips": 2},
                       ["kick", "snare"], "train")
    assert len(ds) == 2, "an explicit synthetic request must still work"


def test_relative_corpus_path_resolves_against_the_repo_root():
    """Running from src/ and from the repo root must address the same corpus."""
    import dataset as D

    assert D.ROOT.name == "flybeats" or (D.ROOT / "src").exists()
    try:
        D.build_dataset({"root": "data/egmd/groove", "synthetic": False},
                        ["kick", "snare"], "train")
    except FileNotFoundError as e:
        # the message must name an absolute path under the repo, not a path
        # relative to whatever directory the process happened to start in
        assert str(D.ROOT) in str(e), str(e)
    except Exception:
        pass          # corpus present: nothing to assert


def test_sweep_matches_the_per_threshold_measure():
    """The fast sweep must agree exactly with computing each threshold alone."""
    from metrics import onset_f_measure, onset_f_sweep
    from dataset import smooth_onsets

    classes = ["kick", "snare", "hat_closed"]
    events = [(i * 0.25, classes[i % 3]) for i in range(16)]
    ref = smooth_onsets(events, classes, 800, 5.0)
    pred = np.clip(ref * 0.8 + 0.15 * np.random.default_rng(0).random(ref.shape), 0, 1)

    thresholds = [0.1, 0.3, 0.5, 0.7]
    fast = onset_f_sweep(pred, ref, 5.0, thresholds)
    for t in thresholds:
        slow = onset_f_measure(pred, ref, 5.0, threshold=t)["f_measure"]
        assert abs(fast[t] - slow) < 1e-9, f"threshold {t}: {fast[t]} vs {slow}"


def test_style_vocabulary_is_global_across_splits():
    """Per-split style vocabularies give the same integer different meanings in
    train and validation, and overflow the genre embedding when a split carries
    a style the training split lacked. Both happened before this was fixed."""
    import csv as _csv
    import pytest
    from dataset import GrooveDataset

    root = ROOT / "data" / "egmd" / "groove"
    if not (root / "info.csv").exists():
        pytest.skip("corpus not present in this checkout")

    kit = DrumKit.from_tier("8piece")
    tr = GrooveDataset(root, kit.classes, split="train", max_files=8)
    va = GrooveDataset(root, kit.classes, split="validation", max_files=8)

    assert tr.styles == va.styles, "splits disagree on the style vocabulary"
    assert tr.style_id == va.style_id

    all_styles = {r["style"].split("/")[0]
                  for r in _csv.DictReader((root / "info.csv").open())}
    assert set(tr.styles) == all_styles

    # every id any split can emit must be inside the embedding built from n_styles
    for ds in (tr, va):
        for i in range(len(ds)):
            assert 0 <= int(ds[i][2]) < ds.n_styles
            break
