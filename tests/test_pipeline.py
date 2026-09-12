"""End-to-end checks on the pieces that glue phases together.

These are the seams where a mistake is invisible: a config that silently drops
a key, a kit that exceeds the confirmed motor-neuron count, a decoder whose
bilateral mask does not actually separate the hemispheres.
"""
import json
import sys
from pathlib import Path

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
    wav, y, style, tempo = ds[0]
    assert wav.shape[0] == 44100
    assert y.shape == (400, 3)
    assert y.max() > 0.5, "targets must contain onsets"
    assert 0 <= int(style) < 4 and float(tempo) > 0
