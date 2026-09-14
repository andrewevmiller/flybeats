"""The Phase A' decision rule, applied by a program instead of by eye.

Five configs x eight classes x two scorings is eighty intervals, and a rule
applied by hand across that many numbers is a rule that finds the answer you
went in with. So the rule lives in scripts/compare_velocity_runs.py -- and the
part of it that can fail silently is the parsing, because a regex that quietly
matches nothing turns "backwards on snare" into "clears the gate".
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from compare_velocity_runs import parse, verdict  # noqa: E402


PROBE = """clips 120  steps 400  drive ch 405  motor units 66  classes 8

== support (y > 0.5) ==
class         n steps  target sd   drive r   motor r   head r     head 95% CI
-----------------------------------------------------------------------------
kick             2499      0.214    -0.014     0.073    0.012  [-0.06, +0.08]  (spans 0)
snare            5224      0.299     0.276     0.198    0.073  [+0.03, +0.12]

head output at hits: mean 0.563 sd 0.9999
target at hits:      mean 0.584 sd 0.2720

== peak (y > 0.95) ==
class         n steps  target sd   drive r   motor r   head r     head 95% CI
-----------------------------------------------------------------------------
kick              837      0.214    -0.174     0.039    0.028  [-0.10, +0.15]  (spans 0)
snare            1785      0.300     0.194     0.145   -0.138  [-0.22, -0.06]
hat_closed        905      0.227     0.126     0.029    0.009  [-0.10, +0.11]  (spans 0)
tom_high           39   too few steps
crash             312      0.194     0.113     0.346    0.349  [+0.17, +0.50]

head output at hits: mean 0.563 sd 0.1046
target at hits:      mean 0.584 sd 0.2720
"""


def test_it_reads_the_peak_section_not_the_support_one():
    """They carry different numbers, and the streaming path reads velocity at
    the peak -- scoring the support rows would answer a different question."""
    assert parse(PROBE, "peak")["head_sd"] == pytest.approx(0.1046)
    assert parse(PROBE, "support")["head_sd"] == pytest.approx(0.9999)


def test_each_interval_becomes_a_sign():
    got = {c["cls"]: c["sign"] for c in parse(PROBE)["classes"]}
    assert got == {"kick": "0", "snare": "-", "hat_closed": "0", "crash": "+"}


def test_a_class_with_too_few_steps_is_not_a_class():
    """It has no interval, so it can neither pass nor fail the gate. Counting
    it as a zero would dilute the count it is absent from."""
    assert "tom_high" not in {c["cls"] for c in parse(PROBE)["classes"]}


def test_one_backwards_class_fails_the_gate_however_many_are_positive():
    ok, why = verdict(parse(PROBE))
    assert not ok and "snare" in why


def test_the_gate_needs_a_positive_class_not_merely_no_negative_one():
    """A head that is flat everywhere is significantly nothing. It is not a
    velocity head, and 'no class is backwards' must not pass it."""
    flat = {"classes": [{"cls": "kick", "sign": "0"}, {"cls": "snare", "sign": "0"}]}
    ok, why = verdict(flat)
    assert not ok and "positive" in why


def test_a_clean_run_passes_and_names_its_classes():
    good = {"classes": [{"cls": "kick", "sign": "0"}, {"cls": "crash", "sign": "+"}]}
    ok, why = verdict(good)
    assert ok and "crash" in why


def test_a_probe_without_the_section_is_an_error_not_an_empty_pass():
    """An unparsed file must not read as a run with no backwards classes."""
    with pytest.raises(ValueError):
        parse("clips 120\n(the probe crashed before it printed anything)\n")


def test_the_real_committed_probes_parse():
    """The fixture above is my transcription; these are the actual files the
    decision will be made from."""
    found = sorted((ROOT / "results").glob("probe_*.txt"))
    if not found:
        pytest.skip("no probe output committed in this checkout")
    for f in found:
        p = parse(f.read_text())
        assert p["classes"], f"{f.name} parsed to no classes at all"
        assert 0.0 < p["target_sd"] < 1.0, f"{f.name} target sd looks wrong"
