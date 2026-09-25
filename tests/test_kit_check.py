"""The kit comparison's pairing and intervals, pinned by hand.

``kit_check.py`` claims each performance is scored on exactly the same
passages in every corpus, and that its 95% range comes from resampling
performances. Get either wrong and a kit effect is either invented or hidden.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kit_check import paired_ci, plan_windows, verdict  # noqa: E402


def test_windows_are_bounded_by_the_shortest_copy():
    starts = plan_windows({"gmd": 10.0, "seen": 9.3, "unseen": 12.0}, 2.0, per_clip=8)
    assert starts == [0.0, 2.0, 4.0, 6.0]               # 9.3 s fits four, not five


def test_windows_are_capped_per_clip():
    assert len(plan_windows({"gmd": 100.0, "seen": 100.0}, 2.0, per_clip=8)) == 8


def test_a_clip_shorter_than_a_window_still_gets_one():
    assert plan_windows({"gmd": 1.2, "seen": 1.2}, 2.0, per_clip=8) == [0.0]


def test_paired_interval_brackets_a_known_shift():
    rng = np.random.default_rng(1)
    diffs = -0.05 + 0.02 * rng.standard_normal(200)
    m, lo, hi = paired_ci(diffs)
    assert lo < m < hi
    assert hi < 0
    assert verdict(m, lo, hi) == "WORSE (confident)"


def test_no_difference_reads_as_no_difference():
    diffs = np.array([0.1, -0.1, 0.05, -0.05, 0.0, 0.02, -0.02])
    m, lo, hi = paired_ci(diffs)
    assert lo < 0 < hi
    assert verdict(m, lo, hi) == "no clear difference"


def test_interval_is_seeded():
    diffs = np.linspace(-0.1, 0.1, 50)
    assert paired_ci(diffs) == paired_ci(diffs)
