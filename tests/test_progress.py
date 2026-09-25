"""The end-of-run "how far along its learning curve" estimate.

The number is read by a person deciding whether to spend another night on a
run, so the failure that matters is a confident wrong answer: calling a curve
that is still climbing done, or the reverse.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from progress import describe, estimate  # noqa: E402


def _curve(k, n=12, A=0.35, B=0.15, noise=0.005, seed=1):
    t = np.arange(1, n + 1)
    return A - B * np.exp(-k * t) + np.random.default_rng(seed).normal(0, noise, n)


def test_a_flat_curve_reads_done():
    est = estimate(_curve(k=1.5))
    assert est["share"] > 0.95 and est["more_to_95"] == 0
    assert "levelled off" in describe(est)


def test_a_still_rising_curve_reads_unfinished():
    est = estimate(_curve(k=0.08))
    assert est["share"] < 0.8
    assert est["more_to_95"] > 0
    assert est["plateau"] > est["best"]
    assert "still rising" in describe(est)


def test_the_interval_brackets_the_estimate():
    for k in (0.08, 0.3, 1.5):
        est = estimate(_curve(k=k))
        assert est["low"] <= est["share"] + 1e-9 and est["share"] <= est["high"] + 1e-9


def test_a_peak_then_a_fall_is_called_overfitting():
    y = list(_curve(k=1.0, n=8)) + [0.30, 0.28, 0.26, 0.24]
    line = describe(estimate(y))
    assert "overfitting" in line and "best.pt kept the peak" in line


def test_too_few_epochs_gives_no_number():
    assert estimate([0.2, 0.25, 0.27]) is None
    assert "too few epochs" in describe(None)


def test_the_global_rng_is_left_alone():
    # A resumed run must match an unbroken one; drawing from the global
    # generator here would shift every later shuffle.
    np.random.seed(123)
    before = np.random.get_state()[1].copy()
    estimate(_curve(k=0.3))
    assert (np.random.get_state()[1] == before).all()


def test_the_log_line_does_not_look_like_the_best_line():
    # The queue scripts grep the last 'best val onset F' line for the score.
    assert "best val onset F" not in describe(estimate(_curve(k=0.08)))
    assert "best val onset F" not in describe(estimate(_curve(k=1.5)))
