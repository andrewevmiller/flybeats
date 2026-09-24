"""The velocity probe's scoring, pinned after it misread two drums.

On 24 September the Phase A' probe reported kick and low tom as confidently
backwards in every arm. Two scoring faults made that up: it scored only the
last 30% of rows, which in info.csv order was almost all one drummer, and its
bootstrap resampled time steps as if every step were an independent witness.
Rescored on every clip with a clip-level bootstrap, both drums were "can't
tell". These pin the three pieces that fix it.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_velocity import boot_ci, clip_folds, hit_peaks, ridge_cv  # noqa: E402


def test_one_row_per_hit_not_every_step_above_threshold():
    t = np.arange(40)
    kernel = np.exp(-0.5 * ((t - 10) / 1.5) ** 2) + np.exp(-0.5 * ((t - 30) / 1.5) ** 2)
    y = kernel[None, :]
    assert (y > 0.8).sum() > 2          # the old mask counts each hit several times
    peaks = hit_peaks(y, 0.8)
    assert peaks.sum() == 2
    assert set(np.flatnonzero(peaks[0])) == {10, 30}


def test_a_hit_on_the_window_edge_still_counts():
    y = np.array([[1.0, 0.7, 0.2, 0.0, 0.3, 0.97]])
    assert list(np.flatnonzero(hit_peaks(y, 0.95)[0])) == [0, 5]


def test_folds_are_shuffled_not_the_tail_of_the_file():
    """Clips arrive grouped by drummer; a contiguous hold-out is one drummer."""
    folds = clip_folds(np.arange(100), k=5, seed=0)
    assert np.bincount(folds).tolist() == [20] * 5
    assert len(set(folds[-20:])) > 1


def test_clip_bootstrap_is_wider_when_rows_are_clustered():
    """Ten clips, each a hundred near-copies of one point: the evidence is ten
    points, and the interval must say so."""
    rng = np.random.default_rng(1)
    base_h, base_t = rng.normal(size=10), rng.normal(size=10)
    clips = np.repeat(np.arange(10), 100)
    h = base_h[clips] + 0.01 * rng.normal(size=1000)
    t = base_t[clips] + 0.01 * rng.normal(size=1000)
    lo, hi = boot_ci(h, t, clips)
    step_lo, step_hi = boot_ci(h, t, np.arange(1000))
    assert (hi - lo) > 3 * (step_hi - step_lo)


def test_ridge_cv_finds_a_real_signal_and_not_a_fake_one():
    rng = np.random.default_rng(0)
    clips = np.repeat(np.arange(40), 25)
    folds = clip_folds(clips, seed=0)
    X = rng.normal(size=(1000, 8)).astype(np.float32)
    y = (X[:, 0] + 0.5 * rng.normal(size=1000)).astype(np.float32)
    assert ridge_cv(X, y, folds) > 0.8
    noise = rng.normal(size=1000).astype(np.float32)
    assert abs(ridge_cv(X, noise, folds)) < 0.15
