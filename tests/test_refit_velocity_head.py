"""The closed-form velocity-head refit must drop into an unmodified nn.Linear.

It solves in standardised coordinates and folds the answer back into raw-rate
weights; the fold is where a sign or a scale goes wrong silently, so these
check the folded weights against the data they were fit on.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from refit import fit_head  # noqa: E402


def _rates(n=2000, k=12, seed=0):
    """Motor-like features: a large shared level, small informative wobble."""
    rng = np.random.default_rng(seed)
    X = 5.0 + 0.1 * rng.normal(size=(n, k))
    return X.astype(np.float32), rng


def test_linear_fit_recovers_a_signal_hidden_under_a_large_mean():
    X, rng = _rates()
    y = 0.5 + 2.0 * (X[:, 3] - 5.0) + 0.02 * rng.normal(size=len(X))
    w, b = fit_head(X, y, "linear", lam=1e-3)
    pred = X @ w + b
    assert np.corrcoef(pred, y)[0, 1] > 0.99
    assert abs(pred.mean() - y.mean()) < 1e-3


def test_sigmoid_fit_is_scored_through_the_sigmoid():
    X, rng = _rates(seed=1)
    y = 1 / (1 + np.exp(-(0.4 + 8.0 * (X[:, 0] - 5.0))))
    w, b = fit_head(X, y, "sigmoid", lam=1e-3)
    pred = 1 / (1 + np.exp(-(X @ w + b)))
    assert np.corrcoef(pred, y)[0, 1] > 0.99
    assert np.abs(pred - y).mean() < 0.01


def test_noise_targets_give_a_near_constant_head():
    X, rng = _rates(seed=2)
    y = rng.uniform(0.3, 0.9, size=len(X))
    w, b = fit_head(X, y, "sigmoid", lam=10.0)
    pred = 1 / (1 + np.exp(-(X @ w + b)))
    assert pred.std() < 0.05
    assert abs(pred.mean() - y.mean()) < 0.02
