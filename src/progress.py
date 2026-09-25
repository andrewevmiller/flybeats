"""How far along its learning curve a run got, estimated from val onset F.

A run always trains for its configured number of epochs, so "finished" only
says the loop ran out. Whether the model had finished *learning* is a separate
question, and the answer is in the validation curve: training loss falls for
as long as there are epochs to spend, while val onset F levels off.

The curve is fitted with a saturating exponential,

    F(t) = A - B * exp(-k * t)          t = epochs completed

and the share of the achievable gain the run collected is 1 - exp(-k * n).
That share depends on k alone, not on A or B, so an optimistic best-threshold
F on one epoch cannot move it much. k is found by a grid search (A and B are
then closed-form least squares), which needs no scipy and cannot fail to
converge. The fit is restricted to rising curves: a run that peaks and then
declines is treated as having learned what it could, which is the right answer
for "would more epochs help".

Twelve points of a noisy score pin k down loosely, so the estimate comes with
an interval from a residual bootstrap. The bootstrap uses its own generator:
touching the global RNG here would make a resumed run diverge from an unbroken
one (tests/test_resume.py).

    python src/progress.py runs/*/history.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

#: Learning rates for the grid, per epoch. 0.02 is a curve still close to a
#: straight line after 12 epochs; 3.0 is one that is flat after the first.
_K = np.geomspace(0.02, 3.0, 400)
#: Fewer epochs than this and any curve fits; the number would be decoration.
MIN_EPOCHS = 4


def _fit(y: np.ndarray, t: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    """Least-squares (k, A, B, fitted curve) over the k grid, with B >= 0."""
    X = np.exp(-_K[:, None] * t[None, :])
    xm = X.mean(1, keepdims=True)
    ym = y.mean()
    slope = ((X - xm) * (y - ym)).sum(1) / ((X - xm) ** 2).sum(1)
    slope = np.minimum(slope, 0.0)                      # slope is -B
    A = ym - slope * xm[:, 0]
    pred = A[:, None] + slope[:, None] * X
    i = int(((y - pred) ** 2).sum(1).argmin())
    return float(_K[i]), float(A[i]), float(-slope[i]), pred[i]


def _share(k: float, B: float, n: int) -> float:
    return 1.0 - math.exp(-k * n) if B > 0 else 1.0


def estimate(scores, n_boot: int = 300) -> dict | None:
    """Estimate from per-epoch val onset F, or None if there are too few epochs."""
    y = np.asarray(scores, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n < MIN_EPOCHS:
        return None
    t = np.arange(1, n + 1, dtype=float)
    k, A, B, pred = _fit(y, t)
    share = _share(k, B, n)

    resid = y - pred
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        kb, _, Bb, _ = _fit(pred + rng.choice(resid, n), t)
        boot.append(_share(kb, Bb, n))
    lo, hi = np.percentile(boot, [10, 90])

    best_ep = int(y.argmax())
    noise = float(resid.std())
    # Epoch by which the fitted curve has 95% of its gain: ln(20) / k.
    ep95 = math.ceil(math.log(20) / k) if B > 0 else 1
    return {
        "share": share, "low": float(lo), "high": float(hi), "epochs": n,
        "plateau": A if B > 0 else float(y.mean()), "best": float(y[best_ep]),
        "best_epoch": best_ep, "last": float(y[-1]), "noise": noise,
        "epoch_95": ep95, "more_to_95": max(ep95 - n, 0),
    }


def describe(est: dict | None) -> str:
    """One plain-language line for the end of a training log."""
    if est is None:
        return f"training progress: too few epochs to estimate (need {MIN_EPOCHS})"
    pct = lambda v: f"{100 * v:.0f}%"  # noqa: E731
    line = f"training progress: ~{pct(est['share'])} of the way to its plateau"
    if pct(est["low"]) != pct(est["high"]):
        line += f" (likely {pct(est['low'])}-{pct(est['high'])})"
    if est["more_to_95"] > 0:
        line += (f"; val onset F still rising toward ~{est['plateau']:.3f} "
                 f"(best {est['best']:.3f}), about {est['more_to_95']} more epochs "
                 f"to reach 95% -- extend with --epochs and --resume")
    else:
        line += f"; levelled off by about epoch {min(est['epoch_95'], est['epochs'])}"
        # A drop from the peak well past the epoch-to-epoch noise is
        # overfitting; best.pt still holds the peak, so it costs time, not model.
        if est["best"] - est["last"] > max(2 * est["noise"], 0.02):
            line += (f", peaked at epoch {est['best_epoch']} ({est['best']:.3f}) and fell "
                     f"to {est['last']:.3f} since -- overfitting; best.pt kept the peak")
        else:
            line += " -- more epochs would not help much"
    return line


def main(paths: list[str]) -> int:
    for p in paths:
        hist = json.loads(Path(p).read_text())
        print(f"{Path(p).parent.name}: {describe(estimate([r['onset_f'] for r in hist]))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
