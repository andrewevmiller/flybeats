"""`train.evaluate` against activations whose answer is known in advance.

TEST_PLAN T1.3. Every onset F this project quotes -- in the README, in
ROADMAP.md, in every `ablation_results.json` and in every model card the
release roadmap promises -- comes out of this one function, and nothing called
it. `test_pipeline.py` covers `onset_f_measure` underneath it; the gap is
`evaluate`'s own aggregation: the threshold sweep, the argmax that picks the
headline number, the curve beside it, and the per-class velocity correlation.

The model is a stub that returns whatever activations the test hands it. That
is the point: with a real model the expected answer would have to be computed
by the code under test.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import smooth_targets  # noqa: E402
from metrics import onset_f_sweep  # noqa: E402
from train import evaluate  # noqa: E402

STEP_MS = 5.0
STEPS = 400          # 2 s at 5 ms
CLASSES = ["kick", "snare", "hat_closed"]
SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def _cfg(threshold=0.3):
    return {"audio": {"step_ms": STEP_MS},
            "eval": {"tolerance_s": 0.05, "threshold": threshold,
                     "threshold_sweep": list(SWEEP)}}


def _clip(seed=0):
    """One clip of targets: onsets at least a refractory window apart."""
    rng = np.random.default_rng(seed)
    events = []
    for c, cls in enumerate(CLASSES):
        for k in range(6):
            t = 0.15 + 0.3 * k + 0.02 * c
            events.append((t, cls, float(rng.uniform(0.3, 0.95))))
    return smooth_targets(events, CLASSES, STEPS, step_ms=STEP_MS)


class _Stub(nn.Module):
    """Returns the activations it was constructed with, as logits.

    `evaluate` sigmoids what it gets, so the probabilities are inverted here
    and come back out unchanged (to within 1e-6, which no peak-pick sees).
    """

    genre = None

    def __init__(self, prob, vel=None):
        super().__init__()
        self.prob = torch.as_tensor(prob, dtype=torch.float32)
        self.vel = None if vel is None else torch.as_tensor(vel, dtype=torch.float32)

    def forward(self, wav, style_id=None, with_velocity=False):
        n = wav.shape[0]
        logits = torch.logit(self.prob[:n].clamp(1e-6, 1 - 1e-6))
        return (logits, None, self.vel[:n] if self.vel is not None else None)


def _loader(y, vel, tempo=120.0):
    """One batch, shaped the way the real DataLoader yields it."""
    n = y.shape[0]
    wav = torch.zeros(n, int(STEPS * STEP_MS * 22.05))
    return [(wav, torch.as_tensor(y), torch.as_tensor(vel),
             torch.zeros(n, dtype=torch.long), torch.full((n,), tempo))]


def _stack(n=2):
    ys, vs = zip(*(_clip(seed) for seed in range(n)))
    return np.stack(ys).astype(np.float32), np.stack(vs).astype(np.float32)


# --- the two answers that are known exactly --------------------------------

def test_a_perfect_copy_scores_one():
    y, vel = _stack()
    out = evaluate(_Stub(y, vel), _loader(y, vel), _cfg(), torch.device("cpu"))
    assert out["onset_f"] == pytest.approx(1.0)
    assert out["onset_f_fixed"] == pytest.approx(1.0)
    assert out["n_clips"] == 2


def test_silence_scores_zero():
    y, vel = _stack()
    out = evaluate(_Stub(np.zeros_like(y), np.zeros_like(vel)),
                   _loader(y, vel), _cfg(), torch.device("cpu"))
    assert out["onset_f"] == 0.0
    assert out["onset_f_fixed"] == 0.0
    # Nothing was predicted, so there is no deviation to report -- and a
    # silent model must not be able to score a flattering timing number.
    assert np.isnan(out["mean_dev_ms"])


# --- the aggregation above onset_f_measure ---------------------------------

def test_the_headline_is_the_curve_argmax_and_the_curve_is_right():
    """The three reported numbers have to agree with each other, and with a
    sweep computed independently clip by clip.

    The prediction is scaled per class so that the classes peak at different
    heights: a single threshold then cannot be best for all of them, which is
    what makes the argmax load-bearing rather than incidental.
    """
    y, vel = _stack()
    pred = y * np.array([0.95, 0.55, 0.35], dtype=np.float32)

    out = evaluate(_Stub(pred, vel), _loader(y, vel), _cfg(), torch.device("cpu"))

    want = {t: float(np.mean([onset_f_sweep(pred[i], y[i], STEP_MS, SWEEP)[t]
                              for i in range(y.shape[0])]))
            for t in SWEEP}

    assert out["onset_f_curve"] == pytest.approx({f"{t:g}": want[t] for t in SWEEP})
    assert out["onset_f"] == pytest.approx(max(want.values()))
    assert out["best_threshold"] == max(want, key=want.get)
    assert out["onset_f"] == pytest.approx(out["onset_f_curve"][f"{out['best_threshold']:g}"])
    assert out["onset_f_fixed"] == pytest.approx(out["onset_f_curve"]["0.3"])
    # The selected maximum can only flatter, never penalise.
    assert out["onset_f"] >= out["onset_f_fixed"]


def test_a_fixed_threshold_outside_the_sweep_is_added_to_it():
    """`onset_f_fixed` is quoted for continuity with older runs, so the sweep
    has to contain it even when a config sets one that is not on the grid."""
    y, vel = _stack()
    pred = y * 0.45
    out = evaluate(_Stub(pred, vel), _loader(y, vel), _cfg(threshold=0.44),
                   torch.device("cpu"))
    assert "0.44" in out["onset_f_curve"]
    assert out["onset_f_fixed"] == pytest.approx(out["onset_f_curve"]["0.44"])


def test_clips_are_averaged_not_pooled():
    """Two clips of different difficulty average, so one dense clip cannot
    dominate the split's score by carrying more onsets."""
    y, vel = _stack()
    easy = _Stub(np.stack([y[0], y[1] * 0.0]), vel)
    out = evaluate(easy, _loader(y, vel), _cfg(), torch.device("cpu"))
    assert out["n_clips"] == 2
    assert out["onset_f"] == pytest.approx(0.5, abs=0.02)


# --- velocity: the per-class distinction the docstring promises ------------

def test_perfect_velocity_reads_as_perfect():
    y, vel = _stack()
    out = evaluate(_Stub(y, vel), _loader(y, vel), _cfg(), torch.device("cpu"))
    assert out["velocity_mae"] == pytest.approx(0.0, abs=1e-6)
    assert out["velocity_r"] == pytest.approx(1.0, abs=1e-6)


def test_a_head_that_predicts_each_class_mean_scores_zero_correlation():
    """The reason `velocity_r` is per class and not pooled.

    A head that learned nothing but each class's average level is deaf to
    dynamics *within* a class -- the whole quantity of interest -- while
    scoring a respectable MAE, because drummers are not that dynamic.
    """
    y, vel = _stack()
    flat = np.zeros_like(vel)
    for c in range(vel.shape[-1]):
        hit = y[..., c] > 0.5
        flat[..., c] = np.where(hit, vel[..., c][hit].mean(), 0.0)

    out = evaluate(_Stub(y, flat), _loader(y, vel), _cfg(), torch.device("cpu"))
    assert out["velocity_r"] == pytest.approx(0.0, abs=1e-6), "per-class r must see through it"
    assert out["velocity_mae"] < 0.25, "while MAE stays respectable -- that is the trap"
