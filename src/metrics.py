"""Metrics. PLAN.md asks for onset F-measure, beat alignment, and groove
similarity -- and for feel to be *measured*, not imposed by a swing slider.
"""
from __future__ import annotations

import numpy as np


def peak_pick(activation: np.ndarray, step_ms: float, threshold: float = 0.3,
              refractory_ms: float = 50.0) -> np.ndarray:
    """Onset times (seconds) from a 1-D activation, with a refractory window."""
    a = np.asarray(activation, dtype=np.float32)
    if a.size < 3:
        return np.array([], dtype=np.float32)
    is_peak = np.zeros_like(a, dtype=bool)
    is_peak[1:-1] = (a[1:-1] >= a[:-2]) & (a[1:-1] > a[2:]) & (a[1:-1] >= threshold)
    idx = np.flatnonzero(is_peak)

    keep, last = [], -np.inf
    gap = refractory_ms / step_ms
    for i in idx[np.argsort(-a[idx])] if False else idx:
        if i - last >= gap:
            keep.append(i)
            last = i
    return np.asarray(keep, dtype=np.float32) * step_ms / 1000.0


def match_onsets(pred: np.ndarray, ref: np.ndarray, tolerance_s: float = 0.05):
    """Greedy nearest-neighbour matching within a tolerance window."""
    if len(pred) == 0 or len(ref) == 0:
        return 0, len(pred), len(ref), np.array([])
    used = np.zeros(len(ref), dtype=bool)
    tp, devs = 0, []
    for p in np.sort(pred):
        d = np.abs(ref - p)
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tolerance_s:
            used[j] = True
            tp += 1
            devs.append(p - ref[j])
    return tp, len(pred) - tp, len(ref) - tp, np.asarray(devs)


def onset_f_measure(pred_act: np.ndarray, ref_act: np.ndarray, step_ms: float,
                    tolerance_s: float = 0.05, threshold: float = 0.3) -> dict:
    """Per-class and micro-averaged onset F, the headline number.

    ``pred_act`` and ``ref_act`` are ``(steps, n_classes)``.
    """
    n_cls = pred_act.shape[1]
    per_class, TP = [], 0
    FP = FN = 0
    all_dev = []
    for c in range(n_cls):
        p = peak_pick(pred_act[:, c], step_ms, threshold)
        r = peak_pick(ref_act[:, c], step_ms, threshold=0.5)
        tp, fp, fn, dev = match_onsets(p, r, tolerance_s)
        TP, FP, FN = TP + tp, FP + fp, FN + fn
        all_dev.append(dev)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        per_class.append({
            "precision": prec, "recall": rec,
            "f": 2 * prec * rec / max(prec + rec, 1e-9),
            "n_ref": tp + fn, "n_pred": tp + fp,
            "mean_dev_ms": float(np.mean(dev) * 1000) if len(dev) else float("nan"),
        })
    prec = TP / max(TP + FP, 1)
    rec = TP / max(TP + FN, 1)
    devs = np.concatenate([d for d in all_dev if len(d)]) if any(len(d) for d in all_dev) else np.array([])
    return {
        "precision": prec, "recall": rec,
        "f_measure": 2 * prec * rec / max(prec + rec, 1e-9),
        "per_class": per_class,
        "mean_dev_ms": float(np.mean(devs) * 1000) if len(devs) else float("nan"),
        "std_dev_ms": float(np.std(devs) * 1000) if len(devs) else float("nan"),
    }


def beat_alignment_error(pred_act: np.ndarray, step_ms: float, tempo_bpm: float) -> float:
    """RMS distance from predicted onsets to the nearest grid line, in ms.

    Low is 'on the grid'. This is descriptive: PLAN.md wants feel measured, so
    a model that sits consistently 8 ms behind the beat should show up as a
    non-zero *mean* deviation with a *small* spread, not be corrected away.
    """
    if tempo_bpm <= 0:
        return float("nan")
    grid = 60.0 / tempo_bpm / 4.0     # 16th notes
    times = np.concatenate([peak_pick(pred_act[:, c], step_ms) for c in range(pred_act.shape[1])]) \
        if pred_act.shape[1] else np.array([])
    if len(times) == 0:
        return float("nan")
    err = times - np.round(times / grid) * grid
    return float(np.sqrt(np.mean(err ** 2)) * 1000)


def swing_ratio(onsets_s: np.ndarray, tempo_bpm: float) -> float:
    """Measured swing: ratio of the long to the short half of each 8th pair.

    1.0 is dead straight, 1.5 is triplet swing. Measured from the model's own
    output -- there is no swing parameter anywhere in this codebase.
    """
    if tempo_bpm <= 0 or len(onsets_s) < 3:
        return float("nan")
    eighth = 60.0 / tempo_bpm / 2.0
    phase = (np.sort(onsets_s) % (2 * eighth)) / eighth
    off = phase[(phase > 0.5) & (phase < 1.5)]
    if len(off) < 2:
        return float("nan")
    mean_off = float(np.mean(off))
    return mean_off / max(2.0 - mean_off, 1e-6)


def groove_similarity(pred_act: np.ndarray, ref_act: np.ndarray, step_ms: float,
                      tempo_bpm: float, subdivisions: int = 16) -> float:
    """Cosine similarity of the two rhythms folded onto one bar.

    This is the standard groove-similarity construction: bin onsets by metric
    position, normalise, compare. It rewards playing the *same groove* even when
    individual hits are missed, which raw onset F does not.
    """
    if tempo_bpm <= 0:
        return float("nan")
    bar = 4 * 60.0 / tempo_bpm
    n_cls = pred_act.shape[1]

    def fold(act: np.ndarray, thr: float) -> np.ndarray:
        v = np.zeros((subdivisions, n_cls), dtype=np.float64)
        for c in range(n_cls):
            for t in peak_pick(act[:, c], step_ms, threshold=thr):
                v[int((t % bar) / bar * subdivisions) % subdivisions, c] += 1.0
        return v.ravel()

    a, b = fold(pred_act, 0.3), fold(ref_act, 0.5)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))
