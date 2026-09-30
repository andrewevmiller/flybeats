"""Phase 3, step 5: the loss, and the hit F1 used for checkpoints until Phase 4's scorecard exists."""
from functools import partial

import numpy as np
import torch
from torch.nn import functional as F


def drum_weights(manifest, pieces, frame_s, max_weight):
    """Per drum: non-hit frames / hit frames over the training split, capped at max_weight."""
    train = manifest[manifest.split == "train"]
    frames = (train.duration / frame_s).sum()
    hits = np.array([train[f"hits_{p}"].sum() for p in pieces], np.float64)
    return torch.tensor(np.minimum((frames - hits) / np.maximum(hits, 1), max_weight), dtype=torch.float32)


def soft_targets(hits, neighbour):
    """1 on the hit frame, `neighbour` on the frame either side. hits: (B, T, D) float."""
    side = torch.zeros_like(hits)
    side[:, 1:] = hits[:, :-1]
    side[:, :-1] = torch.maximum(side[:, :-1], hits[:, 1:])
    return torch.maximum(hits, neighbour * side)


def loss_fn(out, hits, vel, mask, weights, neighbour, velocity_weight):
    """out from FlyNet; hits/vel (B, T, D); mask (B, T) False on warm-up frames; weights (D,)."""
    target = soft_targets(hits, neighbour)
    w = torch.where(target > 0, weights, torch.ones_like(weights)) * mask[..., None]
    bce = F.binary_cross_entropy_with_logits(out["hit_logits"], target, weight=w, reduction="sum") / w.sum()
    on = hits * mask[..., None]
    mse = ((out["vel"] - vel) ** 2 * on).sum() / on.sum().clamp_min(1)
    return bce + velocity_weight * mse, {"hit_bce": float(bce.detach()), "vel_mse": float(mse.detach())}


def make_loss(cfg, weights):
    """loss(out, hits, vel, mask) with the locked settings and these drum weights."""
    lc = cfg["loss"]
    return partial(loss_fn, weights=weights, neighbour=lc["neighbour_target"], velocity_weight=lc["velocity_weight"])


def peaks(prob, threshold):
    """Frames where prob crosses threshold and is a local maximum. prob: (T,) numpy."""
    left = np.r_[-np.inf, prob[:-1]]
    right = np.r_[prob[1:], -np.inf]
    return np.flatnonzero((prob >= threshold) & (prob >= left) & (prob > right))


def tolerance_frames(frames, beats_s, frame_s, fraction, default_beat_s=0.5):
    """Tolerance at each frame: fraction x the local beat length (50 ms at 120 BPM)."""
    b = np.asarray(beats_s)
    if len(b) < 2:
        return np.full(len(frames), fraction * default_beat_s / frame_s)
    lengths = np.diff(b)
    j = np.clip(np.searchsorted(b, frames * frame_s) - 1, 0, len(lengths) - 1)
    return fraction * lengths[j] / frame_s


def match_counts(pred, true, tol):
    """Greedy one-to-one matching of predicted to true frames; tol is per true frame.
    Returns (tp, fp, fn)."""
    used = np.zeros(len(true), bool)
    tp = 0
    for p in pred:
        if not len(true):
            break
        d = np.abs(true - p).astype(float)
        d[used | (d > tol)] = np.inf
        k = int(np.argmin(d))
        if np.isfinite(d[k]):
            used[k] = True
            tp += 1
    return tp, len(pred) - tp, len(true) - tp


def hit_counts(hit_logits, hits, mask, beats_list, frame_s, fraction, threshold):
    """Summed (tp, fp, fn) per drum over a batch. hit_logits/hits: (B, T, D) tensors."""
    prob = torch.sigmoid(hit_logits).detach().cpu().numpy()
    hits = hits.detach().cpu().numpy()
    mask = mask.detach().cpu().numpy()
    counts = np.zeros((prob.shape[2], 3), np.int64)
    for i in range(prob.shape[0]):
        live = np.flatnonzero(mask[i])
        for d in range(prob.shape[2]):
            pred = np.intersect1d(peaks(prob[i, :, d], threshold), live)
            true = np.intersect1d(np.flatnonzero(hits[i, :, d]), live)
            tol = tolerance_frames(true, beats_list[i], frame_s, fraction)   # one per true hit
            counts[d] += match_counts(pred, true, tol)
    return counts


def f1_from_counts(counts):
    tp, fp, fn = counts.T.astype(float)
    return np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(2 * tp + fp + fn, 1), np.nan)
