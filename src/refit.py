"""Closed-form refit of the velocity head, run after training.

The head trained by SGD barely leaves its initialisation. On every Phase A'
checkpoint its weights matched the other arms' to a cosine of 0.99 and its
bias had not moved, while a least-squares fit of the *same* function class on
the same motor features read snare and both toms at r +0.30 to +0.39 on
validation. Refit, the four A' arms went from 1-3 classes confidently right to
2-4, with none backwards (results/local/handoff-2026-09-24.md). That is an
optimisation failure, not missing information, and this is the direct way
round it: run the frozen network over the training split, take one row per
hit at the kernel peak (where the streaming path reads velocity), and solve
for the head.

The head stays what it was -- one linear layer over the motor pool, through
its hemisphere mask, with its own activation and its own input affine if the
decoder standardises. Only ``vel_readout`` changes. The fit is ridge in
standardised coordinates, folded back into the head's input coordinates,
because raw motor rates are dominated by their mean and an unstandardised
solve is ill-conditioned. A sigmoid head is fit on logit targets, then refined
by damped Gauss-Newton on squared error in 0..1, the quantity the training
loss uses. Validation is never touched.
"""
from __future__ import annotations

import numpy as np
import torch


def hit_peaks(y, thresh):
    """One step per hit: the kernel's local maximum, where it clears ``thresh``.

    ``y > 0.95`` alone keeps two or three steps of every hit, all carrying the
    same target, which counts each hit two or three times over.
    """
    lo = np.pad(y, ((0, 0), (1, 0)), constant_values=-np.inf)[:, :-1]
    hi = np.pad(y, ((0, 0), (0, 1)), constant_values=-np.inf)[:, 1:]
    return (y > thresh) & (y >= lo) & (y > hi)


def fit_head(X, y, activation="sigmoid", lam=1.0, gn_iters=30):
    """Weights and bias for ``act(X @ w + b) ~ y``, in X's own coordinates.

    Solved in standardised coordinates for conditioning, then folded back so
    the result drops into an unmodified ``nn.Linear``.
    """
    X = np.asarray(X, np.float64)
    y = np.asarray(y, np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    reg = lam * np.eye(Z.shape[1])
    reg[-1, -1] = 0.0                                    # never shrink the bias

    if activation == "linear":
        beta = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
    elif activation == "sigmoid":
        t = np.clip(y, 0.02, 0.98)
        beta = np.linalg.solve(Z.T @ Z + reg, Z.T @ np.log(t / (1 - t)))
        damp = 1e-3
        loss = _sig_loss(Z, beta, y, reg)
        for _ in range(gn_iters):
            p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
            J = Z * (p * (1 - p))[:, None]
            g = J.T @ (p - y) + reg @ beta
            H = J.T @ J + reg
            cand = beta - np.linalg.solve(H + damp * np.diag(np.diag(H) + 1e-12), g)
            new = _sig_loss(Z, cand, y, reg)
            if new < loss:
                beta, loss, damp = cand, new, damp / 3
            else:
                damp *= 10
    else:
        raise ValueError(f"unknown activation {activation!r}")

    w = beta[:-1] / sd
    b = beta[-1] - float((beta[:-1] * mu / sd).sum())
    return w.astype(np.float32), np.float32(b)


def _sig_loss(Z, beta, y, reg):
    p = 1.0 / (1.0 + np.exp(-(Z @ beta)))
    return 0.5 * float(((p - y) ** 2).sum()) + 0.5 * float(beta @ reg @ beta)


@torch.no_grad()
def collect_motor(model, loader):
    """Motor rates, velocity targets and onset targets over a loader, on CPU."""
    device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    M, V, Y = [], [], []
    try:
        for wav, y, vel_y, style, _ in loader:
            wav = wav.to(device)
            sid = style.to(device) if model.genre is not None else None
            drive = model.encoder(wav)
            tonic = model.genre(sid) if sid is not None else None
            full, _ = model.rnn(drive, tonic=tonic, return_all=True)
            n = min(drive.shape[1], y.shape[1])
            M.append(full[:, :n, model.rnn.motor_idx].float().cpu().numpy())
            V.append(vel_y[:, :n].numpy())
            Y.append(y[:, :n].numpy())
    finally:
        model.train(was_training)
    return tuple(np.concatenate(a) for a in (M, V, Y))


@torch.no_grad()
def refit_velocity_head(model, loader, lam: float = 1.0, min_hits: int = 30,
                        peak: float = 0.95) -> dict:
    """Refit ``model.decoder.vel_readout`` in place from ``loader``'s clips.

    Classes with fewer than ``min_hits`` peaks keep their trained weights.
    Returns a per-class report: hits, whether it was refit, and train r.
    """
    dec = model.decoder
    if not dec.has_velocity:
        raise ValueError("this model has no velocity head to refit")
    M, V, Y = collect_motor(model, loader)
    Z = dec._inputs(torch.from_numpy(M).to(dec.mask.device)).cpu().numpy()
    mask = dec.mask.cpu().numpy().astype(bool)
    W = dec.vel_readout.weight.detach().cpu().numpy().copy()
    B = dec.vel_readout.bias.detach().cpu().numpy().copy()
    act = dec.velocity_activation
    report = {"clips": int(M.shape[0]), "lam": lam, "classes": {}}
    for c, name in enumerate(dec.kit.classes):
        pk = hit_peaks(Y[:, :, c], peak)
        if pk.sum() < min_hits:
            report["classes"][name] = {"hits": int(pk.sum()), "refit": False}
            continue
        X, t = Z[:, :, mask[c]][pk], V[:, :, c][pk]
        w, b = fit_head(X, t, act, lam=lam)
        W[c] = 0.0
        W[c, mask[c]] = w
        B[c] = b
        pred = X @ w + b
        pred = 1 / (1 + np.exp(-pred)) if act == "sigmoid" else pred
        r = float(np.corrcoef(pred, t)[0, 1]) if pred.std() > 1e-8 and t.std() > 1e-8 else 0.0
        report["classes"][name] = {"hits": int(pk.sum()), "refit": True, "train_r": round(r, 4)}
    dec.vel_readout.weight.copy_(torch.from_numpy(W).to(dec.vel_readout.weight.device))
    dec.vel_readout.bias.copy_(torch.from_numpy(B).to(dec.vel_readout.bias.device))
    return report
