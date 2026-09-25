"""Could the velocity head do better with the features it already has?

Runs a checkpoint over its train and validation splits, then compares, per
class at one row per hit on validation:
  head      -- the trained head as saved
  ls_mask   -- the head's own function class (linear on raw motor rates,
               through its hemisphere mask) fit by least squares on train
               peaks: a head trained to convergence on the frames it is read at
  ls_supp   -- the same, fit on train *support* steps weighted by the onset
               target, which is the objective train.py actually minimises
  ridge_all -- standardised ridge on all 66 motor units, fit on train
               (a fair, train-fit version of the probe's ceiling)
and prints how the motor features are conditioned at hits.

    OMP_NUM_THREADS=1 .venv/Scripts/python.exe results/local/diag-2026-09-24/head_refit.py runs/velocity_probe_cpu/best.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train as T                                   # noqa: E402
from build import build_model, get_subgraph         # noqa: E402
from decoder import DrumKit                         # noqa: E402
from probe_velocity import _r, boot_ci, hit_peaks   # noqa: E402

torch.set_num_threads(1)
ck = torch.load(ROOT / sys.argv[1], map_location="cpu", weights_only=False)
cfg = ck["config"]
cfg["train"]["workers"] = 0
np.random.seed(0); torch.manual_seed(0)
kit = DrumKit(ck["kit"])
model, _ = build_model(cfg, get_subgraph(cfg), n_styles=max(ck["n_styles"], 1))
model.load_state_dict(ck["model"]); model.eval()
tr, va, _ = T.build_loaders(cfg, kit)


def collect(loader):
    M, V, Y = [], [], []
    with torch.no_grad():
        for wav, y, vel_y, style, _ in loader:
            sid = style if model.genre is not None else None
            drive = model.encoder(wav)
            tonic = model.genre(sid) if sid is not None else None
            full, _ = model.rnn(drive, tonic=tonic, return_all=True)
            n = min(drive.shape[1], y.shape[1])
            M.append(full[:, :n, model.rnn.motor_idx].numpy())
            V.append(vel_y[:, :n].numpy()); Y.append(y[:, :n].numpy())
    return (np.concatenate(a) for a in (M, V, Y))


Mt, Vt, Yt = collect(tr)
Mv, Vv, Yv = collect(va)
print(f"train clips {Mt.shape[0]}  val clips {Mv.shape[0]}")
mask = model.decoder.mask.numpy().astype(bool)
with torch.no_grad():
    Hv = model.decoder.velocity(torch.from_numpy(Mv)).numpy()


def lstsq(X, y, w=None, lam=1e-3):
    X1 = np.hstack([X, np.ones((len(X), 1), np.float32)])
    if w is None:
        w = np.ones(len(y), np.float32)
    A = (X1 * w[:, None]).T @ X1 + lam * np.eye(X1.shape[1], dtype=np.float32)
    return np.linalg.solve(A, (X1 * w[:, None]).T @ y)


def apply(X, beta):
    return np.hstack([X, np.ones((len(X), 1), np.float32)]) @ beta


def ridge_std(Xtr, ytr, Xte, lam=1.0):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    beta = lstsq((Xtr - mu) / sd, ytr, lam=lam)
    return apply((Xte - mu) / sd, beta)


clips_v = np.repeat(np.arange(Yv.shape[0])[:, None], Yv.shape[1], 1)
print(f"\n{'class':<11}{'hits':>6}{'head':>8}{'ls_mask':>9}{'ls_supp':>9}{'ridge_all':>11}"
      f"   ls_mask 95% CI     mean^2/var  top-sv share")
for c, name in enumerate(kit.classes):
    pt, pv = hit_peaks(Yt[:, :, c], 0.95), hit_peaks(Yv[:, :, c], 0.95)
    if pv.sum() < 30:
        print(f"{name:<11}{int(pv.sum()):>6}   too few hits"); continue
    m = mask[c]
    Xt, yt = Mt[:, :, m][pt], Vt[:, :, c][pt]
    Xv, yv = Mv[:, :, m][pv], Vv[:, :, c][pv]
    ls = apply(Xv, lstsq(Xt, yt))
    st = Yt[:, :, c] > 0.01
    supp = apply(Xv, lstsq(Mt[:, :, m][st], Vt[:, :, c][st], w=Yt[:, :, c][st]))
    ra = ridge_std(Mt[pt], yt, Mv[pv])
    lo, hi = boot_ci(ls, yv, clips_v[pv])
    # conditioning of the head's raw inputs at hits: how much of each unit's
    # second moment is its mean, and how much of the centred variance the
    # single largest direction holds
    mu2 = (Xt.mean(0) ** 2).sum() / Xt.var(0).sum()
    s = np.linalg.svd(Xt - Xt.mean(0), compute_uv=False)
    print(f"{name:<11}{int(pv.sum()):>6}{_r(Hv[:, :, c][pv], yv):>+8.2f}{_r(ls, yv):>+9.2f}"
          f"{_r(supp, yv):>+9.2f}{_r(ra, yv):>+11.2f}   [{lo:+.2f}, {hi:+.2f}]"
          f"{mu2:>12.0f}{(s[0]**2 / (s**2).sum()):>12.2f}")
