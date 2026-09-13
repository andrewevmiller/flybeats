"""Where does hit strength stop being decodable?

The velocity head can only read what reaches the motor pool. When its
correlation with the target sits near zero, that is either a head that failed
to learn or a signal that never arrived, and the two want opposite fixes. A
linear probe separates them:

  1. the information never enters       -> drive probe fails too
  2. the recurrence washes it out       -> drive probe works, motor probe fails
  3. the joint loss under-weights it    -> both probes work, only the head fails

Ridge is the fairest ceiling for a linear head: the same function class the
head has, fit in closed form rather than by SGD against a competing objective.
It also standardises its inputs, so a signal that is present but tiny relative
to the mean still shows up -- which is what separates "washed out" from merely
"small".

    python scripts/probe_velocity.py --checkpoint runs/v1_8piece_cpu/best.pt

Run it on any checkpoint whose ``velocity_r`` disappoints, before touching the
head. As of the sanity run recorded in the README, the answer is (2).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as T                                    # noqa: E402
from build import build_model, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit                          # noqa: E402


def ridge_r(X, y, lam=1.0, split=0.7):
    """Fit on the first `split` of rows, report correlation on the rest."""
    n = len(X)
    k = int(n * split)
    Xtr, ytr, Xte, yte = X[:k], y[:k], X[k:], y[k:]
    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    Xtr = np.hstack([Xtr, np.ones((len(Xtr), 1), np.float32)])
    Xte = np.hstack([Xte, np.ones((len(Xte), 1), np.float32)])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
    w = np.linalg.solve(A, Xtr.T @ ytr)
    pred = Xte @ w
    if pred.std() < 1e-8 or yte.std() < 1e-8:
        return 0.0, float(np.abs(pred - yte).mean())
    return float(np.corrcoef(pred, yte)[0, 1]), float(np.abs(pred - yte).mean())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "runs" / "sanity_3piece_run" / "best.pt")
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args(argv)
    if not args.checkpoint.exists():
        raise SystemExit(f"no checkpoint at {args.checkpoint}")

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    kit = DrumKit(ck["kit"])
    sg = get_subgraph(cfg)
    model, _ = build_model(cfg, sg, n_styles=max(ck["n_styles"], 1))
    model.load_state_dict(ck["model"])
    model.eval()
    if not model.decoder.has_velocity:
        print("this checkpoint has no velocity head; probing the pool anyway")

    _, va, _ = T.build_loaders(cfg, kit)

    D, M, V, Y = [], [], [], []
    with torch.no_grad():
        for wav, y, vel_y, style, _ in va:
            sid = style if model.genre is not None else None
            drive = model.encoder(wav)
            tonic = model.genre(sid) if sid is not None else None
            full, _ = model.rnn(drive, tonic=tonic, return_all=True)
            motor = full[:, :, model.rnn.motor_idx]
            n = min(drive.shape[1], y.shape[1])
            D.append(drive[:, :n].float().numpy())
            M.append(motor[:, :n].float().numpy())
            V.append(vel_y[:, :n].numpy())
            Y.append(y[:, :n].numpy())

    D, M, V, Y = (np.concatenate(a) for a in (D, M, V, Y))
    print(f"clips {D.shape[0]}  steps {D.shape[1]}  "
          f"drive ch {D.shape[2]}  motor units {M.shape[2]}  classes {V.shape[2]}")

    # Two scorings. "support" is every step inside a hit's +/-3 sigma kernel,
    # which is what the loss weights; "peak" is only the steps at the kernel's
    # centre, which is where the streaming path actually reads velocity. The
    # drive swings through its whole envelope across the support while the
    # target stays flat, so a probe over the support measures the head's
    # ability to be invariant to envelope phase -- not its ability to read how
    # hard the hit was.
    for label, thresh in (("support (y > 0.5)", 0.5), ("peak (y > 0.95)", 0.95)):
        print(f"\n== {label} ==")
        _table(kit, D, M, V, Y, model, thresh)


def _table(kit, D, M, V, Y, model, thresh):
    print(f"{'class':<12}{'n steps':>9}{'target sd':>11}"
          f"{'drive r':>10}{'motor r':>10}{'head r':>9}")
    print("-" * 61)
    for c, name in enumerate(kit.classes):
        hit = Y[:, :, c] > thresh
        if hit.sum() < 50:
            print(f"{name:<12}{int(hit.sum()):>9}   too few steps")
            continue
        target = V[:, :, c][hit].astype(np.float32)
        dr, _ = ridge_r(D[hit].astype(np.float32), target)
        mr, _ = ridge_r(M[hit].astype(np.float32), target)
        hr = float("nan")
        if model.decoder.has_velocity:
            with torch.no_grad():
                head = model.decoder.velocity(torch.from_numpy(M))[:, :, c].numpy()[hit]
            hr = (float(np.corrcoef(head, target)[0, 1])
                  if head.std() > 1e-8 else 0.0)
        print(f"{name:<12}{int(hit.sum()):>9}{target.std():>11.3f}"
              f"{dr:>10.3f}{mr:>10.3f}{hr:>9.3f}")

    # How much does the head's own output actually move?
    if not model.decoder.has_velocity:
        return
    with torch.no_grad():
        head_all = model.decoder.velocity(torch.from_numpy(M)).numpy()
    hit_any = Y > 0.5
    print(f"\nhead output at hits: mean {head_all[hit_any].mean():.3f} "
          f"sd {head_all[hit_any].std():.4f}")
    print(f"target at hits:      mean {V[hit_any].mean():.3f} "
          f"sd {V[hit_any].std():.4f}")
    print("\nA head sd far below the target sd is a head predicting the mean.")


if __name__ == "__main__":
    sys.exit(main())
