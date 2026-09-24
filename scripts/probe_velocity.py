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
from refit import hit_peaks                          # noqa: E402  (one row per hit)


def clip_folds(clips, k=5, seed=0):
    """Assign each clip to one of ``k`` folds, shuffled.

    The ridge probes need rows they were not fit on, and those rows must be
    whole clips: steps inside one clip share a drummer, a kit and a mix, so a
    probe fit on half a clip and scored on the other half is scored on its own
    training data. The split must also be shuffled. Validation rows arrive in
    info.csv order, which is grouped by drummer, and the old "last 30% of rows"
    hold-out was 29 of 36 clips from drummer7 -- every verdict in the table was
    a verdict on one drummer's playing.
    """
    ids = np.unique(clips)
    order = np.random.default_rng(seed).permutation(ids)
    fold_of = {c: i % k for i, c in enumerate(order)}
    return np.array([fold_of[c] for c in clips])


def ridge_cv(X, y, folds, lam=1.0):
    """Cross-validated ridge: every row predicted by a fit that never saw its clip.

    Scoring on all rows rather than a hold-out slice puts the probes on exactly
    the rows the head is scored on, so the motor probe stays a fair upper bound
    on a linear head read from the same pool.
    """
    pred = np.zeros(len(y), np.float32)
    for f in np.unique(folds):
        tr, te = folds != f, folds == f
        Xtr, Xte = X[tr], X[te]
        mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
        Xtr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1), np.float32)])
        Xte = np.hstack([(Xte - mu) / sd, np.ones((len(Xte), 1), np.float32)])
        A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1], dtype=np.float32)
        pred[te] = Xte @ np.linalg.solve(A, Xtr.T @ y[tr])
    return _r(pred, y)


def _r(a, b):
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def boot_ci(a, b, clips, n_boot=2000, seed=0):
    """A 95% interval on a correlation, resampling whole clips.

    Rows inside one clip are not independent draws: one drummer, one kit, one
    groove. Resampling rows treats a clip's hundred hits as a hundred witnesses
    and gave intervals several times too narrow -- kick and low tom read as
    confidently backwards on intervals that, resampled by clip, span zero.
    """
    if len(a) < 20 or a.std() < 1e-8 or b.std() < 1e-8:
        return float("nan"), float("nan")
    ids = np.unique(clips)
    if len(ids) < 5:
        return float("nan"), float("nan")
    rows_of = [np.flatnonzero(clips == c) for c in ids]
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        pick = np.concatenate([rows_of[i] for i in rng.integers(0, len(ids), len(ids))])
        x, y = a[pick], b[pick]
        if x.std() > 1e-8 and y.std() > 1e-8:
            rs.append(np.corrcoef(x, y)[0, 1])
    if not rs:
        return float("nan"), float("nan")
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path,
                    default=ROOT / "runs" / "sanity_3piece_run" / "best.pt")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0,
                    help="GrooveDataset picks a random window per clip per call, "
                         "so without this the probe is not reproducible and small "
                         "correlations move between runs of the same checkpoint")
    args = ap.parse_args(argv)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
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
    # which is what the loss weights; "peak" is one step per hit, at the
    # kernel's centre, which is where the streaming path actually reads
    # velocity. The drive swings through its whole envelope across the support
    # while the target stays flat, so a probe over the support measures the
    # head's ability to be invariant to envelope phase -- not its ability to
    # read how hard the hit was.
    #
    # Every column is scored on every clip. The head is not fit here, so it
    # needs no hold-out; the ridge probes are cross-validated by clip, so they
    # are scored on the same rows as the head and stay its upper bound.
    clips = np.repeat(np.arange(Y.shape[0])[:, None], Y.shape[1], axis=1)
    folds_all = clip_folds(np.arange(Y.shape[0]), seed=args.seed)
    for label, rows in (("support (y > 0.5)", lambda y: y > 0.5),
                        ("peak (y > 0.95)", lambda y: hit_peaks(y, 0.95))):
        print(f"\n== {label} ==")
        _table(kit, D, M, V, Y, model, rows, clips, folds_all, args)


def _table(kit, D, M, V, Y, model, rows, clips, folds_all, args):
    print(f"{'class':<12}{'n rows':>8}{'clips':>7}{'target sd':>11}"
          f"{'drive r':>10}{'motor r':>10}{'head r':>9}"
          f"{'  head 95% CI':>16}")
    print("-" * 83)
    head_all = None
    if model.decoder.has_velocity:
        with torch.no_grad():
            head_all = model.decoder.velocity(torch.from_numpy(M)).numpy()
    for c, name in enumerate(kit.classes):
        hit = rows(Y[:, :, c])
        cid = clips[hit]
        n_clips = len(np.unique(cid))
        if hit.sum() < 30 or n_clips < 5:
            print(f"{name:<12}{int(hit.sum()):>8}{n_clips:>7}   too few hits")
            continue
        target = V[:, :, c][hit].astype(np.float32)
        folds = folds_all[cid]
        dr = ridge_cv(D[hit].astype(np.float32), target, folds, args.ridge)
        mr = ridge_cv(M[hit].astype(np.float32), target, folds, args.ridge)
        hr, lo, hi = float("nan"), float("nan"), float("nan")
        if head_all is not None:
            head = head_all[:, :, c][hit]
            hr = _r(head, target)
            lo, hi = boot_ci(head, target, cid, seed=args.seed)
        ci = f"  [{lo:+.2f}, {hi:+.2f}]" if not np.isnan(lo) else "  --"
        crosses = not np.isnan(lo) and lo <= 0.0 <= hi
        print(f"{name:<12}{int(hit.sum()):>8}{n_clips:>7}{target.std():>11.3f}"
              f"{dr:>10.3f}{mr:>10.3f}{hr:>9.3f}{ci:>16}"
              f"{'  (spans 0)' if crosses else ''}")

    # How much does the head's own output actually move?
    if head_all is None:
        return
    hit_any = Y > 0.5
    print(f"\nhead output at hits: mean {head_all[hit_any].mean():.3f} "
          f"sd {head_all[hit_any].std():.4f}")
    print(f"target at hits:      mean {V[hit_any].mean():.3f} "
          f"sd {V[hit_any].std():.4f}")
    print("\nA head sd far below the target sd is a head predicting the mean.")
    print("All clips scored; probes cross-validated by clip; 95% CI resamples clips.")


if __name__ == "__main__":
    sys.exit(main())
