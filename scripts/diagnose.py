"""Why isn't onset F moving?

When training loss falls while onset F stays flat, the usual cause is a model
that has learned each class's *marginal* onset rate and nothing about timing.
That looks like progress on the loss curve and is worth zero.

This separates the candidate causes by measuring, in order:

  1. Is the model beating a constant predictor at all? If its BCE is no better
     than predicting the per-class base rate, it has learned the base rate.
  2. Do the predictions vary over time, or are they flat?
  3. Do they correlate with the target at all?
  4. Are the recurrent dynamics alive -- do motor rates vary over time? If they
     do not, the problem is upstream of the loss.
  5. Is the drive from the encoder time-varying? If not, nothing downstream can be.
  6. If the drive varies and the motor pool does not, *where* between them is the
     modulation lost? Reported per hop from the JO afferents, because "the
     recurrence washes it out" and "the first synapse washes it out" call for
     completely different fixes.

A model can fail at (4) with a perfectly good loss, and fail at (1) with
perfectly healthy dynamics. The layers have to be told apart before anything is
tuned.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as train_mod  # noqa: E402
from build import build_model, device_of, get_subgraph  # noqa: E402
from decoder import DrumKit  # noqa: E402
from subgraph import hops_from  # noqa: E402


def constant_baseline_bce(target: np.ndarray, pos_weight: float) -> tuple[float, np.ndarray]:
    """BCE of the best constant per-class predictor, and those constants.

    With a weighted BCE the optimal constant is not the mean of the target: the
    positive term is scaled, so it sits at  p*w / (p*w + (1-p))  for target mean p.
    """
    p = target.reshape(-1, target.shape[-1]).mean(axis=0)
    q = p * pos_weight / (p * pos_weight + (1.0 - p))
    q = np.clip(q, 1e-6, 1 - 1e-6)
    bce = -(pos_weight * target * np.log(q) + (1 - target) * np.log(1 - q))
    return float(bce.mean()), q


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--clips", type=int, default=16)
    a = ap.parse_args(argv)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    device = torch.device("cpu")
    sg = get_subgraph(cfg)
    kit = DrumKit(ck["kit"])
    model, _ = build_model(cfg, sg, n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()

    _, val_loader, _ = train_mod.build_loaders(cfg, kit)
    pos_weight = cfg["train"].get("pos_weight", 8.0)

    probs, targs, motor_all, drive_all = [], [], [], []
    all_rates = None                      # one batch, every neuron, for section 6
    seen = 0
    with torch.no_grad():
        for wav, y, _vel_y, style, tempo in val_loader:
            sid = style if model.genre is not None else None
            drive = model.encoder(wav)
            tonic = model.genre(sid) if sid is not None else None
            full, _ = model.rnn(drive, tonic=tonic, return_all=True)
            rates = full[:, :, model.rnn.motor_idx]
            logits = model.decoder(rates)
            n = min(logits.shape[1], y.shape[1])
            probs.append(torch.sigmoid(logits[:, :n].float()).numpy())
            targs.append(y[:, :n].numpy())
            motor_all.append(rates[:, :n].float().numpy())
            drive_all.append(drive[:, :n].float().numpy())
            if all_rates is None:
                all_rates = full[:, :n].float().numpy()
            seen += wav.shape[0]
            if seen >= a.clips:
                break

    P = np.concatenate(probs)          # (clips, steps, classes)
    T = np.concatenate(targs)
    M = np.concatenate(motor_all)      # (clips, steps, motor)
    D = np.concatenate(drive_all)      # (clips, steps, sensory)

    print(f"checkpoint : {a.checkpoint}")
    print(f"clips      : {P.shape[0]}   steps: {P.shape[1]}   classes: {P.shape[2]}\n")

    # 1 -- is it beating a constant predictor?
    model_bce = float(-(pos_weight * T * np.log(np.clip(P, 1e-6, 1))
                        + (1 - T) * np.log(np.clip(1 - P, 1e-6, 1))).mean())
    base_bce, q = constant_baseline_bce(T, pos_weight)
    print("1. Against a constant per-class predictor")
    print(f"   model BCE              {model_bce:.4f}")
    print(f"   best-constant BCE      {base_bce:.4f}")
    gap = base_bce - model_bce
    print(f"   gain over constant     {gap:+.4f}"
          f"   <-- near zero means it learned the base rate\n")

    # 2/3 -- does the output vary, and does it track the target?
    print("2/3. Per class: does the prediction move, and does it track the target?")
    head = f"   {'class':<12}{'pred mean':>10}{'pred sd':>9}{'range':>16}{'corr(pred,targ)':>17}{'targ mean':>11}"
    print(head)
    print("   " + "-" * (len(head) - 3))
    for c, name in enumerate(kit.classes):
        p = P[:, :, c].ravel()
        t = T[:, :, c].ravel()
        corr = float(np.corrcoef(p, t)[0, 1]) if p.std() > 1e-9 and t.std() > 1e-9 else float("nan")
        print(f"   {name:<12}{p.mean():>10.4f}{p.std():>9.4f}"
              f"{f'{p.min():.3f}-{p.max():.3f}':>16}{corr:>17.4f}{t.mean():>11.4f}")

    # 4 -- are the recurrent dynamics alive?
    m_t = M.std(axis=1).mean()          # variation over time, averaged over units
    m_u = M.std(axis=2).mean()          # variation across units
    print(f"\n4. Recurrent dynamics")
    print(f"   motor rate: mean {M.mean():.4f}   sd over time {m_t:.6f}   sd across units {m_u:.6f}")
    dead = int((M.std(axis=1).mean(axis=0) < 1e-6).sum())
    print(f"   motor units with no temporal variation: {dead} / {M.shape[2]}"
          f"   <-- all of them means the dynamics are dead")

    # 5 -- is the drive time-varying, and has the encoder silenced itself?
    d_t = D.std(axis=1).mean()
    print(f"\n5. Encoder drive")
    print(f"   drive: mean {D.mean():.4f}   sd over time {d_t:.6f}")
    print(f"   drive channels with no temporal variation: "
          f"{int((D.std(axis=1).mean(axis=0) < 1e-9).sum())} / {D.shape[2]}")

    w = model.encoder.to_jo.weight.detach().numpy()
    neg_frac = float((w < 0).mean())
    enc = model.encoder
    constraint = "constrained >= 0" if getattr(enc, "nonneg", False) else "unconstrained"
    calib = ("calibrated" if bool(getattr(enc, "calibrated", torch.zeros(())))
             else "NOT calibrated" if getattr(enc, "standardize", False) else "off")
    print(f"   to_jo weights: mean {w.mean():+.4f}, {neg_frac:.0%} negative "
          f"(initialised non-negative from the JO zone prior; {constraint})")
    print(f"   feature standardisation: {calib}")
    dc_ratio = abs(float(D.mean())) / max(d_t, 1e-12)
    print(f"   drive DC / temporal sd = {dc_ratio:.2f}"
          f"   <-- >>1 means the input is mostly a constant offset")

    # 6 -- where between the ears and the wings is the modulation lost?
    print("\n6. Signal by depth from the JO afferents")
    dist = hops_from(sg, sg.role("sensory").astype(np.int64))
    motor_set = set(sg.role("motor").astype(np.int64).tolist())
    rel_t = all_rates.std(axis=1).mean(axis=0) / np.maximum(
        np.abs(all_rates.mean(axis=(0, 1))), 1e-12)       # per neuron, relative
    head = f"   {'hop':<6}{'neurons':>9}{'mean rate':>11}{'sd over time':>14}{'relative':>11}"
    print(head)
    print("   " + "-" * (len(head) - 3))
    prev = None
    for h in sorted({int(d) for d in np.unique(dist) if d >= 0}):
        m = dist == h
        if not m.any():
            continue
        r = float(rel_t[m].mean())
        drop = f"   /{prev / r:,.0f}" if prev and r > 0 else ""
        print(f"   {h:<6}{int(m.sum()):>9}{all_rates[:, :, m].mean():>11.4f}"
              f"{all_rates[:, :, m].std(axis=1).mean():>14.6f}{r:>11.5f}{drop}")
        prev = r
    unreached = int((dist < 0).sum())
    n_motor_by_hop = {h: sum(1 for i in motor_set if dist[i] == h)
                      for h in sorted({int(dist[i]) for i in motor_set})}
    print(f"   motor pool sits at hops {n_motor_by_hop}"
          f"   |   never reached from sensory: {unreached}")

    # -- verdict. Every test is RELATIVE to the signal's own scale: an absolute
    # threshold calls a prediction with sd 0.0014 around a mean of 0.31 "varying",
    # which is how the first version of this script misread exactly that case.
    rel = lambda x, m: float(x) / max(abs(float(m)), 1e-12)
    rel_pred, rel_motor, rel_drive = rel(P.std(axis=1).mean(), P.mean()), \
        rel(m_t, M.mean()), rel(d_t, D.mean())
    corrs = np.array([abs(np.corrcoef(P[:, :, c].ravel(), T[:, :, c].ravel())[0, 1])
                      for c in range(P.shape[2])
                      if P[:, :, c].std() > 1e-12 and T[:, :, c].std() > 1e-12])

    print(f"\n   relative temporal variation: drive {rel_drive:.3f}   "
          f"motor {rel_motor:.4f}   prediction {rel_pred:.4f}")
    print(f"   mean |corr(pred, target)|: {corrs.mean():.4f}" if len(corrs) else "")

    print("\nReading:")
    verdict = []
    if gap < 0.01:
        verdict.append("the model is at or below the best constant predictor -- it has "
                       "learned\n     the per-class base rate, and onset F cannot move "
                       "until that changes")
    if rel_drive < 1e-3:
        verdict.append("the encoder drive is effectively constant: the fault is in the "
                       "front end")
    elif rel_motor < 0.05:
        verdict.append(f"drive varies ({rel_drive:.2f} relative) but motor rates barely do "
                       f"({rel_motor:.4f}):\n     the time-varying signal is being washed "
                       f"out in the recurrence")
    if neg_frac > 0.8 and dc_ratio > 1.0:
        verdict.append("the encoder has learned near-uniformly negative weights and a large "
                       "DC\n     offset -- it is suppressing its own sensory input, which is "
                       "the trivial\n     solution when a regulariser rewards silence")
    if len(corrs) and corrs.mean() < 0.1:
        verdict.append("predictions do not track the target at all (|corr| < 0.1)")
    if not verdict:
        verdict.append("output varies and tracks the target; likely undertrained rather "
                       "than structurally broken")
    for v in verdict:
        print(f"   * {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
