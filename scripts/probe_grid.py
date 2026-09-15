"""What can each Phase 4 arm carry, before any training, at every operating point?

``scripts/propagation.py`` sweeps the target spectral radius on the real arm and
scores it with a *proxy* -- relative temporal variation of the motor pool,
referenced to the JO afferents' own variation. Two problems with that, both of
which this script exists to fix:

1. The proxy floats with a calibration constant. ``calibrate_encoder`` sets the
   affine that decides the afferents' variation, so "motor modulation >= 0.5x
   the afferents' own" is measured against a number the encoder happens to be
   scaled to. A criterion that moves with a calibration is not a criterion.

2. It is measured on one arm. The Phase 4 comparison pins every arm to a common
   spectral radius, and rho is a statistic about a hub: the leading eigenvector
   of the real 30k subgraph has a participation ratio of ~191 neurons out of
   30,000, the degree-matched rewire ~11,800. Matching a number carried by 0.6%
   of one network and 39% of another does not equalise them. Measured at
   rho=10, the median neuron's total input gain is 2.10 in the real arm and
   29.04 in a rewired draw -- a 13.8x difference that no amount of training is
   asked to undo, and that the arms are then compared across.

So: score every arm at every scale with a *linear probe* -- ridge from the wing
motor pool to the smoothed onset targets, scored with the repo's own peak
picking and onset F. The decoder really is a single linear layer
(``MotorToDrums``), so the probe's score is a genuine ceiling on what a frozen
core could achieve, in the units of the headline metric, rather than a proxy
for it. Saturation penalises itself: a pinned unit carries no variance.

Forward passes only, untrained models. The point is to find out whether the
design is sound before spending a ~50 GPU-hour campaign on it:

    python scripts/probe_grid.py --config configs/v1_8piece_cpu.yaml \
        --radii 1 2 5 10 25 50 100 250 --clips 12

Read the output three ways:

* **Arms peak at different scales** -- rho-matching is unfair and the campaign
  should give each arm its own operating point.
* **Curves flat over an order of magnitude** -- the whole question is moot;
  keep rho-matching and say so with the curve as evidence.
* **Real below rewired everywhere** -- the hypothesis is in trouble, and it
  cost two GPU-hours to find out instead of a hundred.

What it cannot do: this is a ceiling on the *frozen* core. Training moves
``log_gain``, which is 97% of the trainable parameters. A good probe score is
necessary, not sufficient.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as train_mod  # noqa: E402
from ablations import degree_matched_rewire, shuffle_signs  # noqa: E402
from build import build_model, device_of, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit  # noqa: E402
from metrics import onset_f_sweep  # noqa: E402


# ----------------------------------------------------------------- arms --

def erdos_renyi(sg, seed: int = 0):
    """Same node and edge count, no degree structure at all.

    Brackets the degree-matched null from the side nobody disputes: a null that
    matches nothing *should* lose. If it does not, the task does not need
    topology and the whole comparison is uninformative. Free to include here
    because it costs one more row of forward passes; it is not worth training.
    """
    rng = np.random.default_rng(seed)
    e = sg.n_edges
    src = rng.integers(0, sg.n_nodes, e).astype(sg.edge_index.dtype)
    dst = rng.integers(0, sg.n_nodes, e).astype(sg.edge_index.dtype)

    real_presign = np.zeros(sg.n_nodes, dtype=np.float32)
    real_presign[sg.edge_index[0]] = sg.edge_sign
    pool = real_presign[np.unique(sg.edge_index[0])]      # the +1/-1 balance

    presign = np.zeros(sg.n_nodes, dtype=np.float32)
    emit = np.unique(src)
    presign[emit] = rng.choice(pool, size=len(emit), replace=True)
    order = rng.permutation(e)
    return replace(
        sg, edge_index=np.stack([src, dst]), weight=sg.weight[order].copy(),
        edge_sign=presign[src],
        meta={**sg.meta, "ablation": "erdos_renyi", "seed": seed},
    )


def build_arms(sg, which: list[str]) -> list[tuple[str, object]]:
    out = []
    for name in which:
        if name == "real":
            out.append(("real", sg))
        elif name.startswith("rewired"):
            s = int(name.split(":")[1]) if ":" in name else 0
            out.append((f"rewired:{s}", degree_matched_rewire(sg, seed=s)))
        elif name.startswith("sign_shuffled"):
            s = int(name.split(":")[1]) if ":" in name else 0
            out.append((f"sign_shuffled:{s}", shuffle_signs(sg, seed=s)))
        elif name.startswith("erdos_renyi"):
            s = int(name.split(":")[1]) if ":" in name else 0
            out.append((f"erdos_renyi:{s}", erdos_renyi(sg, seed=s)))
        else:
            raise SystemExit(f"unknown arm {name!r}")
    return out


# ---------------------------------------------------------------- probe --

def with_context(x: np.ndarray, k: int) -> np.ndarray:
    """(clips, steps, n) -> (clips, steps, n*(2k+1)), edge-padded.

    A drum onset is an event in time and the motor pool is a low-pass filter of
    the drive, so a probe reading one frame is scoring the wrong thing. The
    decoder proper sees one frame, but it is trained; this probe is not, and a
    small symmetric window is the cheapest way to stop the ceiling being an
    artefact of phase lag.
    """
    if k <= 0:
        return x
    pads = []
    for d in range(-k, k + 1):
        pads.append(np.stack([np.roll(c, d, axis=0) for c in x]))
    out = np.concatenate(pads, axis=-1)
    out[:, :k] = out[:, k:k + 1]              # kill the wrap-around
    out[:, -k:] = out[:, -k - 1:-k]
    return out


def probe_f(motor: np.ndarray, y: np.ndarray, step_ms: float,
            thresholds, tol: float, k: int = 2, lam: float = 1.0) -> float:
    """Ridge from the motor pool to the onset targets; onset F on held-out clips.

    Fit on the first half of the clips, score on the second. Returns the best F
    over the threshold sweep -- the same "best of sweep" the headline metric
    uses, so the number is directly comparable to a trained model's.
    """
    X = with_context(motor, k)
    n = len(X)
    half = max(1, n // 2)
    Xtr = X[:half].reshape(-1, X.shape[-1]).astype(np.float64)
    ytr = y[:half].reshape(-1, y.shape[-1]).astype(np.float64)

    mu, sd = Xtr.mean(0, keepdims=True), Xtr.std(0, keepdims=True) + 1e-8
    Xtr = np.hstack([(Xtr - mu) / sd, np.ones((len(Xtr), 1))])
    A = Xtr.T @ Xtr + lam * np.eye(Xtr.shape[1])
    w = np.linalg.solve(A, Xtr.T @ ytr)

    best = 0.0
    scored = 0
    per_thr = {t: [] for t in thresholds}
    for c in range(half, n):
        Xc = np.hstack([(X[c].astype(np.float64) - mu) / sd, np.ones((len(X[c]), 1))])
        pred = np.clip(Xc @ w, 0.0, None)
        if pred.std() < 1e-9:
            continue
        got = onset_f_sweep(pred.astype(np.float32), y[c], step_ms,
                            thresholds, tolerance_s=tol)
        for t, v in got.items():
            per_thr[t].append(v)
        scored += 1
    if not scored:
        return float("nan")
    for t, vals in per_thr.items():
        if vals:
            best = max(best, float(np.mean(vals)))
    return best


def participation_ratio(m: np.ndarray) -> float:
    """PR of the motor pool's temporal covariance spectrum.

    A trajectory collapsed onto one or two modes cannot carry eight drum
    classes whatever the readout is, and no modulation statistic would say so.
    """
    x = m.reshape(-1, m.shape[-1])
    x = x - x.mean(0, keepdims=True)
    if x.shape[0] < 2:
        return float("nan")
    ev = np.linalg.eigvalsh(np.cov(x, rowvar=False))
    ev = np.clip(ev, 0.0, None)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / s2) if s2 > 0 else float("nan")


# ----------------------------------------------------------------- main --

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece_cpu.yaml")
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0])
    ap.add_argument("--arms", nargs="+",
                    default=["real", "rewired:0", "rewired:1",
                             "sign_shuffled:0", "erdos_renyi:0"])
    ap.add_argument("--clips", type=int, default=12)
    ap.add_argument("--steps", type=int, default=None, help="default: the whole clip")
    ap.add_argument("--context", type=int, default=2, help="+/- frames into the probe")
    ap.add_argument("--batch", type=int, default=4, help="clips per forward pass")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "local" / "probe_grid.json")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    device = device_of(cfg)
    sg = get_subgraph(cfg)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    loader, _val, n_styles = train_mod.build_loaders(cfg, kit)
    n_styles = max(int(n_styles or 1), 1)

    ds = loader.dataset
    n_clips = min(a.clips, len(ds))
    if n_clips < 4:
        raise SystemExit(f"need >=4 clips to fit and score a probe, have {n_clips}")
    # Spread across the corpus rather than taking the first n. info.csv is
    # ordered by drummer and session, so ds[:n] is one drummer in one room --
    # which is fine for a smoke test and wrong for a probe that is meant to
    # bound what the architecture can carry on this dataset.
    pick = np.linspace(0, len(ds) - 1, n_clips).astype(int)
    wavs = torch.stack([torch.as_tensor(ds[i][0]) for i in pick])
    ys = np.stack([np.asarray(ds[i][1]) for i in pick])

    step_ms = float(cfg["audio"]["step_ms"])
    ecfg = cfg.get("eval", {})
    thresholds = list(ecfg.get("threshold_sweep", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    tol = float(ecfg.get("tolerance_s", 0.05))
    clip_at = float(cfg["model"].get("state_clip", 20.0))
    pinned_level = np.log1p(np.exp(clip_at - 1.0))

    print(f"{sg.summary().splitlines()[0]}")
    print(f"{n_clips} clips | probe context +/-{a.context} | thresholds {thresholds}")

    rows = []
    for arm_name, arm_sg in build_arms(sg, a.arms):
        for rho in a.radii:
            torch.manual_seed(cfg["train"].get("seed", 0))
            np.random.seed(cfg["train"].get("seed", 0))
            c = {**cfg, "model": {**cfg["model"],
                                  "spectral_radius": rho, "gain_scale": "auto"}}
            model, _ = build_model(c, arm_sg, n_styles=n_styles)
            model = model.to(device).eval()
            train_mod.calibrate_encoder(model, ds, cfg, device)

            # The other candidate normalisation target, reported alongside rho
            # so both are visible on one axis: total |incoming gain| per neuron.
            with torch.no_grad():
                w = model.rnn.edge_weight().abs()
                # edge_index is stored [post, pre] (the matrix convention, see
                # model.py) -- NOT [pre, post] like the constructor argument.
                # Index [0] for incoming. Getting this backwards measures how
                # much each neuron SENDS and calls it input, which makes a
                # terminal readout pool look starved when it is well fed.
                dst = model.rnn.edge_index[0].long()
                gain = torch.zeros(model.rnn.n_nodes, device=w.device)
                gain.scatter_add_(0, dst, w)
                med_gain = float(gain.median())
                motor_gain = float(gain[model.rnn.motor_idx].median())

            motor_all, pinned = [], None
            n_steps = None
            with torch.no_grad():
                for b0 in range(0, n_clips, a.batch):
                    wav = wavs[b0:b0 + a.batch].to(device)
                    drive = model.encoder(wav)
                    if a.steps:
                        drive = drive[:, : a.steps]
                    rates, _ = model.rnn(drive, return_all=True)
                    n_steps = rates.shape[1]
                    hit = (rates > pinned_level).float().sum(dim=(0, 1))
                    pinned = hit if pinned is None else pinned + hit
                    motor_all.append(rates[:, :, model.rnn.motor_idx].float().cpu().numpy())
                    del rates, drive
            motor = np.concatenate(motor_all, axis=0)
            frac = (pinned / (n_clips * n_steps)).cpu().numpy()

            steps = min(motor.shape[1], ys.shape[1])
            f = probe_f(motor[:, :steps], ys[:, :steps], step_ms,
                        thresholds, tol, k=a.context)

            dead = float((frac > 0.5).mean())
            motor_dead = int((frac[model.rnn.motor_idx.cpu().numpy()] > 0.5).sum())
            pr = participation_ratio(motor[:, :steps])
            ok = dead < 0.01 and motor_dead == 0 and pr >= len(kit.classes)

            rows.append({"arm": arm_name, "rho": rho, "probe_f": f,
                         "median_gain": med_gain, "motor_gain": motor_gain,
                         "pinned_frac": dead, "motor_pinned": motor_dead,
                         "motor_pr": pr, "passes_guards": bool(ok)})
            print(f"  {arm_name:<18} rho {rho:>7.1f}  probe_F {f:>6.3f}  "
                  f"gain(med/motor) {med_gain:>8.3f}/{motor_gain:>8.3f}  "
                  f"pinned {dead:>6.2%}  motorPR {pr:>6.2f}  "
                  f"{'ok' if ok else 'GUARD'}", flush=True)

    # Baseline: the encoder drive straight to the targets. If the recurrence
    # cannot beat its own input, it is subtracting information -- which is a
    # finding, and one worth having before training rather than after.
    torch.manual_seed(cfg["train"].get("seed", 0))
    model, _ = build_model({**cfg}, sg, n_styles=n_styles)
    model = model.to(device).eval()
    train_mod.calibrate_encoder(model, ds, cfg, device)
    drives = []
    with torch.no_grad():
        for b0 in range(0, n_clips, a.batch):
            d = model.encoder(wavs[b0:b0 + a.batch].to(device))
            drives.append(d[:, : a.steps].float().cpu().numpy() if a.steps
                          else d.float().cpu().numpy())
    drive = np.concatenate(drives, axis=0)
    steps = min(drive.shape[1], ys.shape[1])
    base = probe_f(drive[:, :steps], ys[:, :steps], step_ms, thresholds, tol, k=a.context)
    print(f"\n  {'BASELINE drive->targets':<18} probe_F {base:>6.3f}"
          "   (no recurrence; any arm below this is subtracting information)")
    rows.append({"arm": "baseline_drive", "rho": None, "probe_f": base})

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {a.out}")

    real = [r for r in rows if r["arm"] == "real" and r.get("passes_guards")]
    if real:
        best = max(real, key=lambda r: r["probe_f"])
        print(f"real peaks at rho {best['rho']:g} (probe_F {best['probe_f']:.3f}); "
              "argmax, not smallest-passing -- a self-penalising objective does "
              "not need the saturation tiebreak the old rule used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
