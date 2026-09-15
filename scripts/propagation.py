"""Can the subgraph carry a time-varying signal from the ears to the wings at all?

``scripts/diagnose.py`` answers "did this checkpoint learn anything"; this
answers the prior question: does an *untrained* arm, at a given spectral
radius, settle into a regime where the motor pool still moves -- before any
training is spent finding out that it does not.

**Fixed 15 Sep 2026.** The previous version ran 300 steps from ``v=0`` and
scored the whole run as one number. That number is dominated by the cold-start
transient: motor modulation measured this way drops 8-10x between steps
0-299 and steps 300-799, at every radius, on both tiers tried. Its guard
("motor modulation >= 0.5x the JO afferents' own, <= 5% of unit-steps
clipped") passed only on the transient -- at steady state no swept radius
satisfied it, and the ``at_clip`` guard was a mean over unit-steps, which
cannot tell "every neuron pinned 5% of the time" (harmless) from "5% of
neurons pinned permanently" (those units are dead). Full account in
``results/local/2026-09-15-probe-grid.md``.

This version:

* runs long enough (>=1,600 steps by default -- the config is widened to
  provide it real audio, not just a longer forward pass over silence) and
  reports the motor statistic in non-overlapping ``--window``-step blocks;
* emits ``t_settle`` as a first-class output: the first window after which
  the motor statistic changes by less than ``--settle-tol`` across
  ``--settle-consec`` consecutive windows. All acceptance statistics are then
  computed on ``[t_settle, T)``, not the whole run;
* replaces the mean-over-unit-steps clip guard with a **per-neuron** pinned
  fraction: reject a radius where >=1% of neurons sit at the clip for more
  than half the settled window, or where *any* motor neuron does;
* adds a motor participation-ratio guard (>= the number of kit classes --
  a trajectory collapsed onto one or two modes cannot carry eight drum
  classes whatever the readout is);
* scores the settled window with the same linear probe
  ``scripts/probe_grid.py`` uses (ridge from the motor pool to the smoothed
  onset targets, scored with the repo's own onset F), and recommends the
  **argmax** of that probe over radii that pass the guards, not the smallest
  radius that passes. A self-penalising objective does not need the
  saturation tiebreak the old "smallest passing" rule existed for.
* draws clips spread across the corpus (``np.linspace`` over the dataset),
  not ``dataset[:n]`` -- ``info.csv`` is ordered by drummer and session, so
  the first few rows are one drummer in one room.

    python scripts/propagation.py --config configs/v1_8piece.yaml \
        --radii 1 2 5 10 25 50 100 250 --clips 8

rho remains a hub statistic (see ``probe_grid.py``'s docstring): this script
still sweeps one arm at a common rho by default, which is enough to fix the
settling-time bug, but it is not by itself a fair cross-arm comparison. Pass
``--arms`` to run it on more than one and see that directly.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as train_mod  # noqa: E402
from build import build_model, device_of, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit  # noqa: E402
from subgraph import hops_from  # noqa: E402

from probe_grid import build_arms, participation_ratio, probe_f  # noqa: E402


def relative_variation(rates: np.ndarray, mask: np.ndarray) -> float:
    """Mean over the masked neurons of (sd over time) / |mean|.

    Relative, because an absolute threshold calls a unit with sd 0.003 around a
    mean of 0.72 "varying" -- which is the mistake the first diagnostic made.
    """
    if not mask.any():
        return float("nan")
    sd = rates.std(axis=1).mean(axis=0)[mask]
    mu = np.abs(rates.mean(axis=(0, 1)))[mask]
    return float((sd / np.maximum(mu, 1e-12)).mean())


def windowed_stat(rates: np.ndarray, mask: np.ndarray, window: int) -> np.ndarray:
    """Per-window relative variation of ``mask``, in non-overlapping blocks.

    The trailing partial block is dropped so every window carries the same
    amount of evidence.
    """
    n_windows = rates.shape[1] // window
    return np.array([relative_variation(rates[:, w * window:(w + 1) * window], mask)
                     for w in range(n_windows)])


def find_t_settle(stat: np.ndarray, window: int, tol: float, consec: int,
                  abs_floor: float = 0.01) -> tuple[int | None, int | None]:
    """First window index after which ``stat`` moves < ``tol`` for ``consec``
    consecutive windows. Returns ``(step, window_index)`` or ``(None, None)``
    if the run never settles by that definition.

    A step counts as settled if the change is < ``tol`` *relative* to the
    previous window, OR smaller than ``abs_floor`` outright. Pure relative
    change is unstable once the statistic itself is near zero -- an arm whose
    motor pool barely moves at all (e.g. a low-rho sign-shuffled draw) can sit
    at 0.0003 -> 0.0005 forever and never clear a 10%-relative bar, even
    though that is exactly what "settled" should mean for a near-flat trace.
    """
    n = len(stat)
    for i in range(max(0, n - consec + 1)):
        seg = stat[i:i + consec]
        if len(seg) < consec:
            break
        prev = np.abs(seg[:-1])
        diff = np.abs(np.diff(seg))
        settled_step = (diff < abs_floor) | (diff < tol * prev)
        if np.all(settled_step):
            return i * window, i
    return None, None


def per_neuron_pinned(rates: np.ndarray, pinned_level: float) -> np.ndarray:
    """Fraction of clip-steps each neuron sits at the state clip."""
    hit = rates > pinned_level
    return hit.reshape(-1, hit.shape[-1]).mean(axis=0)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece_cpu.yaml")
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0])
    ap.add_argument("--arms", nargs="+", default=["real"],
                    help="real | rewired[:seed] | sign_shuffled[:seed] | erdos_renyi[:seed]")
    ap.add_argument("--clips", type=int, default=8)
    ap.add_argument("--batch", type=int, default=4, help="clips per forward pass")
    ap.add_argument("--steps", type=int, default=1600,
                    help="forward-pass length; the config's data.seconds is widened "
                         "to provide this much real audio if it is shorter")
    ap.add_argument("--window", type=int, default=100, help="non-overlapping window, steps")
    ap.add_argument("--settle-tol", type=float, default=0.10,
                    help="max relative change between consecutive windows to call it settled")
    ap.add_argument("--settle-consec", type=int, default=3,
                    help="consecutive windows required to satisfy --settle-tol")
    ap.add_argument("--settle-abs-floor", type=float, default=0.01,
                    help="a window-to-window change below this counts as settled "
                         "even if it fails the relative --settle-tol check")
    ap.add_argument("--pinned-neuron-frac", type=float, default=0.01,
                    help="reject a radius where more than this share of neurons are pinned")
    ap.add_argument("--pinned-window-frac", type=float, default=0.5,
                    help="a neuron counts as 'pinned' if it sits at the clip more than "
                         "this fraction of the settled window")
    ap.add_argument("--context", type=int, default=2, help="+/- frames into the probe")
    ap.add_argument("--clip-seed", type=int, default=None,
                    help="seed for the corpus loader's random crop offsets "
                         "(default: train.seed). Vary it to draw a genuinely "
                         "independent sample of audio from the same clips -- that is "
                         "how this tool's noise floor must be measured, and it "
                         "dominates --seed by an order of magnitude.")
    ap.add_argument("--seed", type=int, default=None,
                    help="override train.seed for model init. Re-running a cell under a "
                         "different seed is how this tool's own noise floor is measured; "
                         "the between-arm differences it reports are of the same order.")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "local" / "propagation_grid.json")
    a = ap.parse_args(argv)

    if a.steps < 1600:
        print(f"warning: --steps {a.steps} < 1600; the settling-time bug this script "
              "was fixed for is largest in exactly this range", file=sys.stderr)

    cfg = load_config(a.config)
    device = device_of(cfg)
    seed = a.seed if a.seed is not None else int(cfg["train"].get("seed", 0))
    clip_seed = a.clip_seed if a.clip_seed is not None else seed
    step_ms = float(cfg["audio"]["step_ms"])
    need_s = a.steps * step_ms / 1000.0 * 1.05
    have_s = float(cfg.get("data", {}).get("seconds", 4.0))
    if need_s > have_s:
        cfg = {**cfg, "data": {**cfg.get("data", {}), "seconds": need_s}}
        print(f"widened data.seconds {have_s:g} -> {need_s:.2f} to cover {a.steps} steps "
              f"of real audio at step_ms={step_ms:g}")

    sg = get_subgraph(cfg)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    n_classes = len(kit.classes)
    loader, _val, n_styles = train_mod.build_loaders(cfg, kit)
    n_styles = max(int(n_styles or 1), 1)

    ds = loader.dataset
    n_clips = min(a.clips, len(ds))
    if n_clips < 4:
        raise SystemExit(f"need >=4 clips to fit and score a probe, have {n_clips}")
    pick = np.linspace(0, len(ds) - 1, n_clips).astype(int)
    # The train split crops a random window per __getitem__ off torch's GLOBAL
    # generator (dataset.py:287). The DataLoader seeds that per worker; indexing
    # loader.dataset directly, as this script does, is not covered by it -- so
    # without this seed every invocation scores the probe on different audio,
    # with the targets moving too. That, not model init, was the "noise floor"
    # of ~0.014 reported on 15 Sep: the same cell re-run four times spans 0.069.
    torch.manual_seed(clip_seed)
    np.random.seed(clip_seed)
    wavs = torch.stack([torch.as_tensor(ds[i][0]) for i in pick])
    ys = np.stack([np.asarray(ds[i][1]) for i in pick])

    ecfg = cfg.get("eval", {})
    thresholds = list(ecfg.get("threshold_sweep", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
    tol = float(ecfg.get("tolerance_s", 0.05))
    clip_at = float(cfg["model"].get("state_clip", 20.0))
    pinned_level = np.log1p(np.exp(clip_at - 1.0))

    print(f"{sg.summary().splitlines()[0]}   |   {n_clips} clips, {a.steps} steps, "
          f"window {a.window}, seed {seed}, clip_seed {clip_seed}")

    all_rows = []
    for arm_name, arm_sg in build_arms(sg, a.arms):
        dist = hops_from(arm_sg, arm_sg.role("sensory").astype(np.int64))
        motor = np.zeros(arm_sg.n_nodes, dtype=bool)
        motor[arm_sg.role("motor").astype(np.int64)] = True
        motor_idx = arm_sg.role("motor").astype(np.int64)
        hops = sorted({int(d) for d in np.unique(dist) if 0 <= d <= 4})

        print(f"\n=== arm {arm_name} ===")
        arm_rows = []
        for rho in a.radii:
            torch.manual_seed(seed)
            np.random.seed(seed)
            c = {**cfg, "model": {**cfg["model"], "spectral_radius": rho, "gain_scale": "auto"}}
            model, _ = build_model(c, arm_sg, n_styles=n_styles)
            model = model.to(device).eval()
            train_mod.calibrate_encoder(model, ds, cfg, device)

            rates_all = []
            with torch.no_grad():
                for b0 in range(0, n_clips, a.batch):
                    wav = wavs[b0:b0 + a.batch].to(device)
                    drive = model.encoder(wav)[:, : a.steps]
                    rates, _ = model.rnn(drive, return_all=True)
                    rates_all.append(rates.float().cpu().numpy())
                    del rates, drive
            r = np.concatenate(rates_all, axis=0)
            n_steps = r.shape[1]

            stat = windowed_stat(r, motor, a.window)
            t_settle, w_settle = find_t_settle(stat, a.window, a.settle_tol,
                                               a.settle_consec, a.settle_abs_floor)
            settle_str = (f"window {w_settle} (step {t_settle})" if t_settle is not None
                          else "never (treating whole run as unsettled)")
            t0 = t_settle if t_settle is not None else 0

            windows_str = " ".join(f"{v:.3f}" for v in stat)
            print(f"\n  rho {rho:g}: per-window motor modulation [{windows_str}]")
            print(f"    t_settle: {settle_str}")

            settled = r[:, t0:]
            hop_row = "".join(f"  hop{h}={relative_variation(settled, dist == h):.4f}"
                              for h in hops)
            motor_settled = relative_variation(settled, motor)

            pinned_per_neuron = per_neuron_pinned(settled, pinned_level)
            stuck = pinned_per_neuron > a.pinned_window_frac
            frac_stuck = float(stuck.mean())
            motor_stuck = int(stuck[motor_idx].sum())
            guard_pinned = frac_stuck < a.pinned_neuron_frac and motor_stuck == 0

            steps_here = min(settled.shape[1], ys.shape[1] - t0) if t0 < ys.shape[1] else 0
            if steps_here > 0:
                y_settled = ys[:, t0:t0 + steps_here]
                motor_trace = settled[:, :steps_here][:, :, motor_idx]
                pr = participation_ratio(motor_trace)
                f = probe_f(motor_trace, y_settled, step_ms, thresholds, tol, k=a.context)
            else:
                pr, f = float("nan"), float("nan")
            guard_pr = np.isfinite(pr) and pr >= n_classes
            ok = guard_pinned and guard_pr

            print(f"    settled ([t_settle, T), {settled.shape[1]} steps):{hop_row}"
                  f"  motor={motor_settled:.4f}")
            print(f"    pinned: {frac_stuck:.2%} of neurons ({motor_stuck} motor) "
                  f"stuck >{a.pinned_window_frac:.0%} of the window  |  motor PR {pr:.2f} "
                  f"(need >= {n_classes})  |  probe_F {f:.3f}  |  "
                  f"{'ok' if ok else 'GUARD'}")

            arm_rows.append({
                "arm": arm_name, "rho": rho, "seed": seed,
                "clip_seed": clip_seed, "t_settle": t_settle,
                "window_stat": stat.tolist(), "motor_settled": motor_settled,
                "pinned_frac": frac_stuck, "motor_pinned": motor_stuck,
                "motor_pr": pr, "probe_f": f, "passes_guards": bool(ok),
            })
        all_rows.extend(arm_rows)

        passing = [r for r in arm_rows if r["passes_guards"] and np.isfinite(r["probe_f"])]
        if passing:
            best = max(passing, key=lambda r: r["probe_f"])
            print(f"\n  {arm_name}: recommend rho={best['rho']:g} "
                  f"(probe_F {best['probe_f']:.3f}, argmax over guard-passing radii, "
                  "not smallest-passing)")
        else:
            finite = [r for r in arm_rows if np.isfinite(r["probe_f"])]
            if finite:
                best = max(finite, key=lambda r: r["probe_f"])
                print(f"\n  {arm_name}: NO radius passes both guards. Best probe_F "
                      f"{best['probe_f']:.3f} at rho={best['rho']:g}, reported for "
                      "reference only -- widen --radii or treat this arm as untestable "
                      "at this scale.")
            else:
                print(f"\n  {arm_name}: no radius produced a finite probe_F.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(all_rows, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
