"""Score trained checkpoints at every peak-picking threshold, on every metric.

``evaluate`` reports onset F at the best of seven thresholds, *chosen on the
split it is scored on*, but computes ``beat_align_ms``, ``groove_sim`` and
``mean_dev_ms`` at a hard-coded 0.3. The B=4 pilot showed that this is not a
neutral pair of choices -- it biases the two families of metric in opposite
directions:

* Selecting the threshold per arm favours whichever arm has the *peakier*
  response. Real gained 0.008 from the selection, a rewired draw gained 0.128 --
  larger than the gap between the arms, so the onset-F ranking was decided by
  the selection rather than by the topology. Two of three rewired draws also
  landed on 0.6, the top of the old grid, so their argmax was not even
  bracketed.
* Fixing 0.3 for the timing metrics does the reverse. Rewired's own optimum is
  0.5-0.6, so at 0.3 it is emitting a shower of low-confidence peaks that no
  operating point it would actually be run at would emit. Its 8 ms rush and its
  2.2 ms worse beat alignment may be an artifact of being scored somewhere it
  does not live.

Neither comparison is fair, and the unfairnesses point opposite ways, so the
pilot cannot say which arm is better. This script removes the choice: it takes
one forward pass per checkpoint, caches the probabilities, and then sweeps
*every* metric across a fine threshold grid on the cached output. The curves
say what each arm does at each operating point, and the arms can then be
compared at matched thresholds, at each one's own optimum, or anywhere else --
without re-running the GPU.

    python scripts/threshold_response.py --out runs/threshold_response.json

Forward passes only, no training. What it cannot settle: the checkpoints are
one real topology against three rewired draws, and three nulls put a floor of
1/(3+1) = 0.25 on any rank-based p-value. This sharpens the measurement; it
does not supply the missing null draws.
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
from ablations import build_arm  # noqa: E402
from build import device_of, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit  # noqa: E402
from metrics import (beat_alignment_error, groove_similarity,  # noqa: E402
                     onset_f_measure, onset_f_sweep)

DEFAULT_CKPTS = [
    ROOT / "runs" / "pilot_b4" / "real.pt",
    ROOT / "runs" / "pilot_b4_rewired" / "rewired.pt",
    ROOT / "runs" / "pilot_b4_spread" / "rewired_seed0.pt",
    ROOT / "runs" / "pilot_b4_spread" / "rewired_seed1.pt",
    ROOT / "runs" / "pilot_b4_spread" / "rewired_seed2.pt",
]


@torch.no_grad()
def collect(model, loader, cfg, device, use_genre: bool = True) -> list:
    """One forward pass, returning per-clip (prob, ref, bpm).

    The whole point of caching is that the threshold sweep costs nothing on the
    GPU: the model's output does not depend on the threshold, so a 19-point
    sweep is 19 numpy passes over an array that was computed once.
    """
    model.eval()
    clips = []
    for wav, y, _vel_y, style, tempo in loader:
        wav = wav.to(device)
        sid = style.to(device) if (use_genre and model.genre is not None) else None
        logits, _, _ = model(wav, style_id=sid, with_velocity=True)
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        ref = y.numpy()
        steps = min(prob.shape[1], ref.shape[1])
        for i in range(prob.shape[0]):
            clips.append((prob[i, :steps], ref[i, :steps], float(tempo[i])))
    return clips


def sweep_metrics(clips, step_ms: float, tol: float, thresholds) -> dict:
    """Every metric at every threshold, from cached probabilities."""
    # onset F comes from onset_f_sweep rather than onset_f_measure so the curve
    # reproduces evaluate()'s own numbers exactly -- same construction, same
    # per-clip micro average -- which makes the recorded onset_f a check on
    # this script rather than a second opinion about it.
    f_by_thr = {t: [] for t in thresholds}
    ba_by_thr = {t: [] for t in thresholds}
    gs_by_thr = {t: [] for t in thresholds}
    dev_by_thr = {t: [] for t in thresholds}

    for prob, ref, bpm in clips:
        for t, f in onset_f_sweep(prob, ref, step_ms, thresholds, tol).items():
            f_by_thr[t].append(f)
        for t in thresholds:
            m = onset_f_measure(prob, ref, step_ms, tolerance_s=tol, threshold=t)
            if not np.isnan(m["mean_dev_ms"]):
                dev_by_thr[t].append(m["mean_dev_ms"])
            if bpm > 0:
                ba = beat_alignment_error(prob, step_ms, bpm, threshold=t)
                gs = groove_similarity(prob, ref, step_ms, bpm, threshold=t)
                if not np.isnan(ba):
                    ba_by_thr[t].append(ba)
                if not np.isnan(gs):
                    gs_by_thr[t].append(gs)

    def mean(d):
        return {f"{t:g}": (float(np.mean(v)) if v else float("nan"))
                for t, v in d.items()}

    return {"onset_f": mean(f_by_thr), "beat_align_ms": mean(ba_by_thr),
            "groove_sim": mean(gs_by_thr), "mean_dev_ms": mean(dev_by_thr)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece.yaml")
    ap.add_argument("--checkpoints", nargs="+", type=Path, default=DEFAULT_CKPTS)
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "threshold_response.json")
    ap.add_argument("--grid", nargs="+", type=float, default=None,
                    help="thresholds to sweep (default 0.05..0.95 by 0.05)")
    a = ap.parse_args(argv)

    thresholds = a.grid or [round(0.05 * i, 2) for i in range(1, 20)]

    cfg = load_config(a.config)
    device = device_of(cfg)
    sg = get_subgraph(cfg)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    _train_loader, val_loader, n_styles = train_mod.build_loaders(cfg, kit)
    n_styles = max(int(n_styles or 1), 1)

    step_ms = cfg["audio"]["step_ms"]
    tol = cfg["eval"].get("tolerance_s", 0.05)
    print(f"device {device} | {len(thresholds)} thresholds "
          f"{thresholds[0]:g}..{thresholds[-1]:g}")

    rows = []
    for path in a.checkpoints:
        # Resolved because --checkpoints is usually given relative to the cwd,
        # and relative_to below needs an absolute path to strip ROOT from.
        path = Path(path).resolve()
        if not path.exists():
            print(f"  skip {path} (missing)")
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        arm, seed = ck["arm"], int(ck.get("seed", 0))
        # Rebuilt with the checkpoint's own arm and seed: the rewired topology
        # is a deterministic function of that seed, so this reconstructs the
        # same graph the weights were trained on. The encoder's standardisation
        # affine is a buffer, so load_state_dict restores the calibration too
        # and nothing has to be re-measured.
        model, _kit = build_arm(arm, cfg, sg, n_styles, seed=seed)
        missing, unexpected = model.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            print(f"  WARN {path.name}: {len(missing)} missing, "
                  f"{len(unexpected)} unexpected keys")
        model = model.to(device)

        label = f"{arm}" + (f"_seed{seed}" if "seed" in path.stem else "")
        clips = collect(model, val_loader, cfg, device)
        curves = sweep_metrics(clips, step_ms, tol, thresholds)
        best = max(curves["onset_f"], key=lambda k: curves["onset_f"][k])
        try:
            rel = str(path.relative_to(ROOT))
        except ValueError:
            rel = str(path)
        rows.append({"label": label, "arm": arm, "seed": seed,
                     "checkpoint": rel,
                     "n_clips": len(clips), "best_threshold": float(best),
                     "curves": curves})
        print(f"  {label:16s} n={len(clips):3d}  argmax F {curves['onset_f'][best]:.4f} "
              f"@ {best}  |  F@0.3 {curves['onset_f'].get('0.3', float('nan')):.4f}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"config": str(a.config), "thresholds": thresholds,
                                 "arms": rows}, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    # Guarded because train.build_loaders spawns dataloader workers, and on
    # Windows spawn re-imports this module in every child.
    sys.exit(main())
