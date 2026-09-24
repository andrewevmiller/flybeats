"""Refit a saved checkpoint's velocity head in closed form.

train.py now does this itself after training and writes ``best_refit.pt``
beside ``best.pt`` (``train.refit_velocity_head``, on by default). This script
is for checkpoints trained before that, or to refit with other settings. The
method and its rationale are in ``src/refit.py``.

    python scripts/refit_velocity_head.py --checkpoint runs/velocity_probe_cpu/best.pt \
        --out runs/refit/velocity_probe_cpu/best.pt

Validation is never touched: score the result with probe_velocity.py.
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as T                                    # noqa: E402
from build import build_model, get_subgraph          # noqa: E402
from decoder import DrumKit                          # noqa: E402
from refit import fit_head, hit_peaks, refit_velocity_head  # noqa: E402,F401  (re-exported)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--min-hits", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.out.resolve() == args.checkpoint.resolve():
        raise SystemExit("--out must differ from --checkpoint; the original is the control")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    kit = DrumKit(ck["kit"])
    model, _ = build_model(cfg, get_subgraph(cfg), n_styles=max(ck["n_styles"], 1))
    model.load_state_dict(ck["model"])
    tr, _, _ = T.build_loaders(cfg, kit)

    t0 = time.time()
    report = refit_velocity_head(model, tr, lam=args.lam, min_hits=args.min_hits)
    print(f"train clips {report['clips']}, {time.time() - t0:.0f}s")
    for name, r in report["classes"].items():
        if r["refit"]:
            print(f"{name:<11} {r['hits']:>5} train hits  train r {r['train_r']:+.3f}")
        else:
            print(f"{name:<11} {r['hits']:>5} train hits -- kept as trained")

    ck["model"] = model.state_dict()
    ck["refit"] = {"source": str(args.checkpoint), **report,
                   "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, args.out)
    print(f"-> {args.out}")


if __name__ == "__main__":
    sys.exit(main())
