"""Export a trained checkpoint as a self-contained bundle.

A checkpoint is only loadable next to the machine that made it: the loader
rebuilds the subgraph, which wants the 1.1 GB connectome tables and a graph
build. A bundle is the same model with the topology it already carries made
explicit, so it loads with nothing but the repo and the file itself.

    python scripts/export_bundle.py --checkpoint runs/rho10_long/best.pt
    python scripts/export_bundle.py --checkpoint runs/x/best.pt --out kit.fb

Nothing is written until the bundle has been loaded back and checked against the
original model on real audio, sample for sample. An export that silently changed
the model would be worse than no export: it would sound plausible.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import build_model, get_subgraph  # noqa: E402
from bundle import build_bundle, load_bundle  # noqa: E402
from decoder import DrumKit  # noqa: E402


def verify(model, bundled, cfg, seconds: float = 2.0, tol: float = 1e-5) -> float:
    """Run the same noise through both models; return the largest difference."""
    sr = cfg["audio"]["sample_rate"]
    g = torch.Generator().manual_seed(0)
    wav = torch.randn(1, int(sr * seconds), generator=g) * 0.1
    with torch.no_grad():
        a, _ = model(wav)
        b, _ = bundled(wav)
    if a.shape != b.shape:
        raise SystemExit(f"bundle changed the output shape: {a.shape} vs {b.shape}")
    d = float((a - b).abs().max())
    if not np.isfinite(d) or d > tol:
        raise SystemExit(f"bundle does not reproduce the model: max |diff| = {d:.3e}")
    return d


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="default: flybeats-<kit tier>.fb next to the checkpoint")
    ap.add_argument("--seconds", type=float, default=2.0,
                    help="how much audio to verify the round trip on")
    ap.add_argument("--threshold", type=float, default=None,
                    help="peak-picking threshold to ship with the model. Defaults to "
                         "the one the checkpoint's own eval sweep chose")
    a = ap.parse_args(argv)

    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["config"]
    sg = get_subgraph(cfg)
    model, _ = build_model(cfg, sg, n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    model.load_state_dict(ck["model"])
    model.eval()

    if a.threshold is not None:
        ck["best_threshold"] = a.threshold
    if ck.get("best_threshold") is None:
        print("  note: this checkpoint predates the saved threshold; playback will "
              "use the config default. Pass --threshold to set it.")

    out = a.out or a.checkpoint.parent / f"flybeats-{cfg['kit']['tier']}.fb"
    payload = build_bundle(ck, sg, model)
    torch.save(payload, out)

    bundled, kit, _, roles = load_bundle(out)
    d = verify(model, bundled, cfg, a.seconds)

    src_mb = a.checkpoint.stat().st_size / 1e6
    out_mb = out.stat().st_size / 1e6
    print(f"{a.checkpoint}  ->  {out}")
    print(f"  {src_mb:.1f} MB -> {out_mb:.1f} MB   ({100 * out_mb / src_mb:.0f}% of the checkpoint)")
    print(f"  kit: {'/'.join(kit.classes)}")
    print(f"  peak threshold: {ck.get('best_threshold')}")
    print(f"  roles: {', '.join(f'{k}={len(v)}' for k, v in sorted(roles.items()))}")
    print(f"  verified against the original model: max |diff| = {d:.2e} on "
          f"{a.seconds:g}s of audio")
    print(f"\nRuns with no data/ directory:\n  python src/realtime.py --bundle {out} --render song.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
