"""Export a trained checkpoint to a brain file on the laptop, where the slice, band stats and runs live.

    .venv\\Scripts\\python.exe -I -m flybrain.cli --set baby --checkpoint runs\\baby_real\\best.pt ^
        --out brains\\baby_real.flybrain --name "real fly, baby-trained" [--control 1] [--clips 20] ^
        [--scoreboard reports\\scoreboard.json] [--input-task unknown] [--history]

Run from the repo root with plugin/export on PYTHONPATH (or `cd plugin/export`). The checkpoint's network is
rebuilt from the slice (real or the control's edges) with the locked settings and the set's band statistics,
loaded strictly, run through the parity harness against PyTorch on `--clips` training clips (50, the plan's number), cliff-calibrated
with the repo's silence test on those clips, and written with provenance hashes. Capability is always
observation_only here: drum_capable is set only through Gate B, by hand, with the chair's memo.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from flybeats.config import load_locked, load_paths  # noqa: E402
from flybeats.connectome.store import control_path, load_slice, slice_paths  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model.ear import load_band_stats  # noqa: E402
from flybeats.model.network import FlyNet  # noqa: E402

from .brainfile import read_brain  # noqa: E402
from .exporter import export, git_commit, load_checkpoint, measure_cliff, runaway_limits, sha256_file  # noqa: E402
from .parity import compare, torch_outputs  # noqa: E402
from .reference import run_clip  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", choices=["baby", "full"], required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--control", type=int, default=None)
    ap.add_argument("--clips", type=int, default=50, help="training clips for parity and the cliff calibration")
    ap.add_argument("--scoreboard", default=None, help="JSON with the Scoreboard payload (chair-approved)")
    ap.add_argument("--input-task", choices=["unknown", "accompaniment", "transcription"], default="unknown")
    ap.add_argument("--history", action="store_true", help="the checkpoint was trained with the history readout")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-cliff", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_locked()
    if args.history:
        cfg["model"]["readout_history"] = {"present": True, "blocks_ms": [50, 50, 50, 50], "profile": "per_drum",
                                           "init": "present_only", "velocity": "same"}
    paths = load_paths()
    neurons, edges = load_slice(paths["work_dir"], args.control)
    stats_path = set_dir(paths["work_dir"], args.set) / "band_stats.json"
    model = FlyNet(neurons, edges, cfg, load_band_stats(stats_path))
    extra = load_checkpoint(model, args.checkpoint)
    model.eval()
    manifest, events, beats = load_manifest(paths["work_dir"], args.set)
    sampler = ClipSampler(unpacked_root(paths["work_dir"], args.set), manifest, events, beats, cfg, "train", seed=4242)
    clips = []
    while len(clips) < args.clips:
        clips += sampler.batch(min(8, len(sampler.songs), args.clips - len(clips)))

    cliff = runaway = None
    if not args.no_cliff:
        print("cliff calibration (repo silence test over a gain grid, then bisection)...", flush=True)
        cliff = measure_cliff(model, clips, cfg, device=args.device)
        runaway = runaway_limits(cliff, cfg)
        print(f"  last pass {cliff['last_pass_gain']}x, first fail {cliff['first_fail_gain']}x")
    n_path, e_path = slice_paths(paths["work_dir"])
    if args.control is not None:
        e_path = control_path(paths["work_dir"], args.control)
    provenance = {"checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256_file(args.checkpoint),
                  "checkpoint_extra": {k: v for k, v in extra.items() if isinstance(v, (int, float, str))},
                  "slice_neurons_sha256": sha256_file(n_path), "slice_edges_sha256": sha256_file(e_path),
                  "control_seed": args.control, "band_stats_sha256": sha256_file(stats_path), "set": args.set,
                  "locked_yaml_sha256": sha256_file(ROOT / "config" / "locked.yaml"),
                  "code_commit": git_commit(ROOT), "locked_version": cfg["version"]}
    scoreboard = json.loads(Path(args.scoreboard).read_text(encoding="utf-8")) if args.scoreboard else None
    digest, meta = export(model.cpu(), neurons, edges, cfg, args.out, name=args.name, capability="observation_only",
                          input_task=args.input_task, provenance=provenance, scoreboard=scoreboard, cliff=cliff,
                          runaway=runaway)
    brain = read_brain(args.out)
    print(f"wrote {args.out}: {brain.n} neurons, {meta['n_edges']} connections, sha256 {digest}")

    print(f"parity against PyTorch on {len(clips)} clips...", flush=True)
    reports, worst = [], None
    for i, c in enumerate(clips):
        tor = torch_outputs(model, c["audio"])
        ref = run_clip(brain, c["audio"], dtype=np.float32)
        rep = compare(ref, tor, cfg["training"]["hit_threshold"])
        reports.append(rep)
        if worst is None or rep["rate_err"] > worst["rate_err"]:
            worst = rep
        print(f"  clip {i}: {'pass' if rep['passes'] else 'FAIL'} rates {rep['rate_err']:.2e} prob {rep['prob_err']:.2e} "
              f"hits {rep['hits_torch']} / {rep['hits_engine']} differences {rep['hit_differences']} "
              f"(ambiguous in torch {rep['ambiguous_in_torch']})")
    passed = all(r["passes"] for r in reports)
    side = Path(args.out).with_suffix(".parity.json")
    side.write_text(json.dumps({"brain_sha256": digest, "clips": len(clips), "passes": passed, "worst": worst,
                                "reports": reports}, indent=1) + "\n", encoding="utf-8")
    print(f"parity {'PASS' if passed else 'FAIL'}; details in {side}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
