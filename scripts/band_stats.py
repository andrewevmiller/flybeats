"""Phase 3, step 3: per-band mean and spread for the ear, from a random sample of training clips
(never the first songs processed). Written once to work_dir/data/<set>/band_stats.json.

    .venv\\Scripts\\python.exe scripts\\band_stats.py --set baby
CPU only.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model.ear import compute_band_stats  # noqa: E402
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], required=True)
ap.add_argument("--clips", type=int, default=64)
args = ap.parse_args()
cfg, paths = load_locked(), load_paths()
out = set_dir(paths["work_dir"], args.set) / "band_stats.json"
if out.exists():
    sys.exit(f"{out} exists; band stats are computed once")

manifest, events, beats = load_manifest(paths["work_dir"], args.set)
sampler = ClipSampler(unpacked_root(paths["work_dir"], args.set), manifest, events, beats, cfg, "train", seed=2024)
clips = []
with tqdm(total=args.clips, unit="clip", desc="reading clips") as bar:
    while len(clips) < args.clips:
        got = sampler.batch(min(8, len(sampler.songs)))
        clips += got
        bar.update(min(len(got), args.clips - bar.n))
stats = compute_band_stats(clips[:args.clips])
out.write_text(json.dumps(stats, indent=1), encoding="utf-8")
print(f"wrote {out} from {stats['clips']} clips")
