"""Phase 2, steps 3 and 5: song manifest, drum events, beat grids, the fixed validation clips,
and reports/data_report_<set>.json.

    .venv\\Scripts\\python.exe scripts\\build_manifest.py --set baby
Songs listed in config/excluded_songs.txt (after a leak check) are left out everywhere.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import ROOT, load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import build_manifest, read_excluded, set_dir  # noqa: E402
from flybeats.data.report import build_report  # noqa: E402
from flybeats.data.sampler import ClipSampler, make_validation_clips  # noqa: E402
from flybeats.data.slakh import list_songs, unpacked_root  # noqa: E402
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], required=True)
args = ap.parse_args()
cfg, paths = load_locked(), load_paths()
root = unpacked_root(paths["work_dir"], args.set)
excluded = read_excluded(ROOT / "config" / "excluded_songs.txt")

skip = set(cfg["data"]["slakh"]["exclude"])
total = sum(len([p for p in songs if p.name not in excluded])
            for split, songs in list_songs(root, args.set).items() if split not in skip)
bar = tqdm(total=total, unit="song", desc="songs")   # live count, rate and time left


def progress(song):
    bar.update(1)
    bar.set_postfix_str(song)


manifest, events, beats, dropped = build_manifest(root, args.set, cfg, excluded, progress)
bar.close()
out = set_dir(paths["work_dir"], args.set)
manifest.to_parquet(out / "manifest.parquet", index=False)
events.to_parquet(out / "events.parquet", index=False)
beats.to_parquet(out / "beats.parquet", index=False)
(out / "dropped_notes.json").write_text(json.dumps({str(k): v for k, v in sorted(dropped.items())}), encoding="utf-8")
print(f"{len(manifest)} songs, {len(events)} kit hits, {sum(dropped.values())} dropped; excluded {len(excluded)}")

val_path = out / "validation_clips.json"
if val_path.exists():
    print(f"kept existing {val_path}")
else:
    val = ClipSampler(root, manifest, events, beats, cfg, "validation")
    print(f"wrote {len(make_validation_clips(val, val_path))} validation clips to {val_path}")

report = build_report(manifest, events, beats, dropped, cfg["kit"]["pieces"], full=args.set == "full")
rpath = reports_dir() / f"data_report_{args.set}.json"
rpath.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"wrote {rpath}")
