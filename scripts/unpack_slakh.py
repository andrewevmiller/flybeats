"""Phase 2, step 2: unpack Slakh under work_dir/slakh, skipping macOS junk and the mix files.

    .venv\\Scripts\\python.exe scripts\\unpack_slakh.py --set baby
    .venv\\Scripts\\python.exe scripts\\unpack_slakh.py --set full     # about 45 GB more; needs DONE OK
Keep the archive parts until the Phase 2 gate passes on the full set.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_paths  # noqa: E402
from flybeats.data.slakh import archive_bytes, list_songs, unpack_baby, unpack_full  # noqa: E402
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--set", choices=["baby", "full"], required=True)
args = ap.parse_args()
paths = load_paths()

free_gb = shutil.disk_usage(paths["work_dir"]).free / 1e9
need_gb = 2 if args.set == "baby" else 50
if free_gb < need_gb:
    sys.exit(f"only {free_gb:.0f} GB free on the work_dir drive; need about {need_gb} GB")

bar = tqdm(total=archive_bytes(paths["slakh_dir"], args.set), unit="B", unit_scale=True, desc="archive read")


def progress(done):
    bar.update(done - bar.n)


unpack = unpack_baby if args.set == "baby" else unpack_full
root = unpack(paths["slakh_dir"], paths["work_dir"], progress)
bar.close()

if not root.exists():
    top = sorted(p.name for p in (paths["work_dir"] / "slakh").iterdir())
    sys.exit(f"expected {root} after unpacking; found {top}. Update SETS in src/flybeats/data/slakh.py.")
for split, songs in list_songs(root, args.set).items():
    print(f"{split:<12} {len(songs):>5} songs")
print(f"unpacked to {root}")
