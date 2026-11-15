"""v0.15 holdout seal (plan section 9): choose confirmation sets A and B from the training split, once.

    .venv\\Scripts\\python.exe scripts\\seal_confirmation.py --dry-run     # prints counts, writes nothing
    .venv\\Scripts\\python.exe scripts\\seal_confirmation.py               # seals (refuses if already sealed)

Writes config/confirmation_A.txt and _B.txt (song ids), config/confirmation_clips_A.json and _B.json (fixed
windows), appends a data.confirmation block to config/locked.yaml, and relabels the sealed songs in the full
manifest (backup of the old one goes to the evidence folder). CPU only. Does not train, score or push.
"""
import argparse
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flybeats.config import load_locked, load_paths  # noqa: E402
from flybeats.data.confirm import (MIN_SONGS, SALT, SEED, CLIPS_PER_SONG, apply_confirmation, draw_clips,  # noqa: E402
                                   read_list, select, sha256_file, sha256_text)
from flybeats.data.manifest import set_dir  # noqa: E402

CHECK_CLIPS = "train_check_clips.json"   # the 12 Nov diagnostic draw, under data/work/data/full/
EVIDENCE = ROOT.parent / ".015" / "evidence" / "pre_seal"

ap = argparse.ArgumentParser()
ap.add_argument("--dry-run", action="store_true")
args = ap.parse_args()

cfg, paths = load_locked(), load_paths()
cfg_conf = cfg["data"].get("confirmation")
lists = {n: ROOT / "config" / f"confirmation_{n[-1].upper()}.txt" for n in ("confirm_a", "confirm_b")}
if cfg_conf is not None or any(p.exists() for p in lists.values()):
    sys.exit("already sealed: config/confirmation_*.txt or data.confirmation exists; sealing is done once")

work = set_dir(paths["work_dir"], "full")
manifest = pd.read_parquet(work / "manifest.parquet")
if set(manifest.split) & {"confirm_a", "confirm_b"}:
    sys.exit("the manifest already has confirm_* rows; refusing to seal twice")
baby = pd.read_parquet(set_dir(paths["work_dir"], "baby") / "manifest.parquet")
baby_ids = set(baby.song)
check_ids = {c["song"] for c in json.loads((work / CHECK_CLIPS).read_text(encoding="utf-8"))}
excluded = baby_ids | check_ids

train = manifest[manifest.split == "train"]
clusters = train.groupby("drum_hash").size()
a, b = select(manifest, excluded)
dur = manifest.set_index("song").duration
clip_s = cfg["training_data"]["clip_seconds"]
print(f"train split {len(train)} songs in {len(clusters)} drum clusters")
print(f"excluded: {len(baby_ids)} BabySlakh songs ({len(baby_ids & set(train.song))} in train), "
      f"{len(check_ids)} songs in {CHECK_CLIPS} ({len(check_ids & set(train.song))} in train)")
print(f"A: {len(a)} songs   B: {len(b)} songs")
for name, songs in (("A", a), ("B", b)):
    if len(songs) < MIN_SONGS:
        sys.exit(f"set {name} has {len(songs)} songs, below {MIN_SONGS}: not sealed")
if args.dry_run:
    print("dry run: nothing written")
    sys.exit(0)

# 1. lists
header = (f"# Confirmation set {{name}} (v0.15, sealed {date.today().isoformat()}). Salt {SALT}; clusters by drum_hash,\n"
          f"# ranked by SHA-256(salt + drum_hash). Song ids, one per line. Do not edit: see config/CHANGELOG.md.\n")
hashes = {}
for name, songs, letter in (("confirm_a", a, "A"), ("confirm_b", b, "B")):
    path = ROOT / "config" / f"confirmation_{letter}.txt"
    path.write_text(header.format(name=name) + "\n".join(songs) + "\n", encoding="utf-8")
    # 2. clip draws, one fixed set per list
    clips = draw_clips(songs, dur, clip_s, SEED, CLIPS_PER_SONG)
    cpath = ROOT / "config" / f"confirmation_clips_{letter}.json"
    cpath.write_text(json.dumps(clips, indent=1) + "\n", encoding="utf-8")
    hashes[name] = {"songs": len(songs), "list_sha256": sha256_file(path), "clips_sha256": sha256_file(cpath),
                    "clips": len(clips)}
    print(f"{name}: list {hashes[name]['list_sha256'][:12]}  clips {hashes[name]['clips_sha256'][:12]}")

# 3. locked.yaml: the data.confirmation block (after the slakh line)
text = (ROOT / "config" / "locked.yaml").read_text(encoding="utf-8")
anchor = "  slakh: {release: slakh2100_redux_16k, split: official_redux, exclude: [omitted]}\n"
assert text.count(anchor) == 1, "locked.yaml anchor moved: stop and edit by hand"
block = (
    anchor
    + "  confirmation:                 # v0.15 holdout (prereg-v5.5, 15 Nov 2026; config/CHANGELOG.md). Sealed once.\n"
    + f"    salt: {SALT}\n"
    + f"    seed: {SEED}\n"
    + "    unit: drum_hash\n"
    + f"    min_songs: {MIN_SONGS}\n"
    + f"    clips_per_song: {CLIPS_PER_SONG}\n"
    + f"    clip_seconds: {clip_s}\n"
    + f"    confirm_a: {{songs: {hashes['confirm_a']['songs']}, list_sha256: {hashes['confirm_a']['list_sha256']}, "
      f"clips_sha256: {hashes['confirm_a']['clips_sha256']}}}\n"
    + f"    confirm_b: {{songs: {hashes['confirm_b']['songs']}, list_sha256: {hashes['confirm_b']['list_sha256']}, "
      f"clips_sha256: {hashes['confirm_b']['clips_sha256']}}}\n"
)
new = text.replace(anchor, block).replace("version: prereg-v5.4", "version: prereg-v5.5", 1)
assert new.count("version: prereg-v5.5") == 1
(ROOT / "config" / "locked.yaml").write_text(new, encoding="utf-8")

# 4. relabel the manifest; keep the pre-seal copy as evidence
EVIDENCE.mkdir(parents=True, exist_ok=True)
shutil.copy2(work / "manifest.parquet", EVIDENCE / "manifest_full_before_seal.parquet")
sealed = apply_confirmation(manifest, a, b)
sealed.to_parquet(work / "manifest.parquet", index=False)
(EVIDENCE / "seal_summary.json").write_text(json.dumps({"A": a, "B": b, "hashes": hashes,
                                                        "salt": SALT, "seed": SEED}, indent=1), encoding="utf-8")
print("sealed. Next: rebuild band stats on the remaining train split, then run the data gate.")
