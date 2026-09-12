"""Phase 0 data acquisition.

Downloads the MaleCNS v1.0 flat-connectome tables from the public GCS bucket.
No neuPrint token is required for this path -- the flat-connectome feathers are
the same v1.0 release the neuPrint server exposes, published under CC-BY.

    python scripts/fetch_data.py            # the three tables we actually need
    python scripts/fetch_data.py --all      # including the huge synapse tables
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

BUCKET = "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"

CORE = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "weights": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}
EXTRA = {
    "weights_traced": "connectome-weights-male-cns-v1.0-minconf-0.5-traced-only.feather",
    "body_stats": "body-stats-male-cns-v1.0-minconf-0.5.feather",
}

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"


def fetch(name: str, dest_dir: Path) -> Path:
    dest = dest_dir / name
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {name} ({dest.stat().st_size / 1e6:.1f} MB already present)")
        return dest
    url = f"{BUCKET}/{name}"
    print(f"  [get ] {name}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as fh:
        while chunk := r.read(1 << 22):
            fh.write(chunk)
    tmp.rename(dest)
    print(f"         -> {dest.stat().st_size / 1e6:.1f} MB")
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true", help="also fetch the large optional tables")
    ap.add_argument("--dest", type=Path, default=RAW)
    args = ap.parse_args(argv)

    args.dest.mkdir(parents=True, exist_ok=True)
    wanted = dict(CORE)
    if args.all:
        wanted.update(EXTRA)

    print(f"MaleCNS v1.0 flat-connectome -> {args.dest}")
    for key, name in wanted.items():
        fetch(name, args.dest)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
