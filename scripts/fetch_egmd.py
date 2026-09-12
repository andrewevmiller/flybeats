"""Fetch a drum corpus for Phase 3.

PLAN.md names E-GMD. Its audio archive is 96 GB, which most working
environments cannot hold, so the default here is GMD (``groove-v1.0.0``,
5.1 GB) -- the same Roland TD-11 recordings, the same style vocabulary, audio
plus sample-aligned MIDI. ``--egmd`` fetches the full E-GMD audio set when you
do have the disk; ``--egmd-midi`` fetches only its MIDI (107 MB), which is
enough to work with the style labels.

Large transfers through a proxy get cut off, so every fetch resumes.
"""
from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "egmd"

URLS = {
    "groove": "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0.zip",
    "groove-midi": "https://storage.googleapis.com/magentadata/datasets/groove/groove-v1.0.0-midionly.zip",
    "egmd": "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip",
    "egmd-midi": "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0-midi.zip",
}


def remote_size(url: str) -> int:
    """Ask the server how big the file is.

    Do not hardcode this. A stale constant that is smaller than the real file
    makes a resuming download stop early and hand back a truncated archive that
    still looks 'complete'.
    """
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def resumable_get(url: str, dest: Path, expect: int, attempts: int = 8) -> Path:
    for i in range(attempts):
        have = dest.stat().st_size if dest.exists() else 0
        if have >= expect:
            print(f"  complete: {have / 1e9:.2f} GB")
            return dest
        print(f"  attempt {i + 1}: resuming from {have / 1e9:.2f} / {expect / 1e9:.2f} GB")
        req = urllib.request.Request(url)
        if have:
            req.add_header("Range", f"bytes={have}-")
        try:
            with urllib.request.urlopen(req) as r, open(dest, "ab" if have else "wb") as fh:
                while chunk := r.read(1 << 22):
                    fh.write(chunk)
        except Exception as e:            # truncated transfer: loop and resume
            print(f"    interrupted ({type(e).__name__}: {e})")
    if not dest.exists() or dest.stat().st_size < expect:
        raise SystemExit(f"could not fetch {url} completely after {attempts} attempts")
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=sorted(URLS), default="groove")
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--keep-zip", action="store_true")
    a = ap.parse_args(argv)

    a.dest.mkdir(parents=True, exist_ok=True)
    url = URLS[a.corpus]
    size = remote_size(url)
    zpath = a.dest / Path(url).name

    free = shutil.disk_usage(a.dest).free
    if free < size * 1.9:
        print(f"WARNING: {free / 1e9:.1f} GB free; {a.corpus} needs ~{size * 1.9 / 1e9:.1f} GB "
              f"to download and extract.")

    print(f"{a.corpus} -> {a.dest}")
    resumable_get(url, zpath, size)

    print("  extracting...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(a.dest)
    if not a.keep_zip:
        zpath.unlink()

    for cand in a.dest.rglob("info.csv"):
        print(f"  corpus ready: {cand.parent}")
        break
    return 0


if __name__ == "__main__":
    sys.exit(main())
