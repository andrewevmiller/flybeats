"""Fetch part of E-GMD: only the chosen recordings, stored at the loader's rate.

E-GMD is GMD's performances re-recorded on the 43 kits of a Roland TD-17. It
is 96.4 GB zipped and 141.3 GB as WAV (measured 24 Sep 2026).
``fetch_egmd.py`` needs the zip and the extracted files on disk together, about
238 GB, which does not fit the machine this runs on. Most of the archive is not
wanted either. E-GMD adds kits, not drumming: its 819 train performances come
from GMD's nine drummers, and each one appears 43 times.

Each file inside the zip can be read on its own by HTTP range request. So this
script reads the archive's index, picks rows, and fetches only those. Each WAV
is resampled to 22,050 Hz with the loader's own interpolation and stored as
16-bit FLAC, about 0.24x the size of the WAV. ``GrooveDataset`` then reads it
without resampling and sees the same samples to within 16-bit rounding (a
0.5-step error, ~1.5e-5). The script writes an ``info.csv`` the loader reads
unchanged.

What gets picked, all from ``--seed``:

* ``--holdout`` kits never appear in train. They are the test of the one thing
  E-GMD is for: finding hits on a kit the model has never heard.
* Each train performance gets ``--kits`` kits: the first k of its own seeded
  shuffle of the other kits. The shuffle is fixed per performance, so a larger
  k is a superset of a smaller one. Re-running with a larger k fetches only
  what is new, which is how the pool grows piece by piece.
* Each validation and test performance gets one kit that train can use, under
  its own split name, and one held-out kit under ``<split>_unseen_kit``. Point
  ``data.val_split`` at whichever one you want scored.
* GMD styles that E-GMD lacks (``highlife``) get a placeholder row with no
  audio. The loader builds its style vocabulary from every row, so the style
  ids and the embedding shape match GMD's, and checkpoints stay interchangeable.

Train rows are listed in a seeded shuffle. A ``max_files`` prefix is then a
random sample, not the first drummer's performances.

Every train row is one file. At ``--kits 4`` an epoch reads 4x the rows GMD
does. Holding it to one kit per performance per epoch needs a sampler in
``train.build_loaders``, which does not exist yet.

The fetch resumes. A file already on disk is skipped, and ``info.csv`` lists
only rows whose audio and MIDI are both complete. It is rewritten at the end of
every run, including an interrupted one.

    python scripts/fetch_egmd_subset.py --plan          # sizes only, fetches ~15 MB of index
    python scripts/fetch_egmd_subset.py                 # 4 kits per train performance
    python scripts/fetch_egmd_subset.py --kits 8        # later: grows the same pool
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import shutil
import struct
import sys
import threading
import time
import urllib.request
import zipfile
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data" / "egmd" / "e-gmd-subset"
URL = "https://storage.googleapis.com/magentadata/datasets/e-gmd/v1.0.0/e-gmd-v1.0.0.zip"
PREFIX = "e-gmd-v1.0.0/"
CSV_ENTRY = PREFIX + "e-gmd-v1.0.0.csv"
SAMPLE_RATE = 22_050

#: GMD's style vocabulary, in the loader's order. E-GMD has no ``highlife``.
#: Without a placeholder its vocabulary is 17 long, every id after ``gospel``
#: shifts by one, and a GMD checkpoint's genre embedding no longer fits.
GMD_STYLES = [
    "afrobeat", "afrocuban", "blues", "country", "dance", "funk", "gospel",
    "highlife", "hiphop", "jazz", "latin", "middleeastern", "neworleans", "pop",
    "punk", "reggae", "rock", "soul",
]

#: Held out of train by default: one kit each from electronic, jazz, metal,
#: lo-fi and quiet acoustic. Their neighbours (909 Simple, Jazz, Heavy Metal)
#: stay in train, so this tests an unseen kit, not an unseen family of kits.
DEFAULT_HOLDOUT = [
    "808 Simple", "Bigga Bop (Jazz)", "Speed Metal", "Cassette (Lo-Fi Compress)",
    "Unplugged",
]

#: Stored FLAC size over the source WAV, measured on 12 files from 9 kits.
FLAC_RATIO = 0.24

INFO_COLUMNS = [
    "drummer", "session", "id", "style", "bpm", "beat_type", "time_signature",
    "midi_filename", "audio_filename", "duration", "split", "kit_name",
]
#: Settings that decide which rows are picked and what their audio is. Re-running
#: into a folder fetched with different ones would mix two corpora silently.
PINNED = ("seed", "holdout", "sample_rate", "cap_seconds")


# --------------------------------------------------------------------- remote


def http_range(url: str, start: int, length: int, attempts: int = 6) -> bytes:
    """Bytes ``[start, start + length)`` of ``url``, retried with backoff."""
    last: Exception | None = None
    for i in range(attempts):
        try:
            req = urllib.request.Request(url)
            req.add_header("Range", f"bytes={start}-{start + length - 1}")
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) != length:
                raise IOError(f"short read: {len(data)} of {length} bytes")
            return data
        except Exception as e:             # network faults: back off and retry
            last = e
            time.sleep(min(30, 2 ** i))
    raise IOError(f"range {start}+{length} failed after {attempts} attempts: {last}")


class RangeFile(io.RawIOBase):
    """A seekable read-only view of a remote file, for ``zipfile`` to parse."""

    def __init__(self, url: str):
        self.url, self.pos = url, 0
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"),
                                    timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.last_modified = r.headers.get("Last-Modified", "")

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def readinto(self, b):
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        data = http_range(self.url, self.pos, n)
        b[:n] = data
        self.pos += n
        return n


def entry_bytes(read_range, info: zipfile.ZipInfo) -> bytes:
    """One member of a zip, from its local header on, with its CRC checked.

    ``read_range(start, length)`` does the reading, so the same code runs
    against the remote archive and against a local zip in the tests. One
    request covers the header and the data. The local header's extra field can
    differ in length from the central directory's, so there is slack, plus a
    second read if the slack falls short.
    """
    name_len = len(info.filename.encode("utf-8"))
    want = 30 + name_len + 512 + info.compress_size
    blob = read_range(info.header_offset, want)
    sig, *_, n_name, n_extra = struct.unpack("<IHHHHHIIIHH", blob[:30])
    if sig != 0x04034B50:
        raise ValueError(f"{info.filename}: no local header at {info.header_offset}")
    start = 30 + n_name + n_extra
    data = blob[start:start + info.compress_size]
    if len(data) < info.compress_size:
        data += read_range(info.header_offset + start + len(data),
                           info.compress_size - len(data))
    if info.compress_type == zipfile.ZIP_STORED:
        out = data
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        out = zlib.decompress(data, -15)
    else:
        raise ValueError(f"{info.filename}: compression type {info.compress_type}")
    if zlib.crc32(out) != info.CRC:
        raise ValueError(f"{info.filename}: CRC mismatch, the transfer was corrupted")
    return out


def read_index(url: str = URL):
    """The archive's member list and its metadata CSV: about 15 MB fetched."""
    raw = RangeFile(url)
    z = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
    members = {i.filename[len(PREFIX):]: i for i in z.infolist()
               if i.filename.startswith(PREFIX)}
    text = z.read(CSV_ENTRY).decode("utf-8")
    rows = list(csv.DictReader(io.StringIO(text)))
    return raw, members, rows


# ------------------------------------------------------------------ selection


def select_rows(rows: list[dict], kits: int | None, holdout: list[str], seed: int,
                splits: tuple[str, ...]) -> list[dict]:
    """Which recordings to fetch, as info.csv rows. Pure, so it is testable.

    ``kits=None`` means every kit that is not held out.
    """
    all_kits = sorted({r["kit_name"] for r in rows})
    unknown = sorted(set(holdout) - set(all_kits))
    if unknown:
        raise ValueError(f"not E-GMD kits: {unknown}")
    seen = [k for k in all_kits if k not in holdout]

    by_perf: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in rows:
        if r["split"] in splits:
            by_perf[r["id"]][r["kit_name"]] = r

    picked = []
    for pid in sorted(by_perf):
        avail = by_perf[pid]
        split = next(iter(avail.values()))["split"]
        # Seeded per performance, never from one shared stream, so a
        # performance's kits do not depend on which others were selected.
        rng = random.Random(f"{seed}:{pid}")
        order = [k for k in rng.sample(seen, len(seen)) if k in avail]
        if split == "train":
            chosen = order if kits is None else order[:kits]
            picked += [dict(avail[k]) for k in chosen]
        else:
            if order:
                picked.append(dict(avail[order[0]]))
            unseen = [k for k in rng.sample(sorted(holdout), len(holdout)) if k in avail]
            if unseen:
                picked.append({**avail[unseen[0]], "split": f"{split}_unseen_kit"})
    for r in picked:
        # "_wav" names the archive member; DictWriter drops it from info.csv.
        r["_wav"] = r["audio_filename"]
        r["audio_filename"] = flac_name(r["audio_filename"])
    return picked


def flac_name(path: str) -> str:
    return str(Path(path).with_suffix(".flac").as_posix())


# ----------------------------------------------------------------- conversion


def to_loader_rate(audio: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    """``GrooveDataset.__getitem__``'s resampling, verbatim.

    Any other resampler would give the model different audio from the WAV it
    replaces. ``test_fetch_egmd_subset`` checks the loader sees the same thing.
    """
    if sr == target:
        return audio.astype(np.float32)
    n_out = int(len(audio) * target / sr)
    return np.interp(
        np.linspace(0, len(audio) - 1, n_out), np.arange(len(audio)), audio
    ).astype(np.float32)


def convert_wav(wav: bytes, sample_rate: int = SAMPLE_RATE,
                cap_seconds: float | None = None) -> np.ndarray:
    import soundfile as sf

    audio, sr = sf.read(io.BytesIO(wav), dtype="float32", always_2d=True)
    audio = to_loader_rate(audio.mean(axis=1), sr, sample_rate)
    if cap_seconds:
        # Cut from the start, so the MIDI needs no change: the loader drops
        # events outside the window, and the window cannot pass the audio.
        audio = audio[: int(cap_seconds * sample_rate)]
    return audio


def write_flac(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    sf.write(tmp, audio, sample_rate, format="FLAC", subtype="PCM_16")
    os.replace(tmp, path)


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.part")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def complete(dest: Path, r: dict) -> bool:
    return (dest / r["audio_filename"]).exists() and (dest / r["midi_filename"]).exists()


def write_info(dest: Path, rows: list[dict], seed: int) -> Path:
    """info.csv: the complete rows, plus a placeholder for each missing style."""
    present = [r for r in rows if complete(dest, r)]
    train = [r for r in present if r["split"] == "train"]
    random.Random(seed).shuffle(train)
    rest = sorted((r for r in present if r["split"] != "train"),
                  key=lambda r: (r["split"], r["id"], r["kit_name"]))
    # From the rows written, not the rows selected: mid-fetch, info.csv holds a
    # few styles, and the vocabulary must still be GMD's 18.
    have = {r["style"].split("/")[0] for r in present}
    placeholders = [{"style": s, "split": "none"} for s in GMD_STYLES if s not in have]
    path = dest / "info.csv"
    tmp = path.with_suffix(".csv.part")
    with tmp.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INFO_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in train + rest + placeholders:
            w.writerow(r)
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------------------- main


def download_bytes(r: dict, members: dict) -> int:
    return members[r["_wav"]].compress_size + members[r["midi_filename"]].compress_size


def summarise(rows, members, dest, cap_seconds=None) -> dict:
    out = {}
    for split in sorted({r["split"] for r in rows}):
        rs = [r for r in rows if r["split"] == split]
        todo = [r for r in rs if not complete(dest, r)]
        kept = lambda r: (min(1.0, cap_seconds / float(r["duration"]))
                          if cap_seconds and float(r["duration"]) > 0 else 1.0)
        out[split] = {
            "rows": len(rs),
            "performances": len({r["id"] for r in rs}),
            "kits": len({r["kit_name"] for r in rs}),
            "present": len(rs) - len(todo),
            "download_gb": sum(download_bytes(r, members) for r in todo) / 1e9,
            "stored_gb_est": sum(members[r["_wav"]].file_size * kept(r)
                                 for r in rs) * FLAC_RATIO / 1e9,
        }
    return out


def free_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", type=Path, default=DEST)
    ap.add_argument("--kits", default="4",
                    help="kits per train performance, or 'all' for every kit not held out")
    ap.add_argument("--holdout", default=",".join(DEFAULT_HOLDOUT),
                    help="comma-separated kit names kept out of train; '' for none")
    ap.add_argument("--splits", default="train,validation,test")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample-rate", type=int, default=SAMPLE_RATE,
                    help="must match data.sample_rate, or the loader resamples twice")
    ap.add_argument("--cap-seconds", type=float, default=None,
                    help="keep only the first N seconds of each recording")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="fetch at most N new rows per split (a smoke test)")
    ap.add_argument("--plan", action="store_true", help="print sizes, fetch no audio")
    a = ap.parse_args(argv)

    kits = None if a.kits == "all" else int(a.kits)
    holdout = [k.strip() for k in a.holdout.split(",") if k.strip()]
    settings = {"seed": a.seed, "holdout": sorted(holdout), "sample_rate": a.sample_rate,
                "cap_seconds": a.cap_seconds}

    manifest_path = a.dest / "manifest.json"
    old = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    clash = [k for k in PINNED if k in old and old[k] != settings[k]]
    if clash:
        print(f"{a.dest} was fetched with different {', '.join(clash)}:")
        for k in clash:
            print(f"  {k}: {old[k]!r} there, {settings[k]!r} now")
        print("Use a different --dest, or delete that folder first.")
        return 2

    print(f"reading the archive index from {URL}")
    raw, members, all_rows = read_index()
    rows = select_rows(all_rows, kits, holdout, a.seed, tuple(a.splits.split(",")))
    if a.cap_seconds:
        for r in rows:
            r["duration"] = f"{min(float(r['duration']), a.cap_seconds):.6f}"
    plan = summarise(rows, members, a.dest, a.cap_seconds)

    print(f"\n{'split':<24}{'rows':>7}{'perfs':>7}{'kits':>6}{'on disk':>9}"
          f"{'to download':>13}{'stored (est)':>14}")
    for split, s in plan.items():
        print(f"{split:<24}{s['rows']:>7}{s['performances']:>7}{s['kits']:>6}"
              f"{s['present']:>9}{s['download_gb']:>10.2f} GB{s['stored_gb_est']:>11.2f} GB")
    need = sum(s["stored_gb_est"] for s in plan.values())
    free = free_gb(a.dest)
    print(f"\nheld out of train: {', '.join(holdout) or 'none'}")
    print(f"disk: {free:.0f} GB free, the whole selection stores to ~{need:.1f} GB")
    if a.plan:
        return 0
    if need * 1.2 > free:
        print("not enough free space for the selection plus 20% headroom")
        return 1
    a.dest.mkdir(parents=True, exist_ok=True)

    todo = [r for r in rows if not complete(a.dest, r)]
    if a.limit is not None:
        taken: Counter = Counter()
        kept = []
        for r in todo:
            if taken[r["split"]] < a.limit:
                taken[r["split"]] += 1
                kept.append(r)
        todo = kept
    total_dl = sum(download_bytes(r, members) for r in todo)
    print(f"fetching {len(todo)} recordings, {total_dl / 1e9:.2f} GB, "
          f"{a.workers} at a time\n")

    # Clamped: a member near the end of the archive plus entry_bytes' slack
    # would otherwise ask for bytes past EOF and read as a short transfer.
    read = lambda start, length: http_range(URL, start, min(length, raw.size - start))
    done, got, failed = 0, 0, []
    t0 = time.time()

    def fetch(r):
        wav_info = members[r["_wav"]]
        mid_info = members[r["midi_filename"]]
        audio = convert_wav(entry_bytes(read, wav_info), a.sample_rate, a.cap_seconds)
        write_bytes(a.dest / r["midi_filename"], entry_bytes(read, mid_info))
        write_flac(a.dest / r["audio_filename"], audio, a.sample_rate)
        return wav_info.compress_size + mid_info.compress_size

    step = max(1, len(todo) // 50)
    try:
        with ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures = {pool.submit(fetch, r): r for r in todo}
            for f in as_completed(futures):
                r = futures[f]
                try:
                    got += f.result()
                    done += 1
                except Exception as e:
                    failed.append(f"{r['audio_filename']}: {e}")
                if (done + len(failed)) % step == 0 or done + len(failed) == len(todo):
                    el = time.time() - t0
                    rate = got / max(el, 1e-9)
                    eta = (total_dl - got) / rate if rate else 0
                    bar = "#" * int(30 * (done + len(failed)) / max(1, len(todo)))
                    print(f"  [{bar:<30}] {done + len(failed)}/{len(todo)}  "
                          f"{got / 1e9:.2f}/{total_dl / 1e9:.2f} GB  "
                          f"{rate / 1e6:.1f} MB/s  eta {eta / 60:.0f} min", flush=True)
    finally:
        info = write_info(a.dest, rows, a.seed)
        present = sum(complete(a.dest, r) for r in rows)
        manifest = {
            **settings, "url": URL, "archive_bytes": raw.size,
            "archive_last_modified": raw.last_modified,
            "kits_per_train_performance": a.kits, "splits": a.splits,
            "rows_selected": len(rows), "rows_present": present,
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "failures": failed[:50],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"\n{done} fetched, {len(failed)} failed, {present}/{len(rows)} selected rows on disk")
    print(f"wrote {info}")
    for line in failed[:10]:
        print(f"  FAILED {line}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
