"""Phase 2: where Slakh lives once unpacked, how to unpack it, and which songs belong to which split."""
import glob
import io
import tarfile
from pathlib import Path

SETS = {"baby": "babyslakh_16k", "full": "slakh2100_redux_16k"}

# BabySlakh ships as one flat folder of 20 songs. It gets a fixed split so the pipeline
# (sampler, leak check, validation clips) can be exercised end to end before the full set.
BABY_SPLIT = {"validation": {"Track00015", "Track00016", "Track00017"},
              "test": {"Track00018", "Track00019", "Track00020"}}


def unpacked_root(work_dir, which):
    return Path(work_dir) / "slakh" / SETS[which]


def skip_member(name):
    """macOS junk, and the drums-included mix, which is never used."""
    return "/._" in name or name.endswith(("mix.flac", "mix.wav"))


def archive_parts(slakh_dir, which):
    if which == "baby":
        return [str(Path(slakh_dir) / "babyslakh_16k.tar.gz")]
    parts = sorted(glob.glob(str(Path(slakh_dir) / "slakh2100_redux_16k.tar.gz.part*")))
    return [p for p in parts if p.rsplit("part", 1)[1].isdigit()]


def archive_bytes(slakh_dir, which):
    """Total size of the archive file(s) to read, for a progress bar."""
    return sum(Path(p).stat().st_size for p in archive_parts(slakh_dir, which))


def extract(parts, out, progress=None):
    """Stream the parts back to back through tarfile. progress(bytes read so far) after each member."""
    out.mkdir(parents=True, exist_ok=True)
    chain = Chain(parts)
    with tarfile.open(fileobj=io.BufferedReader(chain, 1 << 20), mode="r|gz") as t:
        for m in t:
            if not skip_member(m.name):
                t.extract(m, out, filter="data")
            if progress:
                progress(chain.bytes_read)


def unpack_baby(slakh_dir, work_dir, progress=None):
    extract(archive_parts(slakh_dir, "baby"), Path(work_dir) / "slakh", progress)
    return unpacked_root(work_dir, "baby")


class Chain(io.RawIOBase):
    """The archive parts read back to back as one stream, so they never need joining on disk."""
    def __init__(self, paths):
        self.files = iter(open(p, "rb") for p in paths)
        self.f = next(self.files)
        self.bytes_read = 0

    def readable(self):
        return True

    def readinto(self, b):
        while True:
            n = self.f.readinto(b)
            if n:
                self.bytes_read += n
                return n
            self.f.close()
            self.f = next(self.files, None)
            if self.f is None:
                return 0


def unpack_full(slakh_dir, work_dir, progress=None):
    log = (Path(slakh_dir) / "parallel_download.log").read_text(encoding="utf-8")
    if "DONE OK" not in log:
        raise RuntimeError("parallel_download.log has no DONE OK; the download is not verified")
    extract(archive_parts(slakh_dir, "full"), Path(work_dir) / "slakh", progress)
    return unpacked_root(work_dir, "full")


def list_songs(root, which):
    """{split: [song folder, ...]} as found on disk. 'omitted' is listed so it can be counted, never used."""
    root = Path(root)
    if which == "baby":
        songs = sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("Track"))
        out = {"train": [], "validation": [], "test": []}
        for p in songs:
            split = next((s for s, ids in BABY_SPLIT.items() if p.name in ids), "train")
            out[split].append(p)
        return out
    return {d.name: sorted(p for p in d.iterdir() if p.is_dir())
            for d in sorted(root.iterdir()) if d.is_dir()}


def stem_audio(song_dir, stem):
    """The stem's audio file (WAV in BabySlakh, FLAC in the full set), or None if it was never rendered."""
    for ext in (".wav", ".flac"):
        p = Path(song_dir) / "stems" / f"{stem}{ext}"
        if p.exists():
            return p
    return None
