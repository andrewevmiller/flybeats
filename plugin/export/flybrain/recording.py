"""Loader for the engine's recordings (Recorder in plugin/engine) and the render tool's frame dumps.

load_recording(bin_path) -> dict like reference.run_clip's: frame, bands, ear, rates, mn, prob, vel, hits, plus
reset, runaway, neutral, muted, serial, and `sidecar` (the JSON next to the file, when present).
load_dump(path) -> the same keys (no serial, no sidecar) for a flybrain_render --dump file.
"""
import json
from pathlib import Path

import numpy as np


def _dtype(nb, n, nm, nd, with_serial):
    fields = [("frame", "<i8", 1), ("reset", "u1", 1), ("runaway", "u1", 1), ("neutral", "u1", 1), ("muted", "u1", 1)]
    if with_serial:
        fields.append(("serial", "<u8", 1))
    fields += [("bands", "<f8", nb), ("ear", "<f8", nb), ("rates", "<f8", n), ("mn", "<f8", nm), ("prob", "<f8", nd),
               ("vel", "<f8", nd), ("hits", "u1", nd)]
    return fields, np.dtype([(name, t, (count,)) if count > 1 else (name, t) for name, t, count in fields])


def open_records(path, with_serial=None):
    """A memory-mapped structured array of the frames (no copy): for recordings too large to load at once.
    Fields as in load_recording. with_serial is read from the magic when None."""
    path = Path(path)
    with open(path, "rb") as f:
        head = f.read(24)
    magic = head[:8]
    if magic not in (b"FLYREC01", b"FLYDUMP1"):
        raise ValueError(f"{path}: not a recording or dump")
    nb, n, nm, nd = np.frombuffer(head[8:24], "<u4").astype(int)
    _, dtype = _dtype(nb, n, nm, nd, magic == b"FLYREC01" if with_serial is None else with_serial)
    frames = (path.stat().st_size - 24) // dtype.itemsize
    return np.memmap(path, dtype, "r", offset=24, shape=(frames,))


def _parse(data, magic, with_serial):
    if data[:8] != magic:
        raise ValueError(f"not a {magic.decode()} file")
    nb, n, nm, nd = np.frombuffer(data[8:24], "<u4").astype(int)
    fields, dtype = _dtype(nb, n, nm, nd, with_serial)
    body = data[24:]
    frames = len(body) // dtype.itemsize
    rec = np.frombuffer(body[:frames * dtype.itemsize], dtype)
    out = {name: np.ascontiguousarray(rec[name]) for name, _, _ in fields}
    out["hits"] = out["hits"].astype(bool)
    out["n"] = dict(n_bands=int(nb), n_neurons=int(n), n_mn=int(nm), n_drums=int(nd))
    return out


def load_recording(path):
    path = Path(path)
    out = _parse(path.read_bytes(), b"FLYREC01", True)
    side = path.with_name(path.name + ".json")
    out["sidecar"] = json.loads(side.read_text(encoding="utf-8")) if side.exists() else None
    return out


def load_dump(path):
    return _parse(Path(path).read_bytes(), b"FLYDUMP1", False)
