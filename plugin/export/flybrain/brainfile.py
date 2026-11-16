"""The .flybrain container: one JSON header plus named little-endian arrays, hashed.

Layout (all integers little-endian):
    bytes 0..8     magic b"FLYBRAIN"
    bytes 8..12    format version, uint32 (FORMAT_VERSION)
    bytes 12..20   header length H, uint64
    bytes 20..20+H JSON header, UTF-8: {"meta": {...}, "arrays": {name: {"dtype", "shape", "offset", "nbytes"}}}
    then each array's raw bytes at its offset (64-byte aligned, measured from byte 0)
    last 32 bytes  SHA-256 of everything before them
A reader checks the magic, the version, the hash and that every array lies inside the file. The hash is also the
brain's identity: the plugin's saved state records it, and a file whose hash differs is a different brain.

The arrays and metadata an engine needs are listed in REQUIRED; the exporter writes them, and verify_brain refuses a
file that lacks one or whose shapes disagree. Everything in meta is JSON so the UI can show it without the arrays.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

MAGIC = b"FLYBRAIN"
FORMAT_VERSION = 1
ALIGN = 64
HASH_BYTES = 32

# name -> (dtype, ndim). Shapes are checked against each other in verify_brain.
REQUIRED = {
    # core: CSR by receiving neuron (row = post), columns = sending neuron (pre)
    "w_indptr": ("int32", 1),       # (n + 1)
    "w_indices": ("int32", 1),      # (nnz) pre index per entry
    "w_values": ("float32", 1),     # (nnz) realised signed weight = sign[pre] * synapses * g * softplus(theta[pair])
    "w_unsigned": ("float32", 1),   # (nnz) synapses * g * softplus(theta[pair]); w_values == sign[pre] * w_unsigned
    "w_pair": ("int32", 1),         # (nnz) type-pair index of each entry (for lesion by pair, display)
    "alpha": ("float32", 1),        # (n) dt / tau per neuron, tau clamped to [tau_min, tau_max]
    "tau_ms": ("float32", 1),       # (n) the clamped time constant per neuron, ms
    "bias": ("float32", 1),         # (n)
    "alive": ("float32", 1),        # (n) 1 for live neurons, 0 for silenced
    "sign": ("int8", 1),            # (n) +1 / -1 / 0 (silenced)
    "type_idx": ("int32", 1),       # (n) cell type per neuron, index into meta.type_names
    # ear
    "ear_idx": ("int32", 1),        # (n_ears) neuron index of each ear neuron
    "ear_band": ("int32", 1),       # (n_ears) band each ear listens to
    "band_mean": ("float32", 1),    # (16)
    "band_std": ("float32", 1),     # (16)
    "band_gain": ("float32", 1),    # (16) learned
    "band_offset": ("float32", 1),  # (16) learned
    "band_matrix": ("float32", 2),  # (257, 16) 0/1 FFT-bin-to-band matrix, exported so no port can drift
    # readout
    "mn_idx": ("int32", 1),         # (n_mn) neuron index of each readout motor neuron
    "mn_drum": ("int32", 1),        # (n_mn) drum index of each readout motor neuron
    "mn_mean": ("float32", 1),      # (n_mn)
    "mn_std": ("float32", 1),       # (n_mn)
    "w_hit": ("float32", 1),        # (n_mn)
    "u_vel": ("float32", 1),        # (n_mn)
    "c_hit": ("float32", 1),        # (n_drums)
    "e_vel": ("float32", 1),        # (n_drums)
    "history_blocks": ("int32", 1), # (n_blocks) frames per history block, empty when the history readout is off
    "p_hit": ("float32", 2),        # (n_drums, 1 + n_blocks) ; (n_drums, 1) all ones when off
    "p_vel": ("float32", 2),        # (n_drums, 1 + n_blocks) ; same shape as p_hit
    # display metadata (per neuron)
    "body_id": ("int64", 1),        # (n) MaleCNS body ID; negative for a mirrored ear copy
    "hop": ("int16", 1),            # (n) hops from the nearest ear along the wiring; 99 = unreachable
    "side": ("int8", 1),            # (n) 0 = L, 1 = R, 2 = midline
    "motor_group": ("int8", 1),     # (n) -1 none, else index into meta.motor_group_names
    "drum_of_neuron": ("int8", 1),  # (n) -1 none, else drum index (readout motor neurons only)
}

REQUIRED_META = ["schema", "name", "capability", "input_task", "n_neurons", "n_edges", "n_drums", "pieces",
                 "gm_notes", "type_names", "motor_group_names", "frame_ms", "sample_rate", "n_fft", "hop",
                 "n_bands", "hit_threshold", "tau_min_ms", "tau_max_ms", "g", "gain_scale", "eigenvalue",
                 "provenance", "cliff", "scoreboard", "band_range", "created"]

CAPABILITIES = ("observation_only", "drum_capable")
INPUT_TASKS = ("unknown", "accompaniment", "transcription")


def _align(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def write_brain(path, meta, arrays):
    """Write meta (JSON-able dict) and arrays ({name: ndarray}) to path. Returns the file's SHA-256 hex."""
    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    for name, (dtype, ndim) in REQUIRED.items():
        if name not in arrays:
            raise ValueError(f"missing array {name!r}")
        a = arrays[name]
        if a.dtype != np.dtype(dtype) or a.ndim != ndim:
            raise ValueError(f"array {name!r}: want {dtype} with {ndim} dims, got {a.dtype} with {a.ndim}")
    for k in REQUIRED_META:
        if k not in meta:
            raise ValueError(f"missing meta {k!r}")
    meta = dict(meta, format_version=FORMAT_VERSION)

    # Lay the arrays out after the header. The header's length depends on the offsets' digits, so fix the
    # header length first by padding the JSON to a size that holds any offsets.
    index = {}
    for name, a in arrays.items():
        index[name] = {"dtype": str(a.dtype), "shape": list(a.shape), "offset": 0, "nbytes": int(a.nbytes)}
    header_len = len(_header_bytes(meta, index)) + 64 * len(arrays) + ALIGN   # room for the real offsets
    header_len = _align(20 + header_len) - 20
    offset = 20 + header_len
    for name, a in arrays.items():
        index[name]["offset"] = offset
        offset = _align(offset + a.nbytes)
    header = _header_bytes(meta, index)
    header = header + b" " * (header_len - len(header))
    out = bytearray(offset)
    out[0:8] = MAGIC
    out[8:12] = np.uint32(FORMAT_VERSION).tobytes()
    out[12:20] = np.uint64(header_len).tobytes()
    out[20:20 + header_len] = header
    for name, a in arrays.items():
        o = index[name]["offset"]
        out[o:o + a.nbytes] = a.astype(a.dtype.newbyteorder("<")).tobytes()
    digest = hashlib.sha256(out).digest()
    Path(path).write_bytes(bytes(out) + digest)
    return digest.hex()


def _header_bytes(meta, index):
    return json.dumps({"meta": meta, "arrays": index}, separators=(",", ":"), sort_keys=True).encode("utf-8")


class Brain:
    """A read brain: .meta (dict), .arrays ({name: ndarray}), .sha256 (hex), .path."""

    def __init__(self, meta, arrays, sha256, path=None):
        self.meta, self.arrays, self.sha256, self.path = meta, arrays, sha256, path

    def __getitem__(self, name):
        return self.arrays[name]

    @property
    def n(self):
        return int(self.meta["n_neurons"])


def read_brain(path, verify=True):
    data = Path(path).read_bytes()
    if len(data) < 20 + HASH_BYTES or data[:8] != MAGIC:
        raise ValueError(f"{path}: not a .flybrain file")
    version = int(np.frombuffer(data[8:12], "<u4")[0])
    if version != FORMAT_VERSION:
        raise ValueError(f"{path}: format version {version}, this reader supports {FORMAT_VERSION}")
    body, digest = data[:-HASH_BYTES], data[-HASH_BYTES:]
    if hashlib.sha256(body).digest() != digest:
        raise ValueError(f"{path}: hash mismatch, the file is corrupt or was edited")
    header_len = int(np.frombuffer(data[12:20], "<u8")[0])
    if 20 + header_len > len(body):
        raise ValueError(f"{path}: header runs past the end of the file")
    header = json.loads(data[20:20 + header_len].decode("utf-8"))
    arrays = {}
    for name, spec in header["arrays"].items():
        o, nb = spec["offset"], spec["nbytes"]
        if o < 20 + header_len or o + nb > len(body):
            raise ValueError(f"{path}: array {name!r} lies outside the file")
        a = np.frombuffer(body[o:o + nb], np.dtype(spec["dtype"]).newbyteorder("<")).reshape(spec["shape"])
        arrays[name] = a.astype(a.dtype.newbyteorder("="))
    brain = Brain(header["meta"], arrays, digest.hex(), str(path))
    if verify:
        verify_brain(brain)
    return brain


def verify_brain(b):
    """Shapes, ranges and metadata agree. Raises ValueError with the first problem found."""
    m, a = b.meta, b.arrays
    for k in REQUIRED_META:
        if k not in m:
            raise ValueError(f"missing meta {k!r}")
    for name, (dtype, ndim) in REQUIRED.items():
        if name not in a:
            raise ValueError(f"missing array {name!r}")
        if a[name].dtype != np.dtype(dtype) or a[name].ndim != ndim:
            raise ValueError(f"array {name!r} has dtype {a[name].dtype} and {a[name].ndim} dims")
    n, nnz, nd = m["n_neurons"], m["n_edges"], m["n_drums"]
    if m["capability"] not in CAPABILITIES or m["input_task"] not in INPUT_TASKS:
        raise ValueError(f"capability {m['capability']!r} / input_task {m['input_task']!r} not allowed")
    if len(m["pieces"]) != nd or len(m["type_names"]) <= int(a["type_idx"].max()):
        raise ValueError("pieces or type_names disagree with the arrays")
    for name in ("alpha", "tau_ms", "bias", "alive", "sign", "type_idx", "body_id", "hop", "side", "motor_group",
                 "drum_of_neuron"):
        if a[name].shape != (n,):
            raise ValueError(f"{name} has shape {a[name].shape}, want ({n},)")
    if a["w_indptr"].shape != (n + 1,) or a["w_indptr"][0] != 0 or a["w_indptr"][-1] != nnz:
        raise ValueError("w_indptr does not describe n rows with n_edges entries")
    if np.any(np.diff(a["w_indptr"]) < 0):
        raise ValueError("w_indptr is not non-decreasing")
    for name in ("w_indices", "w_values", "w_unsigned", "w_pair"):
        if a[name].shape != (nnz,):
            raise ValueError(f"{name} has shape {a[name].shape}, want ({nnz},)")
    if nnz and (a["w_indices"].min() < 0 or a["w_indices"].max() >= n):
        raise ValueError("w_indices out of range")
    for r in range(n):                                   # columns sorted and unique within each row
        cols = a["w_indices"][a["w_indptr"][r]:a["w_indptr"][r + 1]]
        if len(cols) > 1 and np.any(np.diff(cols) <= 0):
            raise ValueError(f"row {r}: columns not strictly increasing")
    ne = a["ear_idx"].shape[0]
    if a["ear_band"].shape != (ne,) or (ne and (a["ear_band"].min() < 0 or a["ear_band"].max() >= m["n_bands"])):
        raise ValueError("ear_band disagrees with ear_idx or n_bands")
    for name in ("band_mean", "band_std", "band_gain", "band_offset"):
        if a[name].shape != (m["n_bands"],):
            raise ValueError(f"{name} has shape {a[name].shape}")
    if a["band_matrix"].shape != (m["n_fft"] // 2 + 1, m["n_bands"]):
        raise ValueError("band_matrix has the wrong shape")
    nm = a["mn_idx"].shape[0]
    for name in ("mn_drum", "mn_mean", "mn_std", "w_hit", "u_vel"):
        if a[name].shape != (nm,):
            raise ValueError(f"{name} has shape {a[name].shape}, want ({nm},)")
    if nm and (a["mn_drum"].min() < 0 or a["mn_drum"].max() >= nd):
        raise ValueError("mn_drum out of range")
    if a["c_hit"].shape != (nd,) or a["e_vel"].shape != (nd,):
        raise ValueError("c_hit / e_vel have the wrong shape")
    nb = a["history_blocks"].shape[0]
    if a["p_hit"].shape != (nd, 1 + nb) or a["p_vel"].shape != (nd, 1 + nb):
        raise ValueError("p_hit / p_vel disagree with history_blocks")
    if nb and a["history_blocks"].min() < 1:
        raise ValueError("history blocks must be at least one frame")
    if not np.all(np.isfinite(a["w_values"])) or not np.all(np.isfinite(a["alpha"])):
        raise ValueError("non-finite weights or alpha")
    if np.any(a["alpha"] <= 0) or np.any(a["alpha"] > 1):
        raise ValueError("alpha must lie in (0, 1]")
    silenced = a["alive"] == 0
    if np.any(a["sign"][silenced] != 0):
        raise ValueError("a silenced neuron has a non-zero sign")
    return True
