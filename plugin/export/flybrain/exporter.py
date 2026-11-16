"""Export a FlyNet (rebuilt from the slice and a checkpoint) to a .flybrain file, verified against PyTorch.

Needs torch and the flybeats package (src/ on sys.path). `export()` is the function; `cli.py` wraps it for the
laptop, where the slice, band stats and checkpoints live.

What a checkpoint does not carry (read from network.py): `g` (initial_gain / largest eigenvalue of the slice)
and `gain_scale` are plain attributes, so the exporter rebuilds FlyNet from the slice and recomputes them;
band statistics come from the set's band_stats.json; the readout normalisation (mn_mean, mn_std) is in the
state_dict as buffers.
"""
import datetime as dt
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from .brainfile import write_brain

EXPORTER_VERSION = "0.1"
SIDES = {"L": 0, "R": 1, "M": 2}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit(root):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:
        return "unknown"


def hops_from_ears(n, pre, post, ear_idx):
    """Hop distance from the nearest ear along the wiring (slice.hops, without scipy)."""
    d = np.full(n, 99, np.int16)
    d[ear_idx] = 0
    frontier = np.zeros(n, bool)
    frontier[ear_idx] = True
    order = np.argsort(pre, kind="stable")
    pre_s, post_s = pre[order], post[order]
    starts = np.searchsorted(pre_s, np.arange(n + 1))
    for k in range(1, 30):
        nxt = np.zeros(n, bool)
        for src in np.flatnonzero(frontier):
            nxt[post_s[starts[src]:starts[src + 1]]] = True
        nxt &= d == 99
        if not nxt.any():
            break
        d[nxt] = k
        frontier = nxt
    return d


@torch.no_grad()
def realised_weights(model, edges):
    """The per-edge values exactly as FlyNet.weight_matrix forms them, plus their unsigned magnitudes (which
    keep the synapse count of silenced senders, whose `fixed` is 0, so they can be woken in the Lab)."""
    scale = model.g * model.gain_scale
    sp = F.softplus(model.theta[model.pair_idx])
    vals = (model.fixed * scale * sp).float().cpu().numpy()
    synapses = torch.from_numpy(edges.synapses.to_numpy(np.float32))
    unsigned = (synapses * scale * sp).float().cpu().numpy()
    return vals, unsigned


@torch.no_grad()
def export(model, neurons, edges, cfg, path, name="flybeats", capability="observation_only", input_task="unknown",
           provenance=None, scoreboard=None, cliff=None, runaway=None, notes=""):
    """Write the brain file. Returns (sha256 hex, meta). model must be on the CPU and in eval mode."""
    model = model.cpu().eval()
    n = model.n
    pre, post = model.pre.numpy(), model.post.numpy()
    if len(set(zip(pre.tolist(), post.tolist()))) != len(pre):
        raise ValueError("duplicate (pre, post) connections: the dense matrix would be ill-defined")
    vals, unsigned = realised_weights(model, edges)
    sign = neurons.sign.to_numpy(np.int8)
    if not np.array_equal(vals, sign[pre] * unsigned):
        raise AssertionError("signed weights are not sign x unsigned bit for bit")
    # CSR by receiving neuron (row = post), columns (pre) increasing within a row
    order = np.lexsort((pre, post))
    rows = post[order]
    indptr = np.zeros(n + 1, np.int64)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr)
    arrays = {
        "w_indptr": indptr.astype(np.int32), "w_indices": pre[order].astype(np.int32),
        "w_values": vals[order].astype(np.float32), "w_unsigned": unsigned[order].astype(np.float32),
        "w_pair": model.pair_idx.numpy()[order].astype(np.int32),
    }
    tau = model.tau_ms()                                                # clamp(exp(log_tau)) per type, float32
    alpha = (model.dt_ms / tau)[model.type_idx]                         # as run() forms it
    arrays.update({
        "alpha": alpha.float().numpy(), "tau_ms": tau[model.type_idx].float().numpy(),
        "bias": model.bias[model.type_idx].float().numpy(), "alive": model.alive.float().numpy(),
        "sign": sign, "type_idx": model.type_idx.numpy().astype(np.int32),
        "ear_idx": model.ear_idx.numpy().astype(np.int32), "ear_band": model.ear_band.numpy().astype(np.int32),
        "band_mean": model.bands.mean.float().numpy(), "band_std": model.bands.std.float().numpy(),
        "band_gain": model.band_gain.float().numpy(), "band_offset": model.band_offset.float().numpy(),
        "band_matrix": model.bands.bands.float().numpy(),
        "mn_idx": model.mn_idx.numpy().astype(np.int32), "mn_drum": model.mn_drum.argmax(1).numpy().astype(np.int32),
        "mn_mean": model.mn_mean.float().numpy(), "mn_std": model.mn_std.float().numpy(),
        "w_hit": model.w_hit.float().numpy(), "u_vel": model.u_vel.float().numpy(),
        "c_hit": model.c_hit.float().numpy(), "e_vel": model.e_vel.float().numpy(),
        "history_blocks": np.array(model.history_blocks, np.int32),
    })
    nd = len(model.pieces)
    if model.history_blocks:
        arrays["p_hit"] = model.p_hit.float().numpy()
        arrays["p_vel"] = (model.p_vel if model.p_vel is not None else model.p_hit.new_zeros(model.p_hit.shape)
                           .index_fill_(1, torch.tensor([0]), 1.0)).float().numpy()
    else:
        arrays["p_hit"] = np.ones((nd, 1), np.float32)
        arrays["p_vel"] = np.ones((nd, 1), np.float32)
    # display metadata
    groups = sorted(set(str(g) for g in neurons.motor_group.dropna().unique()))
    drum_of = neurons.drum.map({p: i for i, p in enumerate(model.pieces)})
    arrays.update({
        "body_id": neurons.bodyId.to_numpy(np.int64),
        "hop": hops_from_ears(n, pre, post, model.ear_idx.numpy()),
        "side": neurons.side.map(SIDES).fillna(2).to_numpy(np.int8),
        "motor_group": neurons.motor_group.map({g: i for i, g in enumerate(groups)}).fillna(-1).to_numpy(np.int8),
        "drum_of_neuron": drum_of.fillna(-1).to_numpy(np.int8),
    })
    mean, std = arrays["band_mean"], arrays["band_std"]
    meta = {
        "schema": "flybrain-1", "name": name, "capability": capability, "input_task": input_task,
        "n_neurons": int(n), "n_edges": int(len(pre)), "n_drums": nd, "pieces": list(model.pieces),
        "gm_notes": cfg["kit"]["gm_notes"], "type_names": list(model.type_names), "motor_group_names": groups,
        "frame_ms": float(model.dt_ms), "sample_rate": 16000, "n_fft": int(model.bands.window.numel()),
        "hop": int(round(16000 * model.dt_ms / 1000)), "n_bands": int(model.bands.mean.numel()),
        "hit_threshold": float(cfg["training"]["hit_threshold"]),
        "tau_min_ms": model.tau_min, "tau_max_ms": model.tau_max,
        "g": float(model.g), "gain_scale": float(model.gain_scale), "eigenvalue": float(model.eigenvalue),
        "locked_version": cfg.get("version"),
        "provenance": dict(provenance or {}, exporter_version=EXPORTER_VERSION, torch_version=torch.__version__,
                           python_version=sys.version.split()[0]),
        "cliff": cliff or {"measured": False, "note": "no cliff calibration: the Cliff Meter must show 'unmeasured'"},
        "scoreboard": scoreboard or {"verdict": "none", "note": "no v0.15 verdict: this brain is observation-only"},
        "band_range": {"low": (mean - 2 * std).tolist(), "high": (mean + 2 * std).tolist(),
                       "rule": "training mean +/- 2 standard deviations of log1p band power"},
        "runaway": runaway or {},
        "readout_history": bool(model.history_blocks),
        "silenced_count": int((arrays["alive"] == 0).sum()),
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "notes": notes,
    }
    digest = write_brain(path, meta, arrays)
    return digest, meta


# ----------------------------------------------------------------------------------------------- calibration

@torch.no_grad()
def measure_cliff(model, clips, cfg, grid=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.75, 2.0), resolution=0.02, device="cpu"):
    """The gain at which the repo's silence test first fails for this network: the grid, then a bisection between
    the last pass and the first fail down to `resolution`. Also returns the music level at gain 1.0, which sets
    the runaway guard's limits. Restores model.gain_scale afterwards."""
    from flybeats.model.training import silence_test
    saved = model.gain_scale
    model = model.to(device)
    results = {}

    def passes(g):
        model.gain_scale = float(g)
        r = silence_test(model, clips, cfg)
        results[f"{g:.4f}"] = {"passes": bool(r["passes"]), "bounded": bool(r["bounded"]),
                               "settled_ratio": float(r["settled_ratio"]), "music_level": float(r["music_level"]),
                               "peak_level": float(r["peak_level"])}
        return bool(r["bounded"]) and r["settled_ratio"] < cfg["silence_test"]["settle_fraction"]

    last_pass, first_fail = None, None
    for g in grid:
        if passes(g):
            last_pass = g
        else:
            first_fail = g
            break
    if last_pass is not None and first_fail is not None:
        lo, hi = last_pass, first_fail
        while hi - lo > resolution:
            mid = round((lo + hi) / 2, 4)
            if passes(mid):
                lo = mid
            else:
                hi = mid
        last_pass, first_fail = lo, hi
    model.gain_scale = saved
    level = results[f"{grid[0]:.4f}"]["music_level"]
    return {"measured": True, "last_pass_gain": last_pass, "first_fail_gain": first_fail,
            "margin": (first_fail / 1.0) if first_fail else None, "grid": list(grid), "resolution": resolution,
            "clips": len(clips), "music_level_at_1x": level, "rule": "repo silence_test: bounded and settled",
            "results": results}


def runaway_limits(cliff, cfg):
    """Runaway-guard limits from the calibration: the repo's own 'bounded' rule (bound_factor x the music level)
    held for `frames` frames, and a per-neuron ceiling far above it so the clamp never touches normal activity."""
    level = cliff["music_level_at_1x"]
    factor = cfg["silence_test"]["bound_factor"]
    return {"mean_rate_limit": factor * level, "rate_ceiling": 1000.0 * factor * level, "frames": 20,
            "rule": f"mean rate above {factor} x music level ({level:.4g}) for 20 frames; per-neuron clamp at 1000x that"}


def load_checkpoint(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    state = ck["model"] if isinstance(ck, dict) and "model" in ck else ck
    model.load_state_dict(state, strict=True)
    return {k: v for k, v in ck.items() if k not in ("model", "opt")} if isinstance(ck, dict) else {}
