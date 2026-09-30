"""Phase 3, steps 6-8: batches as tensors, one training step, validation, and the silence/stability test."""
import numpy as np
import torch

from ..config import load_locked, load_paths
from ..connectome.store import load_slice
from ..data.manifest import load_manifest, set_dir
from ..data.sampler import ClipSampler
from ..data.slakh import unpacked_root
from .ear import load_band_stats
from .loss import drum_weights, f1_from_counts, hit_counts, loss_fn
from .network import FlyNet

RESPOND_STD = 1e-3        # a motor neuron "responds" if its rate varies at least this much during music
SETTLE_FRACTION = 0.05    # activity must fall below 5% of its music level ...
SETTLE_SECONDS = 1.0      # ... within 1 s of silence, and stay there
BOUND_FACTOR = 50.0       # "bounded": finite, and never above 50x the average music level


def setup(which, control_seed=None, device="cuda", cfg=None):
    """Everything a run needs: cfg, model, training and validation samplers, drum weights."""
    cfg = cfg or load_locked()
    paths = load_paths()
    neurons, edges = load_slice(paths["work_dir"], control_seed)
    manifest, events, beats = load_manifest(paths["work_dir"], which)
    root = unpacked_root(paths["work_dir"], which)
    stats_path = set_dir(paths["work_dir"], which) / "band_stats.json"
    if not stats_path.exists():
        raise FileNotFoundError(f"{stats_path} missing: run scripts/band_stats.py --set {which}")
    model = FlyNet(neurons, edges, cfg, load_band_stats(stats_path)).to(device)
    train = ClipSampler(root, manifest, events, beats, cfg, "train")
    val = ClipSampler(root, manifest, events, beats, cfg, "validation")
    weights = drum_weights(manifest, cfg["kit"]["pieces"], cfg["training_data"]["frame_ms"] / 1000).to(device)
    return {"cfg": cfg, "paths": paths, "model": model, "train": train, "val": val, "weights": weights,
            "manifest": manifest, "root": root}


def to_tensors(clips, device):
    stack = lambda k, dt: torch.as_tensor(np.stack([c[k] for c in clips]), dtype=dt, device=device)
    return {"audio": stack("audio", torch.float32), "hits": stack("hits", torch.float32),
            "vel": stack("vel", torch.float32), "mask": stack("mask", torch.float32),
            "beats": [c["beats"] for c in clips]}


def train_step(model, opt, batch, weights, segment=0, clip_norm=1.0):
    """Backpropagate through the whole clip, or through `segment`-frame pieces with the state carried over."""
    opt.zero_grad(set_to_none=True)
    ear_all = model.ear_input(batch["audio"])
    t_len = ear_all.shape[1]
    seg = segment or t_len
    v, total, parts = None, 0.0, {}
    n_segments = (t_len + seg - 1) // seg
    for s in range(0, t_len, seg):
        sl = slice(s, s + seg)
        ear = ear_all[:, sl] if n_segments == 1 else model.ear_input(batch["audio"])[:, sl]
        out, v = model.run(ear, v)
        loss, parts = loss_fn(out, batch["hits"][:, sl], batch["vel"][:, sl], batch["mask"][:, sl], weights)
        (loss / n_segments).backward()
        v = v.detach()
        total += float(loss.detach()) / n_segments
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
    opt.step()
    return {"loss": total, **parts, "mean_rate": float(out["mean_rate"].mean()),
            "peak_rate": float(out["mean_rate"].max())}


@torch.no_grad()
def evaluate(model, clips, weights, frame_s, batch_size=8, progress=None):
    """Loss and per-drum hit F1 (within 10% of a beat) over a fixed list of clips."""
    counts, losses = 0, []
    dev = weights.device
    for i in range(0, len(clips), batch_size):
        b = to_tensors(clips[i:i + batch_size], dev)
        out, _ = model(b["audio"])
        losses.append(float(loss_fn(out, b["hits"], b["vel"], b["mask"], weights)[0]))
        counts = counts + hit_counts(out["hit_logits"], b["hits"], b["mask"], b["beats"], frame_s)
        if progress:
            progress(len(clips[i:i + batch_size]))
    f1 = f1_from_counts(counts)
    return {"loss": float(np.mean(losses)), "f1": dict(zip(model.pieces, np.round(f1, 3).tolist())),
            "mean_f1": float(np.nanmean(f1))}


@torch.no_grad()
def silence_test(model, clips, frame_s, silence_s=4.0, warmup_frames=400, batch_size=4, progress=None):
    """Music then silence. Passes if activity stays finite, drops below 5% of its music level within
    1 s of silence, at least half the readout motor neurons respond, and every drum has one that does."""
    dev = next(model.parameters()).device
    sr = 16000
    pad = int(silence_s * sr)
    rates, mn = [], []
    for i in range(0, len(clips), batch_size):
        audio = torch.as_tensor(np.stack([np.r_[c["audio"], np.zeros(pad, np.float32)] for c in clips[i:i + batch_size]]),
                                device=dev)
        out, _ = model(audio)
        rates.append(out["mean_rate"].cpu())
        mn.append(out["mn_rates"].cpu())
        if progress:
            progress(len(clips[i:i + batch_size]))
    rate = torch.cat(rates).numpy()            # (clips, frames)
    mn_rates = torch.cat(mn).numpy()           # (clips, frames, readout neurons)
    music_end = rate.shape[1] - int(silence_s / frame_s)
    music = rate[:, warmup_frames:music_end]
    music_level = float(music.mean())
    after = rate[:, music_end + int(SETTLE_SECONDS / frame_s):]
    finite = bool(np.isfinite(rate).all())
    bounded = finite and float(rate.max()) <= BOUND_FACTOR * music_level
    settled_ratio = float(after.max() / music_level) if music_level > 0 else float("inf")
    responds = mn_rates[:, warmup_frames:music_end].std(1).max(0) > RESPOND_STD
    drums = model.mn_drum.cpu().numpy().T @ responds.astype(np.float32)
    result = {"finite": finite, "bounded": bounded, "music_level": music_level, "peak_level": float(np.nanmax(rate)),
              "settled_ratio": settled_ratio, "responding_share": float(responds.mean()),
              "responding_per_drum": dict(zip(model.pieces, drums.astype(int).tolist()))}
    result["passes"] = (bounded and settled_ratio < SETTLE_FRACTION and result["responding_share"] >= 0.5
                        and bool((drums >= 1).all()))
    return result
