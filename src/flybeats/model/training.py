"""Phase 3, steps 6-8: batches as tensors, one training step, validation, and the silence/stability test."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ..config import load_locked, load_paths, reports_dir
from ..connectome.store import load_slice
from ..data.manifest import load_manifest, set_dir
from ..data.sampler import ClipSampler
from ..data.slakh import unpacked_root
from .ear import load_band_stats
from .loss import drum_weights, f1_from_counts, hit_counts, make_loss
from .network import FlyNet


def setup(which, control_seed=None, device="cuda", cfg=None):
    """Everything a run needs: cfg, model, training and validation samplers, drum weights and the loss."""
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
    fit_readout_norm(model, ClipSampler(root, manifest, events, beats, cfg, "train", seed=cfg["model"]["readout_norm"]["seed"]),
                     cfg, device)
    weights = drum_weights(manifest, cfg["kit"]["pieces"], cfg["training_data"]["frame_ms"] / 1000,
                           cfg["loss"]["max_drum_weight"]).to(device)
    return {"cfg": cfg, "paths": paths, "model": model, "train": train, "val": val, "weights": weights,
            "loss": make_loss(cfg, weights), "manifest": manifest, "root": root}


def fit_readout_norm(model, sampler, cfg, device, batch_size=8):
    """Standardise the readout's input with fixed stats from locked.yaml's number of random training clips
    through this network, untrained. Its own sampler, so the training clip sequence is unchanged."""
    rn = cfg["model"]["readout_norm"]
    clips = []
    while len(clips) < rn["clips"]:
        clips += sampler.batch(min(batch_size, len(sampler.songs), rn["clips"] - len(clips)))
    batches = [torch.as_tensor(np.stack([c["audio"] for c in clips[i:i + batch_size]]), device=device)
               for i in range(0, len(clips), batch_size)]
    warmup = int(round(cfg["training_data"]["warmup_ignored_seconds"] * 1000 / cfg["training_data"]["frame_ms"]))
    model.fit_readout_norm(batches, warmup, rn["std_floor"])


def make_optimiser(model, cfg):
    tc = cfg["training"]
    if tc["optimiser"] != "adam":
        raise ValueError(f"unknown optimiser {tc['optimiser']!r}")
    return torch.optim.Adam(model.parameters(), lr=tc["learning_rate"])


def make_schedule(opt, steps, cfg):
    """The locked learning-rate schedule over a run of `steps` steps; call .step() after each optimiser step.
    cosine_to_zero: from the locked rate down to 0 at the last step. train.py and memorise.py both use this."""
    if cfg["training"]["schedule"] != "cosine_to_zero":
        raise ValueError(f"unknown schedule {cfg['training']['schedule']!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)


def to_tensors(clips, device):
    stack = lambda k, dt: torch.as_tensor(np.stack([c[k] for c in clips]), dtype=dt, device=device)
    return {"audio": stack("audio", torch.float32), "hits": stack("hits", torch.float32),
            "vel": stack("vel", torch.float32), "mask": stack("mask", torch.float32),
            "beats": [c["beats"] for c in clips]}


def train_step(model, opt, batch, loss_of, clip_norm, segment=0):
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
        loss, parts = loss_of(out, batch["hits"][:, sl], batch["vel"][:, sl], batch["mask"][:, sl])
        (loss / n_segments).backward()
        v = v.detach()
        total += float(loss.detach()) / n_segments
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)   # the size before clipping
    opt.step()
    return {"loss": total, **parts, "mean_rate": float(out["mean_rate"].mean()),
            "peak_rate": float(out["mean_rate"].max()), "grad_norm": float(grad_norm)}


class GraphedStep:
    """train_step over the whole clip, with the forward pass, loss and backward pass recorded once as a CUDA graph
    and replayed every step. A 16 s clip is 3,200 frames and about 100,000 small GPU operations per step; run one
    at a time from Python, the GPU spends much of each step waiting for the next one. Replayed, they go as one job.

    The same operations in the same order as train_step; gradient clipping and the optimiser step stay ordinary
    PyTorch calls outside the graph, so the locked optimiser runs exactly as it does there. The inputs are fixed
    tensors: load() copies a new batch into them (same shapes). The warm-up passes (needed before recording) run
    forward and backward only, so the weights do not move. scripts/check_cuda_graph.py checks that the results
    match train_step's."""

    def __init__(self, model, opt, batch, loss_of, clip_norm, warmup=2):
        self.model, self.opt, self.loss_of, self.clip_norm = model, opt, loss_of, clip_norm
        self.static = {k: batch[k].clone() for k in ("audio", "hits", "vel", "mask")}
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                opt.zero_grad(set_to_none=True)
                self._forward_backward()
        torch.cuda.current_stream().wait_stream(side)
        # Recorded with no gradients present, so the backward pass writes them fresh on every replay rather than
        # adding to them; they must never be set to None afterwards (the graph keeps writing to these tensors)
        opt.zero_grad(set_to_none=True)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._forward_backward()

    def _forward_backward(self):
        b = self.static
        out, _ = self.model.run(self.model.ear_input(b["audio"]))
        loss, parts = self.loss_of(out, b["hits"], b["vel"], b["mask"], as_tensors=True)
        loss.backward()
        return {"loss": loss.detach(), **parts, "mean_rate": out["mean_rate"].detach()}

    def load(self, batch):
        """Copy a new batch into the recorded inputs (train.py; memorisation keeps the same one)."""
        for k, t in self.static.items():
            t.copy_(batch[k])

    def __call__(self):
        self.graph.replay()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)   # before clipping
        self.opt.step()
        o = self.out
        return {"loss": float(o["loss"]), "hit_bce": float(o["hit_bce"]), "vel_mse": float(o["vel_mse"]),
                "mean_rate": float(o["mean_rate"].mean()), "peak_rate": float(o["mean_rate"].max()),
                "grad_norm": float(grad_norm)}


STEP_CODE = ["network.py", "ear.py", "loss.py", "training.py"]


def step_code_fingerprint():
    """A hash of the files the training step is made from: a CUDA graph check vouches only for the code it ran on."""
    here = Path(__file__).resolve().parent
    return hashlib.sha256(b"".join((here / f).read_bytes() for f in STEP_CODE)).hexdigest()[:16]


def graph_checked(history_readout=False):
    """(use the recorded step?, why): only after scripts/check_cuda_graph.py passed on exactly this code; with
    history_readout, only if its readout-history part passed too."""
    path = reports_dir() / "cuda_graph_check.json"
    if not path.exists():
        return False, "not checked yet (scripts/check_cuda_graph.py)"
    r = json.loads(path.read_text(encoding="utf-8"))
    if not r.get("passes"):
        return False, "its check failed"
    if r.get("code") != step_code_fingerprint():
        return False, "the model or training code changed since its check; run scripts/check_cuda_graph.py again"
    if history_readout and not (r.get("history_readout") or {}).get("passes"):
        return False, "the check has no passing readout-history part; run scripts/check_cuda_graph.py again"
    return True, f"checked, {r['speedup']:.2f}x as fast"


def audio_blind_loss(batch, loss_of, n_drums):
    """The lowest loss on this batch from outputs that ignore the audio: one constant hit logit and one
    constant velocity per drum. A model whose loss sits here has learned only how often each drum plays."""
    b, t = batch["hits"].shape[:2]
    dev = batch["hits"].device
    c = torch.full((n_drums,), -4.0, device=dev, requires_grad=True)
    e = torch.zeros(n_drums, device=dev, requires_grad=True)
    opt = torch.optim.LBFGS([c, e], max_iter=500, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        out = {"hit_logits": c.expand(b, t, n_drums), "vel": torch.sigmoid(e).expand(b, t, n_drums)}
        loss, _ = loss_of(out, batch["hits"], batch["vel"], batch["mask"])
        loss.backward()
        return loss

    opt.step(closure)
    return float(closure().detach())


@torch.no_grad()
def evaluate(model, clips, loss_of, cfg, batch_size=8, progress=None):
    """Loss and per-drum hit F1 (within the locked fraction of a beat) over a fixed list of clips."""
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
    threshold = cfg["training"]["hit_threshold"]
    counts, losses = 0, []
    dev = next(model.parameters()).device
    for i in range(0, len(clips), batch_size):
        b = to_tensors(clips[i:i + batch_size], dev)
        out, _ = model(b["audio"])
        losses.append(float(loss_of(out, b["hits"], b["vel"], b["mask"])[0]))
        counts = counts + hit_counts(out["hit_logits"], b["hits"], b["mask"], b["beats"], frame_s, fraction, threshold)
        if progress:
            progress(len(clips[i:i + batch_size]))
    f1 = f1_from_counts(counts)
    return {"loss": float(np.mean(losses)), "f1": dict(zip(model.pieces, np.round(f1, 3).tolist())),
            "mean_f1": float(np.nanmean(f1))}


@torch.no_grad()
def silence_test(model, clips, cfg, batch_size=4, progress=None):
    """Music then silence. Passes if activity stays bounded, drops below settle_fraction of its music level
    within settle_seconds of silence, enough readout motor neurons respond, and every drum has one that does.
    Thresholds from cfg["silence_test"]; the warm-up is the locked unscored start of each clip."""
    st = cfg["silence_test"]
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    silence_s = st["silence_seconds"]
    warmup_frames = int(round(cfg["training_data"]["warmup_ignored_seconds"] / frame_s))
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
    after = rate[:, music_end + int(st["settle_seconds"] / frame_s):]
    finite = bool(np.isfinite(rate).all())
    bounded = finite and float(rate.max()) <= st["bound_factor"] * music_level
    settled_ratio = float(after.max() / music_level) if music_level > 0 else float("inf")
    responds = mn_rates[:, warmup_frames:music_end].std(1).max(0) > st["respond_std"]
    drums = model.mn_drum.cpu().numpy().T @ responds.astype(np.float32)
    result = {"finite": finite, "bounded": bounded, "music_level": music_level, "peak_level": float(np.nanmax(rate)),
              "settled_ratio": settled_ratio, "responding_share": float(responds.mean()),
              "responding_per_drum": dict(zip(model.pieces, drums.astype(int).tolist()))}
    result["passes"] = (bounded and settled_ratio < st["settle_fraction"]
                        and result["responding_share"] >= st["responding_share"]
                        and bool((drums >= 1).all()))
    return result
