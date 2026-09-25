"""Phase 3 training.

Truncated BPTT over the rate model, per-class BCE against Gaussian-smoothed
onset targets, plus a firing-rate regulariser that keeps the network off its
saturation ceiling.

Truncation already bounds activation memory in sequence length; gradient
checkpointing (``train.grad_checkpoint``) trades ~30% more compute to halve
what one chunk holds, which is what buys a longer chunk or a larger batch on a
fixed GPU. bfloat16 autocast (``train.bf16``) is honoured on CUDA and ignored
on CPU, where it costs more than it saves. Both default off here.

The loss is deliberately plain. Everything interesting is supposed to be in the
recurrent core, so anything clever here would muddy what the Phase 4 ablations
are comparing.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build import build_model, device_of, get_subgraph, load_config
from dataset import build_dataset
from metrics import (beat_alignment_error, groove_similarity, onset_f_measure,
                     onset_f_sweep)
from progress import describe, estimate

ROOT = Path(__file__).resolve().parents[1]


def onset_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Per-class BCE. Onsets are rare, so positives are up-weighted."""
    pw = torch.full((logits.shape[-1],), pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def velocity_loss(pred: torch.Tensor, target: torch.Tensor,
                  onsets: torch.Tensor, peak_only: float = 0.0) -> torch.Tensor:
    """How hard, scored only where there is a hit to be that hard.

    Weighted by the onset target itself, so a step contributes in proportion to
    how much of a hit is there. Steps between hits carry no velocity to predict
    and must not be scored: unweighted, 98% of the loss would come from silence
    and the head would learn the mean of nothing. The weights are the same
    smoothed plane the BCE sees, so both heads agree on where the hits are.

    ``peak_only`` drops every step whose onset target sits below it, so the
    head is scored on the frames it is actually read at. The streaming path
    takes velocity at the picked peak and nowhere else, while this weighting
    spreads over the kernel's whole +/-3 sigma support -- roughly 25 steps per
    hit, all carrying the same flat velocity target but wildly different drive.
    Training across all of them asks the head to be invariant to where in the
    envelope it is, which is a different and harder task than the one that
    matters, and averaging is the cheapest way to satisfy it.

    Returns 0 for a chunk with no onsets in it at all, which happens on quiet
    intros and would otherwise divide by zero.
    """
    w = onsets if peak_only <= 0.0 else onsets * (onsets >= peak_only)
    denom = w.sum()
    if float(denom) <= 0.0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return ((pred - target).pow(2) * w).sum() / denom


def velocity_mass(onsets: torch.Tensor, peak_only: float = 0.0) -> torch.Tensor:
    """The denominator ``velocity_loss`` uses, computed over any span.

    ``run_epoch`` needs the whole clip's mass to weight each TBPTT chunk's
    contribution, and it has to be exactly the quantity velocity_loss puts in
    its own denominator. Kept next to it so the two cannot drift apart.
    """
    w = onsets if peak_only <= 0.0 else onsets * (onsets >= peak_only)
    return w.sum()


def rate_penalty(rates: torch.Tensor, ceiling: float) -> torch.Tensor:
    """Penalise mean activity *above* ``ceiling``. Nothing below it is penalised.

    The job of this term is to keep the softplus units off their saturation
    ceiling, where the readout sees a constant. It is not to hold activity at
    some particular level, and the earlier two-sided version did exactly that,
    against a number with no meaning: ``target_rate_hz`` was converted as
    ``hz * step_ms / 1000`` -- 0.025 for 5 Hz at a 5 ms step -- as if
    ``softplus(v - theta)`` were spikes per step. It is a dimensionless
    activation, and it sits near 0.74 at initialisation. So the term was a
    constant ~30x downward pull on all activity, and the cheapest way to
    satisfy it was for the encoder to silence its own sensory input. It did:
    96% of ``to_jo`` went negative, the drive collapsed to DC, and the model
    settled on the best constant predictor while the loss fell. See the README.

    One-sided removes the incentive to be quiet, and the ceiling is measured
    from the network's own activity at initialisation (``rate_ceiling: auto``)
    rather than invented, so it means "several times as loud as this network
    starts" in the units the network actually uses.
    """
    if rates is None:
        return torch.zeros((), device="cpu")
    return (rates.mean() - float(ceiling)).clamp(min=0.0).pow(2)


def resolve_rate_ceiling(model, tb: dict, rates: torch.Tensor) -> float:
    """Resolve ``rate_ceiling: auto`` against the untrained network's activity.

    Called on the first chunk of the first epoch, before any optimiser step, so
    ``rates`` is that model's initial operating point. It is cached on the model
    and not in the config: the Phase 4 arms share one config but not one
    initialisation, and each arm has to be judged against its own starting
    activity for the same reason each arm is normalised to a common spectral
    radius.
    """
    cached = getattr(model, "rate_ceiling", None)
    if cached is not None:
        return float(cached)
    want = tb.get("rate_ceiling", "auto")
    if want == "auto":
        ceiling = float(rates.detach().float().mean()) * float(tb.get("rate_headroom", 4.0))
        print(f"  [rate] ceiling = {ceiling:.4f} activation "
              f"({tb.get('rate_headroom', 4.0)}x this model's initial mean)")
    else:
        ceiling = float(want)
    model.rate_ceiling = ceiling
    return ceiling


def calibrate_encoder(model, dataset, cfg, device, n_clips: int = 16) -> dict:
    """Measure the encoder's feature standardisation on real training audio.

    Deterministic on purpose -- the corpus loader crops clips at random, so
    both random streams it can draw from are seeded and restored around the
    sample. Every Phase 4 arm then calibrates to exactly the same affine, and a
    re-run of a config gets the same encoder it got last time. Torch is in that
    list because the training split's window offsets come from torch's
    generator now; seeding numpy alone would have left the calibration audio
    drifting between arms while this docstring still claimed it did not.
    """
    if not getattr(model.encoder, "standardize", False):
        return {"calibrated": False}
    seed = int(cfg["train"].get("seed", 0))
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        wavs = [torch.as_tensor(dataset[i][0]).unsqueeze(0).to(device)
                for i in range(min(n_clips, len(dataset)))]
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    return model.encoder.calibrate(wavs)


@torch.no_grad()
def calibrate_decoder(model, dataset, cfg, device, n_clips: int = 16) -> dict:
    """Measure the decoder's motor standardisation on the untrained network.

    The same seeded clips as :func:`calibrate_encoder`, run through the
    calibrated encoder and the untrained core, so it must be called after the
    encoder is calibrated and before the first optimiser step. A no-op unless
    ``kit.standardize_motor`` is set. Each model calibrates against its own
    initial activity, as the rate ceiling does, because an ablation arm with a
    different core has a different motor operating point.
    """
    if not getattr(model.decoder, "standardize_motor", False):
        return {"calibrated": False}
    seed = int(cfg["train"].get("seed", 0))
    np_state, torch_state = np.random.get_state(), torch.random.get_rng_state()
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        items = [dataset[i] for i in range(min(n_clips, len(dataset)))]
    finally:
        np.random.set_state(np_state)
        torch.random.set_rng_state(torch_state)
    was_training = model.training
    model.eval()
    try:
        rates = []
        for item in items:
            wav = torch.as_tensor(item[0]).unsqueeze(0).to(device)
            style = torch.as_tensor(item[3]).reshape(1).to(device)
            drive = model.encoder(wav)
            tonic = model.genre(style) if model.genre is not None else None
            full, _ = model.rnn(drive, tonic=tonic, return_all=True)
            rates.append(full[:, :, model.rnn.motor_idx])
    finally:
        model.train(was_training)
    return model.decoder.calibrate(rates)


def resolve_bf16(setting, device) -> bool:
    """Resolve ``train.bf16`` -- ``true``, ``false``, or ``auto`` -- for a device.

    ``auto`` means bf16 only where the silicon actually has it: compute
    capability 8.0 (Ampere) and up. Below that torch will still *run* bf16, by
    emulation, and on a Turing card (7.5) that measured 1.4x SLOWER than fp32
    -- 6.49 ms vs 4.63 ms on a 2048^3 GEMM. So ``auto`` is not the cautious
    choice, it is the fast one, and it is why this is not simply ``false``:
    on an Ampere box the same config should still get bf16.

    An explicit ``true`` is honoured on any card. ``SparseSpMM`` pins its own
    operands to fp32, so the bf16 path runs below 8.0 rather than raising --
    it just has nothing to win. Keeping ``true`` meaningful everywhere is what
    lets the bf16 tests run on this machine instead of skipping.
    """
    if device.type != "cuda":
        return False
    if isinstance(setting, str):
        if setting.strip().lower() != "auto":
            raise SystemExit(
                f"train.bf16 must be true, false, or 'auto' -- got {setting!r}"
            )
        return torch.cuda.get_device_capability(device)[0] >= 8
    return bool(setting)


def audio_gap_mask(n_samples: int, sample_rate: int, gaps: dict,
                   gen: torch.Generator) -> torch.Tensor:
    """Boolean ``(n_samples,)`` mask, True where the audio is to be silenced.

    Random spans of ``min_ms``..``max_ms`` are laid down until about
    ``fraction`` of the window is covered. Spans may overlap; the loop stops
    once coverage reaches the fraction (or after a bounded number of draws).
    """
    frac = float(gaps.get("fraction", 0.5))
    mask = torch.zeros(n_samples, dtype=torch.bool)
    if frac <= 0.0 or n_samples == 0:
        return mask
    lo = max(1, int(round(float(gaps.get("min_ms", 250)) * sample_rate / 1000)))
    hi = max(lo, int(round(float(gaps.get("max_ms", 1000)) * sample_rate / 1000)))
    lo, hi = min(lo, n_samples), min(hi, n_samples)
    goal = min(frac, 1.0) * n_samples
    for _ in range(1000):
        if int(mask.sum()) >= goal:
            break
        span = int(torch.randint(lo, hi + 1, (1,), generator=gen))
        start = int(torch.randint(0, n_samples - span + 1, (1,), generator=gen))
        mask[start: start + span] = True
    return mask


def apply_audio_gaps(wav: torch.Tensor, cfg: dict, gen: torch.Generator) -> torch.Tensor:
    """Zero random spans of each clip in a ``(B, samples)`` batch. Training
    only: the drum targets are left as they are, so the model has to carry the
    groove through the silence from its own state and the style drive."""
    gaps = cfg["train"]["audio_gaps"]
    sr = int(cfg["audio"]["sample_rate"])
    keep = torch.stack([~audio_gap_mask(wav.shape[-1], sr, gaps, gen)
                        for _ in range(wav.shape[0])])
    return wav * keep.to(device=wav.device, dtype=wav.dtype)


def audio_gap_generator(cfg: dict, epoch: int) -> torch.Generator:
    """One generator per epoch, from ``train.seed`` and the epoch number, so a
    resumed run silences the same spans an uninterrupted one would have."""
    seed = int(cfg["train"].get("seed", 0))
    return torch.Generator().manual_seed(seed * 1_000_003 + int(epoch) + 7919)


def run_epoch(model, loader, opt, cfg, device, train: bool = True, use_genre: bool = True,
              epoch: int = 0):
    model.train(train)
    tb = cfg["train"]
    gap_gen = audio_gap_generator(cfg, epoch) if (train and tb.get("audio_gaps")) else None
    chunk = int(tb.get("tbptt_steps", 150))
    amp = resolve_bf16(tb.get("bf16", False), device)
    ckpt = bool(tb.get("grad_checkpoint", False)) and train
    totals = {"loss": 0.0, "bce": 0.0, "rate": 0.0, "vel": 0.0, "n": 0}
    if "target_rate_hz" in tb:
        raise SystemExit(
            "train.target_rate_hz is gone: it compared a dimensionless softplus "
            "activation against a spikes-per-step number and penalised all "
            "activity. Use train.rate_ceiling (activation units, or 'auto') "
            "with train.rate_headroom."
        )

    for wav, y, vel_y, style, _tempo in loader:
        if gap_gen is not None:
            wav = apply_audio_gaps(wav, cfg, gap_gen)
        wav, y, vel_y = wav.to(device), y.to(device), vel_y.to(device)
        style = style.to(device) if (use_genre and model.genre is not None) else None

        state = model.rnn.initial_state(wav.shape[0], device)
        steps = min(int(wav.shape[-1] // model.encoder.hop), y.shape[1])
        n_chunks = 0
        if train:
            opt.zero_grad(set_to_none=True)
        # Clip-level velocity mass. velocity_loss divides by the onset mass
        # inside its own chunk, so under a flat 1/n_total a chunk holding one
        # hit gave that hit ~30x the gradient of a hit in a chunk holding
        # thirty -- and where the boundary falls is an artefact of clip length,
        # not of the music. Normalising by the clip total makes every hit carry
        # the same weight wherever it lands.
        vel_mass = float(velocity_mass(y[:, :steps], tb.get("velocity_peak_only", 0.0)))

        for a in range(0, steps, chunk):
            b = min(a + chunk, steps)
            # This chunk's share of the clip, not 1/n_chunks. Each chunk's loss
            # is already a mean over its own elements, and the last chunk is a
            # remainder -- 800 steps at 150 leaves 50 -- so a flat 1/n_total
            # handed those 50 steps 3x the per-step gradient of the other 750,
            # on every clip, deterministically.
            frac = (b - a) / steps
            # Truncated BPTT: carry the state forward but cut the graph, so
            # memory stays flat in sequence length. The encoder and the genre
            # bias are rebuilt per chunk for the same reason -- a single shared
            # graph would be freed by the first chunk's backward pass.
            state = state.detach()
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.enable_grad()
            with ctx:
                drive = model.encoder.forward_window(wav, a, b - a)
                tonic = model.genre(style) if style is not None else None
                if ckpt:
                    # Recompute this chunk's timestep activations during
                    # backward instead of holding them all.
                    rates, state = torch.utils.checkpoint.checkpoint(
                        lambda d, st, tn: model.rnn(d, state=st, tonic=tn, return_all=True),
                        drive, state, tonic, use_reentrant=False,
                    )
                else:
                    rates, state = model.rnn(drive, state=state, tonic=tonic, return_all=True)
                motor = rates[:, :, model.rnn.motor_idx]
                logits = model.decoder(motor)
                bce = onset_loss(logits.float(), y[:, a:b], tb.get("pos_weight", 8.0))
                reg = rate_penalty(rates.float(), resolve_rate_ceiling(model, tb, rates))
                loss = bce + tb.get("rate_weight", 0.1) * reg
                vloss = torch.zeros((), device=loss.device, dtype=loss.dtype)
                vshare = 0.0
                if model.decoder.has_velocity:
                    peak_only = tb.get("velocity_peak_only", 0.0)
                    vloss = velocity_loss(model.decoder.velocity(motor).float(),
                                          vel_y[:, a:b], y[:, a:b],
                                          peak_only=peak_only)
                    if vel_mass > 0.0:
                        vshare = float(velocity_mass(y[:, a:b], peak_only)) / vel_mass
                    loss = loss + tb.get("velocity_weight", 1.0) * vloss
                # loss is what gets reported; grad_loss is what gets
                # differentiated. They differ only in the chunk weighting, so
                # the printed numbers stay comparable with earlier histories.
                grad_loss = (frac * (bce + tb.get("rate_weight", 0.1) * reg)
                             + tb.get("velocity_weight", 1.0) * vloss * vshare)

            if train:
                grad_loss.backward()
            totals["loss"] += float(loss.detach())
            totals["bce"] += float(bce.detach())
            totals["rate"] += float(reg.detach())
            totals["vel"] += float(vloss.detach())
            n_chunks += 1

        if train:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tb.get("grad_clip", 1.0))
            opt.step()
            # Projected step for the encoder's non-negativity constraint. Here
            # rather than in the model so every entry point that trains --
            # train.py and the Phase 4 arms -- gets it from one place.
            model.encoder.project_()
        totals["n"] += max(n_chunks, 1)

    n = max(totals["n"], 1)
    return {k: v / n for k, v in totals.items() if k != "n"}


@torch.no_grad()
def evaluate(model, loader, cfg, device, use_genre: bool = True) -> dict:
    """Score the model, sweeping the peak-picking threshold.

    A single fixed threshold makes onset F a function of the model's output
    *scale* as much as its timing: during a sanity run the loss fell
    monotonically while F at a fixed 0.3 bounced between 0.41 and 0.65. Worse,
    for Phase 4 it is unfair -- arms that settle at different output scales get
    scored at a point that suits one of them. So F is reported at its best
    threshold per model (chosen on this split, and the threshold is reported
    alongside so the choice is visible), with the fixed-threshold value kept as
    ``onset_f_fixed`` for continuity.
    """
    model.eval()
    step_ms = cfg["audio"]["step_ms"]
    tol = cfg["eval"].get("tolerance_s", 0.05)
    fixed = cfg["eval"].get("threshold", 0.3)
    sweep = cfg["eval"].get("threshold_sweep",
                            [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    if fixed not in sweep:
        sweep = sorted(sweep + [fixed])

    by_thr = {t: [] for t in sweep}
    ba_all, gs_all, dev_all = [], [], []
    vel_err = []
    # Kept per class, not pooled. A drummer's crashes are loud and their hats
    # are quiet, so pooling every class into one correlation rewards a head
    # that has learned nothing but each class's average level -- it would score
    # well while being deaf to dynamics within a class, which is the whole
    # quantity of interest. Both are reported so the gap between them is
    # visible.
    vel_pred: dict[int, list] = {}
    vel_ref: dict[int, list] = {}

    for wav, y, vel_y, style, tempo in loader:
        wav = wav.to(device)
        sid = style.to(device) if (use_genre and model.genre is not None) else None
        logits, _, vhat = model(wav, style_id=sid, with_velocity=True)
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        ref = y.numpy()
        steps = min(prob.shape[1], ref.shape[1])
        if vhat is not None:
            # Score velocity only where a hit is, and weight by how much of one
            # -- the same weighting the loss uses, so the reported error is the
            # quantity being optimised rather than a differently-shaped cousin.
            vp = vhat.float().cpu().numpy()[:, :steps]
            vt = vel_y.numpy()[:, :steps]
            w = ref[:, :steps]
            hit = w > 0.5
            if hit.any():
                vel_err.append(float(np.abs(vp - vt)[hit].mean()))
                for c in range(vp.shape[-1]):
                    m = hit[..., c]
                    if m.any():
                        vel_pred.setdefault(c, []).append(vp[..., c][m])
                        vel_ref.setdefault(c, []).append(vt[..., c][m])
        for i in range(prob.shape[0]):
            fs = onset_f_sweep(prob[i, :steps], ref[i, :steps], step_ms, sweep, tol)
            for t, f in fs.items():
                by_thr[t].append(f)
            m = onset_f_measure(prob[i, :steps], ref[i, :steps], step_ms,
                                tolerance_s=tol, threshold=fixed)
            if not np.isnan(m["mean_dev_ms"]):
                dev_all.append(m["mean_dev_ms"])
            bpm = float(tempo[i])
            if bpm > 0:
                ba = beat_alignment_error(prob[i, :steps], step_ms, bpm)
                gs = groove_similarity(prob[i, :steps], ref[i, :steps], step_ms, bpm)
                if not np.isnan(ba):
                    ba_all.append(ba)
                if not np.isnan(gs):
                    gs_all.append(gs)

    means = {t: (float(np.mean(v)) if v else 0.0) for t, v in by_thr.items()}
    best_t = max(means, key=means.get)
    # A head that has collapsed to one constant velocity still scores a
    # respectable MAE, because drummers are not that dynamic. Correlation is
    # what separates "predicts the mean" from "predicts the dynamics", so it is
    # the number to watch -- within class, for the reason given above.
    def _r(a, b):
        return (float(np.corrcoef(a, b)[0, 1])
                if a.std() > 1e-6 and b.std() > 1e-6 else 0.0)

    vel_r, vel_r_pooled = float("nan"), float("nan")
    if vel_pred:
        per_class = [_r(np.concatenate(vel_pred[c]), np.concatenate(vel_ref[c]))
                     for c in sorted(vel_pred)]
        vel_r = float(np.mean(per_class))
        vel_r_pooled = _r(np.concatenate([np.concatenate(vel_pred[c]) for c in sorted(vel_pred)]),
                          np.concatenate([np.concatenate(vel_ref[c]) for c in sorted(vel_ref)]))
    return {
        "onset_f": means[best_t],
        "velocity_mae": float(np.mean(vel_err)) if vel_err else float("nan"),
        "velocity_r": vel_r,
        "velocity_r_pooled": vel_r_pooled,
        "onset_f_fixed": means[fixed],
        "best_threshold": best_t,
        # The whole sweep, not just its argmax. Reporting only the max hides
        # how much of a score is the model and how much is the selection: in
        # the B=4 pilot the real arm gained 0.008 from being scored at its own
        # best threshold while a rewired draw gained 0.128 -- larger than the
        # gap between the arms, so the ranking was decided by the selection
        # rather than by the topology. Keeping the curve makes that visible,
        # and lets a common operating point be chosen after the fact without
        # re-running. String keys because this is serialised to JSON.
        "onset_f_curve": {f"{t:g}": means[t] for t in sorted(means)},
        "beat_align_ms": float(np.mean(ba_all)) if ba_all else float("nan"),
        "groove_sim": float(np.mean(gs_all)) if gs_all else float("nan"),
        "mean_dev_ms": float(np.mean(dev_all)) if dev_all else float("nan"),
        "n_clips": len(by_thr[fixed]),
    }


def _seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker its own seeded numpy and random state.

    Torch seeds its own per-worker generator; numpy's global RNG it does not
    touch, so forked workers inherit one state and hand back correlated
    "random" choices. Nothing in a loss curve would show that.
    """
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def reseed_loader(loader, seed: int) -> None:
    """Put a loader's shuffle stream back to the start of a known sequence.

    A loader is built once and reused -- across ablation arms, or across two
    passes of one comparison. Its generator keeps advancing, so the second user
    starts wherever the first stopped and gets different batches in a different
    order. That is invisible: both runs look fine, and the difference between
    them gets attributed to whatever the comparison was actually about.
    """
    gen = getattr(loader, "generator", None)
    if gen is not None:
        gen.manual_seed(int(seed))


def build_loaders(cfg, kit):
    dcfg = dict(cfg["data"])
    dcfg.setdefault("step_ms", cfg["audio"]["step_ms"])
    dcfg.setdefault("sample_rate", cfg["audio"]["sample_rate"])
    dcfg.setdefault("window_seed", cfg["train"].get("seed", 0))
    tr = build_dataset(dcfg, kit.classes, "train")
    va = build_dataset(dcfg, kit.classes, cfg["data"].get("val_split", "validation"))
    bs = cfg["train"].get("batch_size", 4)
    # Shuffle order and worker seeds both come off this generator, so two runs
    # of one config see the batches in the same order and the same windows
    # inside them. Without it the arms of an ablation differ by data order as
    # well as by topology, and the difference is invisible.
    gen = torch.Generator().manual_seed(int(cfg["train"].get("seed", 0)))
    train_loader = DataLoader(tr, batch_size=bs, shuffle=True,
                              num_workers=cfg["train"].get("workers", 0), drop_last=True,
                              generator=gen, worker_init_fn=_seed_worker)
    val_loader = DataLoader(va, batch_size=bs, shuffle=False,
                            num_workers=cfg["train"].get("workers", 0),
                            worker_init_fn=_seed_worker)
    return train_loader, val_loader, int(getattr(tr, "n_styles", 1))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rebuild-subgraph", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the pipeline")
    ap.add_argument("--resume", action="store_true",
                    help="continue from OUT/last.pt if there is one, else start fresh. "
                         "A run stops cleanly after its current epoch when OUT/STOP exists")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.smoke:
        cfg["data"].update(synthetic=True, n_clips=8, seconds=1.0)
        cfg["train"].update(batch_size=2, epochs=1, tbptt_steps=50)
    if a.epochs is not None:
        cfg["train"]["epochs"] = a.epochs

    device = device_of(cfg)
    torch.manual_seed(cfg["train"].get("seed", 0))
    np.random.seed(cfg["train"].get("seed", 0))

    sg = get_subgraph(cfg, rebuild=a.rebuild_subgraph)
    print(sg.summary())

    from decoder import DrumKit
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    train_loader, val_loader, n_styles = build_loaders(cfg, kit)

    model, kit = build_model(cfg, sg, n_styles=max(int(n_styles or 1), 1))
    model = model.to(device)
    print(f"device={device} | styles={n_styles} | kit={kit.classes}")
    print(f"params: {model.rnn.n_params()}")

    # The encoder's standardisation is part of the trained artefact: live
    # playback has to see the affine training saw, so it is measured before the
    # first step and saved with the weights.
    stats = calibrate_encoder(model, train_loader.dataset, cfg, device)
    print(f"encoder calibration: {stats}")
    dstats = calibrate_decoder(model, train_loader.dataset, cfg, device)
    if dstats.get("calibrated"):
        print(f"decoder calibration: {dstats}")

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"].get("lr", 3e-3), weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    out = a.out or ROOT / "runs" / a.config.stem
    out.mkdir(parents=True, exist_ok=True)

    n_epochs = int(cfg["train"].get("epochs", 10))
    history, best, start = [], -1.0, 0
    last = out / "last.pt"
    if a.resume and last.exists():
        start, history, best = resume_from(last, model, opt, train_loader, cfg, device)
        print(f"resumed from {last}: {start} of {n_epochs} epochs already done")
        # Reprinted in the live format, so a log that restarts on resume still
        # lists every epoch and runstat's progress bar counts them all.
        for rec in history:
            print(_epoch_line(rec))

    for ep in range(start, n_epochs):
        t0 = time.time()
        tr = run_epoch(model, train_loader, opt, cfg, device, train=True, epoch=ep)
        ev = evaluate(model, val_loader, cfg, device)
        rec = {"epoch": ep, **{f"train_{k}": v for k, v in tr.items()}, **ev,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(_epoch_line(rec))

        if ev["onset_f"] > best:
            best = ev["onset_f"]
            # The sweep already found where this model's outputs sit. Dropping
            # that and letting playback re-guess at a fixed 0.3 is how a
            # checkpoint that scores 0.31 renders a wall of notes.
            torch.save({"model": model.state_dict(), "config": cfg,
                        "kit": kit.classes, "n_styles": n_styles,
                        "best_threshold": float(ev["best_threshold"]),
                        "rate_ceiling": getattr(model, "rate_ceiling", None)}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))
        save_last(last, model, opt, history, best, train_loader, cfg)

        stop = out / "STOP"
        if ep + 1 < n_epochs and stop.exists():
            stop.unlink()
            print(f"stopped after epoch {ep + 1} of {n_epochs} on request ({stop.name} file); "
                  f"continue with --resume")
            print(describe(estimate([r["onset_f"] for r in history])))
            return 0

    print(f"best val onset F: {best:.4f} -> {out}")
    print(describe(estimate([r["onset_f"] for r in history])))
    if (best >= 0.0 and model.decoder.has_velocity
            and cfg["train"].get("refit_velocity_head", True)):
        refit_best(model, out, train_loader, cfg)
    return 0


def _epoch_line(rec: dict) -> str:
    """One epoch's log line. runstat.ps1 and the queue parse this format."""
    vel = ""
    if not np.isnan(rec.get("velocity_mae", float("nan"))):
        # r, not MAE, is the number that says the head learned dynamics:
        # drummers are not that dynamic, so predicting one constant
        # velocity scores a respectable MAE and a correlation of zero.
        vel = (f" vel mae {rec['velocity_mae']:.3f} r {rec['velocity_r']:+.3f}"
               f" (pooled {rec['velocity_r_pooled']:+.3f})")
    return (f"ep{rec['epoch']:>3} loss {rec['train_loss']:.4f} (bce {rec['train_bce']:.4f}) | "
            f"val onset_F {rec['onset_f']:.4f} @thr {rec['best_threshold']:.2f} "
            f"(fixed {rec['onset_f_fixed']:.4f}) groove {rec['groove_sim']:.3f}"
            f"{vel} | {rec['seconds']}s")


def save_last(path: Path, model, opt, history: list, best: float, loader, cfg) -> None:
    """Everything a run needs to carry on from the end of its latest epoch.

    Weights alone are not enough to resume: AdamW's moment estimates are half
    the optimiser, and restarting them from zero is a different training run.
    Neither is the global RNG: the train loader's shuffle order and its worker
    seeds come off the loader's own generator. With all of it restored, a run
    stopped and resumed matches an unbroken one (tests/test_resume.py). Written
    to a temporary name and swapped in, so a crash mid-write cannot leave a
    truncated file where the last good one was.
    """
    gen = getattr(loader, "generator", None)
    state = {
        "model": model.state_dict(), "opt": opt.state_dict(), "config": cfg,
        "history": history, "best": best, "epochs_done": len(history),
        "rate_ceiling": getattr(model, "rate_ceiling", None),
        "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
                "python": random.getstate(),
                "loader": gen.get_state() if gen is not None else None},
    }
    tmp = path.with_suffix(".pt.tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def resume_from(path: Path, model, opt, loader, cfg, device) -> tuple[int, list, float]:
    """Restore a run from ``last.pt``; return (epochs done, history, best F).

    Refuses a checkpoint from a different config: resuming one run's weights
    under another run's settings would be neither run. ``train.epochs`` is
    exempt, so a finished run can be extended.
    """
    ck = torch.load(path, map_location=device, weights_only=False)
    mine, theirs = json.loads(json.dumps(cfg)), json.loads(json.dumps(ck["config"]))
    mine["train"].pop("epochs", None)
    theirs["train"].pop("epochs", None)
    if mine != theirs:
        raise SystemExit(f"{path} was written by a different config; not resuming. "
                         f"Move it aside to start this run from scratch.")
    model.load_state_dict(ck["model"])
    opt.load_state_dict(ck["opt"])
    model.rate_ceiling = ck.get("rate_ceiling")
    rng = ck["rng"]
    torch.set_rng_state(rng["torch"].cpu())
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])
    gen = getattr(loader, "generator", None)
    if gen is not None and rng.get("loader") is not None:
        gen.set_state(rng["loader"].cpu())
    return int(ck["epochs_done"]), list(ck["history"]), float(ck["best"])


def refit_best(model, out: Path, loader, cfg) -> dict:
    """Write ``best_refit.pt``: ``best.pt`` with its velocity head solved in closed form.

    The SGD head does not converge in a run of this length -- it ends within a
    cosine of 0.99 of its initialisation -- while a closed-form fit of the same
    head on the same frozen network gets classes confidently right that the
    trained head could not (see ``src/refit.py``). So every run with a velocity
    head gets both. ``best.pt`` is left exactly as trained, so every earlier
    comparison and every tool that reads it means what it always did;
    ``best_refit.pt`` is the one to probe, bundle and play. Only the train
    split is used, and ``train.refit_velocity_head: false`` turns this off.
    """
    from refit import refit_velocity_head

    device = next(model.parameters()).device
    ck = torch.load(out / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    seed = int(cfg["train"].get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    t0 = time.time()
    report = refit_velocity_head(model, loader)
    ck["model"] = model.state_dict()
    ck["refit"] = {"source": "best.pt", **report, "seconds": round(time.time() - t0, 1),
                   "when": time.strftime("%Y-%m-%dT%H:%M:%S")}
    torch.save(ck, out / "best_refit.pt")
    fitted = {k: v["train_r"] for k, v in report["classes"].items() if v["refit"]}
    print(f"velocity head refit on {report['clips']} train clips "
          f"({ck['refit']['seconds']}s), train r {fitted} -> {out / 'best_refit.pt'}")
    return report


if __name__ == "__main__":
    raise SystemExit(main())
