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
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from build import build_model, device_of, get_subgraph, load_config
from dataset import build_dataset
from metrics import beat_alignment_error, groove_similarity, onset_f_measure

ROOT = Path(__file__).resolve().parents[1]


def onset_loss(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    """Per-class BCE. Onsets are rare, so positives are up-weighted."""
    pw = torch.full((logits.shape[-1],), pos_weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)


def rate_penalty(rates: torch.Tensor, target_hz: float, step_ms: float) -> torch.Tensor:
    """Keep mean firing near a target. Without this the softplus units drift up
    until everything saturates and the readout sees a constant."""
    if rates is None:
        return torch.zeros((), device="cpu")
    target = target_hz * step_ms / 1000.0
    return (rates.mean() - target).pow(2)


def run_epoch(model, loader, opt, cfg, device, train: bool = True, use_genre: bool = True):
    model.train(train)
    tb = cfg["train"]
    chunk = int(tb.get("tbptt_steps", 150))
    amp = bool(tb.get("bf16", False)) and device.type == "cuda"
    ckpt = bool(tb.get("grad_checkpoint", False)) and train
    totals = {"loss": 0.0, "bce": 0.0, "rate": 0.0, "n": 0}

    for wav, y, style, _tempo in loader:
        wav, y = wav.to(device), y.to(device)
        style = style.to(device) if (use_genre and model.genre is not None) else None

        state = model.rnn.initial_state(wav.shape[0], device)
        steps = min(int(wav.shape[-1] // model.encoder.hop), y.shape[1])
        n_chunks = 0
        if train:
            opt.zero_grad(set_to_none=True)
        n_total = max(1, (steps + chunk - 1) // chunk)

        for a in range(0, steps, chunk):
            b = min(a + chunk, steps)
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
                reg = rate_penalty(rates.float(), tb.get("target_rate_hz", 5.0),
                                   cfg["audio"]["step_ms"])
                loss = bce + tb.get("rate_weight", 0.1) * reg

            if train:
                (loss / n_total).backward()
            totals["loss"] += float(loss.detach())
            totals["bce"] += float(bce.detach())
            totals["rate"] += float(reg.detach())
            n_chunks += 1

        if train:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tb.get("grad_clip", 1.0))
            opt.step()
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
    sweep = cfg["eval"].get("threshold_sweep", [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
    if fixed not in sweep:
        sweep = sorted(sweep + [fixed])

    by_thr = {t: [] for t in sweep}
    ba_all, gs_all, dev_all = [], [], []

    for wav, y, style, tempo in loader:
        wav = wav.to(device)
        sid = style.to(device) if (use_genre and model.genre is not None) else None
        logits, _ = model(wav, style_id=sid)
        prob = torch.sigmoid(logits.float()).cpu().numpy()
        ref = y.numpy()
        steps = min(prob.shape[1], ref.shape[1])
        for i in range(prob.shape[0]):
            for t in sweep:
                m = onset_f_measure(prob[i, :steps], ref[i, :steps], step_ms,
                                    tolerance_s=tol, threshold=t)
                by_thr[t].append(m["f_measure"])
                if t == fixed and not np.isnan(m["mean_dev_ms"]):
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
    return {
        "onset_f": means[best_t],
        "onset_f_fixed": means[fixed],
        "best_threshold": best_t,
        "beat_align_ms": float(np.mean(ba_all)) if ba_all else float("nan"),
        "groove_sim": float(np.mean(gs_all)) if gs_all else float("nan"),
        "mean_dev_ms": float(np.mean(dev_all)) if dev_all else float("nan"),
        "n_clips": len(by_thr[fixed]),
    }


def build_loaders(cfg, kit):
    dcfg = dict(cfg["data"])
    dcfg.setdefault("step_ms", cfg["audio"]["step_ms"])
    dcfg.setdefault("sample_rate", cfg["audio"]["sample_rate"])
    tr = build_dataset(dcfg, kit.classes, "train")
    va = build_dataset(dcfg, kit.classes, cfg["data"].get("val_split", "validation"))
    bs = cfg["train"].get("batch_size", 4)
    return (
        DataLoader(tr, batch_size=bs, shuffle=True, num_workers=cfg["train"].get("workers", 0),
                   drop_last=True),
        DataLoader(va, batch_size=bs, shuffle=False, num_workers=cfg["train"].get("workers", 0)),
        int(getattr(tr, "n_styles", 1)),
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece.yaml")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--rebuild-subgraph", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny run to prove the pipeline")
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

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["train"].get("lr", 3e-3), weight_decay=cfg["train"].get("weight_decay", 0.0),
    )
    out = a.out or ROOT / "runs" / a.config.stem
    out.mkdir(parents=True, exist_ok=True)

    history, best = [], -1.0
    for ep in range(cfg["train"].get("epochs", 10)):
        t0 = time.time()
        tr = run_epoch(model, train_loader, opt, cfg, device, train=True)
        ev = evaluate(model, val_loader, cfg, device)
        rec = {"epoch": ep, **{f"train_{k}": v for k, v in tr.items()}, **ev,
               "seconds": round(time.time() - t0, 1)}
        history.append(rec)
        print(f"ep{ep:>3} loss {tr['loss']:.4f} (bce {tr['bce']:.4f}) | "
              f"val onset_F {ev['onset_f']:.4f} @thr {ev['best_threshold']:.2f} "
              f"(fixed {ev['onset_f_fixed']:.4f}) groove {ev['groove_sim']:.3f} | "
              f"{rec['seconds']}s")

        if ev["onset_f"] > best:
            best = ev["onset_f"]
            torch.save({"model": model.state_dict(), "config": cfg,
                        "kit": kit.classes, "n_styles": n_styles}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(history, indent=2))

    print(f"best val onset F: {best:.4f} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
