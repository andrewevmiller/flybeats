"""Spec 1 (fix 01): calibrate the memorisation gate with two unrestricted models on the CPU.

    .venv\\Scripts\\python.exe scripts\\gate_calibration.py --size small            # one size (for running both at once)
    .venv\\Scripts\\python.exe scripts\\gate_calibration.py --size large --publish  # also updates the website page
    .venv\\Scripts\\python.exe scripts\\gate_calibration.py                          # both sizes, one after the other
Options: --threads N (CPU threads per size, default 4), --no-chart (no live chart), --trial (a few steps; writes
nothing). Two sizes at once, one per terminal, finish in about the time of the slower one: 4 threads each.

RULES, fixed in this file before the run (change them here before running, never after):
  Data       The memorisation check's own fixed clips: scripts/memorise.py's 8 clips (locked.yaml memorisation section,
             from the BabySlakh training songs), drawn by flybeats.data.sampler.fixed_clips.
  Models     flybeats.model.ear -> 2-layer GRU -> a hit logit and a velocity per drum, no connectome.
             small: 64 units per layer (about 41,000 parameters, the real network's dial count).
             large: 128 units per layer (about 160,000 parameters).
  Training   the locked loss, Adam at training.learning_rate, cosine to zero over memorisation.steps, gradient
             clipping at training.grad_clip_norm, no stem dropout. Same initial weights for both sizes (init_seed).
  Score      the locked hit F1 at every memorisation.check_every steps, as memorise.py does (the same counts):
             a check passes when kick, snare and closed hi-hat each reach memorisation.pass_f1.
  Verdict    small passes at some check:  the gate is attainable at this size.
             only large passes:            the gate is a capacity test at this size.
             neither passes:               the gate is mis-set for a play-along task; Andrew decides.

Writes reports/gate_calibration/<size>/ (log.jsonl, result.json) and, once both sizes have a result,
reports/gate_calibration.json with the verdict. Changes nothing else.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler, fixed_clips  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model import ear as model_ear  # noqa: E402
from flybeats.model.loss import (drum_weights, f1_from_counts, hit_counts, make_loss, match_counts,  # noqa: E402
                                 peaks, tolerance_frames)
from flybeats.model.training import audio_blind_loss, to_tensors  # noqa: E402

SIZES = {"small": 64, "large": 128}
LAYERS = 2
SET = "baby"                                  # the memorisation check's set
DRUMS = ["kick", "snare", "hihat_closed"]

ap = argparse.ArgumentParser()
ap.add_argument("--size", choices=["small", "large", "both"], default="both")
ap.add_argument("--threads", type=int, default=4, help="CPU threads (only with --device cpu)")
ap.add_argument("--device", default="cuda", help="cuda (the default, the RTX 2060) or cpu")
ap.add_argument("--publish", action="store_true", help="also update the website page (one size only)")
ap.add_argument("--no-chart", action="store_true", help="no live chart and no website updates")
ap.add_argument("--trial", action="store_true", help="a few steps to check the script; writes nothing")
ap.add_argument("--loss-weight", choices=["ratio", "sqrt"], default="ratio",
                help="sqrt: the square-root drum weights (Fix 04 arm); output goes to <size>-loss-sqrt")
ap.add_argument("--gap", type=int, default=10, help="minimum gap in frames for the min-gap score (10 = 50 ms)")
args = ap.parse_args()
if args.publish and args.size == "both":
    ap.error("--publish needs one size (the website shows one run at a time): --size small or --size large")
if args.device == "cpu":
    torch.set_num_threads(args.threads)


class GateModel(nn.Module):
    """The ear, then a causal GRU, then one hit logit and one velocity per drum. Same ear and heads as feasibility.py."""

    def __init__(self, hidden, n_drums, mean, std, hit_bias):
        super().__init__()
        self.ear = model_ear.Bands(mean, std)
        self.gru = nn.GRU(model_ear.N_BANDS, hidden, LAYERS, batch_first=True)
        self.hit = nn.Linear(hidden, n_drums)
        self.vel = nn.Linear(hidden, n_drums)
        nn.init.constant_(self.hit.bias, hit_bias)

    def forward(self, audio):
        h, _ = self.gru(self.ear(audio))
        return {"hit_logits": self.hit(h), "vel": torch.sigmoid(self.vel(h))}


@torch.no_grad()
def peak_stats(hit_logits, batch, cfg):
    """Peaks above the threshold per true hit, per drum: a hit inside two peaks' tolerance is one extra false hit
    (the preregistered second-peak problem). hits_with_0/1/2plus count the true hits by how many peaks they get."""
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
    thr = cfg["training"]["hit_threshold"]
    prob = torch.sigmoid(hit_logits).cpu().numpy()
    hits, mask = batch["hits"].cpu().numpy(), batch["mask"].cpu().numpy()
    out = {}
    for d, name in enumerate(cfg["kit"]["pieces"]):
        n_peaks = n_true = 0
        per_hit = []
        for i in range(prob.shape[0]):
            live = np.flatnonzero(mask[i])
            pred = np.intersect1d(peaks(prob[i, :, d], thr), live)
            true = np.intersect1d(np.flatnonzero(hits[i, :, d]), live)
            n_peaks += len(pred)
            n_true += len(true)
            if len(true):
                tol = tolerance_frames(true, batch["beats"][i], frame_s, fraction)
                per_hit += [int(np.sum(np.abs(pred - t) <= tl)) for t, tl in zip(true, tol)]
        ph = np.array(per_hit)
        out[name] = {"peaks": n_peaks, "true_hits": n_true,
                     "peaks_per_hit": round(float(ph.mean()), 2) if len(ph) else None,
                     "hits_with_0": int((ph == 0).sum()), "hits_with_1": int((ph == 1).sum()),
                     "hits_with_2plus": int((ph >= 2).sum())}
    return out


def min_gap_peaks(p, threshold, gap):
    """Locked peaks, highest first; a peak within `gap` frames of a kept one is dropped (as min_gap_rescore.py)."""
    pk = peaks(p, threshold)
    if gap <= 0 or len(pk) < 2:
        return pk
    kept = []
    for q in pk[np.argsort(-p[pk], kind="stable")]:
        if all(abs(int(q) - k) > gap for k in kept):
            kept.append(int(q))
    return np.sort(np.array(kept, dtype=pk.dtype))


@torch.no_grad()
def min_gap_f1(hit_logits, batch, cfg):
    """Per-drum hit F1 with the min-gap peak rule. Scoring only: the locked check uses the locked rule."""
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
    thr = cfg["training"]["hit_threshold"]
    prob = torch.sigmoid(hit_logits).cpu().numpy()
    hits, mask = batch["hits"].cpu().numpy(), batch["mask"].cpu().numpy()
    pieces = cfg["kit"]["pieces"]
    tp = np.zeros(len(pieces)); fp = np.zeros(len(pieces)); fn = np.zeros(len(pieces))
    for i in range(prob.shape[0]):
        live = np.flatnonzero(mask[i])
        for d in range(len(pieces)):
            pred = np.intersect1d(min_gap_peaks(prob[i, :, d], thr, args.gap), live)
            true = np.intersect1d(np.flatnonzero(hits[i, :, d]), live)
            tol = tolerance_frames(true, batch["beats"][i], frame_s, fraction)
            a, f, m = match_counts(pred, true, tol)
            tp[d] += a; fp[d] += f; fn[d] += m
    return {p: round(float(2 * tp[d] / max(2 * tp[d] + fp[d] + fn[d], 1)), 3) for d, p in enumerate(pieces)}


@torch.no_grad()
def score(model, batch, cfg):
    """Per-drum hit F1 with the locked counts (as memorise.py's score), plus precision, recall and the counts behind
    them, so a low F1 can be read as missed hits or extra hits (re-examining the gate)."""
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
    pieces = cfg["kit"]["pieces"]
    model.eval()
    out = model(batch["audio"])
    counts = hit_counts(out["hit_logits"], batch["hits"], batch["mask"], batch["beats"], frame_s, fraction,
                        cfg["training"]["hit_threshold"])
    model.train()
    tp, fp, fn = counts.T.astype(float)
    ratio = lambda a, n: {p: round(float(x / y), 3) if y > 0 else None for p, x, y in zip(pieces, a, n)}
    return {"f1": dict(zip(pieces, np.round(f1_from_counts(counts), 3).tolist())),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "hits": {p: {"found": int(t), "extra": int(f), "missed": int(m)} for p, (t, f, m) in zip(pieces, counts)},
            "peaks": peak_stats(out["hit_logits"], batch, cfg),
            "f1_min_gap": min_gap_f1(out["hit_logits"], batch, cfg)}


def arm_name(size):
    """Output folder. The earlier results stay in the plain folders (small, large); this re-run writes beside them."""
    return size + ("-loss-sqrt" if args.loss_weight == "sqrt" else "-readouts")


def run(size, cfg, batch, loss_of, mean, std, floor, songs, steps, every):
    hidden = SIZES[size]
    torch.manual_seed(cfg["model"]["init_seed"])
    model = GateModel(hidden, len(cfg["kit"]["pieces"]), mean, std, cfg["model"]["hit_bias_init"]).to(args.device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=cfg["training"]["learning_rate"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    must, pass_f1 = cfg["memorisation"]["must_pass"], cfg["memorisation"]["pass_f1"]
    subtitle = (f"A gate calibration: a {size} network ({hidden} units per layer, {n_params:,} parameters) practises "
                f"on the same 8 drum clips as the memorisation check, with no connectome.")
    log = None
    if not args.trial:
        out_dir = reports_dir() / "gate_calibration" / arm_name(size)
        out_dir.mkdir(parents=True, exist_ok=True)
        log = open(out_dir / "log.jsonl", "w", encoding="utf-8")
        log.write(json.dumps({"meta": {"size": size, "hidden": hidden, "layers": LAYERS, "parameters": n_params,
                                       "steps": steps, "check_every": every, "must_pass": must, "pass_f1": pass_f1,
                                       "loss_weight": args.loss_weight, "gap_frames": args.gap,
                                       "audio_blind_floor": floor, "device": args.device, "title": f"Gate calibration · {arm_name(size)} ({hidden} units)",
                                       "subtitle": subtitle, "version": cfg["version"]}}) + "\n")
        log.flush()
    chart = None
    if not args.trial and not args.no_chart:
        from live_chart import LiveChart   # the chart reads this run's log; publishes only when asked (--publish)
        chart = LiveChart(gate=size, publish=args.publish, own_process=True, say=print)
    checks, t0 = [], time.time()
    try:
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True)
            out = model(batch["audio"])
            loss, _ = loss_of(out, batch["hits"], batch["vel"], batch["mask"])
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip_norm"])
            opt.step()
            sched.step()
            rec = {"step": step, "loss": float(loss.detach()), "grad_norm": float(grad_norm),
                   "elapsed_s": round(time.time() - t0, 1), "t": round(time.time(), 1)}
            if step % every == 0 or step == steps:
                res = score(model, batch, cfg)
                f1 = res["f1"]
                ok = all(f1[p] >= pass_f1 for p in must)
                checks.append({"step": step, "f1": f1, "f1_min_gap": res["f1_min_gap"], "passes": ok,
                               "precision": res["precision"], "recall": res["recall"], "hits": res["hits"]})
                rec |= res
                print(f"[{size}] step {step}: loss {rec['loss']:.4f}  F1 {f1}  {'PASS' if ok else ''}", flush=True)
            if log is not None:
                log.write(json.dumps(rec) + "\n")
                log.flush()
    finally:
        if log is not None:
            log.close()
        if chart is not None:
            chart.finish()
    result = {"size": size, "hidden": hidden, "parameters": n_params, "steps": steps, "songs": songs,
              "passes_at_any_check": any(c["passes"] for c in checks),
              "best_check": max(checks, key=lambda c: min(c["f1"][p] for p in must)), "final": checks[-1],
              "audio_blind_floor": floor, "seconds": round(time.time() - t0)}
    if not args.trial:
        (reports_dir() / "gate_calibration" / arm_name(size) / "result.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8")
        torch.save({"model": model.state_dict(), "step": steps},
                   reports_dir() / "gate_calibration" / arm_name(size) / "final.pt")   # kept: the only copy
    return result


def combine(cfg, steps, songs, floor):
    """Once both sizes have a result: the verdict, in reports/gate_calibration.json."""
    folder = reports_dir() / "gate_calibration"
    results = {}
    for size in SIZES:
        p = folder / size / "result.json"
        if p.exists():
            results[size] = json.loads(p.read_text(encoding="utf-8"))
    if set(results) != set(SIZES):
        print(f"waiting for the other size: have {sorted(results)}")
        return
    if results["small"]["passes_at_any_check"]:
        verdict = "attainable at this size: the small model reaches the gate"
    elif results["large"]["passes_at_any_check"]:
        verdict = "capacity test at this size: only the large model reaches the gate"
    else:
        verdict = "mis-set: neither model reaches the gate; Andrew decides"
    report = {"version": cfg["version"], "set": SET, "songs": songs, "audio_blind_floor": floor,
              "drums": DRUMS, "pass_f1": cfg["memorisation"]["pass_f1"], "must_pass": cfg["memorisation"]["must_pass"],
              "steps": steps, "results": results, "verdict": verdict}
    out = reports_dir() / "gate_calibration.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verdict: {verdict}\nwrote {out}")


cfg, paths = load_locked(), load_paths()
mc = cfg["memorisation"]
steps, every = (20, 10) if args.trial else (mc["steps"], mc["check_every"])
manifest, events, beats = load_manifest(paths["work_dir"], SET)
root = unpacked_root(paths["work_dir"], SET)
train = ClipSampler(root, manifest, events, beats, cfg, "train")
songs, clips = fixed_clips(train, mc["songs"], mc["clips_per_song"], mc["clip_seed"])
batch = to_tensors(clips, args.device)
frame_s = cfg["training_data"]["frame_ms"] / 1000
weights = drum_weights(manifest, cfg["kit"]["pieces"], frame_s, cfg["loss"]["max_drum_weight"],
                       mode=args.loss_weight).to(args.device)
loss_of = make_loss(cfg, weights)
mean, std = model_ear.load_band_stats(set_dir(paths["work_dir"], SET) / "band_stats.json")
floor = audio_blind_loss(batch, loss_of, len(cfg["kit"]["pieces"]))
print(f"clips: {songs}; audio-blind floor {floor:.4f}; {steps} steps, check every {every}; "
      f"{args.device}" + (f", {args.threads} threads" if args.device == "cpu" else ""), flush=True)

for size in (SIZES if args.size == "both" else [args.size]):
    run(size, cfg, batch, loss_of, mean, std, floor, songs, steps, every)
if not args.trial and args.size == "both":
    print("the verdict in reports/gate_calibration.json is from the earlier plain-folder results; not rewritten here")
else:
    print("trial: nothing written")
