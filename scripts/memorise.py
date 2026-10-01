"""Phase 3, step 9: the memorisation check. Train on fixed clips from 4 BabySlakh training songs until
kick, snare and closed hi-hat reach hit F1 >= 0.9 on those same clips. If it cannot, something is broken.
Optimiser, loss, gradient clipping and learning-rate schedule are the locked training ones (the schedule runs over
this check's steps); the fixed clips, step limit and pass rule are this check's own, in locked.yaml's memorisation
section (since prereg-v5). --steps and --songs make a trial run, marked as not the locked check.
Besides F1 it records, for diagnosis: each step's learning rate and gradient size before clipping; at each check,
the size of what the readout receives per drum (next to the untrained network's, in the log's first line); and the
weights as the run ended (final.pt), to re-score or look into without rerunning.
Each run has a name (--name; by default "real", or "control-N" with --control N) and its own folder,
reports/memorise/<name>/: memorise.json, log.jsonl with one line per step (flushed as it goes) and the machine
readings. --pair runs "real" and then control 1 (--pair N: control N) one after the other from one terminal and
prints both results at the end (side by side on one GPU they were slower in total: 17.1 s/step each against 5.4
alone). The log's first line holds the
audio-blind floor: the best loss possible without listening, which the loss has to go clearly below. The starting
weights come from locked.yaml's init_seed, as in train.py, so runs start from the same point.

While it trains, the live chart (scripts/live_chart.py) runs inside it on the first free port from 8765 (it prints
which), and if config/live.local.yaml has upload commands, the run's page on the website (<url>/<name>/) is updated
every minute and once more at the end, however the run ends. --no-chart turns all of that off.
Each step is the recorded one (a CUDA graph, training.GraphedStep) once scripts/check_cuda_graph.py has shown on
this exact code that it trains like the ordinary one; until then, or with --no-graph, the ordinary step. The
first line of the log says which.
Every 250 steps it checks hit F1 on the clips, and with it each drum's precision (share of predicted hits that are
real), recall (share of real hits found) and the counts behind them (found, extra, missed), in the log and the report.

--wide-target (a trial, never the locked check; not part of any preregistration) trains on a wider hit target:
1 on the hit frame, then from 0.5 on the frames either side down in a straight line to the edge of the scoring
tolerance (10% of the local beat, about 14 frames), the larger value where two hits' targets overlap. Drum weights
apply to every frame of it, as they do to the locked target's neighbours. Velocity loss and scoring are unchanged.
Its run is named "<name>-wide-target" by default.
--history-readout (a trial in the same way) turns on the readout history proposed in
reports/history-readout-proposal.md: each drum weighs its motor neurons' present frame and four 50 ms blocks before
it through a learned profile, starting present-only. It uses the recorded step only if the CUDA graph check's
readout-history part passed on this code. Its run is named "<name>-history-readout" by default; with --wide-target too, both apply.

    .venv\\Scripts\\python.exe scripts\\memorise.py [--pair [N] | --control N] [--name NAME] [--steps 3000] [--no-chart] [--no-graph] [--wide-target] [--history-readout]
Uses the GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, reports_dir  # noqa: E402
from flybeats.model.loss import f1_from_counts, hit_counts, tolerance_frames  # noqa: E402
from flybeats.model.training import (GraphedStep, audio_blind_loss, graph_checked, make_optimiser,  # noqa: E402
                                     make_schedule, setup, to_tensors, train_step)
from live_chart import LiveChart  # noqa: E402
from tqdm import tqdm  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=None, help="trial runs only; default and the locked check: locked.yaml")
ap.add_argument("--control", type=int, default=None)
ap.add_argument("--songs", type=int, default=None, help="trial runs only; default and the locked check: locked.yaml")
ap.add_argument("--device", default="cuda")
ap.add_argument("--no-chart", action="store_true", help="no live chart and no website updates")
ap.add_argument("--no-graph", action="store_true", help="the ordinary training step, not the recorded CUDA graph")
ap.add_argument("--name", help='this run\'s name; default "real", or "control-N" with --control N')
ap.add_argument("--pair", type=int, nargs="?", const=1, metavar="N",
                help="run the real network, then control N (default 1), one after the other")
ap.add_argument("--wide-target", action="store_true",
                help="trial only: train on a hit target that spans the scoring tolerance (see the top of this file)")
ap.add_argument("--history-readout", action="store_true",
                help="trial only: the readout history from reports/history-readout-proposal.md")
args = ap.parse_args()


def run_pair(control):
    """Run "real", then "control-N", one after the other in this terminal (side by side they share the GPU and
    together get through fewer steps per hour than one at a time). Each is a copy of this script with its own bar,
    live chart and website page. Ctrl+C stops the run that is going (it sends its last website update) and the
    other is not started. Ends with both results; the exit code is the real run's."""
    import subprocess
    forward = ["--device", args.device] + ["--no-chart"] * args.no_chart + ["--no-graph"] * args.no_graph
    forward += ["--steps", str(args.steps)] * (args.steps is not None) + ["--songs", str(args.songs)] * (args.songs is not None)
    runs = {"real": ["--name", "real"], f"control-{control}": ["--control", str(control)]}
    ended = {}
    for i, (name, extra) in enumerate(runs.items(), 1):
        print(f"\n=== run {i} of {len(runs)}: {name} ===", flush=True)
        started = time.time()
        proc = subprocess.Popen([sys.executable, __file__, *forward, *extra])
        try:
            code = proc.wait()
        except KeyboardInterrupt:           # Ctrl+C reached the run too (same console): let it finish ending
            code = proc.wait()
            ended[name] = (code, started, True)
            break
        ended[name] = (code, started, False)
    print("\n=== results ===")
    for name in runs:
        if name not in ended:
            print(f"{name}: not started")
            continue
        code, started, stopped = ended[name]
        report = reports_dir() / "memorise" / name / "memorise.json"
        if report.exists() and report.stat().st_mtime >= started:
            rep = json.loads(report.read_text(encoding="utf-8"))
            print(f"{name}: {'PASS' if rep['passes'] else 'FAIL'} at step {rep['steps']}, "
                  f"final loss {rep['final_loss']:.4f}, F1 {rep['f1']}")
        else:
            print(f"{name}: {'stopped with Ctrl+C' if stopped else f'ended without a result (exit code {code})'}")
    return ended.get("real", (1,))[0]


if args.pair is not None:
    if args.name or args.control is not None or args.wide_target or args.history_readout:
        ap.error("--pair names its two runs itself; leave out --name, --control, --wide-target, --history-readout")
    sys.exit(run_pair(args.pair))

name = args.name or (("real" if args.control is None else f"control-{args.control}")
                     + "-wide-target" * args.wide_target + "-history-readout" * args.history_readout)
out_dir = reports_dir() / "memorise" / name
out_dir.mkdir(parents=True, exist_ok=True)
cfg = load_locked()
mc = cfg["memorisation"]
if mc["schedule"] != cfg["training"]["schedule"]:
    sys.exit(f"memorisation.schedule {mc['schedule']!r} differs from training.schedule; the check runs training's")
MUST_PASS, PASS_F1 = mc["must_pass"], mc["pass_f1"]
steps = args.steps or mc["steps"]
as_locked = (args.steps is None and args.songs is None and not args.wide_target
             and not args.history_readout)                  # False: a trial run, not the locked check
HISTORY = {"present": True, "blocks_ms": [50, 50, 50, 50], "profile": "per_drum", "init": "present_only",
           "velocity": "same"}                              # as in reports/history-readout-proposal.md
if args.history_readout:
    if cfg["model"].get("readout_history"):
        sys.exit("locked.yaml already sets model.readout_history; --history-readout is for trials before it is locked")
    cfg["model"]["readout_history"] = HISTORY
torch.manual_seed(cfg["model"]["init_seed"])                # the same starting weights every run, as in train.py
s = setup("baby", args.control, args.device, cfg)
model, loss_of, train = s["model"], s["loss"], s["train"]

rng = np.random.default_rng(mc["clip_seed"])
songs = sorted(train.songs.index)[:args.songs or mc["songs"]]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s)))
         for song in songs * mc["clips_per_song"]]
batch = to_tensors(clips, args.device)                      # 8 fixed clips, 2 per song, no stem dropout


def wide_loss(batch):
    """--wide-target's loss: loss.loss_fn with the wider target, built once for the fixed batch. Same signature."""
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
    hits = batch["hits"].cpu().numpy()
    target = np.zeros_like(hits)
    n_clips, n_frames, n_drums = hits.shape
    for i in range(n_clips):
        for d in range(n_drums):
            true = np.flatnonzero(hits[i, :, d])
            for f, tol in zip(true, tolerance_frames(true, batch["beats"][i], frame_s, fraction)):
                k = np.arange(-int(tol), int(tol) + 1)
                v = np.where(k == 0, 1.0, 0.5 * (1 - (np.abs(k) - 1) / max(tol, 1)))
                ok = (f + k >= 0) & (f + k < n_frames)
                target[i, f + k[ok], d] = np.maximum(target[i, f + k[ok], d], v[ok])
    target = torch.tensor(target, device=batch["hits"].device)
    weights, vel_w = s["weights"], cfg["loss"]["velocity_weight"]

    def loss(out, hits, vel, mask, as_tensors=False):
        w = torch.where(target > 0, weights, torch.ones_like(weights)) * mask[..., None]
        bce = F.binary_cross_entropy_with_logits(out["hit_logits"], target, weight=w, reduction="sum") / w.sum()
        on = hits * mask[..., None]
        mse = ((out["vel"] - vel) ** 2 * on).sum() / on.sum().clamp_min(1)
        parts = {"hit_bce": bce.detach(), "vel_mse": mse.detach()}
        return bce + vel_w * mse, parts if as_tensors else {k: float(v) for k, v in parts.items()}
    return loss


if args.wide_target:
    loss_of = wide_loss(batch)
    print("trial: --wide-target, the hit target spans the scoring tolerance; scoring is unchanged")
opt = make_optimiser(model, cfg)
sched = make_schedule(opt, steps, cfg)                      # training's schedule, over this check's steps
floor = audio_blind_loss(batch, loss_of, len(model.pieces))
print(f"audio-blind floor: {floor:.4f} (the loss must go clearly below this)")
if not as_locked:
    print("trial run: --steps or --songs differ from locked.yaml, so this is not the locked check")
use_graph, why = ((False, "turned off (--no-graph)") if args.no_graph else
                  (False, "not on the GPU") if args.device != "cuda" else
                  graph_checked(history_readout=args.history_readout))
graphed = GraphedStep(model, opt, batch, loss_of, cfg["training"]["grad_clip_norm"]) if use_graph else None
print(f"training step: {'recorded (CUDA graph)' if use_graph else 'ordinary'}, {why}")

@torch.no_grad()
def score(model, clips):
    """training.evaluate's hit F1 (the same counts, so the same numbers), plus each drum's precision (share of
    predicted hits that match a real one), recall (share of real hits found) and the counts behind them. Kept here,
    not in training.py, because training.py is part of the code the CUDA graph check vouches for."""
    frame_s = cfg["training_data"]["frame_ms"] / 1000
    fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
    b = to_tensors(clips, args.device)
    out, _ = model(b["audio"])
    counts = hit_counts(out["hit_logits"], b["hits"], b["mask"], b["beats"], frame_s, fraction,
                        cfg["training"]["hit_threshold"])
    tp, fp, fn = counts.T.astype(float)
    ratio = lambda a, n: {p: round(float(x / y), 3) if y > 0 else None for p, x, y in zip(model.pieces, a, n)}  # None: nothing to count
    # What the readout receives: each drum's motor-neuron rates after the fixed scaling, over the scored frames.
    # At setup they are standardised (typical size about 1); growing activity makes them larger.
    z = ((out["mn_rates"] - model.mn_mean) / model.mn_std)[b["mask"] > 0]           # (scored frames, readout neurons)
    readout_inputs = {}
    for d, p in enumerate(model.pieces):
        zd = z[:, model.mn_drum[:, d] > 0]
        readout_inputs[p] = {"rms": round(float(zd.pow(2).mean().sqrt()), 3), "mean": round(float(zd.mean()), 3),
                             "max_abs": round(float(zd.abs().max()), 3)}
    return {"f1": dict(zip(model.pieces, np.round(f1_from_counts(counts), 3).tolist())),
            "precision": ratio(tp, tp + fp), "recall": ratio(tp, tp + fn),
            "hits": {p: {"found": int(t), "extra": int(f), "missed": int(m)} for p, (t, f, m) in zip(model.pieces, counts)},
            "readout_inputs": readout_inputs}


model.eval()
at_start = score(model, clips)["readout_inputs"]              # the untrained network, for comparison with each check
model.train()
log = open(out_dir / "log.jsonl", "w", encoding="utf-8")
log.write(json.dumps({"meta": {"script": "memorise", "name": name, "steps": steps, "audio_blind_floor": floor,
                               "must_pass": MUST_PASS, "pass_f1": PASS_F1, "control": args.control,
                               "version": cfg["version"], "as_locked": as_locked, "cuda_graph": use_graph,
                               "readout_inputs_at_start": at_start, "wide_target": args.wide_target,
                               "history_readout": args.history_readout}}) + "\n")
log.flush()

result, step = None, 0
chart = None if args.no_chart else LiveChart(memorise=name, own_process=True, say=tqdm.write)   # shows this run's log
try:
    t0 = time.time()
    bar = tqdm(range(1, steps + 1), unit="step", desc="memorising")
    for step in bar:
        lr = opt.param_groups[0]["lr"]                            # the rate this step uses
        stats = graphed() if graphed else train_step(model, opt, batch, loss_of, cfg["training"]["grad_clip_norm"])
        sched.step()
        bar.set_postfix(loss=f"{stats['loss']:.4f}")
        rec = {"step": step, "lr": lr, **stats, "elapsed_s": round(time.time() - t0, 1), "t": round(time.time(), 1)}
        if step % mc["check_every"] == 0 or step == steps:
            model.eval()
            result = score(model, clips)
            model.train()
            rec |= {k: result[k] for k in ("f1", "precision", "recall", "hits", "readout_inputs")}
            bar.write(f"step {step}: loss {stats['loss']:.4f}  F1 {result['f1']}")
            bar.write("  precision / recall: " + ", ".join(f"{p} {result['precision'][p]} / {result['recall'][p]}"
                                                          for p in MUST_PASS))
            bar.write("  readout input size (rms; about 1 at setup): " +
                      ", ".join(f"{p} {v['rms']}" for p, v in result["readout_inputs"].items()))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if "f1" in rec and all(result["f1"][p] >= PASS_F1 for p in MUST_PASS):
            break
    bar.close()
    log.close()
    passed = all(result["f1"][p] >= PASS_F1 for p in MUST_PASS)
    report = {"songs": songs, "steps": step, "f1": result["f1"], "precision": result["precision"],
              "recall": result["recall"], "hits": result["hits"], "readout_inputs": result["readout_inputs"],
              "readout_inputs_at_start": at_start, "passes": passed, "control": args.control,
              "audio_blind_floor": floor, "final_loss": stats["loss"], "cuda_graph": use_graph,
              "version": cfg["version"], "as_locked": as_locked, "wide_target": args.wide_target,
              "history_readout": args.history_readout}
    if args.history_readout:                                # each drum's learned time profile, for the write-up
        report["history_profiles"] = {"hit": dict(zip(model.pieces, model.p_hit.detach().cpu().tolist())),
                                      "velocity": dict(zip(model.pieces, model.p_vel.detach().cpu().tolist()))}
    (out_dir / "memorise.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
finally:                        # finished, failed, crashed or Ctrl+C: the website shows how it ended
    if step:                    # the weights as they ended, to re-score or look into without rerunning
        torch.save({"model": model.state_dict(), "step": step, "version": cfg["version"], "name": name},
                   out_dir / "final.pt")
    if chart:
        chart.finish()
print("PASS" if passed else "FAIL: the model cannot memorise; something is broken")
sys.exit(0 if passed else 1)
