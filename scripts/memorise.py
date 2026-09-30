"""Phase 3, step 9: the memorisation check. Train on fixed clips from 4 BabySlakh training songs until
kick, snare and closed hi-hat reach hit F1 >= 0.9 on those same clips. If it cannot, something is broken.
Optimiser, loss and gradient clipping are the locked training ones; the fixed clips and step limit are this
check's own.
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

    .venv\\Scripts\\python.exe scripts\\memorise.py [--pair [N] | --control N] [--name NAME] [--steps 3000] [--no-chart] [--no-graph]
Uses the GPU.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, reports_dir  # noqa: E402
from flybeats.model.training import (GraphedStep, audio_blind_loss, evaluate, graph_checked, make_optimiser,  # noqa: E402
                                     setup, to_tensors, train_step)
from live_chart import LiveChart  # noqa: E402
from tqdm import tqdm  # noqa: E402

MUST_PASS = ["kick", "snare", "hihat_closed"]

ap = argparse.ArgumentParser()
ap.add_argument("--steps", type=int, default=3000)
ap.add_argument("--control", type=int, default=None)
ap.add_argument("--songs", type=int, default=4)
ap.add_argument("--device", default="cuda")
ap.add_argument("--no-chart", action="store_true", help="no live chart and no website updates")
ap.add_argument("--no-graph", action="store_true", help="the ordinary training step, not the recorded CUDA graph")
ap.add_argument("--name", help='this run\'s name; default "real", or "control-N" with --control N')
ap.add_argument("--pair", type=int, nargs="?", const=1, metavar="N",
                help="run the real network, then control N (default 1), one after the other")
args = ap.parse_args()


def run_pair(control):
    """Run "real", then "control-N", one after the other in this terminal (side by side they share the GPU and
    together get through fewer steps per hour than one at a time). Each is a copy of this script with its own bar,
    live chart and website page. Ctrl+C stops the run that is going (it sends its last website update) and the
    other is not started. Ends with both results; the exit code is the real run's."""
    import subprocess
    forward = ["--steps", str(args.steps), "--songs", str(args.songs), "--device", args.device]
    forward += ["--no-chart"] * args.no_chart + ["--no-graph"] * args.no_graph
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
    if args.name or args.control is not None:
        ap.error("--pair names its two runs itself; leave out --name and --control")
    sys.exit(run_pair(args.pair))

name = args.name or ("real" if args.control is None else f"control-{args.control}")
out_dir = reports_dir() / "memorise" / name
out_dir.mkdir(parents=True, exist_ok=True)
cfg = load_locked()
torch.manual_seed(cfg["model"]["init_seed"])                # the same starting weights every run, as in train.py
s = setup("baby", args.control, args.device, cfg)
model, loss_of, train = s["model"], s["loss"], s["train"]

rng = np.random.default_rng(7)
songs = sorted(train.songs.index)[:args.songs]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s))) for song in songs * 2]
batch = to_tensors(clips, args.device)                      # 8 fixed clips, 2 per song, no stem dropout
opt = make_optimiser(model, cfg)
floor = audio_blind_loss(batch, loss_of, len(model.pieces))
print(f"audio-blind floor: {floor:.4f} (the loss must go clearly below this)")
use_graph, why = ((False, "turned off (--no-graph)") if args.no_graph else
                  (False, "not on the GPU") if args.device != "cuda" else graph_checked())
graphed = GraphedStep(model, opt, batch, loss_of, cfg["training"]["grad_clip_norm"]) if use_graph else None
print(f"training step: {'recorded (CUDA graph)' if use_graph else 'ordinary'}, {why}")
log = open(out_dir / "log.jsonl", "w", encoding="utf-8")
log.write(json.dumps({"meta": {"script": "memorise", "name": name, "steps": args.steps, "audio_blind_floor": floor,
                               "must_pass": MUST_PASS, "control": args.control, "version": cfg["version"],
                               "cuda_graph": use_graph}}) + "\n")
log.flush()

result = None
chart = None if args.no_chart else LiveChart(memorise=name, own_process=True, say=tqdm.write)   # shows this run's log
try:
    t0 = time.time()
    bar = tqdm(range(1, args.steps + 1), unit="step", desc="memorising")
    for step in bar:
        stats = graphed() if graphed else train_step(model, opt, batch, loss_of, cfg["training"]["grad_clip_norm"])
        bar.set_postfix(loss=f"{stats['loss']:.4f}")
        rec = {"step": step, **stats, "elapsed_s": round(time.time() - t0, 1), "t": round(time.time(), 1)}
        if step % 250 == 0 or step == args.steps:
            model.eval()
            result = evaluate(model, clips, loss_of, cfg)
            model.train()
            rec["f1"] = result["f1"]
            bar.write(f"step {step}: loss {stats['loss']:.4f}  F1 {result['f1']}")
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if "f1" in rec and all(result["f1"][p] >= 0.9 for p in MUST_PASS):
            break
    bar.close()
    log.close()
    passed = all(result["f1"][p] >= 0.9 for p in MUST_PASS)
    report = {"songs": songs, "steps": step, "f1": result["f1"], "passes": passed, "control": args.control,
              "audio_blind_floor": floor, "final_loss": stats["loss"], "cuda_graph": use_graph}
    (out_dir / "memorise.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
finally:                        # finished, failed, crashed or Ctrl+C: the website shows how it ended
    if chart:
        chart.finish()
print("PASS" if passed else "FAIL: the model cannot memorise; something is broken")
sys.exit(0 if passed else 1)
