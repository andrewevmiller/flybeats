"""Locked win-rule scoring (prereg-v5.4): the real network and each random control, scored with the locked scorecard
(prereg-v5.2) on one split, then the main win rule (prereg-v5.1) and the supporting rule (prereg-v5.2).

    .venv\\Scripts\\python.exe scripts\\score_win.py --split validation            (check only)
    .venv\\Scripts\\python.exe scripts\\score_win.py --split test --draw           (test: draws test clips once, then scores)

--split test is the one-time locked evaluation (locked.yaml win_rule.evaluated_on: test_split_once). The test clips are
drawn once into data/work/data/baby/test_clips.json (4 windows per song, 16 s, numpy seed --draw-seed) and refuse to be
redrawn. Runs on the CPU, so it does not compete with a GPU training run. Writes reports/win_<split>.json.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler, load_validation_clips  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model.training import setup  # noqa: E402
from flybeats.scoring.scorecard import (COUNTED, FACETS, Song, aggregate, balance, check_scorecard,  # noqa: E402
                                        hits_from_probs, supporting_win)
from flybeats.scoring.win_rule import main_win  # noqa: E402

# Machine readings (CPU and GPU usage, temperature, power, memory), from scripts/machine.py: set by the run that uses it.
MACHINE = None
# Website progress page for scoring runs. Its own folder and file names, so it never touches the training chart.
# The upload command and the site address come from config/live.local.yaml (the same settings as the training chart).
SITE_DIR = ROOT / "runs" / "_live_score"
STATE = {"title": "", "status": "starting", "updated": "", "networks": {}, "main_win": None, "supporting": None}
UPLOAD, SITE_URL = None, None
PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flybeats scoring</title><style>
:root{--bg:#fbfaf7;--fg:#1d1d1b;--mut:#6b6a65;--line:#d9d6ce;--ok:#2d6a4f;--bad:#9b2226}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#1b1b19;--fg:#ecebe6;--mut:#9c9a93;--line:#3a3a36;--ok:#7fc8a0;--bad:#e07a7a}}
:root[data-theme="dark"]{--bg:#1b1b19;--fg:#ecebe6;--mut:#9c9a93;--line:#3a3a36;--ok:#7fc8a0;--bad:#e07a7a}
body{margin:0;padding:16px;background:var(--bg);color:var(--fg);font:16px/1.5 system-ui,sans-serif}
h1{font-size:20px;margin:0 0 6px}p{color:var(--mut);margin:4px 0 12px}
table{width:100%;border-collapse:collapse;font-size:15px}td,th{padding:6px 4px;border-bottom:1px solid var(--line);text-align:left}
th{color:var(--mut);font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--ok)}.bad{color:var(--bad)}
</style></head><body>
<h1 id="t">Scoring</h1><p id="s"></p><table><thead><tr><th>Network</th><th class="num">Step</th><th class="num">Beat alignment</th><th class="num">Songs</th></tr></thead><tbody id="rows"></tbody></table>
<p id="w"></p>
<script>
function esc(x){return String(x).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
fetch('score.json?'+Date.now(),{cache:'no-store'}).then(r=>r.json()).then(d=>{
 document.getElementById('t').textContent=d.title||'Scoring';
 document.getElementById('s').textContent='Status: '+d.status+' · updated '+d.updated;
 const rows=Object.entries(d.networks||{}).map(([n,v])=>'<tr><td>'+esc(n)+'</td><td class="num">'+esc(v.step)+'</td><td class="num">'+Number(v.beat_alignment).toFixed(3)+'</td><td class="num">'+esc(v.songs)+'</td></tr>');
 document.getElementById('rows').innerHTML=rows.join('');
 let w='';
 if(d.main_win){w='Main win rule: <b class="'+(d.main_win.wins?'ok':'bad')+'">'+(d.main_win.wins?'met':'not met')+'</b> (real '+Number(d.main_win.real).toFixed(3)+', best control '+Number(d.main_win.best_control).toFixed(3)+', spread '+Number(d.main_win.seed_spread).toFixed(3)+')';}
 if(d.error){w+=' <b class="bad">Error: '+esc(d.error)+'</b>';}
 document.getElementById('w').innerHTML=w;
}).catch(()=>{document.getElementById('s').textContent='Could not load the scoring data yet.';});
</script></body></html>"""


def load_site():
    """Upload command and address from config/live.local.yaml; empty if the file is missing."""
    import yaml
    p = ROOT / "config" / "live.local.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {} if p.exists() else {}


def push(say=print):
    """Write the progress page and its data, then upload the folder. Returns (ok, why)."""
    SITE_DIR.mkdir(parents=True, exist_ok=True)
    STATE["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if MACHINE is not None and MACHINE.latest:
        s = MACHINE.latest
        c, g, r = s["cpu"], s["gpu"] or {}, s["run"]
        STATE["machine"] = {"cpu_usage_pct": c.get("usage_pct"), "cpu_clock_mhz": c.get("clock_mhz"),
                            "cpu_temp_c": c.get("temp_c"), "cpu_power_w": c.get("power_w"),
                            "gpu_name": g.get("name"), "gpu_usage_pct": g.get("usage_pct"),
                            "gpu_clock_mhz": g.get("clock_mhz"), "gpu_temp_c": g.get("temp_c"),
                            "gpu_power_w": g.get("power_w"), "gpu_mem_used_mb": g.get("mem_used_mb"),
                            "gpu_mem_total_mb": g.get("mem_total_mb"), "run_cpu_pct": r.get("cpu_pct"),
                            "run_gpu_pct": r.get("gpu_pct"), "run_ram_mb": r.get("ram_mb"),
                            "run_gpu_mem_mb": r.get("gpu_mem_mb"), "reading_taken": time.strftime("%H:%M:%S")}
    (SITE_DIR / "score.json").write_text(json.dumps(STATE, indent=1, default=float), encoding="utf-8")
    (SITE_DIR / "score.html").write_text((ROOT / "scripts" / "live_score.html").read_text(encoding="utf-8"),
                                         encoding="utf-8")
    if not UPLOAD:
        say("scoring page written locally only (no upload command in config/live.local.yaml)")
        return None, "no upload command"
    cmd = UPLOAD.replace("{dir}", f'"{SITE_DIR}"').replace("{name}", "scoring")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
    ok = r.returncode == 0
    say(f"scoring page upload {'working' if ok else 'FAILED: ' + r.stderr[-200:]}"
        + (f" - {SITE_URL}score.html" if ok and SITE_URL else ""))
    return ok, r.stderr[-200:]


# network name -> (run folder, control seed or None). Controls use the seeds in locked.yaml, in its order.
RUNS = {"real": ("v53_real", None), "control_1": ("v53_control1", 1), "control_2": ("v53_control2", 2),
        "control_3": ("v53_control3", 3), "control_5": ("v53_control5", 5), "control_6": ("v54_control6", 6)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["validation", "test"])
    ap.add_argument("--draw", action="store_true", help="draw test_clips.json once (refused if it exists)")
    ap.add_argument("--draw-seed", type=int, default=2026)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--set", default="baby", choices=["baby", "full"],
                    help="data set the split's songs come from; the networks always load the baby setup, as trained")
    args = ap.parse_args()

    cfg, paths = load_locked(), load_paths()
    check_scorecard(cfg)
    td, sc = cfg["training_data"], cfg["scorecard"]
    work = paths["work_dir"]
    manifest, events, beats = load_manifest(work, args.set)
    root = unpacked_root(work, args.set)
    pieces = cfg["kit"]["pieces"]
    piece_idx = {p: i for i, p in enumerate(pieces)}
    frame_s = td["frame_ms"] / 1000
    first, last = round(td["warmup_ignored_seconds"] / frame_s), round(td["clip_seconds"] / frame_s)
    threshold = cfg["training"]["hit_threshold"]

    split_rows = manifest[manifest.split == args.split].set_index("song").sort_index()
    names = list(split_rows.index)
    if args.split == "validation":
        clips_path = set_dir(work, args.set) / "validation_clips.json"
        clips = load_validation_clips(clips_path)
    else:
        clips_path = set_dir(work, args.set) / "test_clips.json"
        if args.draw:
            if clips_path.exists():
                sys.exit(f"{clips_path} exists; test clips are drawn once")
            rng = np.random.default_rng(args.draw_seed)
            clips = [{"song": s, "start": round(float(rng.uniform(0, split_rows.duration[s] - td["clip_seconds"])), 3)}
                     for s in names for _ in range(sc["test_clips"]["per_song"])]
            clips_path.write_text(json.dumps(clips, indent=1), encoding="utf-8")
            print(f"drew {len(clips)} test clips into {clips_path} (seed {args.draw_seed})", flush=True)
        clips = load_validation_clips(clips_path)
    print(f"{args.split}: {len(names)} songs, {len(clips)} clips", flush=True)
    global UPLOAD, SITE_URL
    site = load_site()
    UPLOAD, SITE_URL = site.get("upload"), site.get("url")
    STATE.update(title=f"Flybeats scoring: {args.split} on {args.set} ({len(names)} songs, {len(clips)} clips)",
                 status="running")
    push()

    # per-song reference parts: the floor is the next song in the split by name, wrapping
    sampler = ClipSampler(root, manifest, events, beats, cfg, args.split)
    ev_by = {s: g for s, g in events.groupby("song")}
    bt_by = {s: g for s, g in beats.groupby("song")}
    song_objs = {}
    for i, s in enumerate(names):
        floor = names[(i + 1) % len(names)]
        ev = ev_by.get(s)
        fev = ev_by.get(floor)
        fbt = bt_by[floor]
        bt = bt_by[s]
        starts = [c["start"] for c in clips if c["song"] == s]
        if not starts:
            continue
        et = ev.time.to_numpy() if ev is not None else np.zeros(0)
        ep = np.array([piece_idx[p] for p in ev.piece]) if ev is not None else np.zeros(0, int)
        ft = fev.time.to_numpy() if fev is not None else np.zeros(0)
        fp = np.array([piece_idx[p] for p in fev.piece]) if fev is not None else np.zeros(0, int)
        song_objs[s] = Song(starts, bt.time.to_numpy(), bt.time[bt.downbeat].to_numpy(), (et, ep), (ft, fp),
                            fbt.time.to_numpy(), float(split_rows.duration[floor]), cfg)
    # the same clip audio the network hears, from the sampler (no stem dropout)
    audio = {(c["song"], c["start"]): sampler.clip(c["song"], c["start"])["audio"] for c in clips if c["song"] in song_objs}

    results = {}
    for name, (run, seed) in RUNS.items():
        torch.manual_seed(cfg["model"]["init_seed"])
        s = setup("baby", seed, args.device, cfg)  # the setup each network was trained under
        model = s["model"]
        ck = torch.load(ROOT / "runs" / run / "best.pt", map_location=args.device, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        model.gain_scale = 1.0
        per_song, detail = {}, {}
        for song, obj in song_objs.items():
            played = []
            with torch.no_grad():
                for st in obj.starts:
                    out, _ = model(torch.as_tensor(audio[(song, st)][None], device=args.device))
                    prob = torch.sigmoid(out["hit_logits"])[0].cpu().numpy()
                    f, p = hits_from_probs(prob, threshold, first, last)
                    played.append((st, f, p))
            per_song[song] = obj.scores(played)
            # per-song detail for diagnosis: hit counts, balance, raw facets and the reference parts
            n = sum(len(f) for _, f, _ in played)
            m = sum(len(f) for _, f, _ in obj.original)
            hits_drum = np.bincount(np.concatenate([p for _, _, p in played]).astype(int), minlength=len(pieces)) \
                if n else np.zeros(len(pieces), int)
            detail[song] = {"n": int(n), "m": int(m), "b": balance(n, m), "hits_per_drum": hits_drum.tolist(),
                            "raw": obj.raw(played), "floor_raw": obj.floor_raw, "ceiling_raw": obj.ceiling_raw,
                            "scores": per_song[song]}
        agg = aggregate(list(per_song.values()), sc["musicianship_weights"], sc["failure_floor"])
        results[name] = {"run": run, "seed": seed, "checkpoint_step": int(ck["step"]),
                         "facets": {f: agg[f] for f in FACETS}, "musicianship": agg["musicianship"],
                         "songs": detail}
        print(f"{name} ({run}, step {ck['step']}): beat_alignment {agg['beat_alignment']['score']:.3f} "
              f"over {agg['beat_alignment']['songs']} songs", flush=True)
        STATE["networks"][name] = {"step": int(ck["step"]), "beat_alignment": agg["beat_alignment"]["score"],
                                   "songs": agg["beat_alignment"]["songs"]}
        push()

    real = results["real"]["facets"]["beat_alignment"]["score"]
    controls = [results[n]["facets"]["beat_alignment"]["score"] for n in RUNS if n != "real"]
    win = main_win(real, controls, cfg)
    sup_real = {f: results["real"]["facets"][f]["score"] for f in COUNTED}
    sup_ctrls = [{f: results[n]["facets"][f]["score"] for f in COUNTED} for n in RUNS if n != "real"]
    sup_ok, sup_detail = supporting_win(sup_real, sup_ctrls)
    out = {"split": args.split, "clips": str(clips_path), "networks": results, "main_win": win,
           "supporting_win": {"wins": sup_ok, "detail": sup_detail}}
    dest = reports_dir() / f"win_{args.split}_{args.set}.json"
    dest.write_text(json.dumps(out, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"main win: {win['wins']} (real {win['real']:.3f}, best control {win['best_control']:.3f}, "
          f"spread {win['seed_spread']:.3f}, margin {win['margin']:.3f})", flush=True)
    print(f"supporting win: {sup_ok} {sup_detail}", flush=True)
    print(f"wrote {dest}", flush=True)
    STATE.update(status="done", main_win=win, supporting={"wins": sup_ok, "detail": sup_detail})
    push()


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        STATE.update(status="failed", error=f"{type(e).__name__}: {e}")
        try:
            push()
        finally:
            raise
