"""Check 2 (diagnostic, scoring only, no training): density-matched rescoring on the validation split (full set, 267 songs).

Each network's hit threshold is set so that it plays the same number of hits per song as the real network does at the
locked threshold (0.5). Then beat alignment, balance and timing without balance are scored as in scripts/score_win.py.
Diagnostic only: this is not a win test and does not change any locked setting.
Posts progress to the website page (runs/_live_score), as score_win.py does.
Writes reports/density_matched_validation.json.

    .venv\\Scripts\\python.exe scripts\\density_matched.py
"""
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import score_win as sw  # noqa: E402  (RUNS, STATE, push, load_site; main() is guarded)
from machine import Sampler  # noqa: E402
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler, load_validation_clips  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model.training import setup  # noqa: E402
from flybeats.scoring.scorecard import Song, aggregate, balance, check_scorecard, hits_from_probs  # noqa: E402

DEVICE = "cuda"
SET = "full"


def heartbeat(every_s=120):
    """Pushes the page every few minutes, so the machine readings on the website stay current."""
    while True:
        time.sleep(every_s)
        try:
            sw.push()
        except Exception as e:  # keep the heartbeat alive; a failed upload is retried on the next beat
            print(f"heartbeat push failed: {e}", flush=True)


def main():
    cfg, paths = load_locked(), load_paths()
    check_scorecard(cfg)
    td, sc = cfg["training_data"], cfg["scorecard"]
    work = paths["work_dir"]
    manifest, events, beats = load_manifest(work, SET)
    root = unpacked_root(work, SET)
    pieces = cfg["kit"]["pieces"]
    piece_idx = {p: i for i, p in enumerate(pieces)}
    frame_s = td["frame_ms"] / 1000
    first, last = round(td["warmup_ignored_seconds"] / frame_s), round(td["clip_seconds"] / frame_s)
    default_thr = cfg["training"]["hit_threshold"]

    split_rows = manifest[manifest.split == "validation"].set_index("song").sort_index()
    names = list(split_rows.index)
    clips = load_validation_clips(set_dir(work, SET) / "validation_clips.json")
    sampler = ClipSampler(root, manifest, events, beats, cfg, "validation")
    ev_by = {s: g for s, g in events.groupby("song")}
    bt_by = {s: g for s, g in beats.groupby("song")}

    # reference parts, exactly as in score_win.py (floor = next song in the split by name)
    song_objs = {}
    for i, s in enumerate(names):
        floor = names[(i + 1) % len(names)]
        ev, fev, fbt, bt = ev_by.get(s), ev_by.get(floor), bt_by[floor], bt_by[s]
        starts = [c["start"] for c in clips if c["song"] == s]
        if not starts:
            continue
        et = ev.time.to_numpy() if ev is not None else np.zeros(0)
        ep = np.array([piece_idx[p] for p in ev.piece]) if ev is not None else np.zeros(0, int)
        ft = fev.time.to_numpy() if fev is not None else np.zeros(0)
        fp = np.array([piece_idx[p] for p in fev.piece]) if fev is not None else np.zeros(0, int)
        song_objs[s] = Song(starts, bt.time.to_numpy(), bt.time[bt.downbeat].to_numpy(), (et, ep), (ft, fp),
                            fbt.time.to_numpy(), float(split_rows.duration[floor]), cfg)
    audio = {(c["song"], c["start"]): sampler.clip(c["song"], c["start"])["audio"] for c in clips if c["song"] in song_objs}
    keys = [(s, st) for s, obj in song_objs.items() for st in obj.starts]

    site = sw.load_site()
    sw.UPLOAD, sw.SITE_URL = site.get("upload"), site.get("url")
    sw.STATE.update(title="Flybeats check 2: density-matched scoring (validation, diagnostic only)", status="running")
    # machine readings for the page: this process's own CPU, GPU and memory, read every 10 s and pushed every 2 min
    sw.MACHINE = Sampler(None, sw.SITE_DIR / "machine_log.jsonl", every=10, own_process=True)
    sw.MACHINE.start()
    threading.Thread(target=heartbeat, daemon=True).start()
    sw.push()

    def hits_per_song(probs, thr):
        """Mean number of hits per song at threshold thr (songs with clips only)."""
        per = {}
        for key in keys:
            f, _ = hits_from_probs(probs[key], thr, first, last)
            per[key[0]] = per.get(key[0], 0) + len(f)
        return float(np.mean(list(per.values())))

    target, results = None, {}
    for name, (run, seed) in sw.RUNS.items():
        torch.manual_seed(cfg["model"]["init_seed"])
        s = setup("baby", seed, DEVICE, cfg)  # the setup each network was trained under
        model = s["model"]
        ck = torch.load(ROOT / "runs" / run / "best.pt", map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        model.gain_scale = 1.0
        probs = {}
        with torch.no_grad():
            for key in keys:
                out, _ = model(torch.as_tensor(audio[key][None], device=DEVICE))
                probs[key] = torch.sigmoid(out["hit_logits"])[0].cpu().numpy().astype(np.float32)

        if target is None:  # the real network sets the target at the locked threshold
            target = hits_per_song(probs, default_thr)
            thr = default_thr
        else:
            lo, hi = 0.0005, 0.9995
            for _ in range(30):  # bisection: hits per song falls as the threshold rises
                mid = (lo + hi) / 2
                if hits_per_song(probs, mid) > target:
                    lo = mid
                else:
                    hi = mid
            thr = (lo + hi) / 2
        reached = hits_per_song(probs, thr)

        per_song, timing, bals = {}, [], []
        for song, obj in song_objs.items():
            played = []
            for st in obj.starts:
                f, p = hits_from_probs(probs[(song, st)], thr, first, last)
                played.append((st, f, p))
            per_song[song] = obj.scores(played)
            n = sum(len(f) for _, f, _ in played)
            m = sum(len(f) for _, f, _ in obj.original)
            b = balance(n, m)
            bals.append(b)
            if b > 0:
                timing.append(obj.raw(played)["beat_alignment"] / b)
        agg = aggregate(list(per_song.values()), sc["musicianship_weights"], sc["failure_floor"])
        results[name] = {"run": run, "seed": seed, "checkpoint_step": int(ck["step"]), "threshold": float(thr),
                         "hits_per_song": reached, "target_hits_per_song": target,
                         "balance_mean": float(np.mean(bals)), "beat_alignment": agg["beat_alignment"]["score"],
                         "songs": agg["beat_alignment"]["songs"], "timing_without_b": float(np.mean(timing))}
        sw.STATE["networks"][name] = {"step": int(ck["step"]), "threshold": round(float(thr), 4),
                                      "hits_mean": round(reached, 1), "balance_mean": round(float(np.mean(bals)), 3),
                                      "beat_alignment": float(agg["beat_alignment"]["score"]),
                                      "songs": int(agg["beat_alignment"]["songs"]),
                                      "timing_without_b": float(np.mean(timing))}
        sw.push()
        print(f"{name} ({run}): threshold {thr:.4f}, hits/song {reached:.1f} (target {target:.1f}), "
              f"beat alignment {agg['beat_alignment']['score']:.3f}, timing {np.mean(timing):.4f}", flush=True)
        del probs

    dest = reports_dir() / "density_matched_validation.json"
    dest.write_text(json.dumps({"set": SET, "split": "validation", "default_threshold": default_thr,
                                "target_hits_per_song": target, "networks": results}, indent=2, default=float) + "\n",
                    encoding="utf-8")
    sw.STATE.update(status="done")
    sw.push()
    print(f"wrote {dest}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except BaseException as e:
        sw.STATE.update(status="failed", error=f"{type(e).__name__}: {e}")
        try:
            sw.push()
        finally:
            raise
