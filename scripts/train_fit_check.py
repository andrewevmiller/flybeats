"""Step 1 (diagnostic, scoring only, no training): do the existing checkpoints place their hits in time on TRAINING clips?

The same six networks as check 2, scored at the locked threshold (0.5) on a fixed draw of training clips: 270 training
songs (numpy seed 2026, sorted names), 4 windows of 16 s each, drawn once into data/work/data/full/train_check_clips.json.
Reports beat alignment, balance, hits per song and timing without balance (random = 0.5). Posts progress to the website
(runs/_live_score) with machine readings, as the other scoring runs do. Writes reports/train_fit_check.json.

If timing is also at chance on training clips, the networks have not fitted timing at all (an optimisation limit).
    .venv\\Scripts\\python.exe scripts\\train_fit_check.py
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
from flybeats.data.sampler import ClipSampler  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model.training import setup  # noqa: E402
from flybeats.scoring.scorecard import Song, aggregate, balance, check_scorecard, hits_from_probs  # noqa: E402

DEVICE = "cuda"
SET = "full"
N_SONGS, PER_SONG, SEED = 270, 4, 2026


def heartbeat(every_s=120):
    while True:
        time.sleep(every_s)
        try:
            sw.push()
        except Exception as e:  # keep the heartbeat alive
            print(f"heartbeat push failed: {e}", flush=True)


def draw_clips(manifest, cfg, path):
    """The fixed training draw: refused if it already exists (written once, like the test clips)."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = manifest[manifest.split == "train"].set_index("song").sort_index()
    clip_s = cfg["training_data"]["clip_seconds"]
    rng = np.random.default_rng(SEED)
    songs = list(rng.choice(list(rows.index), size=N_SONGS, replace=False))
    songs = [s for s in sorted(songs) if rows.duration[s] > clip_s]
    clips = [{"song": s, "start": round(float(rng.uniform(0, rows.duration[s] - clip_s)), 3)}
             for s in songs for _ in range(PER_SONG)]
    path.write_text(json.dumps(clips, indent=1), encoding="utf-8")
    return clips


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
    thr = cfg["training"]["hit_threshold"]

    clips_path = set_dir(work, SET) / "train_check_clips.json"
    clips = draw_clips(manifest, cfg, clips_path)
    split_rows = manifest[manifest.split == "train"].set_index("song").sort_index()
    names = sorted({c["song"] for c in clips})
    sampler = ClipSampler(root, manifest, events, beats, cfg, "train")
    ev_by = {s: g for s, g in events.groupby("song")}
    bt_by = {s: g for s, g in beats.groupby("song")}
    song_objs = {}
    full_names = list(split_rows.index)  # the floor is the next song in the TRAIN split by name, as for scoring
    for s in names:
        i = full_names.index(s)
        floor = full_names[(i + 1) % len(full_names)]
        ev, fev, fbt, bt = ev_by.get(s), ev_by.get(floor), bt_by[floor], bt_by[s]
        starts = [c["start"] for c in clips if c["song"] == s]
        et = ev.time.to_numpy() if ev is not None else np.zeros(0)
        ep = np.array([piece_idx[p] for p in ev.piece]) if ev is not None else np.zeros(0, int)
        ft = fev.time.to_numpy() if fev is not None else np.zeros(0)
        fp = np.array([piece_idx[p] for p in fev.piece]) if fev is not None else np.zeros(0, int)
        song_objs[s] = Song(starts, bt.time.to_numpy(), bt.time[bt.downbeat].to_numpy(), (et, ep), (ft, fp),
                            fbt.time.to_numpy(), float(split_rows.duration[floor]), cfg)
    audio = {(c["song"], c["start"]): sampler.clip(c["song"], c["start"])["audio"] for c in clips}
    keys = [(s, st) for s, obj in song_objs.items() for st in obj.starts]

    site = sw.load_site()
    sw.UPLOAD, sw.SITE_URL = site.get("upload"), site.get("url")
    sw.STATE.update(title="Flybeats step 1: fit on training clips (existing checkpoints, locked threshold, diagnostic only)",
                    status="running")
    sw.MACHINE = Sampler(None, sw.SITE_DIR / "machine_log.jsonl", every=10, own_process=True)
    sw.MACHINE.start()
    threading.Thread(target=heartbeat, daemon=True).start()
    sw.push()

    results = {}
    for name, (run, seed) in sw.RUNS.items():
        torch.manual_seed(cfg["model"]["init_seed"])
        s = setup("baby", seed, DEVICE, cfg)  # the setup each network was trained under
        model = s["model"]
        ck = torch.load(ROOT / "runs" / run / "best.pt", map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"])
        model.eval()
        model.gain_scale = 1.0
        per_song, timing, bals, hits = {}, [], [], []
        for song, obj in song_objs.items():
            played = []
            with torch.no_grad():
                for st in obj.starts:
                    out, _ = model(torch.as_tensor(audio[(song, st)][None], device=DEVICE))
                    prob = torch.sigmoid(out["hit_logits"])[0].cpu().numpy()
                    f, p = hits_from_probs(prob, thr, first, last)
                    played.append((st, f, p))
            per_song[song] = obj.scores(played)
            n = sum(len(f) for _, f, _ in played)
            m = sum(len(f) for _, f, _ in obj.original)
            b = balance(n, m)
            bals.append(b)
            hits.append(n)
            if b > 0:
                timing.append(obj.raw(played)["beat_alignment"] / b)
        agg = aggregate(list(per_song.values()), sc["musicianship_weights"], sc["failure_floor"])
        results[name] = {"run": run, "checkpoint_step": int(ck["step"]), "threshold": thr,
                         "hits_per_song": float(np.mean(hits)), "balance_mean": float(np.mean(bals)),
                         "beat_alignment": agg["beat_alignment"]["score"], "songs": agg["beat_alignment"]["songs"],
                         "timing_without_b": float(np.mean(timing))}
        sw.STATE["networks"][name] = {"step": int(ck["step"]), "hits_mean": round(float(np.mean(hits)), 1),
                                      "balance_mean": round(float(np.mean(bals)), 3),
                                      "beat_alignment": float(agg["beat_alignment"]["score"]),
                                      "songs": int(agg["beat_alignment"]["songs"]),
                                      "timing_without_b": float(np.mean(timing))}
        sw.push()
        print(f"{name} ({run}): hits/song {np.mean(hits):.1f}, beat alignment {agg['beat_alignment']['score']:.3f}, "
              f"timing {np.mean(timing):.4f}", flush=True)
        del model

    dest = reports_dir() / "train_fit_check.json"
    dest.write_text(json.dumps({"set": SET, "split": "train (fixed draw, 270 songs x 4 clips)", "threshold": thr,
                                "networks": results}, indent=2, default=float) + "\n", encoding="utf-8")
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
