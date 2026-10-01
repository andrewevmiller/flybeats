"""How much of the drum part can be predicted from the band? The model hears only the non-drum stems and has to play
the drums along with them, from what it has heard so far. This measures that from the data alone: no network, CPU only.
The audio features are exactly the model's input (the locked ear, with its band statistics).

1. Memorisation ceiling, on the 8 clips memorise.py uses. A model that works only from the last W of sound gives the
   same answer wherever that stretch of sound is the same. So if two moments sound (nearly) the same over the last W
   but the drummer played differently, no such model can get both right. For each real hit, it finds the moment
   elsewhere in the 8 clips whose last W of band audio is closest (not overlapping it), and checks:
     twin        that moment is near-identical (distance below eps; see below)
     conflict    a twin where the drummer did not play that drum there (within the scoring tolerance)
   and the same for moments with no hit whose twin has one. W: 50 ms, 200 ms (about the model's memory: its longest
   time constant), 1 s, 2 s. "Near-identical" has no natural cut-off, so eps is shown at 3 levels: 5%, 10% and 20% of
   the median distance between random pairs of moments.
2. Predictability across songs: a simple predictor (logistic regression, one per drum) from the last 200 ms or 1 s of
   band audio, fitted on 5 training songs and scored on 5 others (the 10 training songs memorise.py does not use;
   validation and test songs are not touched). Reported: average precision per drum, against the base rate (what
   predicting without audio gives).

Writes reports/drum_predictability.json.
    .venv\\Scripts\\python.exe scripts\\drum_predictability.py [--trial]
A few minutes on the CPU. --trial runs a small version to check the script, and writes nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import ClipSampler  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model.ear import Bands, load_band_stats  # noqa: E402
from flybeats.model.loss import tolerance_frames  # noqa: E402

MEMORY_MS = [50, 200, 1000, 2000]        # part 1
PROBE_MS = [200, 1000]                   # part 2
EPS = [0.05, 0.10, 0.20]                 # near-identical: fraction of the median random-pair distance
BLOCKS = 40                              # a context is its window averaged into at most this many equal blocks

ap = argparse.ArgumentParser()
ap.add_argument("--trial", action="store_true", help="a small version to check the script runs; writes nothing")
args = ap.parse_args()
t0 = time.time()
torch.set_num_threads(4)
cfg, paths = load_locked(), load_paths()
frame_ms = cfg["training_data"]["frame_ms"]
frame_s = frame_ms / 1000
frac = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
manifest, events, beats = load_manifest(paths["work_dir"], "baby")
train = ClipSampler(unpacked_root(paths["work_dir"], "baby"), manifest, events, beats, cfg, "train")
pieces = train.pieces
ear = Bands(*load_band_stats(set_dir(paths["work_dir"], "baby") / "band_stats.json"))


def ear_of(audio):
    """The model's ear input for one clip: (frames, 16)."""
    with torch.no_grad():
        return ear(torch.from_numpy(audio)[None])[0].numpy()


def context(x, t, frames):
    """The last `frames` frames of x up to and including t, averaged into at most BLOCKS blocks, flattened."""
    w = x[t - frames + 1:t + 1]
    k = min(BLOCKS, frames)
    return w[: frames // k * k].reshape(k, frames // k, -1).mean(1).ravel()


report = {"trial": args.trial}

# ---------------------------------------------------------------- 1. memorisation ceiling
mc = cfg["memorisation"]
rng = np.random.default_rng(mc["clip_seed"])
mem_songs = sorted(train.songs.index)[:mc["songs"]]
clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s)))
         for song in mem_songs * mc["clips_per_song"]]
if args.trial:
    clips = clips[:4]
feats = [ear_of(c["audio"]) for c in clips]
first = int(round(cfg["training_data"]["warmup_ignored_seconds"] / frame_s))     # the scored frames, as in the check
T = clips[0]["hits"].shape[0]
tols = [np.full(T, 0.0) for _ in clips]
for i, c in enumerate(clips):                                  # the scoring tolerance at every frame of every clip
    tols[i] = tolerance_frames(np.arange(T), c["beats"], frame_s, frac)
near_hit = []                                                  # (clip, frame, drum) -> a hit of that drum within tolerance
for i, c in enumerate(clips):
    h = c["hits"].astype(bool)
    nh = np.zeros_like(h)
    for d in range(len(pieces)):
        for f in np.flatnonzero(h[:, d]):
            r = int(np.ceil(tols[i][f]))
            nh[max(f - r, 0):f + r + 1, d] = True
    near_hit.append(nh)

print(f"1. memorisation ceiling: {len(clips)} clips, {sum(int(c['hits'][first:].sum()) for c in clips)} scored hits "
      f"({time.time() - t0:.0f} s)")
report["ceiling"] = {}
qrng = np.random.default_rng(0)
for w_ms in MEMORY_MS:
    wf = w_ms // frame_ms
    frames = [(i, t) for i in range(len(clips)) for t in range(max(first, wf - 1), T)]
    ctx = torch.as_tensor(np.stack([context(feats[i], t, wf) for i, t in frames]), dtype=torch.float32)
    where = np.array(frames)
    hit_q = [(k, d) for k, (i, t) in enumerate(frames) for d in np.flatnonzero(clips[i]["hits"][t])]
    quiet = [k for k, (i, t) in enumerate(frames) if not near_hit[i][t].any()]
    quiet_q = list(qrng.choice(quiet, min(len(quiet), 3000), replace=False))
    a = qrng.integers(0, len(frames), 4000)
    b = qrng.integers(0, len(frames), 4000)
    median_pair = float(torch.linalg.vector_norm(ctx[a] - ctx[b], dim=1).median())

    def nearest(k):
        """Closest other moment whose window does not overlap k's (other clips, or far enough in the same clip)."""
        dist = torch.linalg.vector_norm(ctx - ctx[k], dim=1)
        i, t = frames[k]
        dist[(where[:, 0] == i) & (np.abs(where[:, 1] - t) < wf + int(np.ceil(tols[i][t])) + 1)] = float("inf")
        j = int(torch.argmin(dist))
        return j, float(dist[j]) / median_pair

    hit_nn = [(k, d, *nearest(k)) for k, d in hit_q]
    quiet_nn = [(k, *nearest(k)) for k in quiet_q]
    res = {"median_random_pair_distance": median_pair, "hits": len(hit_nn), "quiet_moments_sampled": len(quiet_nn),
           "nearest_distance_median_hits": float(np.median([x[3] for x in hit_nn])), "by_eps": {}}
    for eps in EPS:
        per = {}
        for d, p in enumerate(pieces):
            hd = [(k, j, r) for k, dd, j, r in hit_nn if dd == d]
            if not hd:
                continue
            twins = [(k, j) for k, j, r in hd if r <= eps]
            conflict = [1 for k, j in twins if not near_hit[frames[j][0]][frames[j][1], d]]
            per[p] = {"hits": len(hd), "with_twin": len(twins), "twin_conflicts": len(conflict)}
        twins_q = [(k, j) for k, j, r in quiet_nn if r <= eps]
        q_conf = sum(1 for k, j in twins_q if clips[frames[j][0]]["hits"][frames[j][1]].any())
        res["by_eps"][eps] = {"per_drum": per, "quiet_with_twin": len(twins_q), "quiet_twin_is_a_hit": q_conf}
    report["ceiling"][w_ms] = res
    print(f"   memory {w_ms:>5} ms: nearest other moment for a hit is typically "
          f"{res['nearest_distance_median_hits']:.2f} x the median random-pair distance ({time.time() - t0:.0f} s)")

print("\n   hits with a near-identical twin elsewhere, and how many of those twins conflict (the drummer did not play "
      "that drum there)")
must = cfg["memorisation"]["must_pass"]
print(f"   {'memory':>7} {'eps':>5}  " + "  ".join(f"{p:>22}" for p in must) + f"  {'all drums':>22}  {'quiet moments':>24}")
for w_ms, res in report["ceiling"].items():
    for eps in EPS:
        e = res["by_eps"][eps]
        cells = []
        for p in must + ["all"]:
            if p == "all":
                h = sum(v["hits"] for v in e["per_drum"].values())
                tw = sum(v["with_twin"] for v in e["per_drum"].values())
                cf = sum(v["twin_conflicts"] for v in e["per_drum"].values())
            else:
                v = e["per_drum"].get(p, {"hits": 0, "with_twin": 0, "twin_conflicts": 0})
                h, tw, cf = v["hits"], v["with_twin"], v["twin_conflicts"]
            cells.append(f"{tw:>4} twins/{h:<4} {cf:>4} conflict")
        q = f"{e['quiet_with_twin']:>4} twins, {e['quiet_twin_is_a_hit']:>4} a hit"
        print(f"   {w_ms:>5}ms {eps:>5.2f}  " + "  ".join(f"{c:>22}" for c in cells) + f"  {q:>24}")

# ---------------------------------------------------------------- 2. predictability across songs
songs = [s for s in sorted(train.songs.index) if s not in set(mem_songs)]
if args.trial:
    songs = songs[:2]
fit_songs, score_songs = songs[0::2], songs[1::2]
print(f"\n2. predictability across songs: fitted on {fit_songs}, scored on {score_songs} ({time.time() - t0:.0f} s)")
STEP = 4                                   # every 4th frame (20 ms), plus every hit frame, to keep the sample small
samples = {w: {"fit": [], "score": []} for w in PROBE_MS}
for song in songs:
    part = "fit" if song in fit_songs else "score"
    for start in np.arange(0, float(train.songs.duration[song]) - train.clip_s, train.clip_s):
        c = train.clip(song, float(start))
        x = ear_of(c["audio"])
        hits = c["hits"].astype(bool)
        soft = np.zeros_like(hits)          # target: a hit of that drum within +-2 frames (10 ms)
        for sh in range(-2, 3):
            soft |= np.roll(hits, sh, axis=0)
        chosen = sorted(set(range(first, T, STEP)) | set(np.flatnonzero(hits[first:].any(1)) + first))
        for w_ms in PROBE_MS:
            wf = w_ms // frame_ms
            for t in chosen:
                if t >= wf - 1:
                    samples[w_ms][part].append((context(x, t, wf), soft[t], t % STEP == 0))
    print(f"   {song} read ({time.time() - t0:.0f} s)", flush=True)


def average_precision(score, y, weight):
    order = np.argsort(-score)
    y, weight = y[order], weight[order]
    tp = np.cumsum(y * weight)
    fp = np.cumsum((1 - y) * weight)
    prec = tp / np.maximum(tp + fp, 1e-12)
    return float((prec * y * weight).sum() / max((y * weight).sum(), 1e-12))


report["across_songs"] = {"fit_songs": fit_songs, "scored_songs": score_songs, "per_memory": {}}
print(f"\n   average precision on the scored songs (base rate = no audio) for a hit within +-10 ms")
print(f"   {'memory':>7}  " + "  ".join(f"{p:>13}" for p in pieces))
for w_ms in PROBE_MS:
    xf = np.stack([s[0] for s in samples[w_ms]["fit"]]); yf = np.stack([s[1] for s in samples[w_ms]["fit"]])
    xs = np.stack([s[0] for s in samples[w_ms]["score"]]); ys = np.stack([s[1] for s in samples[w_ms]["score"]])
    grid = np.array([s[2] for s in samples[w_ms]["score"]])     # frames on the regular grid: unbiased base rate
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    xft = torch.as_tensor((xf - mu) / sd, dtype=torch.float32)
    xst = torch.as_tensor((xs - mu) / sd, dtype=torch.float32)
    out = {}
    for d, p in enumerate(pieces):
        y = torch.as_tensor(yf[:, d], dtype=torch.float32)
        if y.sum() < 20 or ys[:, d].sum() < 5:
            continue
        lin = torch.nn.Linear(xft.shape[1], 1)
        pw = (len(y) - y.sum()) / y.sum()
        opt = torch.optim.LBFGS(lin.parameters(), max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xft)[:, 0], y, pos_weight=pw) \
                + 1e-3 * lin.weight.pow(2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            s = lin(xst)[:, 0].numpy()
        # hit frames were added on top of the 1-in-STEP grid: weight grid frames by STEP so the sample stands for
        # every frame (hit frames off the grid count once)
        wgt = np.where(grid, STEP, 1.0)
        base = float(ys[grid, d].mean())
        out[p] = {"average_precision": average_precision(s, ys[:, d].astype(float), wgt), "base_rate": base}
    report["across_songs"]["per_memory"][w_ms] = out
    print(f"   {w_ms:>5}ms  " + "  ".join(
        f"{out[p]['average_precision']:>5.2f} ({out[p]['base_rate']:.2f})" if p in out else f"{'-':>13}"
        for p in pieces))

if args.trial:
    sys.exit(f"\ntrial run: nothing written ({time.time() - t0:.0f} s)")
out_path = reports_dir() / "drum_predictability.json"
out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"\nwrote {out_path} ({time.time() - t0:.0f} s)")
