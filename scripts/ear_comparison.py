"""Would a different ear let the fly hear more of what predicts the drums? The model hears the band without the drums
and plays the drum part. This compares ears, the fixed front end from audio to the 150 ear neurons, by how well a
simple predictor can foresee the drummer's hits from each ear's output. No network, CPU only.

Ears (all causal: each 5 ms frame's window ends at that frame):
  locked         16 bands, 40-4,000 Hz, 32 ms window: exactly the model's ear now
  1 to 8 kHz     16 bands, 40-8,000 Hz (the audio's full range)
  2 32 bands     32 bands, 40-4,000 Hz
  4 short high   16 bands, 40-4,000 Hz; bands from 1 kHz up use an 8 ms window
  3 onsets       16 bands, 40-4,000 Hz: each band's level and its rise since the previous frame (32 signals)
  5 8 kHz + onsets   1 and 3 together
Listed in order of how small a change each is (the order the decision rule uses).

Songs: the 10 BabySlakh training songs the memorisation check does not use, in two halves. Each half is fitted on
and the other scored (2-fold); validation and test songs are not touched.
Predictor: logistic regression, one per drum, from the ear's last 200 ms or 1 s (averaged into 40 blocks); target:
a hit of that drum within 10 ms. Score: average precision.

DECISION RULE, fixed before the run (Andrew, 30 Sep 2026): the score is the average precision averaged over kick,
snare and closed hi-hat, both memory lengths and both folds. An ear qualifies if its score is at least 25% higher
than the locked ear's. If several qualify, the smallest change in the order above is chosen. If none does, the ear
stays as locked. The script applies the rule and prints the outcome; it changes nothing.

Writes reports/ear_comparison.json.
    .venv\\Scripts\\python.exe scripts\\ear_comparison.py [--trial]
About 15-25 minutes on the CPU. --trial runs a small version to check the script, and writes nothing.
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
from flybeats.data.manifest import load_manifest  # noqa: E402
from flybeats.data.sampler import ClipSampler  # noqa: E402
from flybeats.data.slakh import unpacked_root  # noqa: E402
from flybeats.model import ear as model_ear  # noqa: E402

SR, HOP = 16000, model_ear.HOP
MEMORY_MS = [200, 1000]
BLOCKS = 40
STEP = 4                                 # every 4th frame (20 ms), plus every hit frame
DRUMS = ["kick", "snare", "hihat_closed"]
THRESHOLD = 0.25


def band_matrix(n_fft, n_bands, lo, hi):
    """model_ear.band_matrix's rule for any window, band count and range."""
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    edges = np.geomspace(lo, hi, n_bands + 1)
    m = np.zeros((len(freqs), n_bands), np.float32)
    for b in range(n_bands):
        inside = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if inside.any():
            m[inside, b] = 1
        else:
            m[np.argmin(np.abs(np.log(freqs[1:] / np.sqrt(edges[b] * edges[b + 1])))) + 1, b] = 1
    return torch.from_numpy(m), edges


def power(audio, n_fft):
    """(frames, bins) power; frame t's window ends at sample (t + 1) * HOP, as in the model's ear."""
    x = torch.nn.functional.pad(torch.from_numpy(audio)[None], (n_fft - HOP, 0))
    spec = torch.stft(x, n_fft, HOP, window=torch.hann_window(n_fft), center=False, return_complex=True)
    return spec.abs().pow(2)[0].t()


class Ear:
    def __init__(self, name, hi=4000.0, n_bands=16, short_above=None, onsets=False, lo=40.0):
        self.name, self.onsets = name, onsets
        m_long, edges = band_matrix(512, n_bands, lo, hi)
        short = np.zeros(n_bands, bool) if short_above is None else edges[:-1] >= short_above
        self.m_long = m_long[:, ~short]
        self.m_short = band_matrix(128, n_bands, lo, hi)[0][:, short] if short.any() else None
        self.order = np.argsort(np.r_[np.flatnonzero(~short), np.flatnonzero(short)])

    def __call__(self, audio):
        parts = [power(audio, 512) @ self.m_long]
        if self.m_short is not None:
            parts.append(power(audio, 128) @ self.m_short)
        level = torch.log1p(torch.cat(parts, 1)[:, self.order]).numpy()
        if not self.onsets:
            return level
        rise = np.zeros_like(level)
        rise[1:] = np.clip(np.diff(level, axis=0), 0, None)
        return np.concatenate([level, rise], 1)


EARS = [Ear("locked"), Ear("1 to 8 kHz", hi=8000.0), Ear("2 32 bands", n_bands=32),
        Ear("4 short high", short_above=1000.0), Ear("3 onsets", onsets=True),
        Ear("5 8 kHz + onsets", hi=8000.0, onsets=True)]


def context(x, t, frames):
    w = x[t - frames + 1:t + 1]
    k = min(BLOCKS, frames)
    return w[: frames // k * k].reshape(k, frames // k, -1).mean(1).ravel()


def average_precision(score, y, weight):
    order = np.argsort(-score)
    y, weight = y[order], weight[order]
    tp, fp = np.cumsum(y * weight), np.cumsum((1 - y) * weight)
    return float(((tp / np.maximum(tp + fp, 1e-12)) * y * weight).sum() / max((y * weight).sum(), 1e-12))


def fit_score(xf, yf, xs, ys, wgt):
    mu, sd = xf.mean(0), xf.std(0) + 1e-6
    xft = torch.as_tensor((xf - mu) / sd, dtype=torch.float32)
    y = torch.as_tensor(yf, dtype=torch.float32)
    lin = torch.nn.Linear(xft.shape[1], 1)
    pw = (len(y) - y.sum()) / y.sum().clamp_min(1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=300, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xft)[:, 0], y, pos_weight=pw) \
            + 1e-3 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        s = lin(torch.as_tensor((xs - mu) / sd, dtype=torch.float32))[:, 0].numpy()
    return average_precision(s, ys.astype(float), wgt)


ap = argparse.ArgumentParser()
ap.add_argument("--trial", action="store_true", help="a small version to check the script runs; writes nothing")
args = ap.parse_args()
t0 = time.time()
torch.set_num_threads(4)
cfg, paths = load_locked(), load_paths()
frame_ms = cfg["training_data"]["frame_ms"]
first = int(round(cfg["training_data"]["warmup_ignored_seconds"] * 1000 / frame_ms))
manifest, events, beats = load_manifest(paths["work_dir"], "baby")
train = ClipSampler(unpacked_root(paths["work_dir"], "baby"), manifest, events, beats, cfg, "train")
pieces = train.pieces
drum_idx = [pieces.index(d) for d in DRUMS]
mem_songs = set(sorted(train.songs.index)[:cfg["memorisation"]["songs"]])
songs = [s for s in sorted(train.songs.index) if s not in mem_songs]
if args.trial:
    songs = songs[:2]
halves = [songs[0::2], songs[1::2]]

probe = np.random.default_rng(0).normal(0, 0.1, SR).astype(np.float32)   # the locked ear must be the model's ear
gap = float(np.abs(EARS[0](probe) - model_ear.Bands(sr=SR).raw(torch.from_numpy(probe)[None])[0].numpy()).max())
if gap > 1e-4:
    sys.exit(f"the 'locked' ear differs from the model's ear by {gap}; stopping")

clips = []                                     # (song, audio, soft target (frames, 3), frames to sample, on-grid)
for song in songs:
    for start in np.arange(0, float(train.songs.duration[song]) - train.clip_s, train.clip_s):
        c = train.clip(song, float(start))
        hits = c["hits"][:, drum_idx].astype(bool)
        soft = np.zeros_like(hits)
        for sh in range(-2, 3):
            soft |= np.roll(hits, sh, axis=0)
        T = hits.shape[0]
        chosen = np.array(sorted(set(range(first, T, STEP)) | set(np.flatnonzero(c["hits"][first:].any(1)) + first)))
        clips.append((song, c["audio"], soft, chosen))
print(f"read {len(clips)} clips from {len(songs)} songs ({time.time() - t0:.0f} s)", flush=True)

report = {"trial": args.trial, "halves": halves, "rule": {"drums": DRUMS, "memories_ms": MEMORY_MS,
                                                          "threshold": THRESHOLD}, "ears": {}}
for e in EARS:
    per = {}
    for w_ms in MEMORY_MS:
        wf = w_ms // frame_ms
        data = {0: [], 1: []}
        for song, audio, soft, chosen in clips:
            x = e(audio)
            half = 0 if song in halves[0] else 1
            for t in chosen[chosen >= wf - 1]:
                data[half].append((context(x, t, wf), soft[t], t % STEP == 0))
        for fold in (0, 1):
            fit, score = data[fold], data[1 - fold]
            xf = np.stack([d[0] for d in fit]); yf = np.stack([d[1] for d in fit])
            xs = np.stack([d[0] for d in score]); ys = np.stack([d[1] for d in score])
            wgt = np.where(np.array([d[2] for d in score]), STEP, 1.0)
            for k, drum in enumerate(DRUMS):
                if yf[:, k].sum() < 20 or ys[:, k].sum() < 5:
                    continue
                per[f"{drum}|{w_ms}|{fold}"] = {"ap": fit_score(xf, yf[:, k], xs, ys[:, k], wgt),
                                                "base": float(ys[np.array([d[2] for d in score]), k].mean())}
        print(f"  {e.name:<18} memory {w_ms:>4} ms done ({time.time() - t0:.0f} s)", flush=True)
    score_all = float(np.mean([v["ap"] for v in per.values()])) if per else float("nan")
    report["ears"][e.name] = {"score": score_all, "cells": per}

locked = report["ears"]["locked"]["score"]
print(f"\nscore = average precision over {', '.join(DRUMS)}, memories {MEMORY_MS} ms and both folds")
print(f"   {'ear':<18} {'score':>7} {'vs locked':>10}   " + "  ".join(f"{d:>12}" for d in DRUMS) + "  (base rate)")
chosen = None
for e in EARS:
    r = report["ears"][e.name]
    rel = r["score"] / locked - 1
    r["vs_locked"] = rel
    r["qualifies"] = e.name != "locked" and rel >= THRESHOLD
    by_drum = []
    for d in DRUMS:
        cells = [v for k, v in r["cells"].items() if k.startswith(d + "|")]
        by_drum.append(f"{np.mean([c['ap'] for c in cells]):.3f} ({np.mean([c['base'] for c in cells]):.3f})"
                       if cells else "-")
    print(f"   {e.name:<18} {r['score']:>7.4f} {rel:>+10.0%}   " + "  ".join(f"{c:>12}" for c in by_drum)
          + ("   qualifies" if r["qualifies"] else ""))
    if r["qualifies"] and chosen is None:
        chosen = e.name
report["outcome"] = chosen or "locked"
print(f"\ndecision rule (25% over the locked ear; smallest qualifying change): "
      f"{'adopt ' + repr(chosen) if chosen else 'no ear qualifies; the ear stays as locked'}")

if args.trial:
    sys.exit(f"trial run: nothing written ({time.time() - t0:.0f} s)")
out = reports_dir() / "ear_comparison.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out} ({time.time() - t0:.0f} s)")
