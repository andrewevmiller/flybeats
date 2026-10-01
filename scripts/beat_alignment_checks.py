"""Reference-part checks for the proposed beat-alignment measure (reports/beat-alignment-proposal.md, not locked).
Drum parts and beat annotations only: no audio, no network, so nothing here can tune the measure to a result.

The measure, as proposed: each hit's on-grid score s = (1 + cos(2 pi p)) / 2, p its position in 16ths of the local
beat (s = 1 on a 16th line, 0 halfway between); raw = mean s x density balance b = min(n / m, m / n) (n hits played,
m in the original part; b = 0 if n = 0). Per song, rescaled 0-100 between the floor (the next validation song's part
by name, laid onto this song's beats and shifted 3/8 beat later) and the ceiling (the original part); songs where
ceiling - floor < 0.1 are left out.

Checks, on the full set's fixed validation clips (4 per song, first 2 s ignored), hits at 5 ms frames as in training:
  1 ceiling against floor per song; how many songs are left out
  2 silence, random times (right number of hits), every 16th on every drum, one on-grid hit per clip, and the
    original part with +/-10 and +/-20 ms timing jitter: where each lands, rescaled
  3 hand-made examples with known answers (run first; the script stops if any fails)
Writes reports/beat_alignment_checks.json.
    .venv\\Scripts\\python.exe scripts\\beat_alignment_checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import load_validation_clips  # noqa: E402

SHIFT_BEATS = 3 / 8
EXCLUDE_BELOW = 0.1


def beat_position(times, beats):
    """Fractional beat index of each time, following the annotated beats; extrapolated past either end."""
    b = np.asarray(beats, float)
    k = np.clip(np.searchsorted(b, times, side="right") - 1, 0, len(b) - 2)
    return k + (times - b[k]) / (b[k + 1] - b[k])


def time_of(position, beats):
    b = np.asarray(beats, float)
    k = np.clip(np.floor(position).astype(int), 0, len(b) - 2)
    return b[k] + (position - k) * (b[k + 1] - b[k])


def on_grid(times, beats):
    return (1 + np.cos(2 * np.pi * 4 * beat_position(times, beats))) / 2


def raw(times, m, beats):
    n = len(times)
    if n == 0 or m == 0:
        return 0.0
    return float(on_grid(times, beats).mean() * min(n / m, m / n))


# ---- 3: hand-made examples ----
steady = np.arange(0, 20, 0.5)                                       # 120 BPM
assert np.allclose(on_grid(np.array([0.0, 0.125, 0.25, 1.375]), steady), 1)          # on 16th lines
assert np.allclose(on_grid(np.array([0.0625, 0.1875, 1.3125]), steady), 0)          # halfway
assert np.allclose(on_grid(np.array([0.03125]), steady), 0.5)                       # a quarter of a 16th off
change = np.r_[np.arange(0, 4, 0.5), 4 + np.arange(0, 8, 0.75)]                     # 120 -> 80 BPM at 4 s
assert np.allclose(on_grid(4 + np.arange(0, 6, 0.1875), change), 1)                 # 80 BPM 16ths after the change
assert raw(np.array([]), 10, steady) == 0
assert np.isclose(raw(np.arange(0, 2, 0.125), 32, steady), 0.5)                     # on grid, half the hits
assert np.isclose(raw(np.arange(0, 8, 0.125), 32, steady), 0.5)                     # on grid, twice the hits
print("hand-made examples: all as expected")

cfg, paths = load_locked(), load_paths()
frame_s = cfg["training_data"]["frame_ms"] / 1000
clip_s = cfg["training_data"]["clip_seconds"]
skip = cfg["training_data"]["warmup_ignored_seconds"]
manifest, events, beats = load_manifest(paths["work_dir"], "full")
clips = load_validation_clips(set_dir(paths["work_dir"], "full") / "validation_clips.json")
pieces = cfg["kit"]["pieces"]
ev = {s: g.time.to_numpy() for s, g in events[events.piece.isin(pieces)].groupby("song")}
pc = {s: g.piece.map({q: j for j, q in enumerate(pieces)}).to_numpy() for s, g in
      events[events.piece.isin(pieces)].groupby("song")}
bt = {s: g.sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
songs = sorted({c["song"] for c in clips})
floor_song = {s: songs[(i + 1) % len(songs)] for i, s in enumerate(songs)}
rng = np.random.default_rng(0)


def frames(times, start, piece=None):
    """Hit times as the labels have them: on 5 ms frames of the clip, inside the scored part; one hit per drum per
    frame (hits on different drums in the same frame each count)."""
    f = np.floor((times - start) / frame_s)
    piece = np.zeros(len(f), int) if piece is None else np.asarray(piece)
    keep = (f >= round(skip / frame_s)) & (f < round(clip_s / frame_s))
    if not keep.any():
        return np.zeros(0)
    pairs = np.unique(np.stack([f[keep], piece[keep]], 1), axis=0)
    return start + pairs[:, 0] * frame_s


per_song = {}
for s in songs:
    b = bt.get(s)
    if b is None or len(b) < 2:
        continue
    out = {k: [] for k in ("ceiling", "floor", "silence", "random", "every_16th", "one_hit", "jitter_10ms",
                           "jitter_20ms")}
    m_total = 0
    for c in (c for c in clips if c["song"] == s):
        lo, hi = c["start"] + skip, c["start"] + clip_s
        orig_t, orig_p = ev.get(s, np.zeros(0)), pc.get(s, np.zeros(0, int))
        inside = (orig_t >= c["start"]) & (orig_t < hi)
        orig_t, orig_p = orig_t[inside], orig_p[inside]
        orig = frames(orig_t, c["start"], orig_p)
        m = len(orig)
        m_total += m
        # floor: the next song's part in beats, from the same beat index, laid onto this song's beats, +3/8 beat
        fs, fb = floor_song[s], bt.get(floor_song[s])
        b0, b1 = beat_position(np.array([lo, hi]), b)
        q = beat_position(ev.get(fs, np.zeros(0)), fb)
        qp = pc.get(fs, np.zeros(0, int))
        span = b1 - b0
        offset = b0 if len(fb) - 1 >= b0 + span else max(0.0, (len(fb) - 1 - span)) * rng.random()
        sel = (q >= offset) & (q < offset + span)
        flo = frames(time_of(q[sel] - offset + b0 + SHIFT_BEATS, b), c["start"], qp[sel])
        grid = time_of(np.arange(np.ceil(b0 * 4), np.floor(b1 * 4) + 1) / 4, b)
        grid = grid[(grid >= lo) & (grid < hi)]
        cases = {"ceiling": orig, "floor": flo, "silence": np.zeros(0),
                 "random": frames(rng.uniform(lo, hi, m), c["start"]),
                 "every_16th": frames(grid, c["start"]),               # scored once; counted once per drum below
                 "one_hit": frames(grid[:1], c["start"]),
                 "jitter_10ms": frames(orig_t + rng.uniform(-0.01, 0.01, len(orig_t)), c["start"], orig_p),
                 "jitter_20ms": frames(orig_t + rng.uniform(-0.02, 0.02, len(orig_t)), c["start"], orig_p)}
        for k, t in cases.items():
            n = len(t) * (len(pieces) if k == "every_16th" else 1)       # every drum: one hit per drum per line
            out[k].append((on_grid(t, b) if len(t) else np.zeros(0), n, m))
    if m_total == 0:
        continue
    raws = {}
    for k, parts in out.items():
        s_all = np.concatenate([p[0] for p in parts])
        n, m = sum(p[1] for p in parts), sum(p[2] for p in parts)
        if k == "every_16th":                                         # the same grid score for each drum's copy
            s_all = np.repeat(s_all, len(pieces))
        raws[k] = 0.0 if n == 0 else float(s_all.mean() * min(n / m, m / n))
    per_song[s] = raws

ceil = np.array([r["ceiling"] for r in per_song.values()])
flo = np.array([r["floor"] for r in per_song.values()])
kept = [s for s, r in per_song.items() if r["ceiling"] - r["floor"] >= EXCLUDE_BELOW]
print(f"\n1  {len(per_song)} validation songs with drums and beats")
print(f"   raw ceiling (original part): median {np.median(ceil):.3f}, 10th percentile {np.percentile(ceil, 10):.3f}, "
      f"min {ceil.min():.3f}")
print(f"   raw floor (next song's part, +3/8 beat): median {np.median(flo):.3f}, 90th percentile "
      f"{np.percentile(flo, 90):.3f}, max {flo.max():.3f}")
print(f"   ceiling - floor: median {np.median(ceil - flo):.3f}; left out (< {EXCLUDE_BELOW}): "
      f"{len(per_song) - len(kept)} songs ({(len(per_song) - len(kept)) / len(per_song):.0%})")

print("\n2  rescaled 0-100 between each song's floor and ceiling, clipped, averaged over the kept songs")
summary = {}
for k in ("ceiling", "floor", "silence", "random", "every_16th", "one_hit", "jitter_10ms", "jitter_20ms"):
    sc = [100 * np.clip((per_song[s][k] - per_song[s]["floor"]) / (per_song[s]["ceiling"] - per_song[s]["floor"]),
                        0, 1) for s in kept]
    rw = [per_song[s][k] for s in kept]
    summary[k] = {"rescaled_mean": float(np.mean(sc)), "rescaled_median": float(np.median(sc)),
                  "raw_median": float(np.median(rw))}
    print(f"   {k:<13} rescaled mean {np.mean(sc):6.1f}  median {np.median(sc):6.1f}   (raw median {np.median(rw):.3f})")

report = {"songs": len(per_song), "kept": len(kept), "left_out": sorted(set(per_song) - set(kept)),
          "ceiling_raw": {"median": float(np.median(ceil)), "p10": float(np.percentile(ceil, 10))},
          "floor_raw": {"median": float(np.median(flo)), "p90": float(np.percentile(flo, 90))},
          "cases": summary, "per_song": per_song}
(reports_dir() / "beat_alignment_checks.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(f"\nwrote {reports_dir() / 'beat_alignment_checks.json'}")
