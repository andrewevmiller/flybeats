"""Reference checks for every facet of the proposed scorecard (reports/scorecard-proposal.md, not locked).
Drum parts and beat annotations only: no audio, no network, so nothing here can tune a facet to a result.

Facets, as proposed (raw numbers; rescaled per song between floor and ceiling, clipped 0-100; a song is left out of
a facet where raw ceiling - raw floor < 0.1):
  beat alignment   mean over hits of (1 + cos(2 pi x 16th position)) / 2, x density balance b
  tempo following  |mean over hits of exp(2 pi i x 16th position)| x b; floor: the next song's part at its own timing
  listening        hit F1 against the original, per drum, locked tolerance (10% of the local beat), pooled
  groove           per drum, correlation (clipped at 0) of hit counts at the bar's 16 positions (nearest, from
                   downbeats), weighted by the original's share of hits, x b
  style fit is reported only, not counted in the supporting rule
  style fit        1 - Jensen-Shannon divergence (base 2) of the share of hits per drum
  b = min(n / m, m / n), n hits played, m in the original; 0 if n = 0.
Test outputs (on the full set's fixed validation clips, 4 per song, first 2 s ignored, 5 ms frames), each set against
the outcome the proposal fixed in advance: the original part, the floor, silence, random times, every 16th on every
drum, +/-10 and +/-20 ms jitter, the original 30 ms late, steady 16ths at another tempo, the original with its drums
permuted, and steady hi-hat 8th notes. Hand-made examples with known answers run first; the script stops if any fails.
Writes reports/scorecard_checks.json.
    .venv\\Scripts\\python.exe scripts\\scorecard_checks.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.data.manifest import load_manifest, set_dir  # noqa: E402
from flybeats.data.sampler import load_validation_clips  # noqa: E402
from flybeats.model.loss import match_counts, tolerance_frames  # noqa: E402

SHIFT_BEATS = 3 / 8
EXCLUDE_BELOW = 0.1
FACETS = ["beat_alignment", "tempo_following", "listening", "groove", "style_fit", "groove_cosine_rejected"]
# groove: revised after the first run showed random timing scoring 44-51 on the cosine version, which is kept as
# groove_cosine_rejected for the record (cosine credits
# hits spread evenly over the bar): per drum, the correlation (clipped at 0) of the two 16-position profiles (an even spread
# correlates 0), averaged with the original's share of hits as weights, x b. Reference parts only.
cfg, paths = load_locked(), load_paths()
frame_s = cfg["training_data"]["frame_ms"] / 1000
clip_s = cfg["training_data"]["clip_seconds"]
skip = cfg["training_data"]["warmup_ignored_seconds"]
fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
pieces = cfg["kit"]["pieces"]
D = len(pieces)


# ---- the measures ----
def beat_position(times, beats):
    b = np.asarray(beats, float)
    k = np.clip(np.searchsorted(b, times, side="right") - 1, 0, len(b) - 2)
    return k + (times - b[k]) / (b[k + 1] - b[k])


def time_of(position, beats):
    b = np.asarray(beats, float)
    k = np.clip(np.floor(position).astype(int), 0, len(b) - 2)
    return b[k] + (position - k) * (b[k + 1] - b[k])


def balance(n, m):
    return 0.0 if n == 0 or m == 0 else min(n / m, m / n)


def bar_slot(times, downbeats):
    d = np.asarray(downbeats, float)
    k = np.clip(np.searchsorted(d, times, side="right") - 1, 0, len(d) - 2)
    return np.round(16 * (times - d[k]) / (d[k + 1] - d[k])).astype(int) % 16          # nearest of the 16 positions


def facets(clips, beats, downbeats):
    """Raw facet values for one song. clips: list of (start, pred_frames, pred_pieces, orig_frames, orig_pieces)."""
    t_pred = np.concatenate([s + f * frame_s for s, f, _, _, _ in clips])
    p_pred = np.concatenate([p for _, _, p, _, _ in clips]).astype(int)
    t_orig = np.concatenate([s + f * frame_s for s, _, _, f, _ in clips])
    p_orig = np.concatenate([p for _, _, _, _, p in clips]).astype(int)
    n, m = len(t_pred), len(t_orig)
    b = balance(n, m)
    out = {}
    pos = beat_position(t_pred, beats) if n else np.zeros(0)
    out["beat_alignment"] = float(((1 + np.cos(2 * np.pi * 4 * pos)) / 2).mean() * b) if n else 0.0
    out["tempo_following"] = float(np.abs(np.exp(2j * np.pi * 4 * pos).mean()) * b) if n else 0.0
    tp = fp = fn = 0
    for s, f, p, fo, po in clips:
        bt = np.asarray(beats) - s
        for d in range(D):
            pred, true = np.unique(f[p == d]), np.unique(fo[po == d])
            c = match_counts(pred, true, tolerance_frames(true, bt, frame_s, fraction))
            tp, fp, fn = tp + c[0], fp + c[1], fn + c[2]
    out["listening"] = 2 * tp / max(2 * tp + fp + fn, 1)
    if n and len(downbeats) > 1:
        hp = np.zeros((D, 16))
        ho = np.zeros((D, 16))
        np.add.at(hp, (p_pred, bar_slot(t_pred, downbeats)), 1)
        np.add.at(ho, (p_orig, bar_slot(t_orig, downbeats)), 1)
        out["groove_cosine_rejected"] = float((hp * ho).sum() / max(np.linalg.norm(hp) * np.linalg.norm(ho), 1e-12) * b)
        corr = []
        for d in range(D):
            x, y = hp[d], ho[d]
            corr.append(0.0 if x.std() == 0 or y.std() == 0 else max(0.0, float(np.corrcoef(x, y)[0, 1])))   # clipped at 0
        w = ho.sum(1) / max(ho.sum(), 1)
        out["groove"] = float(np.dot(w, corr) * b)
    else:
        out["groove_cosine_rejected"] = out["groove"] = 0.0
    if n and m:
        a, c = np.bincount(p_pred, minlength=D) / n, np.bincount(p_orig, minlength=D) / m
        mid = (a + c) / 2
        kl = lambda x: float(np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / np.where(mid > 0, mid, 1)), 0)))
        out["style_fit"] = 1 - (kl(a) + kl(c)) / 2
    else:
        out["style_fit"] = 0.0
    return out


# ---- hand-made examples ----
steady = np.arange(0, 40, 0.5)                                       # 120 BPM, 4/4
bars = steady[::4]
grid_f = (np.arange(2, 14, 0.125) / frame_s).round().astype(int)     # every 16th from 2 s to 14 s, in frames
kick = np.zeros(len(grid_f), int)
same = [(0.0, grid_f, kick, grid_f, kick)]
r = facets(same, steady, bars)
assert all(np.isclose(r[k], 1) for k in FACETS if k != "groove"), r   # one slot per bar position: no spread                   # identical to the original: 1 everywhere
late = [(0.0, grid_f + 6, kick, grid_f, kick)]                       # 30 ms late, on a 125 ms 16th
r = facets(late, steady, bars)
assert np.isclose(r["tempo_following"], 1) and r["beat_alignment"] < 0.6, r
half = [(0.0, (grid_f + 12.5).astype(int), kick, grid_f, kick)]      # 62.5 ms: halfway between grid lines
assert facets(half, steady, bars)["beat_alignment"] < 0.01
silent = [(0.0, np.zeros(0, int), np.zeros(0, int), grid_f, kick)]
assert all(v == 0 for v in facets(silent, steady, bars).values())
other = [(0.0, grid_f, np.full(len(grid_f), 2), grid_f, kick)]       # right times, wrong drum
r = facets(other, steady, bars)
assert r["listening"] == 0 and r["groove_cosine_rejected"] == 0 and r["style_fit"] == 0 and r["groove"] == 0 and np.isclose(r["beat_alignment"], 1), r
print("hand-made examples: all as expected")

# ---- reference outputs on the validation songs ----
manifest, events, beats = load_manifest(paths["work_dir"], "full")
clips = load_validation_clips(set_dir(paths["work_dir"], "full") / "validation_clips.json")
ev = events[events.piece.isin(pieces)]
ev_t = {s: g.time.to_numpy() for s, g in ev.groupby("song")}
ev_p = {s: g.piece.map({q: j for j, q in enumerate(pieces)}).to_numpy() for s, g in ev.groupby("song")}
bt = {s: g.sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
dbt = {s: g[g.downbeat].sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
songs = sorted({c["song"] for c in clips})
floor_song = {s: songs[(i + 1) % len(songs)] for i, s in enumerate(songs)}
dur = manifest.set_index("song").duration
rng = np.random.default_rng(0)
first, last = round(skip / frame_s), round(clip_s / frame_s)
PERMUTE = np.roll(np.arange(D), 3)                                   # each drum played as another, fixed in advance


def to_frames(times, start, piece):
    f = np.floor((np.asarray(times) - start) / frame_s).astype(int)
    piece = np.asarray(piece, int)
    keep = (f >= first) & (f < last)
    if not keep.any():
        return np.zeros(0, int), np.zeros(0, int)
    pairs = np.unique(np.stack([f[keep], piece[keep]], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]


CASES = ["ceiling", "floor", "floor_own_timing", "silence", "random", "every_16th", "jitter_10ms", "jitter_20ms",
         "late_30ms", "other_tempo_16ths", "drums_permuted", "hihat_8ths"]
hh = pieces.index("hihat_closed")
raws = {}
for s in songs:
    b, dbs = bt.get(s), dbt.get(s)
    if b is None or len(b) < 2 or s not in ev_t:
        continue
    per_case = {k: [] for k in CASES}
    for c in (c for c in clips if c["song"] == s):
        st, lo, hi = c["start"], c["start"] + skip, c["start"] + clip_s
        sel = (ev_t[s] >= st) & (ev_t[s] < hi)
        ot, op = ev_t[s][sel], ev_p[s][sel]
        of, opp = to_frames(ot, st, op)
        m = len(of)
        fs = floor_song[s]
        fb, ft, fp = bt[fs], ev_t.get(fs, np.zeros(0)), ev_p.get(fs, np.zeros(0, int))
        b0, b1 = beat_position(np.array([lo, hi]), b)
        span = b1 - b0
        off = b0 if len(fb) - 1 >= b0 + span else max(0.0, len(fb) - 1 - span) * rng.random()
        q = beat_position(ft, fb)
        w = (q >= off) & (q < off + span)
        fstart = st if dur[fs] >= st + clip_s else max(0.0, dur[fs] - clip_s) * rng.random()
        w2 = (ft >= fstart) & (ft < fstart + clip_s)
        grid = time_of(np.arange(np.ceil(b0 * 4), np.floor(b1 * 4) + 1) / 4, b)
        grid = grid[(grid >= lo) & (grid < hi)]
        local16 = np.median(np.diff(grid)) if len(grid) > 1 else 0.125
        other = np.arange(lo, hi, local16 * 1.13)                    # steady 16ths, 13% slower than the song
        outs = {
            "ceiling": (of, opp),
            "floor": to_frames(time_of(q[w] - off + b0 + SHIFT_BEATS, b), st, fp[w]),
            "floor_own_timing": to_frames(ft[w2] - fstart + st, st, fp[w2]),
            "silence": (np.zeros(0, int), np.zeros(0, int)),
            "random": to_frames(rng.uniform(lo, hi, m), st, rng.permutation(opp)),
            "every_16th": to_frames(np.repeat(grid, D), st, np.tile(np.arange(D), len(grid))),
            "jitter_10ms": to_frames(ot + rng.uniform(-0.01, 0.01, len(ot)), st, op),
            "jitter_20ms": to_frames(ot + rng.uniform(-0.02, 0.02, len(ot)), st, op),
            "late_30ms": to_frames(ot + 0.030, st, op),
            "other_tempo_16ths": to_frames(other, st, rng.choice(opp, len(other)) if m else np.zeros(len(other), int)),
            "drums_permuted": (of, PERMUTE[opp]),
            "hihat_8ths": to_frames(grid[::2], st, np.full(len(grid[::2]), hh)),
        }
        for k, (f, p) in outs.items():
            per_case[k].append((st, f, p, of, opp))
    raws[s] = {k: facets(v, b, dbs if dbs is not None else b[::4]) for k, v in per_case.items()}

report = {"songs": len(raws), "facets": {}, "cases": {}}
kept = {}
for fct in FACETS:
    floor_case = "floor_own_timing" if fct == "tempo_following" else "floor"
    ceil = np.array([raws[s]["ceiling"][fct] for s in raws])
    flo = np.array([raws[s][floor_case][fct] for s in raws])
    kept[fct] = [s for s in raws if raws[s]["ceiling"][fct] - raws[s][floor_case][fct] >= EXCLUDE_BELOW]
    report["facets"][fct] = {"floor_case": floor_case, "ceiling_median": float(np.median(ceil)),
                             "floor_median": float(np.median(flo)), "gap_median": float(np.median(ceil - flo)),
                             "kept": len(kept[fct]), "left_out": len(raws) - len(kept[fct])}
print(f"\n{len(raws)} validation songs with drums and beats")
print(f"{'facet':<17}{'floor used':<19}{'raw ceiling':>12}{'raw floor':>10}{'gap':>7}{'left out':>10}")
for fct, r in report["facets"].items():
    print(f"{fct:<17}{r['floor_case']:<19}{r['ceiling_median']:>12.3f}{r['floor_median']:>10.3f}"
          f"{r['gap_median']:>7.3f}{r['left_out']:>6} ({r['left_out'] / len(raws):.0%})")

print("\nrescaled 0-100 (mean over each facet's kept songs)")
print(f"{'output':<19}" + "".join(f"{f:>17}" for f in FACETS))
for case in CASES:
    row = {}
    for fct in FACETS:
        floor_case = report["facets"][fct]["floor_case"]
        sc = [100 * np.clip((raws[s][case][fct] - raws[s][floor_case][fct])
                            / (raws[s]["ceiling"][fct] - raws[s][floor_case][fct]), 0, 1) for s in kept[fct]]
        row[fct] = float(np.mean(sc)) if sc else float("nan")
    report["cases"][case] = row
    print(f"{case:<19}" + "".join(f"{row[f]:>17.1f}" for f in FACETS))

report["per_song_raw"] = raws
(reports_dir() / "scorecard_checks.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(f"\nwrote {reports_dir() / 'scorecard_checks.json'}")
