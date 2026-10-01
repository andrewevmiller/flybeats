"""Reference checks for the locked scorecard (prereg-v5.2; flybeats.scoring.scorecard), on drum parts only: no audio,
no network, so nothing here can tune a facet to a result.

On the full set's fixed validation clips (4 per song, first 2 s ignored, 5 ms frames), each test output is scored
with the implemented facets against the outcome fixed in advance in reports/scorecard-proposal.md: the original part,
the floor, silence, random times, every 16th on every drum, +/-10 and +/-20 ms jitter, the original 30 ms late,
steady 16ths at another tempo, the original with its drums permuted, and steady hi-hat 8th notes.
(Before the facets were implemented this script computed them itself; those runs, including the rejected cosine
groove, are in git history: commits 9c54642 and 3a46b06.)
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
from flybeats.scoring.scorecard import FACETS, Song, beat_position, check_scorecard, time_of, to_part  # noqa: E402

cfg, paths = load_locked(), load_paths()
check_scorecard(cfg)
pieces = cfg["kit"]["pieces"]
D = len(pieces)
manifest, events, beats = load_manifest(paths["work_dir"], "full")
clips = load_validation_clips(set_dir(paths["work_dir"], "full") / "validation_clips.json")
ev = events[events.piece.isin(pieces)]
ev_t = {s: g.time.to_numpy() for s, g in ev.groupby("song")}
ev_p = {s: g.piece.map({q: j for j, q in enumerate(pieces)}).to_numpy() for s, g in ev.groupby("song")}
bt = {s: g.sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
dbt = {s: g[g.downbeat].sort_values("time").time.to_numpy() for s, g in beats.groupby("song")}
dur = manifest.set_index("song").duration
songs = sorted({c["song"] for c in clips})
rng = np.random.default_rng(0)
PERMUTE = np.roll(np.arange(D), 3)                                   # each drum played as another, fixed in advance
HH = pieces.index("hihat_closed")
CASES = ["ceiling", "floor", "floor_own_timing", "silence", "random", "every_16th", "jitter_10ms", "jitter_20ms",
         "late_30ms", "other_tempo_16ths", "drums_permuted", "hihat_8ths"]

per_song, kept = {}, {f: [] for f in FACETS}
for i, s in enumerate(songs):
    if s not in ev_t or s not in bt or len(bt[s]) < 2:
        continue
    fs = songs[(i + 1) % len(songs)]
    song = Song([c["start"] for c in clips if c["song"] == s], bt[s], dbt.get(s, []), (ev_t[s], ev_p[s]),
                (ev_t.get(fs, np.zeros(0)), ev_p.get(fs, np.zeros(0, int))), bt[fs], float(dur[fs]), cfg)
    part = lambda t, p, st: (st, *to_part(t, p, st, song.frame_s, song.first, song.last))
    outs = {k: [] for k in CASES}
    for (st, of, op) in song.original:
        lo, hi = st + song.first * song.frame_s, st + song.clip_s
        ot = st + of * song.frame_s
        b0, b1 = beat_position(np.array([lo, hi]), song.beats)
        grid = time_of(np.arange(np.ceil(b0 * 4), np.floor(b1 * 4) + 1) / 4, song.beats)
        grid = grid[(grid >= lo) & (grid < hi)]
        other = np.arange(lo, hi, (np.median(np.diff(grid)) if len(grid) > 1 else 0.125) * 1.13)   # 13% slower
        m = len(of)
        outs["silence"].append((st, np.zeros(0, int), np.zeros(0, int)))
        outs["random"].append(part(rng.uniform(lo, hi, m), rng.permutation(op), st))
        outs["every_16th"].append(part(np.repeat(grid, D), np.tile(np.arange(D), len(grid)), st))
        outs["jitter_10ms"].append(part(ot + rng.uniform(-0.01, 0.01, m), op, st))
        outs["jitter_20ms"].append(part(ot + rng.uniform(-0.02, 0.02, m), op, st))
        outs["late_30ms"].append(part(ot + 0.030, op, st))
        outs["other_tempo_16ths"].append(part(other, rng.choice(op, len(other)) if m else np.zeros(len(other), int), st))
        outs["drums_permuted"].append((st, of, PERMUTE[op]))
        outs["hihat_8ths"].append(part(grid[::2], np.full(len(grid[::2]), HH), st))
    outs["ceiling"], outs["floor"], outs["floor_own_timing"] = song.original, song.floor, song.floor_own_timing
    per_song[s] = {"ceiling_raw": song.ceiling_raw, "floor_raw": song.floor_raw,
                   "scores": {k: song.scores(v) for k, v in outs.items()}}
    for f in FACETS:
        if song.kept(f):
            kept[f].append(s)

report = {"songs": len(per_song), "facets": {}, "cases": {}}
print(f"{len(per_song)} validation songs with drums and beats")
print(f"{'facet':<17}{'raw ceiling':>12}{'raw floor':>10}{'gap':>7}{'left out':>10}")
for f in FACETS:
    ceil = np.array([r["ceiling_raw"][f] for r in per_song.values()])
    flo = np.array([r["floor_raw"][f] for r in per_song.values()])
    out = len(per_song) - len(kept[f])
    report["facets"][f] = {"ceiling_median": float(np.median(ceil)), "floor_median": float(np.median(flo)),
                           "gap_median": float(np.median(ceil - flo)), "kept": len(kept[f]), "left_out": out}
    print(f"{f:<17}{np.median(ceil):>12.3f}{np.median(flo):>10.3f}{np.median(ceil - flo):>7.3f}"
          f"{out:>6} ({out / len(per_song):.0%})")
print("\nrescaled 0-100 (mean over each facet's kept songs; tempo following's floor is the own-timing part)")
print(f"{'output':<19}" + "".join(f"{f:>17}" for f in FACETS))
for case in CASES:
    row = {f: float(np.mean([per_song[s]["scores"][case][f] for s in kept[f]])) for f in FACETS}
    report["cases"][case] = row
    print(f"{case:<19}" + "".join(f"{row[f]:>17.1f}" for f in FACETS))
report["per_song"] = {s: {"ceiling_raw": r["ceiling_raw"], "floor_raw": r["floor_raw"]} for s, r in per_song.items()}
(reports_dir() / "scorecard_checks.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(f"\nwrote {reports_dir() / 'scorecard_checks.json'}")
