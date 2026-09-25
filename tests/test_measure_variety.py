"""The variety measures, pinned on patterns whose answers are known by hand.

``measure_variety.py`` reports how much the fly's drumming changes and how
varied it is. Those numbers will be quoted as "the style dial changes X% of
hits", so each measure is checked here against a case worked out by hand.
"""
import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from measure_variety import (  # noqa: E402
    bar_patterns, bar_variety, busyness, change, groove_similarity, pick_songs,
)

DRUMS = ["kick", "snare", "hat_closed"]
BPM = 120.0                     # one 4/4 bar = 2 s, one 16th = 0.125 s


def groove(bars: int, fill_bar: int | None = None):
    """Kick on 1 and 3, snare on 2 and 4, eighth-note hats; one bar can be a fill."""
    hits = []
    for b in range(bars):
        t0 = 2.0 * b
        if b == fill_bar:
            hits += [("snare", t0 + 0.125 * k, 0.8) for k in range(16)]
            continue
        hits += [("kick", t0, 0.9), ("kick", t0 + 1.0, 0.9),
                 ("snare", t0 + 0.5, 0.8), ("snare", t0 + 1.5, 0.8)]
        hits += [("hat_closed", t0 + 0.25 * k, 0.5) for k in range(8)]
    return hits


def test_identical_renders_change_nothing_and_disjoint_ones_everything():
    g = groove(4)
    assert change(g, g) == 0.0
    assert change([], []) == 0.0
    # 125 ms is half the hat spacing, so no hit lands within 50 ms of any hit
    # of its own drum. (A 200 ms shift would put each hat 50 ms from the next.)
    shifted = [(d, t + 0.125, v) for d, t, v in g]
    assert change(g, shifted) == 1.0
    assert change(g, [(d, t + 0.03, v) for d, t, v in g]) == 0.0   # within it


def test_change_counts_the_share_of_hits_that_differ():
    a = [("kick", 0.0, 1.0), ("kick", 1.0, 1.0)]
    b = [("kick", 0.0, 1.0), ("snare", 1.0, 1.0)]          # one of two hits swapped
    assert change(a, b) == 0.5


def test_a_steady_groove_has_one_distinct_bar_and_a_fill_adds_one():
    steady = bar_variety(bar_patterns(groove(4), BPM, 8.0, DRUMS))
    assert steady["bars"] == 4
    assert steady["distinct_share"] == 0.25
    assert steady["mean_bar_difference"] == 0.0
    fill = bar_variety(bar_patterns(groove(4, fill_bar=3), BPM, 8.0, DRUMS))
    assert fill["distinct_share"] == 0.5
    assert 0.0 < fill["mean_bar_difference"] < 1.0


def test_a_hit_just_before_the_downbeat_lands_on_the_next_bar():
    grid = bar_patterns([("kick", 1.99, 1.0)], BPM, 4.0, DRUMS)
    assert not grid[0].any() and grid[1][0]


def test_incomplete_bars_are_not_counted():
    assert bar_patterns(groove(3), BPM, 5.0, DRUMS).shape[0] == 2


def test_groove_similarity_ignores_which_bar_but_not_where_in_the_bar():
    g = groove(4)
    assert np.isclose(groove_similarity(g, g, BPM, DRUMS), 1.0)
    one_bar = [h for h in g if h[1] < 2.0]
    assert np.isclose(groove_similarity(g, one_bar, BPM, DRUMS), 1.0)
    offbeat = [(d, t + 0.125, v) for d, t, v in g]         # every hit a 16th late
    assert groove_similarity(g, offbeat, BPM, DRUMS) < 0.2


def test_busyness():
    b = busyness(groove(2), 4.0, DRUMS)
    assert b["hits_per_s"] == 24 / 4.0
    assert np.isclose(sum(b["share"].values()), 1.0)
    assert np.isclose(b["share"]["hat_closed"], 16 / 24)


def test_songs_are_4_4_grooves_spread_across_styles_and_seeded(tmp_path):
    rows = []
    for i, (style, beat, ts, dur) in enumerate([
        ("rock/1", "beat", "4-4", 30), ("rock/2", "beat", "4-4", 30),
        ("funk/1", "beat", "4-4", 30), ("jazz/1", "fill", "4-4", 30),
        ("latin/1", "beat", "6-8", 30), ("soul/1", "beat", "4-4", 5),
    ]):
        (tmp_path / f"{i}.wav").write_bytes(b"")
        rows.append({"id": str(i), "style": style, "beat_type": beat, "time_signature": ts,
                     "duration": str(dur), "split": "test", "audio_filename": f"{i}.wav"})
    with (tmp_path / "info.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    picked = pick_songs(tmp_path, 2, 20.0, seed=0)
    assert {r["style"].split("/")[0] for r in picked} == {"rock", "funk"}
    assert pick_songs(tmp_path, 2, 20.0, seed=0) == picked
    assert len(pick_songs(tmp_path, 10, 20.0, seed=0)) == 3   # only three qualify
