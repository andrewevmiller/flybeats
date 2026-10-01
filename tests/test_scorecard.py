import copy

import numpy as np
import pytest

from flybeats.scoring.scorecard import (FACETS, Song, aggregate, balance, beat_position, check_scorecard,
                                        hits_from_probs, supporting_win, to_part)

KICK, SNARE, HH = 0, 2, 3                                    # kit order: kick, hihat_pedal, snare, hihat_closed, ...


def pattern(beat_s, seconds):
    """A rock beat: kick on 1 and 3, snare on 2 and 4, closed hi-hat on 8th notes."""
    t, p = [], []
    for k in range(int(seconds / beat_s)):
        b = k * beat_s
        t += [b, b + beat_s / 2]
        p += [HH, HH]
        t.append(b)
        p.append(KICK if k % 2 == 0 else SNARE)
    return np.array(t), np.array(p)


@pytest.fixture(scope="module")
def song(cfg):
    beats = np.arange(0, 60, 0.5)                            # 120 BPM, 4/4
    floor_beats = np.arange(0, 60, 0.6)                      # the floor song: 100 BPM
    ft, fp = pattern(0.6, 60)
    fp = np.where(fp == HH, 7, fp)                           # the floor song rides the ride: a different drum mix
    return Song([4.0, 20.0], beats, beats[::4], pattern(0.5, 60), (ft, fp), floor_beats, 60.0, cfg)


def shifted(song, seconds=0.0, pieces=None):
    """The original part moved by `seconds` (and its drums replaced by `pieces`), as a played part."""
    out = []
    for s, f, p in song.original:
        t = s + f * song.frame_s + seconds
        out.append((s, *to_part(t, p if pieces is None else pieces[p], s, song.frame_s, song.first, song.last)))
    return out


def test_locked_scorecard_is_the_one_implemented(cfg):
    check_scorecard(cfg)
    bad = copy.deepcopy(cfg)
    bad["scorecard"]["facets"]["groove"] = "cosine"
    with pytest.raises(ValueError, match="groove"):
        check_scorecard(bad)
    bad = copy.deepcopy(cfg)
    bad["win_rule"]["supporting"] = "real_beats_random_median_on_4_of_6_other_facets"
    with pytest.raises(ValueError, match="supporting"):
        check_scorecard(bad)


def test_original_scores_100_and_floor_and_silence_0(song):
    assert all(v == pytest.approx(100) for v in song.scores(song.original).values())
    silence = [(s, np.zeros(0, int), np.zeros(0, int)) for s in song.starts]
    assert all(v == 0 for v in song.scores(silence).values())
    floor = song.scores(song.floor)
    assert all(floor[f] == pytest.approx(0) for f in FACETS if f != "tempo_following")
    assert song.scores(song.floor_own_timing)["tempo_following"] == pytest.approx(0)


def test_floor_is_shifted_3_8_beat_onto_this_songs_beats(song):
    pos = beat_position(np.concatenate([s + f * song.frame_s for s, f, _ in song.floor]), song.beats)
    assert np.allclose((pos * 4) % 1, 0.5, atol=0.045)       # halfway between 16th lines (5 ms frames: <= 0.04 of a 16th)
    assert song.floor_raw["beat_alignment"] < 0.01


def test_tempo_following_floor_keeps_its_own_tempo(song):
    assert song.raw(song.floor_own_timing)["tempo_following"] < 0.3   # 100 BPM against 120: no common pulse
    assert song.raw(song.floor)["tempo_following"] > 0.9              # the locked floor follows the tempo


def test_late_part_keeps_the_pulse_but_leaves_the_grid(song):
    raw = song.raw(shifted(song, 0.030))                    # 30 ms late on a 125 ms 16th
    assert raw["tempo_following"] == pytest.approx(1, abs=0.01)
    assert raw["beat_alignment"] < 0.6
    assert raw["listening"] == pytest.approx(1)               # inside the 50 ms tolerance


def test_wrong_drums_keep_the_timing_and_lose_the_rest(song):
    swap = np.arange(8)
    swap[[KICK, SNARE, HH]] = [5, 6, 7]                      # kick -> tom, snare -> crash, hi-hat -> ride
    raw = song.raw(shifted(song, pieces=swap))
    assert raw["beat_alignment"] == pytest.approx(1, abs=0.01) and raw["tempo_following"] == pytest.approx(1, abs=0.01)
    assert raw["listening"] == 0 and raw["groove"] == 0 and raw["style_fit"] == 0


def test_density_balance(song):
    assert balance(0, 10) == 0 and balance(5, 10) == 0.5 and balance(20, 10) == 0.5
    half = [(s, f[::2], p[::2]) for s, f, p in song.original]             # every other hit
    assert song.raw(half)["beat_alignment"] == pytest.approx(0.5, abs=0.02)


def test_part_must_be_on_the_songs_clips(song):
    with pytest.raises(ValueError, match="clips"):
        song.raw([(0.0, np.zeros(0, int), np.zeros(0, int))])


def test_hits_from_probs_uses_the_locked_peak_rule():
    prob = np.zeros((100, 8))
    prob[[10, 50, 70], 0] = [0.9, 0.6, 0.4]                  # 0.4: below the threshold
    prob[[51, 52], 3] = [0.7, 0.8]                           # one peak, at 52
    f, p = hits_from_probs(prob, 0.5, first=20)              # frame 10 is in the ignored start
    assert f.tolist() == [50, 52] and p.tolist() == [0, 3]


def test_aggregate_skips_left_out_songs_and_weights_the_musicianship_score():
    a = dict.fromkeys(FACETS, 80.0)
    b = dict.fromkeys(FACETS, 10.0)
    b["style_fit"] = None                                    # left out of style fit
    weights = {"beat_alignment": 35, "tempo_following": 25, "listening": 15, "style_fit": 15, "groove": 10}
    out = aggregate([a, b], weights, failure_floor=20)
    assert out["beat_alignment"]["score"] == 45 and out["beat_alignment"]["failed_share"] == 0.5
    assert out["style_fit"]["score"] == 80 and out["style_fit"]["songs"] == 1
    assert out["musicianship"] == pytest.approx((85 * 45 + 15 * 80) / 100)


def test_supporting_win_needs_all_3_strictly():
    controls = [dict(tempo_following=v, listening=v, groove=v) for v in (10, 20, 30, 40, 50)]
    assert supporting_win(dict(tempo_following=31, listening=31, groove=31), controls)[0]
    assert not supporting_win(dict(tempo_following=31, listening=31, groove=29), controls)[0]
    assert not supporting_win(dict(tempo_following=31, listening=31, groove=30), controls)[0]   # a tie is not a win
