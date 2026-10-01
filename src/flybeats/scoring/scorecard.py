"""The scorecard (config/locked.yaml scorecard, defined in prereg-v5.2; reports/scorecard-proposal.md).

A song's part is a list of clips, each (start_s, frames, pieces): hits as frame indices within the clip (5 ms frames)
and drum indices, inside the scored part of the clip (after the ignored first seconds), one hit per drum per frame.
The network's part comes from its outputs (`hits_from_probs`); the original part, the floor and the tempo-following
floor from the song's drum events and beats (`Song`).

Raw facets (b = min(n/m, m/n), n network hits and m original hits over the song, all drums; 0 if n = 0):
  beat_alignment   mean over hits of (1 + cos(2 pi x 16th position)) / 2, x b
  tempo_following  |mean over hits of exp(2 pi i x 16th position)|, x b
  listening        hit F1 against the original, per drum, locked tolerance, counts pooled over drums and clips
  groove           per drum, the correlation (clipped at 0) of hits at the nearest of the bar's 16 positions,
                   averaged with the original's share of hits as weights, x b
  style_fit        1 - Jensen-Shannon divergence (base 2) of the share of hits per drum (reported only)
Each is rescaled per song between the floor and the ceiling (the original part), clipped to 0-100; a song is left
out of a facet where the raw ceiling is within 0.1 of the raw floor. Tempo following uses the other song's part at
its own timing as its floor; every other facet, that part laid onto this song's beats and shifted 3/8 beat later.
"""
import numpy as np

from ..model.loss import match_counts, peaks, tolerance_frames

FACETS = ["beat_alignment", "tempo_following", "listening", "groove", "style_fit"]
COUNTED = ["tempo_following", "listening", "groove"]           # the supporting rule's facets
SHIFT_BEATS = 3 / 8
LOCKED = {                                                     # exactly the prereg-v5.2 wording this code implements
    "hits": "locked_peak_rule_per_drum_5ms_frames",
    "grid": "sixteenths_from_annotated_beats",
    "floor_song": "next_in_split_by_name_wrapping",
    "exclude_song_from_facet_if_raw_ceiling_minus_floor_below": 0.1,
    "facets": {
        "beat_alignment": "mean_over_hits_of_(1+cos(2pi*16th_position))/2_x_b",
        "tempo_following": "abs_mean_over_hits_of_exp(2pi*i*16th_position)_x_b",
        "tempo_following_floor": "next_song_part_at_its_own_timing",
        "listening": "hit_f1_vs_original_per_drum_locked_tolerance_counts_pooled",
        "groove": "per_drum_correlation_clipped_at_0_of_hits_at_nearest_of_16_bar_positions_weighted_by_original_share_x_b",
        "style_fit": "1_minus_jensen_shannon_base2_of_share_of_hits_per_drum",
    },
}


def check_scorecard(cfg):
    """Raise unless locked.yaml's scorecard is the one this code implements."""
    sc = cfg["scorecard"]
    for key, want in LOCKED.items():
        got = sc.get(key)
        if key == "facets":
            for f, w in want.items():
                if (got or {}).get(f) != w:
                    raise ValueError(f"scorecard.facets.{f} is {(got or {}).get(f)!r}; this code implements {w!r}")
        elif got != want:
            raise ValueError(f"scorecard.{key} is {got!r}; this code implements {want!r}")
    if cfg["win_rule"].get("supporting") != "real_beats_random_median_on_all_3_of_tempo_following_listening_groove":
        raise ValueError(f"win_rule.supporting {cfg['win_rule'].get('supporting')!r} is not the one implemented")


# ---- time on the song's beats ----
def beat_position(times, beats):
    """Fractional beat index of each time, following the annotated beats; extrapolated past either end."""
    b = np.asarray(beats, float)
    k = np.clip(np.searchsorted(b, times, side="right") - 1, 0, len(b) - 2)
    return k + (np.asarray(times, float) - b[k]) / (b[k + 1] - b[k])


def time_of(position, beats):
    b = np.asarray(beats, float)
    position = np.asarray(position, float)
    k = np.clip(np.floor(position).astype(int), 0, len(b) - 2)
    return b[k] + (position - k) * (b[k + 1] - b[k])


def bar_slot(times, downbeats):
    """Nearest of the bar's 16 positions, bars from the annotated downbeats."""
    d = np.asarray(downbeats, float)
    k = np.clip(np.searchsorted(d, times, side="right") - 1, 0, len(d) - 2)
    return np.round(16 * (np.asarray(times, float) - d[k]) / (d[k + 1] - d[k])).astype(int) % 16


def balance(n, m):
    return 0.0 if n == 0 or m == 0 else min(n / m, m / n)


# ---- parts ----
def to_part(times, pieces, start, frame_s, first, last):
    """Hit times and drums -> (frames, pieces) inside the scored frames [first, last), one hit per drum per frame."""
    f = np.floor((np.asarray(times, float) - start) / frame_s + 1e-9).astype(int)
    p = np.asarray(pieces, int)
    keep = (f >= first) & (f < last)
    if not keep.any():
        return np.zeros(0, int), np.zeros(0, int)
    pairs = np.unique(np.stack([f[keep], p[keep]], 1), axis=0)
    return pairs[:, 0], pairs[:, 1]


def hits_from_probs(prob, threshold, first, last=None):
    """A network's clip output (frames, drums) of hit probabilities -> (frames, pieces) by the locked peak rule."""
    prob = np.asarray(prob)
    last = prob.shape[0] if last is None else last
    frames, pieces = [], []
    for d in range(prob.shape[1]):
        pk = peaks(prob[:, d], threshold)
        pk = pk[(pk >= first) & (pk < last)]
        frames.append(pk)
        pieces.append(np.full(len(pk), d))
    f, p = np.concatenate(frames).astype(int), np.concatenate(pieces).astype(int)
    order = np.lexsort((p, f))
    return f[order], p[order]


class Song:
    """One song's reference parts on given clip starts: the original part, the floor and the tempo-following floor.
    events: (times, pieces) of the song's drum hits; floor_events/floor_beats/floor_duration: the floor song's."""

    def __init__(self, starts, beats, downbeats, events, floor_events, floor_beats, floor_duration, cfg):
        td = cfg["training_data"]
        self.frame_s = td["frame_ms"] / 1000
        self.clip_s = td["clip_seconds"]
        self.first = round(td["warmup_ignored_seconds"] / self.frame_s)
        self.last = round(self.clip_s / self.frame_s)
        self.fraction = cfg["scorecard"]["onset_tolerance"]["fraction_of_local_beat"]
        self.n_drums = len(cfg["kit"]["pieces"])
        self.starts = list(starts)
        self.beats = np.asarray(beats, float)
        self.downbeats = np.asarray(downbeats, float) if len(downbeats) > 1 else self.beats[::4]
        et, ep = (np.asarray(x) for x in events)
        ft, fp = (np.asarray(x) for x in floor_events)
        fb = np.asarray(floor_beats, float)
        part = lambda t, p, s: to_part(t, p, s, self.frame_s, self.first, self.last)
        self.original, self.floor, self.floor_own_timing = [], [], []
        fq = beat_position(ft, fb) if len(ft) else np.zeros(0)
        for s in self.starts:
            lo, hi = s + self.first * self.frame_s, s + self.clip_s
            sel = (et >= s) & (et < hi)
            self.original.append((s, *part(et[sel], ep[sel], s)))
            # the floor: the floor song's part in beats from the same beat index (wrapping if it is too short),
            # laid onto this song's beats and shifted 3/8 beat later
            b0, b1 = beat_position(np.array([lo, hi]), self.beats)
            span, n_fb = b1 - b0, len(fb) - 1
            off = b0 if n_fb >= b0 + span else b0 % max(n_fb - span, 1.0)
            w = (fq >= off) & (fq < off + span)
            self.floor.append((s, *part(time_of(fq[w] - off + b0 + SHIFT_BEATS, self.beats), fp[w], s)))
            # the tempo-following floor: the floor song's part at its own timing, from the same clip start
            fs = s if floor_duration >= s + self.clip_s else s % max(floor_duration - self.clip_s, 1.0)
            w2 = (ft >= fs) & (ft < fs + self.clip_s)
            self.floor_own_timing.append((s, *part(ft[w2] - fs + s, fp[w2], s)))
        self.ceiling_raw = self.raw(self.original)
        self.floor_raw = self.raw(self.floor)
        self.floor_raw["tempo_following"] = self.raw(self.floor_own_timing)["tempo_following"]

    def raw(self, played):
        """Raw facet values for a part played on this song's clips (a list of (start, frames, pieces), in order)."""
        if [c[0] for c in played] != self.starts:
            raise ValueError("the part must be on this song's clips, in order")
        fs, D = self.frame_s, self.n_drums
        t_pred = np.concatenate([s + f * fs for s, f, _ in played])
        p_pred = np.concatenate([p for _, _, p in played]).astype(int)
        t_orig = np.concatenate([s + f * fs for s, f, _ in self.original])
        p_orig = np.concatenate([p for _, _, p in self.original]).astype(int)
        n, m = len(t_pred), len(t_orig)
        b = balance(n, m)
        out = dict.fromkeys(FACETS, 0.0)
        if n:
            pos = beat_position(t_pred, self.beats)
            out["beat_alignment"] = float(((1 + np.cos(2 * np.pi * 4 * pos)) / 2).mean() * b)
            out["tempo_following"] = float(np.abs(np.exp(2j * np.pi * 4 * pos).mean()) * b)
        tp = fp_ = fn = 0
        for (s, f, p), (_, fo, po) in zip(played, self.original):
            bt = self.beats - s
            for d in range(D):
                true = np.unique(fo[po == d])
                c = match_counts(np.unique(f[p == d]), true, tolerance_frames(true, bt, fs, self.fraction))
                tp, fp_, fn = tp + c[0], fp_ + c[1], fn + c[2]
        out["listening"] = 2 * tp / max(2 * tp + fp_ + fn, 1)
        if n and m:
            hp, ho = np.zeros((D, 16)), np.zeros((D, 16))
            np.add.at(hp, (p_pred, bar_slot(t_pred, self.downbeats)), 1)
            np.add.at(ho, (p_orig, bar_slot(t_orig, self.downbeats)), 1)
            corr = [0.0 if hp[d].std() == 0 or ho[d].std() == 0 else max(0.0, float(np.corrcoef(hp[d], ho[d])[0, 1]))
                    for d in range(D)]
            out["groove"] = float(np.dot(ho.sum(1) / m, corr) * b)
            a, c = np.bincount(p_pred, minlength=D) / n, np.bincount(p_orig, minlength=D) / m
            mid = (a + c) / 2
            kl = lambda x: float(np.sum(x[x > 0] * np.log2(x[x > 0] / mid[x > 0])))
            out["style_fit"] = 1 - (kl(a) + kl(c)) / 2
        return out

    def kept(self, facet, below=LOCKED["exclude_song_from_facet_if_raw_ceiling_minus_floor_below"]):
        return self.ceiling_raw[facet] - self.floor_raw[facet] >= below

    def scores(self, played):
        """Rescaled 0-100 per facet; None for a facet this song is left out of."""
        raw = self.raw(played)
        return {f: (float(100 * np.clip((raw[f] - self.floor_raw[f]) / (self.ceiling_raw[f] - self.floor_raw[f]), 0, 1))
                    if self.kept(f) else None) for f in FACETS}


def aggregate(song_scores, weights, failure_floor):
    """Per-song scores -> each facet's mean over its kept songs, the share of those songs under failure_floor, and
    the musicianship score (the weighted mean of the facet means; reported only)."""
    out = {}
    for f in FACETS:
        vals = np.array([s[f] for s in song_scores if s[f] is not None], float)
        out[f] = {"score": float(vals.mean()) if len(vals) else float("nan"), "songs": int(len(vals)),
                  "failed_share": float((vals < failure_floor).mean()) if len(vals) else float("nan")}
    total = sum(weights.values())
    out["musicianship"] = float(sum(weights[f] * out[f]["score"] for f in FACETS) / total)
    return out


def supporting_win(real, controls):
    """real: {facet: score}; controls: list of 5 such. True if the real network beats the controls' median on all
    of COUNTED (strictly; a tie is not a win). Returns (wins, {facet: (real, median)})."""
    detail = {f: (real[f], float(np.median([c[f] for c in controls]))) for f in COUNTED}
    return all(r > m for r, m in detail.values()), detail
