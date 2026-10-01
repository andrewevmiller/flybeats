# Proposal: the full scorecard

Status: **accepted and locked as prereg-v5.2** (Andrew, 30 Sep 2026), together with the beat-alignment definition.
Not yet implemented in `flybeats.scoring`. No network has been scored with any facet. It takes in `reports/beat-alignment-proposal.md` (beat alignment, unchanged here) and is
meant to be locked with it as one tag, before anything is scored.

**Revised 30 Sep 2026 after the reference checks** (`scripts/scorecard_checks.py`, drum parts only, no network), as
this proposal allows: groove changed from cosine similarity to a clipped correlation (the cosine version gave random
timing 51 of 100); style fit reported only, not counted in the supporting rule (a third of songs left out); one
expected outcome corrected. Andrew chose the supporting rule's count: **all 3** of the other counted facets. Results
of the checks are in "Reference checks: results" below.

## The gap

`locked.yaml` names five facets and their weights, a floor and ceiling frame, and two win rules:

```yaml
scorecard:
  onset_tolerance: {fraction_of_local_beat: 0.10}
  floor: other_song_drum_part_same_tempo_offset_3_8_beat
  ceiling: original_drum_part
  clip_each_facet: [0, 100]
  failure_floor: 20
  musicianship_weights: {beat_alignment: 35, tempo_following: 25, listening: 15, style_fit: 15, groove: 10}
  automatic_score: rescale_without_listening
win_rule:
  main: real_beats_all_5_random_on_beat_alignment_by_more_than_seed_spread
  supporting: real_beats_random_median_on_4_of_6_other_facets
```

None of the five facets has a measure. The supporting rule counts "6 other facets", but only 4 others are named.
`failure_floor` has no stated use. This proposal defines all of them.

## Shared rules (all facets)

1. **Hits:** the locked peak rule (local peak of hit probability >= 0.5), per drum, at 5 ms frames. Velocity, where
   used, is the network's loudness output at the hit frame.
2. **Songs and clips:** test songs on fixed clips, 4 windows of 16 s per song, drawn once into `test_clips.json`
   (as validation's were); the first 2 s of each clip ignored. Clips are pooled per song.
3. **Grid:** each song's 16th notes, from its annotated beats (each beat interval divided into 4), so tempo changes
   are followed. Bars come from the annotated downbeats.
4. **Rescaling:** per song and facet, 100 x (raw - floor) / (ceiling - floor), clipped to 0-100. A facet's song
   score is left out where ceiling - floor < 0.1 (raw), decided from the reference parts alone before any network
   is scored, and listed. A facet's score is the mean over its kept songs, each song weighted equally.
5. **Floor part:** the next song in the same split, sorted by name (the last wraps to the first); its part written
   in beats from its own annotated beats, laid onto this song's beats from the same beat index, shifted 3/8 beat
   later. (One exception, tempo following, below.)
6. **Density balance** b = min(n / m, m / n), n the network's hits and m the original's, all drums; b = 0 if n = 0.
   Used where a facet would otherwise reward playing far too much or far too little.
7. **Musicianship score:** the weighted mean of the five facets with the locked weights (35 / 25 / 15 / 15 / 10).
   Reported, not used by either win rule.
8. **`failure_floor: 20`:** a facet's song score under 20 counts as a failure on that song; the share of songs
   failed is reported per facet. Not used by either win rule.

## The five facets

### Beat alignment (35): hits on the song's grid
As in `reports/beat-alignment-proposal.md`: each hit's on-grid score s = (1 + cos(2 pi p)) / 2, p its position in
16ths (1 on a grid line, 0 halfway between), averaged over hits, times b.

Reference checks already run (`scripts/beat_alignment_checks.py`, 267 validation songs, no network): ceiling median
0.986, floor median 0.021, no song left out. Rescaled: original +/-10 ms jitter 96.5, +/-20 ms 91.2, random times
49.3, every 16th on every drum 6.9, one on-grid hit per clip 0.9, silence 0.

**What it does not measure:** whether the hits are the drummer's. Steady 8th notes at the right density score near
100. That is what the other facets are for.

### Tempo following (25): keeping the song's pulse, at any offset
**Measure:** R = |mean over hits of exp(2 pi i x 4 p)|, p each hit's position in beats on the song's annotated beat
axis: 1 if every hit sits at the same place within the 16th, wherever that is; near 0 if hits drift against the
song's pulse. Raw = R x b.

**Its floor is the one exception:** the next song's part **at its own timing** (not laid onto this song's beats),
from the same clip start. The locked floor is at this song's tempo by construction, so it would follow the tempo
perfectly and score as high as the ceiling: the locked frame cannot measure tempo following. This exception is a
change to the locked frame, stated as one.

**How it differs from beat alignment:** a network that keeps the song's pulse but sits a fixed amount off the grid
(say consistently 30 ms late) scores high here and lower on beat alignment. One that plays at its own steady tempo
scores low on both.

### Listening (15): playing the drummer's hits
**Measure:** hit F1 against the original part, per drum, matched one to one within the locked onset tolerance (10%
of the local beat), counts pooled over drums: the measure training already validates with. A network only scores by
playing the right drum at the right moment, which it can only learn from what the band does.

The floor (3/8 beat later) is outside the tolerance, so it scores near 0; the ceiling is 1.

### Groove (10): the drummer's pattern within the bar
**Measure:** for the network and the original, count hits per drum at each of the 16 positions in the bar, each hit
at the **nearest** position (bars from the annotated downbeats), over the song's clips. Per drum, the correlation of
the two 16-position profiles, **clipped at 0** (0 if either profile is flat or the network plays none of that drum);
raw = the average over drums weighted by the original's share of hits, x b.

It rewards the right pattern even when single hits miss, which listening doesn't. The floor's 3/8-beat shift moves
every hit to another position, so it scores low.

**Changed after the reference checks.** First proposed as the cosine similarity of the two 8 x 16 counts (v0.0's
`metrics.groove_similarity`): random timing scored 51, because cosine credits hits spread evenly over the bar. A
correlation gives an even spread 0. Unclipped, the floor's raw score was slightly negative (median -0.083), so
silence (0) scored above the floor; clipping each drum's correlation at 0 fixes that. Positions were first taken by
rounding down, which put grid lines on the bin edges, so hits a few ms early fell into the previous position; the
nearest position fixes that.

### Style fit (15): the right drums, in the right proportions
**Measure:** raw = 1 - (Jensen-Shannon divergence, base 2, between the network's and the original's share of hits
per drum over the song's clips); 0 if the network plays nothing. 1 = the same mix of kick, snare, hi-hats, toms,
cymbals.

**Reported only; not counted in the supporting rule.** The reference checks confirmed it is weak: another song's
drum mix is usually close to this song's (raw floor median 0.825), so 90 of 270 songs (33%) fall under the 0.1 rule;
and it looks only at the mix, so random timing with the right drums scores 100. It stays in the musicianship score
with its locked weight, and is reported per song.

## The supporting rule: all 3 counted facets (decided)

Only 4 facets besides beat alignment are named, and style fit is reported only, so 3 are counted: tempo following,
listening and groove. **The real network must beat the 5 controls' median on all 3** (Andrew, 30 Sep 2026).

Why all 3 rather than 2 of 3: beating the median of 5 controls happens by chance half the time on any one facet when
the real wiring is no better. The chance of a false supporting win:

| Rule | If the facets were independent | If perfectly correlated |
|---|---|---|
| Locked "4 of 6" | 34% | 50% |
| 2 of 3 | 50% | 50% |
| **All 3** | **12.5%** | 50% |

Listening and groove both measure playing the drummer's part, so they move together; the real figure for all 3 is
somewhere around 12.5-25%. Each facet's comparison is reported on its own whatever the rule says, so a specific
advantage (say, tempo following alone) stays visible.

Considered and not chosen: **2 of 3** (a coin flip with no true effect, and weaker than the locked rule it replaces);
**adding two facets to make 6** (dynamics, arrangement), which would invent measures to fit a count whose origin isn't
recorded.

## Reference checks before locking (drum parts only, no network)

`scripts/beat_alignment_checks.py` grows into `scripts/scorecard_checks.py`, on the 267 validation songs:

| Output | Expected |
|---|---|
| Original part | 100 on every facet |
| Floor | 0 on every facet |
| Silence | 0 on every facet |
| Random times, right number of hits | middling on beat alignment; low on tempo following, listening and groove (corrected: first written "middling on tempo following", but random times keep no pulse, so low is right) |
| Every 16th on every drum | low on all |
| Original +/-10 / +/-20 ms jitter | high on beat alignment, tempo following and groove; listening still high (inside the tolerance) |
| Original shifted a fixed 30 ms late | lower on beat alignment, high on tempo following |
| Steady 16ths at a different tempo | low on beat alignment and tempo following |
| Original with drums permuted | high on beat alignment and tempo following, low on listening, groove and style fit |
| Steady 8th notes on the hi-hat at the original density | high on beat alignment, low on listening and groove |

Plus hand-made unit tests with known answers for each facet, and per facet the number of songs left out by the 0.1
rule. If a facet behaves differently from the table, the proposal is revised before locking, never after scoring.

## Reference checks: results

`scripts/scorecard_checks.py` on the full set's 270 validation songs (4 fixed clips each), drum parts only.
Hand-made examples all gave their known answers.

| Facet | Raw ceiling | Raw floor | Songs left out |
|---|---|---|---|
| Beat alignment | 0.986 | 0.021 | 3 (1%) |
| Tempo following (own-timing floor) | 0.989 | 0.016 | 11 (4%) |
| Listening | 1.000 | 0.009 | 3 (1%) |
| Groove (as revised) | 1.000 | 0.004 | 3 (1%) |
| Style fit | 1.000 | 0.825 | 90 (33%) |

Rescaled 0-100, mean over each facet's kept songs:

| Output | Beat alignment | Tempo following | Listening | Groove | Style fit |
|---|---|---|---|---|---|
| Original | 100 | 100 | 100 | 100 | 100 |
| Floor | 0 | 0 | 0 | 0 | 0 |
| Silence | 0 | 0 | 0 | 0 | 0 |
| Random times | 50 | 5 | 18 | 8 | 100 |
| Every 16th, every drum | 7 | 11 | 13 | 1 | 18 |
| +/-10 ms jitter | 97 | 95 | 99 | 98 | 100 |
| +/-20 ms jitter | 91 | 85 | 99 | 97 | 100 |
| 30 ms late | 67 | 99 | 97 | 96 | 100 |
| Steady 16ths, another tempo | 35 | 0 | 19 | 6 | 93 |
| Drums permuted | 100 | 100 | 12 | 20 | 5 |
| Hi-hat 8ths | 60 | 64 | 20 | 6 | 8 |

Everything matches the expected table except style fit's weakness (above). The locked floor would have scored 55 on
tempo following, confirming its exception. The rejected cosine groove scored random times 51.

## If accepted

1. **`locked.yaml`:** a `scorecard.facets` section with each measure as above, the tempo-following floor exception,
   the test clips, the 0.1 rule, style fit as reported only, and `win_rule.supporting:
   real_beats_random_median_on_all_3_of_tempo_following_listening_groove`; a new tag and CHANGELOG line.
2. **Code:** `flybeats.scoring` (one function per facet, the rescaling and the musicianship score) with unit tests;
   `scripts/scorecard_checks.py`.
3. **Then:** the feasibility model scored on validation songs with every facet (CPU), and the power estimate rerun
   on beat alignment.

## What changes from the locked text, in one place

| Locked | Proposed | Why |
|---|---|---|
| Facets named, not defined | Each defined above | They had no measure |
| One floor for every facet | Tempo following uses the other song's part at its own timing | The locked floor follows the tempo by construction |
| "4 of 6 other facets" | Beats the controls' median on all 3 of tempo following, listening and groove | Only 4 others are named; style fit is too weak to count |
| Style fit counted like the others | Reported only (still in the musicianship score) | 33% of songs left out; scores only the drum mix |
| `failure_floor: 20` | Share of songs under 20 reported per facet; no rule uses it | It had no stated use |
