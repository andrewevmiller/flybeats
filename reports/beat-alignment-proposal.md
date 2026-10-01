# Proposal: define beat alignment, the main win rule's metric

Status: **accepted and locked as prereg-v5.2** (Andrew, 30 Sep 2026), with the rest of the scorecard
(`reports/scorecard-proposal.md`). Not yet implemented in `flybeats.scoring`. No network has been scored with it.

## The gap

The main win rule (prereg-v5.1) is decided on beat alignment alone, but `locked.yaml` only names it:

```yaml
scorecard:
  onset_tolerance: {fraction_of_local_beat: 0.10}
  floor: other_song_drum_part_same_tempo_offset_3_8_beat
  ceiling: original_drum_part
  clip_each_facet: [0, 100]
  failure_floor: 20
  musicianship_weights: {beat_alignment: 35, tempo_following: 25, listening: 15, style_fit: 15, groove: 10}
  automatic_score: rescale_without_listening
```

So the frame is fixed: each facet is a raw number, rescaled to 0-100 between a floor (another song's drum part at
this song's tempo, shifted by 3/8 of a beat) and a ceiling (the song's original drum part), and clipped. **The raw
number itself is not defined anywhere.** The v0.0 version (`metrics.beat_alignment_error`: RMS distance from each
onset to a 16th-note grid built from one fixed tempo) is not carried over: it ignores the song's annotated beats, and
nothing stops a network scoring well by playing very few, or very many, hits on the grid.

**It has to be locked before anything is scored with it,** including the feasibility model, so the definition
can't drift towards whatever shows the most room.

## What the locked frame implies

The floor shifts a real drum part by 3/8 of a beat: one and a half 16th notes, so every hit that was on the 16th
grid lands exactly halfway between two grid lines. A measure that fits this frame must score that low and the
original part high. So beat alignment is about **whether the hits sit on the song's own 16th-note grid** (from its
annotated beats, `beats.parquet`, which come from each song's MIDI tempo map), not about which drum or rhythm is
played; those belong to the other facets.

## Requirements

1. Uses the song's annotated beats, so tempo changes are followed.
2. High for the original part, low for the 3/8-beat floor.
3. **Silence does not beat the floor,** and neither do very sparse or very dense outputs: one hit on beat 1 per
   clip, or a hit on every 16th on every drum, must not score near the ceiling.
4. No new threshold to choose, where possible.
5. Computed from the locked peak rule's hits (local peak of hit probability >= 0.5), all drums together.

## Recommended definition: on-grid score x density balance

For each clip:

1. **Grid:** the song's 16th-note positions: each annotated beat interval divided into 4 equal parts (so it
   follows the local tempo).
2. **On-grid score of each hit:** with d its distance to the nearest grid line and L the local 16th length,
   s = (1 + cos(2 pi d / L)) / 2. So s = 1 on a grid line, 0.5 a quarter of a 16th off, 0 halfway between grid
   lines. Smooth; no tolerance to choose.
3. **Density balance:** with n the number of predicted hits and m the original part's hits in the clip (all drums,
   scored frames only), b = min(n / m, m / n), and b = 0 if n = 0.
4. **Raw beat alignment = (mean s over the predicted hits) x b.** Between 0 and 1.

Per song: the same raw number for the original part (ceiling) and the floor part; the network's score is
100 x (raw - floor) / (ceiling - floor), clipped to 0-100 (the locked frame), from the song's clips pooled.
The main metric is the average over songs, each song weighted equally.

**What the reference outputs score** (what the definition is built to do; to be checked on data, below):

| Output | Mean s | b | Raw | Rescaled |
|---|---|---|---|---|
| Original part (ceiling), quantised | about 1 | 1 | about 1 | 100 |
| Floor: another song's part, shifted 3/8 beat | about 0 | about 1 | about 0 | 0 |
| Silence | none | 0 | 0 | 0 |
| Random times, right number of hits | about 0.5 | about 1 | about 0.5 | about 50 |
| Every 16th on every drum | 1 | small (about m / n) | small | low |
| One hit per clip, on the grid | 1 | about 1 / m | about 0 | about 0 |

## Details that have to be fixed with it

- **The floor part:** for each song, the drum part of the next song in the same split, sorted by name (the last
  wraps to the first), written in beats using its own annotated beats, laid onto this song's beats from the clip
  start (so its tempo is this song's), and shifted later by 3/8 of a beat.
- **Clips:** the test songs are scored on fixed clips made the way validation's were: 4 windows of 16 s per song,
  drawn once with a fixed seed into `test_clips.json`, never regenerated. The first 2 s of each clip are ignored,
  as in training.
- **Songs where the frame can't separate:** where the ceiling is not clearly above the floor (raw ceiling minus raw
  floor under 0.1), the song is left out of beat alignment. Example: a part in triplets is off the straight 16th
  grid, so its ceiling is low, and shifting it by 3/8 beat can land it near grid lines. Which songs are left out is
  decided from the reference parts alone, before any network is scored, and listed.
- **Swing:** swung parts sit partly off the straight grid, which lowers their ceiling; the per-song rescaling
  measures each network against the song's own original, so swing songs are not counted as failures. Swing is not
  corrected or modelled.

## Alternatives considered

- **Share of hits within a tolerance of the grid, x the same density balance.** Simpler to explain, but needs a
  tolerance. The locked onset tolerance (10% of the beat) is too loose for this: it covers 80% of all time around
  16th grid lines, so random hits would score 0.8. A tighter one (5%) would be a new number to choose.
- **Onset F-measure against the original part, all drums together.** Measures copying the drummer's rhythm, not
  sitting on the grid; it overlaps the listening and groove facets. Not beat alignment.
- **Beat-tracking F-measure (do the hits mark the quarter-note beats).** Penalises legitimate off-beat playing
  (8th-note hi-hats), so even the ceiling scores poorly.
- **v0.0's RMS distance to a fixed-tempo grid.** Ignores tempo changes and has no density guard.
- **A grid of 16ths and triplets together.** Gives triplet parts their due, but the 3/8-beat shift then lands
  close to triplet positions (0.375 against 0.333 of a beat), so the floor would score well. Rejected.

## Checks before locking (reference parts only, no network)

These test the measure on drum parts, not on any network, so they can be run before locking without tuning to a
result:

1. Raw ceiling and raw floor on the validation songs: ceiling clearly above floor for most songs; the count left out
   by the 0.1 rule.
2. Silence, random times with the right density, every-16th, and one-hit-per-clip: each lands where the table above
   says.
3. Hand-made examples with known answers as unit tests: hits on grid lines (s = 1), halfway (s = 0), a tempo change
   mid-clip followed by the grid.

## If accepted

1. **`locked.yaml`:** under `scorecard`, `beat_alignment:` with `grid: sixteenths_from_annotated_beats`,
   `hit_score: raised_cosine_of_distance_to_grid`, `density_balance: min_ratio_of_hit_counts`,
   `floor_song: next_in_split_by_name`, `exclude_if_ceiling_minus_floor_below: 0.1`, `test_clips: 4_per_song`; a new
   tag and CHANGELOG line.
2. **Code:** `flybeats.scoring.beat_alignment` with the unit tests above, and the reference-part checks as a script.
3. **Then:** the feasibility model's weights scored with it on validation songs (CPU), and the power estimate rerun on
   the real metric.

**Not covered here:** the other facets (tempo following, listening, style fit, groove) are also undefined, and the
supporting rule's "4 of 6 other facets" names more facets than the 5 listed. They need the same treatment before the
supporting rule is used; the main rule needs only this one.
