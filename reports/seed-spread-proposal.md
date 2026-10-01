# Proposal: define "seed spread" in the main win rule

Status: **proposal, not locked.** Nothing in `config/locked.yaml` or the code has changed.

## The gap in the locked rule

`config/locked.yaml`:

```yaml
win_rule:
  main: real_beats_all_5_random_on_beat_alignment_by_more_than_seed_spread
  supporting: real_beats_random_median_on_4_of_6_other_facets
  evaluated_on: test_split_once
```

"Seed spread" is not defined anywhere: not in `locked.yaml`, the CHANGELOG or the scripts. Two readings are both
natural, and they make the rule differ in strictness by about half again (below). It has to be settled before any
network is scored for the win rule, and preferably before any trained control is scored on validation songs, so
the choice can't be influenced by how the controls turn out.

## What it is not about

Choosing now is not tuning to a result. No real-against-control score on the main metric exists yet: beat alignment
is part of the scorecard, which isn't built, and no network has been trained beyond memorisation. The only evidence
used below is a simulation of the rule itself (`scripts/power_estimate.py`), which says nothing about which network
is better.

## The readings

The real network has one wiring and one training run (init seed 0). Only the 5 random controls (control seeds 1-5)
give a spread, so "seed spread" is the spread of the 5 controls' scores on the same metric and test songs.

"Beats all 5 by more than the seed spread" is read as: **real score > highest control score + spread.**

| Reading | Spread | False wins with no true effect (simulated) | True advantage for an 80% chance of winning, control spread 0.01 (hit F1 stand-in) |
|---|---|---|---|
| A. Standard deviation | sample sd of the 5 control scores (divisor n - 1 = 4) | about 6% | about 0.041 (47% of the headroom) |
| B. Range | highest minus lowest control score | about 1.5% | roughly half as much again |

From `reports/power_estimate.txt`: hit F1 on kick, snare and closed hi-hat standing in for beat alignment;
song-sampling noise from the feasibility check, scaled to the 151 test songs; the real spread between controls
unknown and swept.

## Recommendation: A, the sample standard deviation

- **It is the usual meaning** of "spread" when results are compared across seeds.
- **Its false-win rate is already strict:** about 6% with no true effect, and the real network also has to beat the
  best of 5 controls outright.
- **The range is much stricter for little gain in protection.** With 5 controls, the range is usually 2-2.5x the
  standard deviation. Under the simulation it takes roughly half as much again of the true advantage to reach the
  same chance of a win, when the power estimate already shows the headroom is small.

**Against A, stated plainly:** a standard deviation from 5 numbers is itself uncertain. One unusual control can make
it large (making the rule stricter) or 5 close controls can make it small. The range has the same weakness, more so.
Neither reading removes it; more controls would, which is a separate, costlier decision.

## Exact wording proposed

```yaml
win_rule:
  main: real_beats_all_5_random_on_beat_alignment_by_more_than_seed_spread
  seed_spread: sample_sd_of_the_5_control_scores   # divisor 4; same metric, same test songs, unrounded
  main_test: real_score > max(control_scores) + seed_spread   # a tie is not a win
  supporting: real_beats_random_median_on_4_of_6_other_facets
  evaluated_on: test_split_once
```

The supporting rule (median of the controls) is unaffected.

## Also recommended (report only, changes no rule)

**Score the 5 trained controls on validation songs before the test split is touched,** and report their spread on
the stand-in and main metrics. It doesn't change the rule or the test; it tells us, before the one test evaluation,
how large an effect the rule could detect. If the controls spread widely, that is known and written down in advance.

## If accepted

1. **`locked.yaml`:** the two lines above, in a new tag (on its own, or alongside prereg-v6 if that is accepted).
2. **CHANGELOG line (draft):**

   > 2026-09-30 - prereg-vN: "seed spread" in the main win rule defined as the sample standard deviation (divisor 4)
   > of the 5 random controls' beat-alignment scores on the test songs; the real network wins if its score exceeds
   > the highest control's by more than that. The term was undefined; settled before any network is scored on the
   > main metric. Simulated false-win rate with no true effect: about 6% (scripts/power_estimate.py, hit F1 standing
   > in for beat alignment). Using the range instead would have cut that to about 1.5% at the cost of roughly half
   > as much again of the true advantage needed to win.
3. **Code:** when the scorecard is built, the win-rule function computes exactly this, with a test on a hand-made
   example for each outcome (win, loss, tie).
