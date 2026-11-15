# Main test result, prereg-v5.4 (scored 12 Nov 2026)

Status: one-time locked evaluation on the full official test split (151 songs, 604 clips). No settings were changed for this scoring. Results written to reports/win_test.json.

## Result

**Main win rule: not met.** The real network did not beat the best random control by more than the control spread.

| Network | Run | Checkpoint | Beat alignment (150 songs) |
|---|---|---|---|
| Control 1 | v53_control1 | step 1000 | 24.079 |
| Real | v53_real | step 1000 | 19.474 |
| Control 2 | v53_control2 | step 1000 | 19.004 |
| Control 5 | v53_control5 | step 500 (best on validation) | 17.449 |
| Control 6 | v54_control6 | step 1000 | 18.076 |
| Control 3 | v53_control3 | step 1000 | 15.403 |

- Best control: 24.079 (control 1).
- Seed spread (sample standard deviation of the five control scores): 3.233.
- Margin: 19.474 − 24.079 − 3.233 = **−7.838**. The rule needs a margin above zero.
- One song is left out of the beat-alignment facet for every network (150 counted), as the scorecard rules require.

**Supporting rule: not met.** The real network must beat the controls' median on all three of tempo following, listening and groove. It beat the median on two of three:

| Facet | Real | Controls' median | Beats median? |
|---|---|---|---|
| Tempo following | 12.060 | 12.892 | No |
| Listening | 11.273 | 10.471 | Yes |
| Groove | 6.783 | 6.328 | Yes |

## Plain-language summary

- We compared the real fly wiring against five versions of the same network with the wiring scrambled. The question was whether the real wiring plays drums in time with a song better than the scrambled versions.
- On the main measure, how well the drum hits line up with the beat, the real wiring scored 19.5. The best scrambled version scored 24.1, and the other scrambled versions ranged from 15.4 to 19.0. The real wiring is not clearly ahead, and on this measure one scrambled version is clearly ahead of it.
- On the supporting measures, the real wiring is ahead on listening (how closely its hits match the original drum part) and on groove (whether its hits repeat in the bar the way the original's do). It is behind on tempo following.
- On this test, the locked rule does not show that the real wiring plays better than scrambled wiring.

## Context and caveats for any write-up

- **Control 4 was refused** at its starting weights by the start check (silence settled 0.556 against the limit 0.05). It was replaced by **seed 6** under prereg-v5.4, decided before any test scoring. See config/CHANGELOG.md.
- **Control 5 is scored from its step-500 checkpoint**, the best on validation under the locked rule. Its step-1000 checkpoint was not used.
- **Test split choice:** the baby set has only three test songs. The full official split (151 songs) was chosen for scoring after training had finished and before any test scoring. The networks were trained only on the baby training songs; the scoring songs did not appear in that training data.
- **Test clips:** drawn once with numpy seed 2026, four 16-second windows per song (data/work/data/full/test_clips.json). The locked settings give the clip length and count but not a seed, so the seed was chosen by me, not fixed in advance.
- **Scoring code:** scripts/score_win.py was written for this evaluation. It was checked on the validation split (3 songs, 12 clips, a code check only, no results used) before the test run. Its model setup is the baby setup each network was trained under.
- **Training runs:** all five trained networks passed the silence test at each check. The validation mean F1 scores were close together and near the floor, as in the earlier locked runs.
- **Not checked by this scoring:** whether the result holds with other clip draws or a different test seed. A second draw would be a new evaluation and would need its own rule.
- Control 6 finished training on 12 November, after the other runs. Its step-1000 checkpoint was used.

## Files

- Scores and win decision: reports/win_test.json
- Scoring script: scripts/score_win.py
- Test clips: data/work/data/full/test_clips.json
- Scoring log: scratchpad/win_test_full.log
- Training runs: runs/v53_real, runs/v53_control1, runs/v53_control2, runs/v53_control3, runs/v53_control5, runs/v54_control6
