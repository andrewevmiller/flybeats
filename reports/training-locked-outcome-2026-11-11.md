# Locked training outcome, 11 November 2026

Status: report of what happened. No settings changed. Written after the locked runs and a labelled trial.

## What was run
- The locked training runs (config/locked.yaml, training section: 1,000 steps, validation every 500, stop rule `activity_unbounded_or_not_settled`) for the real network and controls 1–5, in sequence, on the GPU.
- A start check was added to scripts/train.py on 11 Nov 2026: before training, the silence test is run at the starting weights, and a network that fails is refused. It is recorded in each run's log.
- A labelled trial (`--no-stop`, not locked): the real network trained for the full 1,000 steps with the stop rule off.

## Outcome of the locked runs

| Network | Start check | Step-500 validation: mean F1 | Silence settled ratio (limit 0.05) | Outcome |
|---|---|---|---|---|
| Real | Pass | 0.066 | 0.152 | Stopped at step 500 |
| Control 1 | Pass | 0.064 | 0.365 | Stopped at step 500 |
| Control 2 | Pass | 0.053 | 0.404 | Stopped at step 500 |
| Control 3 | Pass | 0.048 | 0.601 | Stopped at step 500 |
| Control 4 | **Refused** (settled 0.556 at start) | — | — | Did not train |
| Control 5 | Pass | 0.077 | 0.215 | Stopped at step 500 |

**No locked run reached 1,000 steps.** Every network that trained stopped at step 500 under the locked stop rule.

## Labelled trial (not locked)
- Real network, stop rule off (`runs/baby_real_trial_nostop`): step 500 mean F1 0.066 (settled 0.152); step 1,000 mean F1 0.071 (settled 0.150). The silence rule failed at both checks, and training continued.

## Checkpoint test (diagnostic, GPU)
- With every cell-type bias set to zero, every saved checkpoint passes the silence test (settled ratio 0.000). Zeroing the ear-band offsets changes nothing.
- The silence failure therefore comes from a learned resting level in the bias, not from activity that keeps itself going. Whether the network still plays without the bias is not yet measured.

## What this means
- The main test (real against controls on new songs, at 1,000 steps) **cannot be read under the locked rule**: no locked 1,000-step result exists, and control 4 never trains.
- At step 500 the scores are close to the floor. The spread across controls is about 0.013 (standard deviation), and the real network (0.066) sits inside it.
- The controls have about twice the adjustable dials of the real network (83,581 against 41,186), so the comparison is not matched in size.

## Disclosure (for any write-up)
"The locked training runs stopped at step 500 under the stop rule, because the silence test failed at that validation for the real network and controls 1, 2, 3 and 5. Control 4 was refused at the starting weights by the start check added on 11 November 2026, before any training. A labelled trial with the stop rule off shows the real network's mean F1 moved from 0.066 to 0.071 between steps 500 and 1,000, with the silence test still failing. A checkpoint test shows the silence failure is removed by zeroing the per-cell-type bias. No setting was changed to produce these results."

## Not decided here
- Whether the stop rule should change (locked change: new tag, CHANGELOG line, disclosure). Its measurement may need revising, since it counts a learned resting level as "not settled".
- Whether to run controls under the labelled no-stop trial.
