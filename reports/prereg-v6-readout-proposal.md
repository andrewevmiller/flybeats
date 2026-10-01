# Proposal: prereg-v6 readout scaling

Status: **proposal, not locked.** Nothing in `config/locked.yaml` or the code has changed. If Andrew accepts it, it
becomes prereg-v6: a new tag and a CHANGELOG line, written before the runs that use it.

Written 30 Sep 2026 from the prereg-v5 memorisation run of the real network (`reports/memorise/real/`, finished 15:37).
Control 1 under prereg-v5 (prereg-v4 wiring) finished at about 18:20; its numbers are in "Control 1's result".

## Why this proposal exists

prereg-v5 set the trigger in advance (`reports/prereg-v5-training-proposal.md`, "Not changed now", A): a readout-scaling
change is proposed only if, by step 3,000, the typical size of the scaled readout inputs is above 5 for any drum. At
setup it is about 1.

**At step 3,000 it was above 5 for every drum:**

| Drum | Typical size at setup | At step 250 | At step 3,000 | Average at step 3,000 |
|---|---|---|---|---|
| Kick | 0.66 | 8.2 | 13.2 | 8.8 |
| Pedal hi-hat | 0.61 | 7.9 | 13.2 | 8.7 |
| Snare | 0.98 | 10.2 | 7.2 | 5.4 |
| Closed hi-hat | 0.98 | 10.9 | 7.3 | 5.2 |
| Open hi-hat | 0.96 | 6.0 | 9.0 | 6.6 |
| Toms | 0.95 | 15.1 | 17.3 | 11.7 |
| Crash | 0.95 | 5.2 | 8.6 | 5.7 |
| Ride | 0.93 | 22.4 | 23.3 | 16.3 |

The prereg-v5 trigger for thin readouts (B) was **not** crossed. It needed failing drums to miss hits while
predicting few false ones. Instead recall is mostly fair and precision poor for every drum:

| | Kick | Snare | Closed hi-hat |
|---|---|---|---|
| F1 at step 3,000 | 0.62 | 0.35 | 0.52 |
| Recall | 0.84 | 0.44 | 0.78 |
| Precision | 0.50 | 0.29 | 0.39 |

The network finds most hits but adds many false ones.

## What the numbers say, and what they don't

**Say:**
- The scaling fitted once at setup no longer fits the network by step 250. The inputs sit 5–16 units off centre and
  spread over 5–17 times the fitted range.
- The drift happened early, mostly in the first 250 steps, and then stayed there.

**Don't:**
- The readout is linear, so any fixed scaling can be absorbed into its weights. In principle the drift doesn't stop
  the readout representing a good answer. What it plausibly changes is how easily the answer is **learned**.
  - Adam moves each weight by about the same amount per step, whatever its input's size. With inputs 7–23 times their
    intended size, each step moves the hit outputs 7–23 times as much as intended.
  - A large shared offset makes every weight also act as a bias.
- That is a reasoned mechanism, not a measured one. Nothing here proves that fixing the scaling will fix precision.
  This proposal changes the scaling because the pre-set trigger fired, and says plainly that the effect is unknown.

## Context found after the trigger fired: what the false hits are

**Added after the proposal was written. It does not change the proposal:** the trigger was set in advance and fired,
and reshaping the change to fit this breakdown would be tuning after the fact. It is recorded so expectations and the
final report are honest.

The real network's final weights (`reports/memorise/real/final.pt`, step 3,000) were run again (`scripts/false_hits.py`) on its 8 clips on the
CPU, and scored exactly as the check does. The F1 scores matched the logged ones: kick 0.623, snare 0.352, closed
hi-hat 0.520. Every false hit was then sorted:

| Kind | Meaning |
|---|---|
| Duplicate | Within tolerance of a real hit of that drum that another prediction already matched |
| Near miss | No real hit of that drum within tolerance, but one within twice it |
| Other drum | Neither of those, but within tolerance of another drum's real hit |
| Stray | Nothing real nearby |

| Drum | Correct / false / missed | Duplicate | Near miss | Other drum | Stray | Peaks per correct hit |
|---|---|---|---|---|---|---|
| Kick | 127 / 129 / 25 | 17% | 19% | 30% | 34% | 1.17 |
| Pedal hi-hat | 12 / 11 / 2 | 9% | 0% | 45% | 45% | 1.08 |
| Snare | 54 / 130 / 69 | 11% | 21% | 42% | 26% | 1.30 |
| Closed hi-hat | 210 / 327 / 60 | 32% | 28% | 10% | 29% | 1.50 |
| Open hi-hat | 18 / 14 / 8 | 21% | 14% | 21% | 43% | 1.17 |
| Toms | 10 / 12 / 2 | 33% | 0% | 50% | 17% | 1.40 |
| Crash | 12 / 16 / 8 | 6% | 12% | 38% | 44% | 1.08 |
| Ride | 33 / 37 / 24 | 41% | 32% | 19% | 8% | 1.45 |
| **All drums, 676 false hits** | | **24%** | **24%** | **23%** | **29%** | |

Drums with under 20 false hits have too few to read much into.

**What it shows:**
- **No single cause dominates.**
- **Duplicates** (one real hit drawing two or more peaks above the 0.5 threshold) matter most for the fast, frequent
  drums: closed hi-hat averages 1.5 peaks per real hit, ride 1.45.
- **Snare's main failure is confusion:** 42% of its false hits land on other drums' hits.
- **Near misses** are timing a little off.
- **Strays** are false alarms from nothing.

**What it means for this proposal:**
- **The readout drift is a plausible cause of duplicates and some strays** (oversized inputs make the output jumpy),
  and a weak explanation for confusion between drums or slightly-off timing. **prereg-v6 may help, but is unlikely to
  fix precision on its own.**
- **The locked scoring counts every local peak above 0.5 as a hit, with no minimum gap between hits,** which is part of
  why duplicates count against precision. Many onset-detection evaluations suppress peaks that are too close together.
  Changing the scoring now would be a design change after results, so it stays as locked. The final report notes it.
- **The same breakdown was run on control 1's prereg-v5 weights** (below). If its false hits were made up
  differently (less snare confusion, say), that would point at the wiring, not the setup.

**Control 1's breakdown** (`reports/memorise/control-1/final.pt`, step 3,000; F1 matched the logged ones):

| Drum | Correct / false / missed | Duplicate | Near miss | Other drum | Stray | Peaks per correct hit |
|---|---|---|---|---|---|---|
| Kick | 148 / 46 / 4 | 43% | 15% | 15% | 26% | 1.14 |
| Pedal hi-hat | 14 / 3 / 0 | 0% | 0% | 67% | 33% | 1.00 |
| Snare | 106 / 53 / 17 | 17% | 25% | 40% | 19% | 1.10 |
| Closed hi-hat | 252 / 51 / 18 | 27% | 37% | 10% | 25% | 1.06 |
| Open hi-hat | 22 / 5 / 4 | 40% | 40% | 20% | 0% | 1.09 |
| Toms | 12 / 7 / 0 | 43% | 0% | 0% | 57% | 1.25 |
| Crash | 17 / 14 / 3 | 21% | 7% | 43% | 29% | 1.18 |
| Ride | 35 / 34 / 22 | 24% | 32% | 29% | 15% | 1.23 |
| **All drums, 213 false hits** | | **28%** | **25%** | **24%** | **23%** | |

- **Control 1 makes about a third as many false hits (213 against 676), but the mix is almost the same:** roughly a
  quarter each. Snare's false hits are still mostly on other drums (40% against 42%).
- **So the kinds of error are a property of the setup, shared by both wirings; the real wiring makes more of every
  kind.** Closed hi-hat shows the largest gap in duplicates: 1.06 peaks per correct hit against 1.50.

### Readout diagnostics, tested and not proposed

Two other readout ideas were tried on both networks' final weights (CPU, the 8 memorisation clips). Neither is part of
this proposal.

| Readout | Real F1 (found / false / missed) | Control 1 F1 (found / false / missed) |
|---|---|---|
| Locked: one frame's peak at 0.5 | 0.521 (476 / 676 / 198) | 0.812 (606 / 213 / 68) |
| Peak at 50/51 = 0.98, the threshold consistent with the hit weight, fixed by formula (real only) | 0.000 (0 / 0 / 674) | not run |
| Window: true-scale probability summed over the tolerance window, at 0.5, fixed by formula | 0.321 (151 / 116 / 523) | 0.351 (151 / 35 / 523) |
| Peak, best threshold from a sweep | 0.527 | 0.813 |
| Window, best threshold from a sweep | 0.605 | 0.867 |

- **The consistent threshold is rejected:** no frame's output reaches 0.98, so it finds nothing.
- **The window readout at its formula threshold is too strict.** At its best threshold it beats the best single-frame
  readout on both networks (+0.08 real, +0.05 control 1). The sweep thresholds were picked on these same clips, so
  these are upper bounds, not scores. Using a window readout would mean fixing its threshold in advance on validation
  songs, as a separate preregistered change; it is not proposed here.

## Proposed change

**Refit the readout scaling every 250 steps, preserving the network's output exactly.**

1. **When:** at every check, every 250 steps (the same fixed schedule as the checks, set now), in memorisation and
   training alike.
2. **What:** recompute each readout motor neuron's mean and spread exactly as at setup: the same 16 fixed training
   clips (`readout_norm.seed`), the same 2 s warm-up and the same spread floor. The difference is that they go through
   the current network, not the untrained one.
3. **Output unchanged:** before the new scaling takes effect, adjust the readout's weights and biases (`w_hit`,
   `u_vel`, `c_hit`, `e_vel`) so every hit and loudness output is exactly what it was the moment before.
   - A neuron whose spread grew by a factor k gets its weight multiplied by k.
   - Each drum's bias absorbs the shift in means.

   So the refit itself changes no prediction. It only changes how the next steps' updates land.
4. **Optimiser state:** Adam's running averages for those four readout parameters no longer match their new scale, so
   they are reset to zero at each refit, with Adam's bias correction restarted for them. Every other parameter's
   optimiser state is left alone.
5. **Records:** each refit is logged (the step, and the typical input size before and after), so the drift and its
   correction stay visible.

**It applies to every network alike:** the real one and all controls, in memorisation and in training. Validation
and the silence test use whatever scaling is current.

## Alternatives considered

- **Make the scaling learned** (mean and spread as parameters). This is an exact reparametrisation of the readout, so
  it doesn't by itself stop the drift, and it adds parameters Adam would also move by fixed-size steps.
- **Normalise within each clip as it plays** (a running mean and spread over the clip). This changes what the model
  computes, making the readout adapt to each song. That is a design change, not a fix for drift.
- **Batch normalisation.** This makes one clip's output depend on the other clips in its batch, which breaks the
  causal, per-clip model.
- **Refit only once, after the first 250 steps.** Simpler, but the drift kept moving for some drums (toms, ride, kick
  grew until the end). A fixed every-check schedule is just as simple to state in advance.

## Checks before any run

- **Exactness:** a unit test that a refit leaves every hit and loudness output unchanged (float32, within rounding),
  and that each readout input's typical size is about 1 afterwards on the fit clips.
- **The CUDA-graph check:** the refit changes buffers and parameters in place, so the recorded step keeps working
  without being re-recorded. The check gains a part that runs 20 steps with a refit at step 10, recorded against
  ordinary, requiring exact agreement as before.
- **Timing:** a refit is one forward pass over 16 clips, about 3 s every 250 steps, about 0.4% of a run.

## If prereg-v6 memorisation still fails: stopping rule B

Chosen by Andrew on 30 Sep 2026, **before** any prereg-v6 run, so the decision does not depend on its result.

1. **No more changes for memorisation.** If the real network does not reach F1 ≥ 0.9 on kick, snare and closed hi-hat
   within 3,000 steps under prereg-v6, the pipeline is not changed again to make it pass: no prereg-v7 driven by
   memorisation.
2. **Training goes ahead anyway,** under prereg-v6 exactly as locked: BabySlakh training of the real network and all
   5 controls, then the full set when its step count is locked. The win rule (real against all 5 controls, on the
   test split, once) is unchanged. It is a relative comparison, so it can still be made when no network memorises
   well.
3. **The failure is reported up front,** not in a footnote. The report states that no network passed memorisation
   (or which did), gives their F1 scores, and says the comparison is between networks that all learn the task
   poorly in absolute terms. Any conclusion is limited to that.
4. **The only exception is a genuine bug:** code that does not do what the locked settings say it does, shown by a
   failing test. A bug may be fixed, with the fix, the test and its effect disclosed in the CHANGELOG and the report.
   A design or tuning change never counts as a bug.
5. **The same applies if memorisation passes.** Training goes ahead under prereg-v6 as locked either way.

Control 1's memorisation result is recorded either way and decides nothing.

## Changes made after seeing results

Every change to the design since prereg-v2, with what prompted it. None came after a comparison of the real network
against controls on validation or test data; there has been none. This list goes into the final report as written.

| Version | Change | What prompted it | Justified whatever the result? |
|---|---|---|---|
| prereg-v3 (29 Sep) | Standardise each readout motor neuron's rate before the readout | The real network's first memorisation run stalled on the audio-blind floor (loss 0.4932 against 0.4934, every F1 near 0) | Partly: untrained rates varied by about 0.001, too little for any readout to use, which is a property of the setup. The trigger was a failed run. |
| prereg-v4 (30 Sep) | Controls rewired only within neuron classes, keeping reciprocal pairs | **Seeing the old control 1 memorise better than the real network** (F1 at step 1,500: kick 0.78 against 0.53). Looking for why found 561–614 ear-to-motor shortcuts in every control, against 0 in the real slice. | Yes: ears never connect to motor neurons in the fly, so such controls are wrong whatever the results. The change made the controls harder to train, not easier. Loop strength was deliberately **not** matched, to avoid copying the real wiring's structure into the controls. **Disclosed because the trigger was a real-vs-control result, on memorisation only.** |
| prereg-v5 (30 Sep) | The memorisation check uses the locked learning-rate schedule; its settings moved into locked.yaml; diagnosis recorded | Both prereg-v3 memorisation runs failed, with the loss overshooting (rising on 36–47% of late steps) | Yes: the check had not been running the schedule it vouches for. Thresholds for further changes were set before the next run. |
| (implementation, 30 Sep) | Connection gradient formed once per clip | Profiling (speed) | Same quantity; checked against a float64 reference before use. No design change. |
| prereg-v6 (30 Sep, proposed) | Readout scaling refitted every 250 steps, output-preserving | The trigger set in prereg-v5 fired on every drum (typical readout input size 7.2–23.3 against 5) | Yes, by the pre-set rule. Its effect is not known in advance. Applies to every network alike. |

## Waiting on

Nothing. Control 1's result is below.

## Control 1's result

**Control 1 under prereg-v5 (prereg-v4 wiring), finished about 18:20: FAIL at step 3,000**, final loss 0.2827. F1:
kick 0.855, snare 0.752, closed hi-hat 0.880 (real: 0.623, 0.352, 0.520). It got further than the real network but
still missed 0.9 on kick and snare.

**Its readout inputs drifted too:** typical size (rms) 11.0–32.6 across drums, against 7.2–23.3 for the real network
and the trigger of 5. It was set beforehand that this would mean the drift is a general property of the setup, not of
the real wiring; the fix applies to every network alike.

## If accepted: the work

1. **`locked.yaml`:** `prereg-v6`; `readout_norm.refit_every: 250`, `refit: output_preserving`,
   `reset_readout_optimiser_state: true`. Under `memorisation`, add
   `if_it_fails: train_anyway_no_further_changes`, which records stopping rule B.
2. **`network.py`:** `refit_readout_norm(...)`, which preserves the output.
3. **`training.py`:** a helper that refits and resets the four readout parameters' Adam state. It is called by
   `memorise.py` and `train.py` at every check.
4. **Tests:** exactness and the typical size after a refit, plus the CUDA-graph check part above.
5. **The runs:** memorisation for the real network and control 1 again under prereg-v6, about 5.3 hours for the pair.

### Draft CHANGELOG line

> 2026-09-30 - prereg-v6: readout scaling, before any real training run. The prereg-v5 trigger fired (set in
> reports/prereg-v5-training-proposal.md): at step 3,000 of the real network's memorisation, the typical size of the
> scaled readout inputs was 7.2–23.3 for every drum, against about 1 at setup and a trigger of 5. It was mostly
> reached within the first 250 steps. Failures were mostly false hits (precision 0.29–0.56, recall 0.44–0.86). The
> scaling is now refitted every 250 steps from the same 16 fit clips through the current network, adjusting the
> readout weights and biases so every output is unchanged at that moment. Adam's state for those four parameters is
> reset at each refit. It applies to every network, in memorisation and training. Its effect on memorisation is not
> known in advance. Stopping rule, chosen before any prereg-v6 run: if memorisation still fails, the pipeline is not
> changed again for it. Training of the real network and all 5 controls goes ahead as locked, with the failure
> reported up front; only a genuine bug, shown by a failing test, may be fixed, and is disclosed. Every change made
> after seeing results is listed in reports/prereg-v6-readout-proposal.md for the final report.
