# Proposal: prereg-v5 training fixes

Status: **accepted on 30 Sep 2026 and locked as prereg-v5** (tag `prereg-v5`, CHANGELOG line). As proposed, with one
detail settled in the code: the locked `memorisation` section records the clips as 4 songs × 2 clips with clip seed 7,
not just "8 clips", so the exact clips are on the record.

Written 30 Sep 2026, from the logs of the two memorisation runs of 29–30 Sep (`reports/memorise/real/` and
`reports/memorise/control-1/`, both prereg-v3). Both failed the check at step 3,000.

## The rule this proposal follows

The results are already in. So a change is proposed only if it is justified whatever those results were. Anything
that looks like a fix but has only a guess behind it gets measured first. Otherwise the settings drift towards
whatever makes this run pass, and the check stops meaning anything. Each fix below says which kind it is.

## What the two runs show

| | Real | Control 1 (old whole-slice swaps) |
|---|---|---|
| F1 at step 3,000: kick / snare / closed hi-hat | 0.56 / 0.38 / 0.53 | 0.82 / 0.64 / 0.77 |
| Loss at 3,000 (audio-blind floor 0.493) | 0.337 | 0.271 |
| Steps where the loss **rose**, 2,000–3,000 | 36% | 47% |
| Step-to-step change against the underlying improvement, 500–1,000 → 2,000–3,000 | 3× → 19× | 3× → 29× |
| Average activity, step 1 → 3,000 | 0.035 → 0.104 (3.0×) | 0.070 → 0.480 (6.8×) |

- **Every step is on the same 8 clips, and the recorded step is deterministic.** So a rise in loss is pure
  overshoot, not sampling noise. In both networks the overshoot grows until each step's jitter is 19–29 times the
  progress it makes.
- **Activity kept growing in both,** well past the level the readout scaling was fitted to at the start.
- **Snare is the weakest drum in both,** even in control 1, where sound reached the readout directly.

## Fix 1: memorisation uses the locked learning-rate schedule

**Kind: justified whatever the results; it corrects a mismatch.**

- **The mismatch:**
  - `locked.yaml` sets `schedule: cosine_to_zero  # over the run's steps`, and `train.py` does this.
  - `memorise.py` keeps the learning rate fixed at 1e-3 for all 3,000 steps.
  - Its docstring says it uses "the locked training ones" (optimiser, loss and gradient clipping), but the schedule
    was left out.
- **Why it's justified:** the check exists to vouch for the training setup, so it should run that setup.
- **The change:** `memorise.py` lowers the learning rate from 1e-3 to 0 on a cosine curve over its step limit (3,000),
  exactly as `train.py` does over its steps.
  - If the check passes early and stops, it simply stops. The schedule is not stretched or shortened to fit.
  - The pass rule (F1 ≥ 0.9 on kick, snare and closed hi-hat), the 8 clips and the 3,000-step limit stay the same.
- **What to expect:** this directly targets the overshoot. Late in the run, the step size falls towards zero while the
  gradient keeps pointing the same way. **Nobody can say in advance whether it gets the real network to 0.9.** At
  step 3,000 the real network was 0.34–0.52 short, and the schedule could plausibly close only part of that.

## Fix 2: record what the next diagnosis will need

**Kind: measurement only. Training does not change.**

Three things were missing when these runs were diagnosed:

1. **Precision and recall per drum.** Already added to `memorise.py` (commit `7454fee`). The next run is the first to
   log them.
2. **The final weights,** saved as `reports/memorise/<name>/final.pt`, so a finished run can be re-scored and looked
   into without rerunning it.
3. **What the readout actually receives, at each 250-step check.** For each drum: the scaled motor-neuron rates the
   readout gets (mean, typical size and largest size), next to the values measured at setup. This directly tests the
   suspicion that the scaling drifted. At setup the typical size is about 1 by construction; if it is 5–50 by step
   3,000, the scaling has drifted.
4. **The gradient size before clipping,** each step. Without it there is no way to tell whether clipping at 1.0 is
   capping every step.

Items 2 and 3 go in `memorise.py`. Item 4 needs the recorded step to return a number it already computes, a small
change to `training.py`. That changes the fingerprint, so the CUDA-graph check must be rerun. It has to be rerun anyway
before `train.py` can use the recorded step.

## Not changed now, and what would justify changing them

Each has a plausible story, but no measurement behind it yet. Changing them now would be tuning to this result. The
thresholds below are set **now**, before the next run, so the decision after it is not made up after the fact.

**A. The readout scaling, fitted once at setup.**
- **The story:** as activity triples (real) or grows sevenfold (control 1), the fixed scaling feeds the readout values
  far larger than intended, which makes each weight step overshoot.
- **What would justify changing it:** by step 3,000, the typical size of the scaled readout inputs (item 3) is
  **above 5** for any drum, in the real network or the control.
- **The fix it would lead to** would apply to every network alike: refitting the scaling on a fixed schedule, or
  making it part of what's learned. That would be prereg-v6, with the measurement quoted.

**B. The thin snare and closed hi-hat readouts** (12 motor neurons, only 5 cell types each).
- **The story:** too few independent signals to place hits accurately. Snare is weakest in both networks.
- **What would justify changing it:** after Fix 1, snare or closed hi-hat stays below 0.9 **mainly because of missed
  hits** (recall below 0.7 with precision above 0.8, from item 1), while the drums with larger readouts pass.
- **Why it would still need care:** the readout assignment (which limb and family plays which drum) is a core design
  choice. Changing it is a bigger step than A and would need its own proposal.

**C. The step limit (3,000) and the 0.9 bar.**
- **Neither changes.** Raising the limit or lowering the bar after failing would be moving the goalposts.
- **If both networks improve steadily but slowly under Fix 1,** that is a finding about how much training this model
  needs, and it gets reported. The bar does not bend to meet it.

**D. The real network's connections starting about 2.5× weaker** (each network scaled to its own critical point).
- **Not a bug.** It follows from the real wiring's stronger loops and applies to every network alike (see prereg-v4).
- **Reported with the results, not removed.**

## After the next memorisation runs

Run the real network and the new prereg-v4 control 1 under prereg-v5. Then:

| Result | Next |
|---|---|
| Real passes | Rerun the CUDA-graph check, then BabySlakh training for the real network and all 5 controls. |
| Real fails, and A's or B's threshold is crossed | A proposal for that fix only (prereg-v6), quoting the measurement. |
| Real fails, and neither threshold is crossed | Stop and look again with the new logs before changing anything. |

The control's result is recorded either way. **It does not decide anything:** the memorisation check is about whether
the setup works, and the win rule is about the comparison.

## If accepted: the work

1. **`locked.yaml`:** `prereg-v5`. No training setting changes. Add under `memorisation` (a new section, since the
   check's own settings were only in `memorise.py` until now):
   - `schedule: cosine_to_zero`
   - `steps: 3000`
   - `clips: 8`
   - `pass_f1: 0.9`
   - `must_pass: [kick, snare, hihat_closed]`

   This puts the check's own settings on the record, and `memorise.py` reads them from there.
2. **`memorise.py`:** the cosine schedule; save `final.pt`; log the readout-input sizes at each check.
3. **`training.py`:** return the gradient size before clipping, from `train_step` and `GraphedStep` alike.
4. **Tests and checks:**
   - A unit test that the schedule reaches 0 at the last step.
   - Rerun `check_cuda_graph.py`, which now also covers new batches.
   - A 1-step CPU run of `memorise.py` to check the new log fields.
5. **Then run memorisation:** the real network and the new control 1 (`memorise.py --pair`), about 9 hours with the
   recorded step.

### Draft CHANGELOG line

> 2026-09-30 - prereg-v5: memorisation check, before any real training run. `memorise.py` now lowers the learning
> rate to 0 on the locked cosine schedule over its 3,000 steps, as `train.py` does. It had kept 1e-3 throughout. In
> the prereg-v3 runs, the loss rose on 36% (real) and 47% (control 1) of the last 1,000 steps, with step-to-step
> jitter 19–29 times the underlying improvement. The check's own settings (8 clips, 3,000 steps, F1 ≥ 0.9 on kick,
> snare and closed hi-hat) move into `locked.yaml` unchanged. The check also records the gradient size, the size of
> the readout's inputs, and the final weights. Readout scaling and readout width are unchanged; the measurements that
> would justify changing them are stated in `reports/prereg-v5-training-proposal.md`.
