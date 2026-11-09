# Proposal: specs for the fixes that leave the connectome untouched

Status: **proposal, not locked.** Nothing in `config/locked.yaml` or the code has changed. Written 9 Nov 2026 from the
logs and code in this repository only; no network was run to write it (the data and weights are not in git).

Scope: the seven fixes that change nothing about the wiring, signs, synapse counts or cell-type tying, in the order
of the integrity ranking, plus one measurement that comes first. The three fixes that change how connection
strengths are expressed (dial groups in controls, the loop-strength cap, per-connection dials) are not specified
here.

## Rules for every spec

- **Off by default.** Each change sits behind a trial option, as `--history-readout` does. With the option off, every
  output and gradient is bit-identical to today's; each spec has a test for that.
- **Out of bounds for a worker:** `config/locked.yaml`, `config/CHANGELOG.md`, `src/flybeats/scoring/`,
  `src/flybeats/connectome/`. Locking is Andrew's step, with a new tag, after a trial result.
- **Fingerprint:** `network.py`, `ear.py`, `loss.py` and `training.py` are the recorded step's code. Any spec that
  edits one of them ends with `scripts/check_cuda_graph.py` passing again before a GPU run uses the recorded step.
- **Every trial runs the real network and control 1,** with the trial's values and its decision rule written into
  the script before the run.
- **Prerequisite, P:** the unit tests load the real slice through `paths`, so they cannot run without the data. Add
  `tests/synthetic.py`: a tiny made-up slice (about 40 neurons, 2 ears per band group, 2 motor neurons per drum, a few
  shared cell types, some reciprocal pairs, one silenced neuron) with exactly the columns `FlyNet` reads. New tests
  use it; existing tests are left alone.

## Spec 0: count each network's dials (measurement only)

- **Why:** `network.py` makes one dial per (sending type, receiving type) pair present in the edge list. The real
  slice has 41,186. A rewired control should have far more (a rough simulation gives about 100,000); nobody has
  counted. A memorisation check rewards free parameters, so this decides how to read every real-against-control
  memorisation result so far.
- **Change:** `scripts/build_controls.py` adds `type_pairs` for the real slice and each control to
  `reports/controls_report.json`; `memorise.py` and `train.py` write `len(model.theta)` in the log's first line.
- **Done when:** the six counts are in the report. No test beyond the count equalling `model.theta.numel()`.

## Spec 1: calibrate the gate

- **Why:** nothing has reached F1 0.9 on the 8 clips, control 1 included, and no model has been shown to.
- **Change:** a new CPU script, `scripts/gate_calibration.py`. The clip-drawing lines of `memorise.py` move into one
  helper in `flybeats/data/sampler.py` that both scripts call, so the 8 clips are the same by construction.
- **Models:** `feasibility.py`'s ear-to-GRU model at two sizes: 2 layers of 128 units (about 160,000 parameters) and
  2 layers of 64 (about 41,000, the real network's dial count).
- **Training and scoring:** exactly the memorisation check's (locked loss, Adam 1e-3, cosine to zero, clip 1.0, 3,000
  steps, a check every 250, the locked peak rule and tolerance).
- **Rule, fixed in the script:** the small model reaches 0.9 on kick, snare and closed hi-hat: the gate is attainable
  at this size. Only the large one does: the gate is a capacity test at this size. Neither does: the gate is mis-set
  for a play-along task and says nothing about the fly network.
- **Output:** `reports/gate_calibration.json` and a log. It changes nothing else.
- **Tests:** the helper returns the same clips as the old inline code (same songs, same start times).

## Spec 2: a short screening run

- **Why:** a full memorisation run takes about 2.5 hours, and only one fits on the GPU at a time.
- **Change:** `memorise.py --screen`: 2 songs x 2 clips, 300 steps, a check every 50, the schedule over those 300
  steps, the real network then control 1, results under `reports/screen/<name>/`, marked `as_locked: false`. It
  combines with the other trial options. About 15 minutes each at 2.9 s a step.
- **Validation before it is trusted:** run it on three setups whose full results are known (locked real, locked
  control 1, real with `--history-readout`). It passes if it reproduces both orderings: control 1 above real on all
  three gate drums, and the history readout above the locked readout on kick and closed hi-hat. If it does not, it is
  not used to screen anything.
- **Tests:** `--screen` with the other options off gives the same step-1 loss as `--songs 2 --steps 300`.

## Spec 3: separate learning rates with a warm-up

- **Why:** in the real run 87% of steps 251-1,000 had gradients above the clip (control 1: 1%), the loss rose on 35%
  of those steps on a fixed batch, and F1 was still rising when the schedule reached zero.
- **Change:** `training.make_optimiser` builds two parameter groups: the network (`theta`, `log_tau`, `bias`,
  `band_gain`, `band_offset`) and the readout (`w_hit`, `u_vel`, `c_hit`, `e_vel`, `p_hit`, `p_vel`).
  `training.make_schedule` gains a linear warm-up before the cosine. `memorise.py` logs both rates.
- **Trial option:** `memorise.py --split-lr`. Values fixed now: network 3e-4, readout 1e-3, warm-up 100 steps.
- **Tests:** option off: one group's worth of behaviour, identical weights after 3 steps; option on: every parameter
  is in exactly one group; both rates are 0 at the last step; the warm-up starts at 0 and reaches the full rate at
  step 100.
- **Read the result by:** share of steps above the clip, share of steps where the loss rose, and F1 at 3,000.

## Spec 4: lighter hit weighting in the loss

- **Why:** hit frames carry up to 50 times a quiet frame's weight while a hit is scored at probability 0.5, so the
  loss favours firing; precision is 0.29-0.50 against recall 0.44-0.84.
- **Change:** `loss.drum_weights` gains a `mode`: `ratio` (today's) or `sqrt` (the square root of the same ratio, so
  at most about 7).
- **Trial option:** `memorise.py --loss-weight sqrt`. The audio-blind floor is recomputed by the existing code.
- **Tests:** `ratio` returns today's numbers exactly; `sqrt` is their square root; the cap still applies before the
  root.
- **Read the result by:** F1, precision and recall per drum. Loss values are not comparable with earlier runs.
  **Guard:** a rare drum (pedal hi-hat, toms, crash) that stops firing altogether counts against the change.

## Spec 5: a minimum gap in the peak rule

- **Why:** the locked rule counts every local peak at or above 0.5; 24% of the real network's false hits are a
  second peak on a hit already matched.
- **Change:** `loss.peaks(prob, threshold, min_gap_frames=0)`: peaks are taken in order of height and any peak
  within the gap of one already kept is dropped. 0 is today's rule.
- **First step needs no training:** a CPU script in the pattern of `scripts/false_hits.py` rescoring the saved
  `final.pt` of the real network and control 1 at a gap of 10 frames (50 ms), fixed now.
- **Tests:** gap 0 equals today's output on random inputs; no two kept peaks are closer than the gap; the higher of
  two close peaks is the one kept.
- **Integrity note:** the prereg-v5.2 scorecard takes its hits from the locked peak rule, so adopting this changes the
  scorecard's input. It is a scoring change made after results and needs its own tag and disclosure.

## Spec 6: the prereg-v6 refit, on its own

- **Why:** readout inputs grow from about 1 to 7-23; the trigger set in prereg-v5 fired. Never tested alone.
- **Change:** as `reports/prereg-v6-readout-proposal.md`: `FlyNet.refit_readout_norm`, a helper in `training.py`
  that refits and resets Adam's state for the four readout parameters, called at every check.
- **Trial option:** `memorise.py --refit-readout`, without `--history-readout`.
- **Tests:** every hit and loudness output is unchanged across a refit (float32 rounding); input size is about 1
  afterwards on the fit clips; Adam's state is zero for the four parameters and untouched for the rest; the
  CUDA-graph check gains its part with a refit at step 10.

## Spec 7: a fixed muscle filter

- **Why:** a readout that weighs the recent past helped kick and closed hi-hat in the history-readout trial; a fixed
  filter would add no learned readout numbers.
- **Change:** `network.readout` smooths each motor neuron's scaled rate with a causal one-pole filter before the
  locked readout. Setting: `model.readout_filter: {tau_ms: ...}`; absent means today's readout.
- **Blocked on a value:** the time constant must come from a cited muscle source, fixed before any run. The history
  proposal found no locked source; this spec does not supply one.
- **Tests:** a time constant below one frame equals the locked readout; causality; each drum still reads only its
  own motor neurons.
- **Warning from the trial:** the learned profiles put negative weight on 105-150 ms back (present minus older past,
  a sharpening). A smoothing filter cannot do that, so the trial's gain is weak evidence for this fix.

## Confidence

Two scores per fix, each the estimated chance, out of 100, of a meaningful improvement for the real network:

- **Memorisation:** at least +0.05 hit F1 averaged over kick, snare and closed hi-hat at step 3,000 of the locked
  check. For scale: the history readout gave +0.056, and passing the gate needs about +0.40.
- **Training:** a higher validation hit F1 at the same step count in BabySlakh training. No training run exists, so
  there is no baseline; these scores rest on argument alone and are capped accordingly.

What moves a score:

1. **Is the mechanism measured?** Seen directly in the logs, inferred from them, or only reasoned.
2. **Does any existing result test it?** A trial or rescoring that already points one way, for or against.
3. **How much of the gap can it reach?** An arithmetic upper bound where one exists.
4. **Does it carry over** from 8 fixed clips to new batches every step?
5. **Can it do harm?** A known way for it to make things worse lowers the score.

| Fix | Memorisation | Training | Main reason |
|---|---|---|---|
| 0. Count the dials | 0 | 0 | Changes no network. Decides how to read the control comparison |
| 1. Calibrate the gate | 0 | 0 | Changes no network. About 90 that it settles whether 0.9 is attainable |
| 2. Screening run | 0 | 0 | Changes no network. About 60 that a 15-minute run reproduces the known orderings |
| 3. Separate learning rates | 40 | 35 | Instability is measured, but prereg-v5's schedule fix barely moved the real network (0.56 / 0.38 / 0.53 to 0.62 / 0.35 / 0.52) |
| 4. Lighter hit weighting | 35 | 30 | The mismatch is arithmetic, but sweeping the threshold on the trained weights gained only 0.006, and the wide-target trial shows loss changes can backfire |
| 5. Minimum gap | 25 | 5 | Certain to raise F1 a little; the bound from the false-hit table is about +0.04 on the three drums. Scoring only, so training does not change |
| 6. Refit alone | 30 | 35 | Drift is measured, but control 1 drifts more (11.0-32.6) and memorises well, so drift is not what holds the real network back |
| 7. Muscle filter | 20 | 20 | The trial's learned profile sharpened; this smooths. No sourced time constant |

No fix here has a realistic chance of passing the 0.9 gate by itself (under 5 each), and all seven together probably
do not either. Snare did not move under the history readout, and the trace puts that down to the wiring.
