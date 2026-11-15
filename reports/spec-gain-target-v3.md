# Spec G v3 (draft, not locked): starting-gain target by active-loop margin, with the locked silence test

Status: proposal, written 11 Nov 2026, **before any v3 run**. Supersedes v2 (reports/spec-gain-target-v2.md) and v1 (reports/spec-gain-target.md), both kept unchanged. Nothing in config/ or src/ changes.

## Why this exists
- At the starting weights the real network's whole-network loop gain is 1.000, but its active-neuron loop gain is 0.46 (control 1: 0.74). The starting scale is set from the whole network, so the real network's drum paths start too quiet (scratchpad: start_gain.json).
- The locked silence test (scripts/stability.py; reports/stability.json) fails the real network at 1.5× gain: its activity does not die away after the music stops.
- So a scale rule that raises the real network's gain has to be checked against the locked silence test, not only against overflow.

## Disclosure (read first; required in any write-up)
The starting-scale rule was proposed on 11 November 2026, after memorisation results in which control 1 scored ahead of the real wiring, and after a sweep showing the real network's active loop gain was 0.46 against 0.74. A first grid (0.6–0.8) passed for both networks; the grid was then extended to 1.0 and the stability test changed to the locked silence test. Prereg-v4 (config/CHANGELOG.md) stated the real network's weaker start would be reported, not removed. This spec removes part of that difference. Results under the original rule (initial_gain 1.0) are reported alongside every result from this spec.

## Networks
- The real network and controls 1–5 (reports/controls_report.json), all on the same locked settings. Stage 1 runs all six.

## The scale formula (one shot, not iterated)
1. Baseline active-loop gain at scale 1.0: largest real eigenvalue of the connection matrix restricted to neurons active on at least 5% of frames (rate above 1e-3), on the 2 memorisation clips (fixed_clips: songs 4, clips per song 2, seed 7).
2. Scale for target T: scale = T ÷ baseline active-loop gain, for that network only.
3. Readout normalisation is refitted at that scale (training.fit_readout_norm), on the 20 stability clips' sampler seed as setup uses it, for each network.

## The stability rule (fixed now)
A network passes at T only if all of these hold on the stability clips at its scale:
1. **Locked silence test** (scripts/stability.py, criteria in the script): `passes` is true, `bounded` is true, and `settled_ratio` is below the locked threshold.
2. No NaN in any rate.
3. Peak rate below 10.
4. Mean rate at most 3× that network's mean at scale 1.0 (an upper limit, not a target).

Stability clips: 20 clips drawn from the training split with ClipSampler seed 11, each followed by 4 s of silence (as stability.py).

## Grid and target
- Grid: T ∈ {0.60, 0.70, 0.80, 0.90, 1.00}.
- T* = the largest grid value that **every** network passes.
- **Safety margin:** the chosen target is one grid step below T* (so if T* = 0.9, the target is 0.8). Training roughly triples mean activity by step 1,000, so a start at the edge is not safe.
- If no grid value passes for every network, the spec stops.

## Stage 1 (CPU, no training, minutes to tens of minutes)
- Stability test at each grid value for each of the six networks, per the rule above.
- Record the actual active-loop gain achieved (it can differ from T as the active set changes) and the whole-network gain.
- Output: a verdict table and T*. Deliberately uses no memorisation or real-versus-control result.

## Stage 2 (GPU, needs the GPU go)
- A 500-step memorisation screen (judged at step 500, a check point under check_every 250) on the 8 fixed clips, for the real network and control 1.
- **Baseline:** each network also runs at scale 1.0 with the same 500 steps, same day, same code. Every difference is against that network's own baseline. Baseline runs are required; they are not optional.
- Runs go to new folders under reports/gate_scale/. Nothing locked is read or written.
- About 25 minutes per run on the GPU; four runs (two networks at the chosen target, two baselines).
- The 0.70 reference run from v2 is dropped.

## Decision rule (fixed now)
A locked change is considered only if all of these hold at step 500:
- (a) the real network's snare and closed hi-hat F1 each rise by at least 0.05 over its own scale-1.0 baseline;
- (b) control 1's F1 on the same two drums falls by no more than 0.02 against its own baseline;
- (c) no run in either network shows NaN, runaway, a silence-test failure, or the locked `stop_if` condition (activity_unbounded_or_not_settled) at the chosen target.

If (c) fails for either network, there is no change, and the failure is reported as a stability result.

## Integrity
- T* and the chosen target are fixed by Stage 1 alone; Stage 2 cannot change them.
- Any locked change needs a new tag, a dated CHANGELOG line that cites the disclosure above, and this spec and its results attached.
- Results under the original rule are reported alongside any result under this spec.
- Optional, recommended: the main unseen-songs test is reported under both the original and the new rule.
- Out of scope: the gate's role (already a diagnostic), the loss, the connectome, the dial counts.

## Costs
- Stage 1: CPU; estimate to be measured on the first network.
- Stage 2: about 100 minutes of GPU for four runs.

## What would stop the plan
- Stage 1 fails for every grid value for any network: stop and report.
- Stage 2 condition (c) fails at the chosen target: no change; report.
