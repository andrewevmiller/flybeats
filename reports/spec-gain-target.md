# Spec G (draft, not locked): a starting-gain target set by active-loop margin

Status: proposal. Written 11 Nov 2026, before any run of this test. Nothing in config/ or src/ changes.

## Why
- At the starting weights, the real network's whole-network loop gain is 1.000, but the part that actually fires has gain 0.46. The current scale rule (1 / largest eigenvalue, `initial_gain: 1.0`) uses the whole network, so the real network's drum paths start too quiet (reports/... bias and loop notes; scratchpad scale_sweep_gpu.json).
- Scaling the whole network: the real network stays normal at 1.5× (active loop 0.70; readout spread about 3.5× larger) and overflows at 2.0×. Control 1 runs away at 1.5× (peak rate 1,667) and overflows at 2.0×.
- So a shared multiplier cannot help the real network without breaking the control. A per-network rule is needed, and it must be the same rule for both.

## The rule (fixed now)
1. Each network's starting scale is set so that its active-loop gain equals a target T.
   - Active loop gain: the largest real eigenvalue of the connection matrix restricted to neurons active on at least 5% of frames (threshold 1e-3), at the starting weights, on the 2 memorisation clips (as start_gain.py).
2. T is chosen from the grid {0.60, 0.70, 0.80} by a stability rule that does not look at memorisation or the real-versus-control result:
   - T is the largest value in the grid for which, at its scale, **both** networks have no NaN, peak rate below 10, and mean rate within 3× of the same network's mean at scale 1.0.
   - If no value in the grid passes for both, the spec fails and nothing is run.
3. The same T applies to both networks. No network gets its own T.

## Test
- **Stage 1 (CPU, no training, minutes):** the sweep at the chosen scales. Records activity, silent readout neurons, peak rate, and active loop gain. Pass = the stability rule above holds at T.
- **Stage 2 (GPU screen, needs the GPU go):** memorisation check at the chosen scales, 300 steps, real and control 1, to see whether training behaves. Screen only; results go to a new folder and change nothing locked.
- **Decision rule for any change (written now):** a locked change is considered only if (a) real snare and closed hi-hat F1 rise by at least 0.05 at 300 steps AND (b) control 1 falls by no more than 0.02 on the same drums. Otherwise no change, and the result is reported.

## Integrity
- The stability rule and the grid are fixed here. The real-versus-control result is not an input to T.
- Any locked change follows the usual route: a new tag, a dated CHANGELOG line, and a disclosure that the scale rule was set after the sweep.
- This spec does not decide the gate, the connectome, or the loss.

## Costs
- Stage 1: about 3 minutes on the CPU.
- Stage 2: about 15 minutes per network on the GPU (screen length), plus setup.

## What would change the plan
- Stage 1 fails for every T in the grid: stop. The scale rule cannot be fixed by gain alone.
- Stage 2 shows no snare gain for the real network: keep the locked rule and report it.
