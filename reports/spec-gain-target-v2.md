# Spec G v2 (draft, not locked): starting-gain target set by active-loop margin

Status: proposal, written 11 Nov 2026, **before any v2 run**. Supersedes the grid in reports/spec-gain-target.md (v1), which is kept unchanged. Nothing in config/ or src/ changes.

## What changed from v1
- The grid is extended to {0.60, 0.70, 0.80, 0.90, 1.00}. v1's top value (0.80) passed the stability rule, so the rule never bound.
- Stage 2 now runs at a fixed reference (0.70) and at the largest target that passes, rather than at one target chosen by the grid's top.
- The v1 stability rule and the v1 decision rule are unchanged.

## Why
- Stage 1 of v1: all of {0.60, 0.70, 0.80} passed for both networks, so the grid ceiling chose the target. The real network's stability limit lies above 0.80; the control's lies near 1.0 (it ran away at scale 1.5, where its active loop gain was 1.00; reports in scratchpad scale_sweep_gpu.json).

## The rule (fixed now)
1. Active-loop gain at the starting weights, as in start_gain.py: largest real eigenvalue of the connection matrix restricted to neurons active on at least 5% of frames (rate above 1e-3), on the 2 memorisation clips.
2. Each network's starting scale is set so its active-loop gain equals T.
3. T is the largest value in {0.60, 0.70, 0.80, 0.90, 1.00} at which **both** networks pass the stability rule: no NaN, peak rate below 10, mean rate within 3× of that network's own mean at scale 1.0.
4. The same T applies to both networks. No network has its own T.
5. If no T passes for both, the spec stops.

## Stage 1 (CPU, no training, about 4 minutes)
- Run the sweep at each grid value for both networks, using the rule above.
- Record the actual active-loop gain achieved (it can differ from T because the active set changes).
- Output: a verdict per T and the chosen T.

## Stage 2 (GPU, needs the GPU go each time)
- Memorisation screen, 300 steps, on the 8 fixed clips, for the real network and control 1, at two targets: the chosen T and the reference 0.70.
- Both runs use the same code as the locked check except the starting scale, and go to new folders (`reports/gate_scale/`).
- About 15 minutes per run on the GPU, so about one hour for four runs.

## Decision (written now)
- **Target selection is fixed by Stage 1 alone.** Stage 2 does not choose among targets; it only decides whether a change is considered at all.
- A locked change is considered only if, at 300 steps, (a) the real network's snare and closed hi-hat F1 each rise by at least 0.05 at the chosen T **and** (b) control 1's F1 on the same drums falls by no more than 0.02.
- Both (a) and (b) use the same T for both networks.
- The 0.70 reference is a robustness check. If the result at 0.70 disagrees with the chosen T, the result is reported, and no change is made.
- If a run overflows or runs away, that is reported as a stability failure for that network at that T.

## Integrity
- Stage 1 uses no memorisation or real-versus-control result.
- The choice of T is fixed by Stage 1. Stage 2 cannot move it.
- Any locked change needs a new tag, a dated CHANGELOG line, and a disclosure that the starting-scale rule was set after the sweep, with this spec and its results attached.
- This spec does not change the gate's role (already a diagnostic), the loss, the connectome, or the dial counts.

## Costs
- Stage 1: about 4 minutes on the CPU.
- Stage 2: about one hour on the GPU (four 300-step runs).

## What would stop the plan
- Stage 1 fails for every T: stop, and report that the starting scale cannot be fixed by gain alone.
- Stage 2 overflows for the real network at the chosen T: keep the locked rule and report the overflow.
