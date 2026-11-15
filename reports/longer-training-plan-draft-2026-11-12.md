# Longer-training diagnostic plan (draft, 12 Nov 2026)

Status: draft only. Nothing trained, scored on the GPU, edited or tagged. Needs Andrew's approval before any step that uses the GPU or changes locked settings.

## Why
Check 2 (reports/density_matched_validation.json) matched every network to the same hit count (107.3 per song). Beat alignment then spans 15.9 to 16.6, and timing sits at 0.505 to 0.517 for all six networks, against 0.5 for random. So:
- The earlier control 1 lead came from hit count, not wiring.
- No network shows timing skill yet.
The likely limit is training (1,000 baby steps, validation F1 near its floor), not the wiring. This plan tests that.

## Step 1: free check, no new training
Score the existing checkpoints on training clips (not the 267 validation songs, which are now used up for tuning).
- If timing is also at 0.5 on training clips, the networks have not fitted timing at all: an optimisation limit.
- If timing is above 0.5 on training clips but not validation, that points to overfitting or a split effect.
- Needs no new training. Can run on the CPU or the GPU (GPU needs approval).

## Step 2: longer-training diagnostic (prereg-v5.5, GPU, needs approval)
- **Networks:** the real network and control 2 (a mid-pack control, chosen before results). Not control 1, which was picked for its score.
- **Data:** the full training set (1,162 songs). Steps: 5,000 per run, about 4 hours each if the 49 minutes per 1,000 steps holds on the full set (not verified). Total about 8 hours for two runs.
- **Logged every 500 steps:** training loss, hits per song, and timing without b on fixed clips.
- **Read-out:** the baby validation clips and training clips. Validation is spent for further tuning, so no new claim is read from it.

Outcomes:
- **Both rise clearly above 0.5 and keep climbing:** training was the limit. Plan the full comparison with all controls under a new tag.
- **Control 2 rises and the real network stays at 0.5:** the real wiring, or its weaker starting scale, may be the limit. One pair cannot separate these; repeats would be needed.
- **Neither rises and the loss is flat:** the setup (loss or readout) is the limit, not wiring or training length.

## Locked-settings changes (needed before any run)
- A new prereg-v5.5 tag with a changelog line, written before the run.
- `training.steps.full` is currently null and is set from the measured cost before the first full run. The full set's cost needs measuring first (reports/cost.json exists for an earlier measurement; not yet checked against the full set).
- Any change to the threshold rule, the readout or the loss would need its own tag. This plan changes none of them.

## Caveats
- Matching was on mean hits per song, not per drum, so balance still differs slightly per network.
- Checked: the matched target of 107.274 hits per song is the mean over all 270 validation songs. The 267 count is only the songs kept for the beat-alignment score, so the target is correct.
- Timing has no error bars, and 0.5 is the theoretical chance value. Loud moments could push a network slightly above 0.5 without any skill.
- Checkpoints were chosen on a few baby songs; control 5 is scored at step 500.
- The test split is spent. Any confirmatory claim needs songs not yet looked at, named in advance.

## Decisions for Andrew
1. Approve Step 1 (checkpoint scoring on training clips), on the CPU or the GPU.
2. Approve the prereg-v5.5 tag for Step 2, and which two networks to train.
3. Approve about 8 GPU hours for Step 2, one run at a time.

## Not verified
- The 4-hour estimate for a 5,000-step run on the full set.
- Whether timing is at chance on training clips (Step 1 would answer this).
- Whether the cost file covers the full set.
