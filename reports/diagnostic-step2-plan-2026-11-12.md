# Diagnostic step 2 plan (draft, 12 Nov 2026)

Status: plan only. Nothing is trained, edited or tagged yet.

## Why
Step 1 (CPU/GPU rescoring on the 267 unused validation songs, reports/win_validation_full.json) showed:
- Control 1 leads (20.4 against the real network's 15.9), but the other controls are close to the real network (15.6 to 17.3, and 13.2 for control 3).
- So the real network is not clearly behind the controls as a group. Control 1 is the outlier.
Step 2 asks whether control 1's lead is wiring or training luck, and whether more control seeds show the same pattern.

## What runs (GPU, about 9 hours in total at about 49 minutes per 1,000-step run)
1. **Training repeats** (wiring fixed, training seed changed): real x3 and control 1 x3, so 6 runs.
   - Needs a way to vary the training seed. Today model.init_seed and training_data.sampling_seed are fixed at 0 and train.py has no override.
2. **More diagnostic control seeds** 7 to 12 (build the wiring on CPU first, then run each start check; about 1 in 6 is expected to fail the start check). Expect 5 runs.

Total: about 11 runs. Each run posts to the website through the existing live chart (train.py publishes by default).

## Locked-settings change (needed before any run)
Under prereg-v5.5 (Andrew allowed locked changes for this test):
- Add `diagnostic_control_seeds: [7, 8, 9, 10, 11, 12]`, kept separate from the five official seeds, so the win rule and the seed spread (divisor 4) do not change.
- Add `diagnostic_training_seeds: [1, 2, 3]` for repeats of the real network and control 1.
- Add a changelog line before any run.
- Record the step-1 result as diagnostic (validation, not a confirmation).

## Posting to the website
- Each training run posts its progress through the live chart (as the v53/v54 runs did).
- Step 2 also needs a summary page: the same design as score.html, showing the repeats and the new controls. Built after the runs finish.

## Decisions for Andrew
1. Approve the prereg-v5.5 change above.
2. Approve the GPU runs, about 9 hours, one at a time, starting with the 6 repeats.
3. Confirm whether the five new control seeds (7 to 12) should be included in this round, or only the repeats.

## Not verified
- Whether 1 in 6 is the right failure rate for new seeds (taken from the earlier 4-of-5 pattern).
- Whether the training-seed change alters the CUDA graph fingerprint (scripts/check_cuda_graph.py would need rerunning).
