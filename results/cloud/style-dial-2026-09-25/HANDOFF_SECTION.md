## Style dial experiment (cloud, 24-25 Sep)

**Verdict: the style dial works once its push goes into pC1.** Switching style now changes 20% of hits (baseline 0%), and timing costs only 0.003. Adding audio gaps did not help: they halved the style effect. Single seed per arm, CPU, cloud machine.

| | baseline | pC1 | pC1+gaps |
|---|---|---|---|
| hits changed by style | 0% | 20% | 9% |
| largest style shift (0-1) | 0.002 | 0.51 | 0.24 |
| timing score (best val onset F) | 0.2834 | 0.2804 | 0.2765 |
| drums that play | snare | kick, snare, hat, 2 toms | snare, hat |
| hits/s (human 8.6) | 10.1 | 8.5 | 9.8 |
| groove match to human | 0.29 | 0.31 | 0.30 |
| held-out kits, timing change | -0.015 | -0.017 | -0.011 |

- **A working path is enough.** The pC1 arm passes the bar (at least ~10% of hits changed, timing drop at most 0.01). It also brought in four more drums and made the drive slider work (10% of hits vs under 1%). Tightness 1.5 no longer silences it.
- **A reason to use style did not help, in this form.** Gaps (half of each clip silenced, targets kept) cut the style effect to 9% and went back to mostly snare.
- Cloud numbers: timing scores compare only within this run, not with the laptop's.

**Built** on branch `claude/style-dial-2026-09-25` (from cuda-path-verified 054b3a7). Code commit d201564:
- `genre.targets` (src/build.py `genre_target_index`, used by src/bundle.py). It is a list of roles and defaults to `["octopaminergic"]`, so old configs and checkpoints build the identical model.
- `train.audio_gaps` (src/train.py `audio_gap_mask` / `apply_audio_gaps` / `audio_gap_generator`). It applies in training epochs only, is seeded from train.seed and the epoch, and is off by default.
- configs/style_pc1_cpu.yaml (targets [pC1], max_current 5.0) and configs/style_pc1_gaps_cpu.yaml (+ audio_gaps 0.5 / 250-1000 ms). Both are based on velocity_probe_cpu.
- tests/test_style_dial.py (10 tests) plus new rows and an arm test in tests/test_config.py. Full suite: 264 passed (before the changes: 247 passed, 2 skipped, no failures).
- scripts/run_style_queue.sh (the POSIX twin of the .ps1, which is unchanged) and scripts/push_style_results.sh.
- Before training, the untrained pC1 model was checked (style_pathway.py). Rates stayed finite and no drum was pinned, even with all of pC1 held at +/-5.

**Machine:** 4 cores, 15 GB RAM, Python 3.11.15, torch 2.14.0+cu130 from PyPI (CPU only). Each run took about 50 min (240-251 s/epoch). The whole queue ran 02:32-05:53 UTC.

**Skipped or failed:** nothing. No takeover, no relaunch, and every step exited 0.

**Files** in `results/cloud/style-dial-2026-09-25/`:
- `reports/`: 00-style-verdict.txt, arm-style-pc1.txt, arm-style-pc1-gaps.txt. Read these first.
- `COMPARE.txt`: the side-by-side measurements.
- `variety/<run>/`: variety.json and report.txt.
- `kit_check/<run>-validation/`: kit_check.json and report.txt.
- `checkpoints/<run>/`: best.pt and history.json, for all three runs.
- `<run>.train.log`, `queue.log`, `runner.out`, `NOTES.md`, `HEARTBEAT.txt`, `state/`.

**Suggested next step (not built):** a second seed of style_pc1_cpu, to confirm the result before building on it.
