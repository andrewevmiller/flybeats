## Style dial, seed-1 confirmation (cloud, 25 Sep)

**Verdict: the effect holds on a second seed, at about half the size.** With the push on pC1, switching style changes 11% of hits on seed 1, against 20% on seed 0. The baseline stays at 0% on both seeds. That clears the ~10% bar only just. Timing does not suffer: pC1 scored +0.018 over its baseline on seed 1 (-0.003 on seed 0).

| | baseline s0 | pC1 s0 | baseline s1 | pC1 s1 |
|---|---|---|---|---|
| hits changed by style | 0% | 20% | 0% | 11% |
| largest style shift (0-1) | 0.002 | 0.51 | 0.007 | 0.33 |
| timing score (best val onset F) | 0.2834 | 0.2804 | 0.2668 | 0.2851 |
| drive slider, hits changed | <1% | 10-11% | <1% | 6% |
| hits/s (human 8.6) | 10.1 | 8.5 | 9.0 | 10.5 |
| groove match to human | 0.29 | 0.31 | 0.32 | 0.32 |
| held-out kits, timing change | -0.015 | -0.017 | -0.011 | -0.023 |

- **Held on both seeds:** the dial works only with the push on pC1, timing costs nothing measurable, the drive slider comes alive, and more drums come in.
- **Did not hold:** the size (use 10-20%, not 20%) and seed 0's human-like hit rate (seed 1's pC1 plays 10.5 hits/s). The tightness-1.5 observation could not be tested, because the seed-1 baseline was not silenced either.
- **Watch:** pC1 loses slightly more timing on held-out kits than its baseline, on both seeds.

**Built** on `claude/intelligent-bohr-r61k7z` (main with `claude/style-dial-2026-09-25` merged in; the session could push only there, not to `claude/style-pc1-seed1`):
- `configs/style_pc1_s1_cpu.yaml`, plus a table row and a seed-pair test in `tests/test_config.py`. Full suite: 285 passed.
- `STYLE_*` overrides on `scripts/run_style_queue.sh` and `scripts/push_style_results.sh` (defaults unchanged), and a wrapper, `scripts/run_style_s1_queue.sh`.

**Machine:** 4 cores, 15 GB, torch 2.14.0+cu130 (CPU). Epochs took ~415 s, against ~245 s on the seed-0 host. The queue ran 19:17-22:51 UTC. Nothing was skipped or failed.

**Files** in `results/cloud/style-pc1-seed1-2026-09-25/`: `reports/00-seed1-verdict.txt` (read first), `COMPARE.txt` (four columns), `variety/`, `kit_check/`, `checkpoints/`, logs, `NOTES.md`.

**Suggested next step (not built):** the size is seed-dependent, so pin it down before building on it. Options: a third seed, or measure_variety over more than 6 songs, since the 11% vs 20% gap may partly be measurement noise.
