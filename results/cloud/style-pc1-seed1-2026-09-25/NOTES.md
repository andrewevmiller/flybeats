# Style dial, seed-1 confirmation: cloud run notes

Session start: 25 Sep 2026 (first session for this run; the earlier attempt
pushed nothing because the git proxy refused pushes). Branch
claude/intelligent-bohr-r61k7z = main + claude/style-dial-2026-09-25 merged in
(the session may push only to this branch, not claude/style-pc1-seed1).

## Question
Does the seed-0 style-dial result (pC1 target: 20% of hits changed by style vs
0% baseline, timing -0.003) hold on a second seed? Bar as before: at least ~10%
of hits changed, timing drop at most 0.01 vs the baseline on the same seed.

## Machine
- nproc 4, RAM 15 GB, 30 GB disk free at start (same shape as the seed-0 machine)
- Python 3.11.15, torch 2.14.0+cu130 from PyPI (CPU only), same as seed 0

## Setup
- venv + requirements-dev, GMD groove-v1.0.0 (5.11 GB), MaleCNS tables (fetch_data.py)
- Added configs/style_pc1_s1_cpu.yaml (style_pc1_cpu, seed 1, window_seed 0);
  velocity_probe_s1_cpu.yaml already existed. STYLE_* overrides on the queue and
  push scripts (defaults unchanged); scripts/run_style_s1_queue.sh wraps them.
- Test suite (-m "not gpu"): 285 passed, 5 deselected, no failures.
- No pre-training style_pathway check: same architecture and target as seed 0,
  which checked safe; only the init seed differs.

## Run
- 19:17 UTC runner started (scripts/run_style_s1_queue.sh), one python job at a time
