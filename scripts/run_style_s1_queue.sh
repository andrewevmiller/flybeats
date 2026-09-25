#!/bin/bash
# Style dial, seed-1 confirmation: scripts/run_style_queue.sh pointed at the
# seed-1 pair. It trains velocity_probe_s1_cpu and style_pc1_s1_cpu, fetches the
# E-GMD subset, measures both, and writes a four-column COMPARE.txt: the seed-0
# baseline and pC1 (from results/cloud/style-dial-2026-09-25/variety/, already
# on the branch) followed by the two seed-1 runs.
#
#   nohup setsid scripts/run_style_s1_queue.sh > runs/style_s1/runner.out 2>&1 < /dev/null &
#
# Same markers and resume as the seed-0 queue, so a relaunch skips finished
# work. Results go to results/cloud/style-pc1-seed1-2026-09-25/ on
# STYLE_BRANCH (set it to the branch this session may push to).
cd "$(dirname "$0")/.."
S0=results/cloud/style-dial-2026-09-25/variety
export STYLE_Q=runs/style_s1
export STYLE_RUNS="velocity_probe_s1_cpu style_pc1_s1_cpu"
export STYLE_DIR=results/cloud/style-pc1-seed1-2026-09-25
export STYLE_LABEL="Style dial seed 1"
export STYLE_BRANCH=${STYLE_BRANCH:?set STYLE_BRANCH to the branch to push results to}
export STYLE_COMPARE_BEFORE="$S0/velocity_probe_cpu/variety.json $S0/style_pc1_cpu/variety.json"
for f in $STYLE_COMPARE_BEFORE; do
  [ -f "$f" ] || { echo "missing seed-0 result $f" >&2; exit 1; }
done
exec scripts/run_style_queue.sh
