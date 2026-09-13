#!/bin/bash
# Run training configs strictly in sequence, probing each.
#
#   scripts/run_velocity_queue.sh [config-name ...]
#
# Sequential is not a preference. Torch takes a thread per core in each
# process, and two training jobs on four cores put one epoch from 76 s to
# 1,415 s -- the oversubscribed threads spin rather than progress. The script
# also waits out any training already in flight rather than racing it, so a
# second batch can be chained behind a first while that one is still going.
#
# Results land in $OUT (default runs/queue): queue.log for progress,
# probe_<run>.txt per run, SUMMARY.txt at the end. Nothing here is in version
# control, so commit anything worth keeping.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
OUT=${OUT:-runs/queue}
PY=${PY:-.venv/bin/python}
mkdir -p "$OUT"

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(velocity_w5_cpu velocity_lin_cpu velocity_peak_cpu)
fi

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/queue.log"; }

probe() {
  if [ -f "runs/$1/best.pt" ]; then
    $PY scripts/probe_velocity.py --checkpoint "runs/$1/best.pt" --seed 0 \
      > "$OUT/probe_$1.txt" 2>&1
    say "probed $1"
  else
    say "no checkpoint for $1; skipping probe"
  fi
}

while pgrep -f "src/train.py" >/dev/null; do sleep 60; done
say "starting batch: ${RUNS[*]}"

for cfg in "${RUNS[@]}"; do
  say "training $cfg"
  $PY -u src/train.py --config "configs/$cfg.yaml" > "$OUT/$cfg.log" 2>&1
  say "$cfg -> $(grep -o 'best val onset F: [0-9.]*' "$OUT/$cfg.log" | tail -1)"
  probe "$cfg"
done

{
  echo "=== queue summary: ${RUNS[*]} ==="
  for r in "${RUNS[@]}"; do
    echo; echo "--- $r  ($(grep -o 'best val onset F: [0-9.]*' "$OUT/$r.log" 2>/dev/null | tail -1)) ---"
    sed -n '/peak (y > 0.95)/,$p' "$OUT/probe_$r.txt" 2>/dev/null || echo "(no probe)"
  done
} > "$OUT/SUMMARY.txt" 2>&1
say "batch done -> $OUT/SUMMARY.txt"
