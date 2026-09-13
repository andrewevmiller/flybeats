#!/bin/bash
# Phase A' queue. Strictly sequential: two training jobs on four cores is the
# documented 76s -> 1415s collapse, so each run waits for the last.
# Override the output directory with OUT=... ; defaults to runs/queue.
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=4
Q=${OUT:-runs/queue}
PY=.venv/bin/python
mkdir -p "$Q"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$Q/queue.log"; }

probe() {   # $1 = run name
  if [ -f "runs/$1/best.pt" ]; then
    say "probing $1"
    $PY scripts/probe_velocity.py --checkpoint "runs/$1/best.pt" --seed 0 \
      > "$Q/probe_$1.txt" 2>&1
    say "probe $1 -> $Q/probe_$1.txt"
  else
    say "no checkpoint for $1, skipping probe"
  fi
}

# A'1 is already running; wait it out rather than racing it.
while pgrep -f "src/train.py" >/dev/null; do sleep 60; done
say "A'1 finished"
probe velocity_w5_cpu

for cfg in velocity_lin_cpu velocity_peak_cpu; do
  say "training $cfg"
  $PY -u src/train.py --config "configs/$cfg.yaml" > "$Q/$cfg.log" 2>&1
  say "$cfg exit $? (best: $(grep -o 'best val onset F: [0-9.]*' "$Q/$cfg.log" | tail -1))"
  probe "$cfg"
done

say "queue done"
{
  echo "=== Phase A' summary ==="
  for r in velocity_probe_cpu velocity_w5_cpu velocity_lin_cpu velocity_peak_cpu; do
    echo; echo "--- $r ---"
    sed -n '/peak (y > 0.95)/,$p' "$Q/probe_$r.txt" 2>/dev/null || echo "(no probe)"
  done
} > "$Q/SUMMARY.txt" 2>&1
say "summary -> $Q/SUMMARY.txt"
