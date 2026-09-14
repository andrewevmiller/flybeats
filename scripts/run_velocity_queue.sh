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
say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/queue.log"; }

# Results are committed as each run lands, not held until the batch ends.
# A container restart took out a run that was two hours in and would have taken
# the finished results with it if they had only existed in $OUT -- runs/ and
# the scratchpad are both outside version control and both die with the box.
# Losing one run's compute is the cost of a restart; losing a result is not.
commit_result() {
  local run=$1
  [ -f "results/probe_$run.txt" ] || return 0
  git add "results/probe_$run.txt" >/dev/null 2>&1 || return 0
  git diff --cached --quiet && return 0
  git -c user.name="Andrew Miller" -c user.email="andrewmiller857@gmail.com" \
      commit -q -m "Queue result: $run

$(grep -o 'best val onset F: [0-9.]*' "$OUT/$run.log" 2>/dev/null | tail -1)

Raw probe output, committed by scripts/run_velocity_queue.sh as the run landed.
Interpretation goes in ROADMAP.md; this is the evidence it rests on.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" >/dev/null 2>&1
  for i in 1 2 3; do
    git pull --rebase -q origin "$(git rev-parse --abbrev-ref HEAD)" >/dev/null 2>&1
    if git push -q origin HEAD >/dev/null 2>&1; then
      say "committed results/probe_$run.txt"
      return 0
    fi
    sleep $((i * 5))
  done
  say "WARNING: could not push results/probe_$run.txt"
}

probe() {
  if [ ! -f "runs/$1/best.pt" ]; then
    say "no checkpoint for $1; skipping probe"
    return 0
  fi
  mkdir -p results
  $PY scripts/probe_velocity.py --checkpoint "runs/$1/best.pt" --seed 0 \
    > "$OUT/probe_$1.txt" 2>&1

  # A probe that died leaves a file -- torch's two sparse warnings and nothing
  # else -- and copying that into results/ records the run as landed. The
  # resume script then skips it forever and the comparator cannot parse it.
  # Exactly that happened once: a run three epochs into twelve was filed as a
  # finished comparison arm.
  if ! grep -q "peak (y > 0.95)" "$OUT/probe_$1.txt"; then
    say "probe of $1 produced no table; NOT recording it (see $OUT/probe_$1.txt)"
    return 0
  fi
  cp "$OUT/probe_$1.txt" "results/probe_$1.txt"
  say "probed $1"
  commit_result "$1"
}

# Is a *real* training job in flight?  Not "does any command line mention
# src/train.py": pgrep -f matches whole command lines, and the shell that
# launches this queue carries the pattern in its own (the startup check that
# reports whether training came up). The naive guard therefore waited on its
# own parent, and a batch of five runs sat in this loop overnight having
# trained nothing. Only a live python interpreter counts as training.
train_running() {
  local p exe
  for p in $(pgrep -f "src/train\.py" 2>/dev/null); do
    [ "$p" = "$$" ] && continue
    exe=$(readlink -f "/proc/$p/exe" 2>/dev/null) || continue
    case "$exe" in */python*) return 0 ;; esac
  done
  return 1
}

main() {
  mkdir -p "$OUT"

  local RUNS=("$@")
  if [ ${#RUNS[@]} -eq 0 ]; then
    RUNS=(velocity_w5_cpu velocity_lin_cpu velocity_peak_cpu)
  fi

  # Bounded, because a guard that can block forever is how the above went
  # unnoticed: after three hours, say so loudly and run anyway.
  waited=0
  while train_running; do
    sleep 60
    waited=$((waited + 60))
    if [ "$waited" -ge 10800 ]; then
      say "WARNING: another training job still up after 3h -- starting anyway"
      break
    fi
  done
  say "starting batch: ${RUNS[*]}"

  for cfg in "${RUNS[@]}"; do
    say "training $cfg"
    $PY -u src/train.py --config "configs/$cfg.yaml" > "$OUT/$cfg.log" 2>&1
    rc=$?
    # An interrupted run still leaves the best.pt of whatever epoch it reached.
    # Probing that and committing the result files a partial run as a finished
    # arm, which is worse than losing it: the numbers look ordinary.
    if [ "$rc" -ne 0 ]; then
      say "$cfg FAILED or was interrupted (exit $rc) -- not probing. Re-running it"
      say "  will resume from runs/$cfg/last.pt; see $OUT/$cfg.log"
      continue
    fi
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
}

# Sourced (by tests/test_queue_guard.py) this defines the functions and stops.
# Executed, it runs the batch.
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  main "$@"
fi
