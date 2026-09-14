#!/bin/bash
# Restart a batch with only the runs that have not landed yet.
#
#   scripts/resume_velocity_queue.sh <config-name> [config-name ...]
#
# Nothing on this box restarts a killed queue, so recovery after a container
# restart is manual -- and the part of it that is easy to get wrong is working
# out what still needs to run. Retraining a config that already finished costs
# ~100 minutes and overwrites the checkpoint the result was measured on.
#
# A run has landed when results/probe_<run>.txt exists. That is the right test
# and the only durable one: run_velocity_queue.sh commits that file as the run
# finishes, so it survives the restart that runs/ and the scratchpad do not.
# A half-trained run leaves a best.pt from whichever epoch it reached, so
# best.pt is NOT evidence of a finished run and is deliberately not consulted.
#
# To leave it running after the shell goes away:
#   OUT=<dir> setsid nohup scripts/resume_velocity_queue.sh <configs> &
# Set DRY_RUN=1 to print the plan and start nothing.
set -u
cd "$(dirname "$0")/.."

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  echo "usage: $(basename "$0") <config-name> [config-name ...]" >&2
  exit 2
fi

TODO=()
for r in "${RUNS[@]}"; do
  if [ -f "results/probe_$r.txt" ]; then
    echo "landed  $r"
  else
    TODO+=("$r")
  fi
done

if [ ${#TODO[@]} -eq 0 ]; then
  echo "nothing left to run"
  exit 0
fi

echo "queue   ${TODO[*]}"
[ "${DRY_RUN:-0}" = "1" ] && exit 0
exec scripts/run_velocity_queue.sh "${TODO[@]}"
