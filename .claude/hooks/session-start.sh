#!/bin/bash
# Put a killed training batch back on its feet when the container comes back.
#
# This box restarts often -- five times in one session, twice only 37 minutes
# apart -- and a restart kills every process while leaving the disk intact.
# Training itself survives that now (runs/<name>/last.pt, resumed per epoch),
# but nothing restarts the *queue*, so the box sits idle until something
# notices. An hourly Routine was doing the noticing until it fired three times
# into a session that had hit its usage limit, and the box idled 2.5 hours.
#
# A SessionStart hook does not depend on the session being able to answer.
#
# INERT BY DEFAULT. It does nothing at all unless .claude/batch.active exists,
# and that file is gitignored -- it is a local statement that a batch is meant
# to be running on this machine, not something a fresh clone inherits. Nobody
# who checks this repository out starts a four-core training job by accident.
set -uo pipefail

cd "${CLAUDE_PROJECT_DIR:-$(dirname "${BASH_SOURCE[0]}")/../..}" || exit 0

MARKER=.claude/batch.active
LOG=.claude/hooks/session-start.log
say() { echo "[$(date -u +%FT%TZ)] $*" >> "$LOG"; }

[ -f "$MARKER" ] || exit 0

RUNS=$(grep -vE '^\s*(#|$)' "$MARKER" | tr '\n' ' ')
[ -n "$RUNS" ] || exit 0

# Nothing outstanding: the batch is done and the marker is stale, so retire it.
# This comes first because it reads files rather than processes -- a stale
# marker should be cleared whatever is or is not running.
REMAINING=0
for r in $RUNS; do
  [ -f "results/probe_$r.txt" ] || REMAINING=$((REMAINING + 1))
done
if [ "$REMAINING" -eq 0 ]; then
  say "all runs landed; removing $MARKER"
  rm -f "$MARKER"
  exit 0
fi

# Already training? Then nothing was lost, and starting a second job would be
# actively harmful: two on four cores is the documented 76 s -> 1,415 s
# collapse. Same /proc/PID/exe test the queue uses, because `pgrep -f` matches
# the command line of whatever is asking -- including this script.
for p in $(pgrep -f "src/train\.py" 2>/dev/null); do
  case "$(readlink -f "/proc/$p/exe" 2>/dev/null)" in
    */python*) exit 0 ;;
  esac
done

say "no training running, $REMAINING of $(echo $RUNS | wc -w) outstanding -- relaunching"
OUT=${OUT:-runs/queue} setsid nohup scripts/resume_velocity_queue.sh $RUNS \
  >> "$LOG" 2>&1 < /dev/null &
exit 0
