#!/bin/bash
# Copy the style-dial queue's results onto the branch and push them.
#
#   scripts/push_style_results.sh             # results so far + markers + heartbeat
#   scripts/push_style_results.sh heartbeat   # heartbeat + current history.json only
#   scripts/push_style_results.sh restore     # takeover: put pushed work back under runs/
#
# runs/ is gitignored, so a cloud machine's work survives only what this copies
# into results/cloud/style-dial-2026-09-25/ and pushes. Every git operation
# holds one lock, so the runner's heartbeat loop and a session editing NOTES.md
# never commit over each other. Pushes only to claude/style-dial-2026-09-25,
# and never forces.
#
# STYLE_BRANCH, STYLE_DIR, STYLE_Q, STYLE_RUNS and STYLE_SESSION override the
# defaults below for another queue (see scripts/run_style_s1_queue.sh).
cd "$(dirname "$0")/.."
MODE=${1:-results}
BRANCH=${STYLE_BRANCH:-claude/style-dial-2026-09-25}
D=${STYLE_DIR:-results/cloud/style-dial-2026-09-25}
Q=${STYLE_Q:-runs/style}
RUNS=${STYLE_RUNS:-"velocity_probe_cpu style_pc1_cpu style_pc1_gaps_cpu"}
LOCK=/tmp/style-push.lock
SESSION=${STYLE_SESSION:-https://claude.ai/code/session_01FSVuPhRocno6vLJuTLPxPG}
LABEL=${STYLE_LABEL:-Style dial}

current_step() { cat "$Q/CURRENT_STEP" 2>/dev/null || echo "idle"; }
current_run()  { cat "$Q/CURRENT_RUN" 2>/dev/null; }

# Complete means history.json holds as many epochs as the config asks for.
run_complete() {
  [ -f "runs/$1/best.pt" ] && [ -f "runs/$1/history.json" ] || return 1
  .venv/bin/python - "$1" <<'EOF' 2>/dev/null
import json, sys
sys.path.insert(0, "src")
from build import load_config
name = sys.argv[1]
want = int(load_config(f"configs/{name}.yaml")["train"].get("epochs", 10))
sys.exit(0 if len(json.load(open(f"runs/{name}/history.json"))) >= want else 1)
EOF
}

write_heartbeat() {
  mkdir -p "$D"
  local run ep=""
  run=$(current_run)
  [ -n "$run" ] && [ -f "$Q/$run.log" ] && ep=$(grep -h "^ep " "$Q/$run.log" | tail -n 1)
  printf '%s\n%s\n%s\n%s\n' "$(date -u +%s)" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" \
    "$(current_step)" "$ep" > "$D/HEARTBEAT.txt"
}

copy_results() {
  mkdir -p "$D/state" "$D/variety" "$D/kit_check" "$D/checkpoints"
  for f in queue.log runner.out COMPARE.txt; do
    [ -f "$Q/$f" ] && cp "$Q/$f" "$D/$f"
  done
  for r in $RUNS; do
    [ -f "$Q/$r.log" ] && cp "$Q/$r.log" "$D/$r.train.log"
    if [ -f "runs/variety/$r/variety.json" ]; then
      mkdir -p "$D/variety/$r"; cp runs/variety/$r/variety.json runs/variety/$r/report.txt "$D/variety/$r/" 2>/dev/null
    fi
    if [ -f "runs/kit_check/$r-validation/kit_check.json" ]; then
      mkdir -p "$D/kit_check/$r-validation"
      cp runs/kit_check/$r-validation/kit_check.json runs/kit_check/$r-validation/report.txt \
        "$D/kit_check/$r-validation/" 2>/dev/null
    fi
    # best.pt only once the run has finished: a mid-run best.pt would be
    # reported under the finished run's name. Never last.pt.
    if run_complete "$r"; then
      mkdir -p "$D/checkpoints/$r"
      cmp -s "runs/$r/best.pt" "$D/checkpoints/$r/best.pt" || cp "runs/$r/best.pt" "$D/checkpoints/$r/best.pt"
      cp "runs/$r/history.json" "$D/checkpoints/$r/history.json"
    fi
  done
  for m in "$Q"/DONE_*; do
    [ -e "$m" ] || continue
    case "$(basename "$m")" in DONE_egmd*) continue ;; esac
    cp "$m" "$D/state/"
  done
}

commit_and_push() {   # $1 = message
  (
    flock 9
    git add -- "$D" $EXTRA_PATHS >/dev/null 2>&1
    if git diff --cached --quiet; then exit 0; fi
    git commit -q -m "$1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: $SESSION" || exit 1
    for i in 1 2 3 4; do
      git push -q origin "HEAD:$BRANCH" 2>&1 && exit 0
      [ "$i" = 4 ] && break
      echo "push rejected (attempt $i); rebasing" >&2
      git pull -q --rebase --autostash origin "$BRANCH" 2>&1 || git rebase --abort 2>/dev/null
      sleep $((2 ** i))
    done
    echo "push failed after retries" >&2
    exit 1
  ) 9>"$LOCK"
}

case "$MODE" in
  heartbeat)
    write_heartbeat
    run=$(current_run)
    if [ -n "$run" ] && [ -f "runs/$run/history.json" ]; then
      mkdir -p "$D/progress/$run"; cp "runs/$run/history.json" "$D/progress/$run/history.json"
    fi
    commit_and_push "$LABEL: heartbeat ($(current_step))"
    ;;
  restore)
    mkdir -p "$Q"
    for m in "$D"/state/DONE_*; do [ -e "$m" ] && cp "$m" "$Q/"; done
    for r in $RUNS; do
      if [ -f "$D/checkpoints/$r/best.pt" ]; then
        mkdir -p "runs/$r"; cp "$D/checkpoints/$r/best.pt" "$D/checkpoints/$r/history.json" "runs/$r/"
      fi
      [ -d "$D/variety/$r" ] && { mkdir -p "runs/variety/$r"; cp "$D/variety/$r"/* "runs/variety/$r/"; }
      [ -d "$D/kit_check/$r-validation" ] && { mkdir -p "runs/kit_check/$r-validation"; cp "$D/kit_check/$r-validation"/* "runs/kit_check/$r-validation/"; }
      [ -f "$D/$r.train.log" ] && [ ! -f "$Q/$r.log" ] && cp "$D/$r.train.log" "$Q/$r.log"
    done
    [ -f "$D/queue.log" ] && [ ! -f "$Q/queue.log" ] && cp "$D/queue.log" "$Q/queue.log"
    [ -f "$D/COMPARE.txt" ] && [ ! -f "$Q/COMPARE.txt" ] && cp "$D/COMPARE.txt" "$Q/COMPARE.txt"
    echo "restored: $(ls "$Q" | tr '\n' ' ')"
    ;;
  results)
    copy_results
    write_heartbeat
    commit_and_push "$LABEL: results so far ($(current_step))"
    ;;
  commit)
    # For a session's own edits (NOTES.md, reports, or code fixes named in
    # EXTRA_PATHS="src/x.py ..."): same lock.
    write_heartbeat
    commit_and_push "${2:-$LABEL: notes}"
    ;;
  *) echo "usage: $0 [heartbeat|restore|commit MSG]" >&2; exit 2 ;;
esac
