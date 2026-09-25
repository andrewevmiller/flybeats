#!/bin/bash
# Style-dial queue, POSIX twin of scripts/run_style_queue.ps1, for the cloud
# (Linux, CPU only). No GO/DISARM/GIVE_UP_AT gate and no Windows Update check:
# those exist because the laptop runner has to be armed by hand. This one is
# started by the session that built the code, and simply runs.
#
#   nohup setsid scripts/run_style_queue.sh > runs/style/runner.out 2>&1 < /dev/null &
#
# Strictly sequential, one python job at a time: two jobs on four cores is the
# documented 76 s -> 1415 s per-epoch collapse.
#   1. train velocity_probe_cpu (the baseline, trained HERE: numbers from
#      another machine are not comparable), style_pc1_cpu, style_pc1_gaps_cpu
#   2. fetch the E-GMD subset for the kit check (keyed on the data, not a
#      marker, since a takeover machine starts with no data)
#   3. per run: measure_variety.py, then kit_check.py
#   4. measure_variety.py --compare -> runs/style/COMPARE.txt
# A DONE_<step> marker follows each step, so a relaunch skips finished work;
# train.py --resume carries on a run cut short on the same machine. After each
# step scripts/push_style_results.sh puts the results on the branch, and a
# side loop pushes a heartbeat every 30 minutes.
cd "$(dirname "$0")/.."
Q=runs/style
mkdir -p "$Q"
echo $$ > "$Q/runner.pid"
PY=.venv/bin/python
NP=$(nproc)
RUNS="velocity_probe_cpu style_pc1_cpu style_pc1_gaps_cpu"

say() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$Q/queue.log"; }
step() { echo "$1" > "$Q/CURRENT_STEP"; echo "${2:-}" > "$Q/CURRENT_RUN"; }
push() { scripts/push_style_results.sh >> "$Q/push.log" 2>&1 || say "push failed (see $Q/push.log)"; }

( while true; do scripts/push_style_results.sh heartbeat >> "$Q/push.log" 2>&1; sleep 1800; done ) &
HB=$!
trap 'kill $HB 2>/dev/null; rm -f "$Q/runner.pid"' EXIT

run_complete() {   # history holds every epoch the config asks for
  [ -f "runs/$1/best.pt" ] && [ -f "runs/$1/history.json" ] || return 1
  $PY - "$1" <<'EOF' 2>/dev/null
import json, sys
sys.path.insert(0, "src")
from build import load_config
name = sys.argv[1]
want = int(load_config(f"configs/{name}.yaml")["train"].get("epochs", 10))
sys.exit(0 if len(json.load(open(f"runs/{name}/history.json"))) >= want else 1)
EOF
}

# Preflight: every config must resolve to CPU.
for r in $RUNS; do
  dev=$($PY -c "import sys; sys.path.insert(0,'src'); from build import load_config; print(load_config('configs/$r.yaml')['train'].get('device'))" 2>/dev/null | tail -n 1)
  if [ "$dev" != "cpu" ]; then say "refused: configs/$r.yaml resolves to device=$dev, not cpu"; exit 1; fi
done
say "runner started (pid $$), OMP_NUM_THREADS=$NP"

# --- 1. training --------------------------------------------------------
for r in $RUNS; do
  if [ -f "$Q/DONE_train_$r" ]; then say "$r already trained, skipping"; continue; fi
  if ! run_complete "$r"; then
    step "train $r" "$r"
    say "training $r"
    OMP_NUM_THREADS=$NP $PY -u src/train.py --config "configs/$r.yaml" --resume >> "$Q/$r.log" 2>&1
    rc=$?
    say "$r exit $rc (best: $(grep -o 'best val onset F: [0-9.]*' "$Q/$r.log" | tail -n 1))"
    if ! run_complete "$r"; then
      say "$r did not finish all its epochs; runs/$r/last.pt holds its place. Stopping."
      push; exit 1
    fi
  fi
  date -u +%FT%TZ > "$Q/DONE_train_$r"
  step "trained $r" ""
  push
done

# --- 2. E-GMD subset for the kit check ----------------------------------
EGMD=data/egmd/e-gmd-subset
have_egmd() { [ -f "$EGMD/info.csv" ] && [ -n "$(find "$EGMD" -name '*.flac' -print -quit 2>/dev/null)" ]; }
# No marker: the fetch skips every file already on disk, so running it again
# costs little when the data is there and finishes the job when it is partial.
step "fetch e-gmd subset" ""
say "fetching E-GMD subset (skips files already present)"
$PY -u scripts/fetch_egmd_subset.py >> "$Q/fetch_egmd_subset.log" 2>&1
say "E-GMD fetch exit $? ($(du -sh "$EGMD" 2>/dev/null | cut -f1))"
have_egmd || say "no E-GMD subset: the kit check will be skipped"
push

# --- 3. measurements ----------------------------------------------------
for r in $RUNS; do
  if [ ! -f "$Q/DONE_variety_$r" ]; then
    step "measure variety $r" ""
    say "measuring variety $r"
    OMP_NUM_THREADS=$NP $PY -u scripts/measure_variety.py --checkpoint "runs/$r/best.pt" \
      --threads "$NP" > "$Q/variety_$r.txt" 2>&1
    rc=$?; say "variety $r exit $rc"
    [ $rc -eq 0 ] && date -u +%FT%TZ > "$Q/DONE_variety_$r"
  fi
  if [ ! -f "$Q/DONE_kit_check_$r" ]; then
    if have_egmd; then
      step "kit check $r" ""
      say "kit check $r"
      OMP_NUM_THREADS=$NP $PY -u scripts/kit_check.py --checkpoint "runs/$r/best.pt" \
        --threads "$NP" --out "runs/kit_check/$r-validation" > "$Q/kit_check_$r.txt" 2>&1
      rc=$?; say "kit check $r exit $rc"
      [ $rc -eq 0 ] && date -u +%FT%TZ > "$Q/DONE_kit_check_$r"
    else
      say "kit check $r skipped: no E-GMD subset"
    fi
  fi
  push
done

# --- 4. comparison ------------------------------------------------------
if [ ! -f "$Q/DONE_compare" ]; then
  step "compare" ""
  files=""
  for r in $RUNS; do [ -f "runs/variety/$r/variety.json" ] && files="$files runs/variety/$r/variety.json"; done
  $PY scripts/measure_variety.py --compare $files > "$Q/COMPARE.txt" 2>"$Q/compare.err"
  rc=$?; say "compare exit $rc -> $Q/COMPARE.txt"
  [ $rc -eq 0 ] && date -u +%FT%TZ > "$Q/DONE_compare"
fi

step "done" ""
date -u +%FT%TZ > "$Q/DONE_all"
say "queue done"
push
