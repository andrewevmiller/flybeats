# Plan: a second seed of the pC1 style arm (cloud)

Written 25 Sep 2026 for Andrew, who will queue it from a new session. Nothing is
scheduled yet. This file has the plan and, at the end, a routine prompt that
can be used as it stands.

## Why

The overnight cloud run (branch `claude/style-dial-2026-09-25`, results in
`results/cloud/style-dial-2026-09-25/` on that branch) found that routing the
style push into pC1 makes the style dial work:

| seed 0, cloud | baseline | pC1 |
|---|---|---|
| hits changed by style | 0% | 20% |
| timing score (best val onset F) | 0.2834 | 0.2804 |

That is one seed. This run asks whether it holds on a second one.

## Design

Two training runs on seed 1, one change between them:

| run | config | status |
|---|---|---|
| baseline, seed 1 | `configs/velocity_probe_s1_cpu.yaml` | exists (seed 1, `window_seed` pinned to 0) |
| pC1, seed 1 | `configs/style_pc1_s1_cpu.yaml` | new: `_base_: style_pc1_cpu.yaml`, `data.window_seed: 0`, `train.seed: 1` |

- **The seed-1 baseline is not optional.** Timing scores move between seeds,
  so "timing drops by no more than 0.01" only means something against a
  baseline on the same seed. Every cloud experiment trains its own baseline.
- **`window_seed` is pinned to 0,** as in `velocity_probe_s1_cpu`, so both
  seed-1 runs are validated on the same windows as seed 0.
- **The gaps arm is not repeated.** Seed 0 showed it halves the style effect,
  and the verdict dropped it.
- **Pass rule, per seed:** switching style changes about 10% or more of hits,
  and timing drops by no more than about 0.01 against that seed's baseline.
  Seed 1 passing too makes the result confirmed. Seed 1 failing means the
  seed-0 result was luck or fragile.
- **Also worth reading:** whether more drums play (seed 0: kick, snare, hat and
  two toms instead of snare only), the drive slider, and whether tightness 1.5
  still silences the model.
- **The comparison table has four columns:** the seed-0 cloud numbers come from
  `results/cloud/style-dial-2026-09-25/variety/*/variety.json` on the old
  branch. All four are cloud runs on the same environment.

## What the cloud session has to build (small)

The code (`genre.targets`, `train.audio_gaps`) and both scripts are already on
`claude/style-dial-2026-09-25`, not yet on `main` or `cuda-path-verified`. So:

1. **Branch** `claude/style-pc1-seed1` from `origin/claude/style-dial-2026-09-25`.
2. **Add** `configs/style_pc1_s1_cpu.yaml`, plus its row in
   `tests/test_config.py`'s table.
3. **Make the two scripts settable,** keeping the current values as defaults so
   last night's behaviour is unchanged:
   - `scripts/run_style_queue.sh`: `RUNS`, `Q`;
   - `scripts/push_style_results.sh`: `RUNS`, `Q`, `BRANCH`, `D`.
4. **Run with** `RUNS="velocity_probe_s1_cpu style_pc1_s1_cpu"`,
   `Q=runs/style_s1`, `BRANCH=claude/style-pc1-seed1`,
   `D=results/cloud/style-pc1-seed1`.

## Timeline (from last night's measured step times)

| step | time |
|---|---|
| setup (venv, torch from PyPI, data) | about 15-20 min |
| 2 training runs, 12 epochs, 240-250 s each | about 100 min |
| E-GMD subset fetch | about 10 min |
| variety and kit check, both runs | about 26 min |
| **total** | **about 2.5 h** |

## How to queue it

One-time routine in the Default environment (`env_01VgP5eFSi1pz43rmNb7p47Q`):
- **Model:** Opus 5.5 (`claude-opus-5-5`).
- **Source:** `https://github.com/andrewevmiller/flybeats`.
- **Connectors:** pass `mcp_connections` with only `Claude_Code_Remote` in the
  create body; otherwise every claude.ai connector is attached.
- **Allowed tools:** the same as last night's routine
  `trig_01C6m6ox7M7pyRuZNFFbfivU`.
- **Takeover checks:** two more one-time routines with the SAME prompt, at
  start + 2 h and start + 4 h. Each exits in seconds if the heartbeat is fresh
  or the run is finished.

## What the 25 Sep run taught about the cloud

These are already in the prompt:
- **The machine stays on after a turn ends only while a Bash task started with
  `run_in_background: true` is pending.** So the session chains waits of
  540 s or less, and never ends a turn without one.
  - This held for 3.7 h last night.
  - With nothing pending, the machine is reclaimed and every process on it
    dies.
- **Foreground `sleep` is blocked.**
- **`download.pytorch.org` is blocked;** install torch from PyPI (CUDA
  libraries included, unused). There is enough disk if the GMD zip is deleted
  after extraction.
- **Cloud sessions push only to `claude/*` branches.**

## Reviewing the results

The branch ends with `results/cloud/style-pc1-seed1/HANDOFF_SECTION.md`,
which is pushed last. From `flybeats\git`:

```
git fetch origin claude/style-pc1-seed1
git worktree add --detach "../review-style-pc1-seed1" origin/claude/style-pc1-seed1
```

Read `reports/00-seed1-verdict.txt` first.

---

## Routine prompt (use as it stands)

````text
You are working for Andrew, the owner of the flybeats project (github.com/andrewevmiller/flybeats), on an experiment he approved on 25 Sep 2026. This is an unattended cloud run on Linux, CPU only. Andrew reviews the results on his Windows laptop. Work autonomously, and do only what is described here.

GOAL
Confirm or refute, on a second seed, that routing flybeats' style push into pC1 makes the style dial work. On seed 0 (cloud, 25 Sep, branch claude/style-dial-2026-09-25) switching style changed 20% of hits against 0% for the baseline, at a timing cost of 0.003. Train a seed-1 baseline and a seed-1 pC1 arm, measure both, and push everything to one branch for review.

START HERE: FIRST SESSION OR TAKEOVER?
Several sessions are scheduled with this same prompt, so that a later one can finish the job if an earlier one dies. Before anything else:
1. `git fetch origin` and `git ls-remote --heads origin claude/style-pc1-seed1`.
2. If that branch does NOT exist, you are the first session. Skip to BRANCH AND GIT and follow the whole prompt.
3. If it exists: `git checkout -B claude/style-pc1-seed1 origin/claude/style-pc1-seed1`. Then:
   a. If results/cloud/style-pc1-seed1/HANDOFF_SECTION.md exists, the experiment is finished. Change nothing, push nothing, and end with one line saying so.
   b. Otherwise read results/cloud/style-pc1-seed1/HEARTBEAT.txt. Its first line is a UTC Unix time. If it is less than 60 minutes older than `date -u +%s`, another session is running the experiment. Change nothing, push nothing, and end with one line saying so.
   c. Otherwise you are a TAKEOVER: the earlier session died. Append "Takeover at <UTC time>; last heartbeat: <its contents>" to results/cloud/style-pc1-seed1/NOTES.md. Do SETUP steps 1-3. Check that WHAT TO BUILD is committed on the branch, build only what is missing, run the tests with -m "not gpu", and commit and push. Then run the restore mode of scripts/push_style_results.sh (with the environment variables under RUN SETTINGS) to put finished work back under runs/, and continue from STAYING WITH THE RUN. Finished steps are skipped; a training run that was cut short starts again from epoch 0, because checkpoints are only pushed when a run finishes.

BRANCH AND GIT
- Start from the first experiment's branch, which holds the code and scripts this run needs: `git checkout -b claude/style-pc1-seed1 origin/claude/style-dial-2026-09-25`.
- Commit and push ONLY to claude/style-pc1-seed1. Never push to any other branch, never open or merge a PR, and never force-push.
- Push the branch straight away, with results/cloud/style-pc1-seed1/HEARTBEAT.txt (format under HEARTBEAT), so later sessions can see you are alive. Update and push the heartbeat after each SETUP step and after the code is committed, until the runner takes that job over.
- Do not change scripts/run_style_queue.ps1 (the laptop's runner), and do not change anything under results/cloud/style-dial-2026-09-25/.

SETUP (note times and versions for NOTES.md)
1. Record `nproc`, `free -g`, `df -h .` and `python3 --version`. (Measured on 25 Sep: 4 cores, 15 GB RAM, 30 GB free disk.)
2. `python3 -m venv .venv`, then `.venv/bin/pip install --no-cache-dir torch`, then `.venv/bin/pip install --no-cache-dir -r requirements-dev.txt`.
   - download.pytorch.org is BLOCKED by this environment's network policy (403 at the proxy); do not try the CPU index. The PyPI wheel brings unused CUDA libraries (about 7 GB on disk); torch runs on CPU without a GPU.
   - Confirm `.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` prints False.
3. Data (not in git):
   - `.venv/bin/python scripts/fetch_data.py`, the MaleCNS tables, 1.1 GB;
   - `.venv/bin/python scripts/fetch_egmd.py`, GMD, 5.1 GB. Once it has extracted, ALWAYS delete the downloaded zip, for disk.
   Last night this whole setup took 17 minutes. The Bash tool's limit is 10 minutes: give long commands timeout 600000, or run them with run_in_background: true and wait for the notification. Foreground `sleep` is blocked in this harness.

WHAT TO BUILD (small; everything else already exists on the branch)
1. configs/style_pc1_s1_cpu.yaml: `_base_: style_pc1_cpu.yaml`, `name: style_pc1_s1_cpu`, `data: {window_seed: 0}`, `train: {seed: 1}`. Give it a header comment in the style of configs/velocity_probe_s1_cpu.yaml: what it repeats, why window_seed is pinned, and that its partner is velocity_probe_s1_cpu. Add its row to the table in tests/test_config.py.
2. Make scripts/run_style_queue.sh and scripts/push_style_results.sh settable from the environment: `RUNS=${RUNS:-<current list>}`, `Q=${Q:-runs/style}`, and in the push script also `BRANCH=${BRANCH:-claude/style-dial-2026-09-25}` and `D=${D:-results/cloud/style-dial-2026-09-25}`.
   - The defaults must stay exactly the current values, so a run without the variables behaves as it did on 25 Sep.
   - The runner must pass the variables through to every push-script call (export them).
   - Update each script's header comment to say so.
   - The runner's compare step should also take extra variety.json files from an environment variable `COMPARE_EXTRA` (default empty) and put them before its own.
3. Run the full suite: `.venv/bin/python -m pytest -q -m "not gpu"`. It was 264 passed on this branch. Commit and push the code.

RUN SETTINGS
Export these for the runner and for every manual call of the push script:
  RUNS="velocity_probe_s1_cpu style_pc1_s1_cpu"
  Q=runs/style_s1
  BRANCH=claude/style-pc1-seed1
  D=results/cloud/style-pc1-seed1
  COMPARE_EXTRA="results/cloud/style-dial-2026-09-25/variety/velocity_probe_cpu/variety.json results/cloud/style-dial-2026-09-25/variety/style_pc1_cpu/variety.json"
(The COMPARE_EXTRA files are the seed-0 cloud measurements, already on this branch. COMPARE.txt then has four columns: baseline s0, pC1 s0, baseline s1, pC1 s1.)

HEARTBEAT
results/cloud/style-pc1-seed1/HEARTBEAT.txt:
- line 1: `date -u +%s`;
- line 2: the UTC time as text;
- line 3: the current step;
- line 4: the newest "ep N ..." line of the current training log, if any.
The runner's side loop refreshes it every 30 minutes.

Expected: about 17 min of setup, about 100 min for the two training runs (240-250 s per epoch, 12 epochs), about 10 min for the E-GMD subset fetch, and about 26 min of measurements. About 2.5 h in all.

RESULTS, in results/cloud/style-pc1-seed1/ on the branch
- queue.log, runner.out, both training logs and COMPARE.txt;
- variety/<run>/ and kit_check/<run>-validation/ for both runs;
- checkpoints/<run>/best.pt and history.json for both runs;
- NOTES.md (machine, versions, epoch times, any takeover, anything skipped), HEARTBEAT.txt and state/;
- reports/ and HANDOFF_SECTION.md.

REPORTS: Andrew reads these first, possibly on his phone. Plain language, minimally technical, numbers in brackets.
- reports/arm-style-pc1-seed1.txt, 150-300 words, against the SEED-1 baseline:
  - does switching style change what the model plays (share of hits changed, largest signal shift)?
  - which drums play;
  - the timing score;
  - groove match and busyness;
  - what the sliders do, including whether tightness 1.5 silences it;
  - whether timing holds on other kits.
- reports/00-seed1-verdict.txt, under 200 words:
  - a four-column table: baseline s0, pC1 s0, baseline s1, pC1 s1. Take the s0 timing scores from results/cloud/style-dial-2026-09-25/queue.log: 0.2834 and 0.2804;
  - a plain answer: does the result hold on a second seed?
  - "Holds" only if seed 1 also changes about 10% or more of hits AND loses no more than about 0.01 of timing against the seed-1 baseline. Otherwise say which part failed.
  - Say which other seed-0 effects repeated: more drums playing, the drive slider working, and tightness 1.5 no longer silencing it.
  - Say plainly that these are cloud numbers, comparable only with other cloud runs.
- HANDOFF_SECTION.md: one "## Style dial, second seed (cloud, <date>)" section, covering what was built (files, commits), the machine, the four-column table and the verdict, anything skipped or failed including any takeover, and where each file is on the branch. It is the "finished" signal, so push it LAST.

HARD RULES
- CPU only. Both configs resolve to device: cpu; the runner's preflight checks this. No CUDA work of any kind.
- Never run two training or measurement jobs at once. Oversubscribed threads caused a documented collapse from 76 s to 1,415 s per epoch. While the runner works, run nothing heavy yourself.
- Minimal, additive edits. Existing configs, checkpoints and the scripts' default behaviour must stay exactly as they are.

STAYING WITH THE RUN: THIS DECIDES WHETHER THE RUN PRODUCES ANYTHING
Measured in this environment on 14 and 25 Sep:
- The machine stays on after your turn ends ONLY while a Bash task you started with run_in_background: true is still pending. When it finishes, the harness wakes you. This carried the 25 Sep run through 3.7 hours.
- With nothing pending, the machine is reclaimed shortly after the turn ends, and every process dies, nohup'd ones included.
So:
1. Launch the runner detached, with the RUN SETTINGS exported: `nohup setsid scripts/run_style_queue.sh > runs/style_s1/runner.out 2>&1 < /dev/null &`. It writes runs/style_s1/runner.pid first; check that the file exists.
2. In the SAME turn, start one WATCH: a Bash call with run_in_background: true and timeout 600000:
   `timeout 540 bash -c 'while kill -0 $(cat runs/style_s1/runner.pid) 2>/dev/null && [ ! -f runs/style_s1/DONE_all ]; do sleep 30; done'; date -u; tail -n 3 runs/style_s1/queue.log; grep -h "^ep " runs/style_s1/*.log 2>/dev/null | tail -n 1`
3. When the watch wakes you:
   - if the runner is alive and DONE_all is absent, start the next WATCH BEFORE doing anything else;
   - then look further only if something changed.
   Keep wake turns short.
4. THE RULE: until the final push, never end a turn without exactly one WATCH pending. That includes SETUP and the build: anything longer than a few minutes goes in a background task whose notification you wait for.
5. Keep each watch at 540 s or less. Do not rely on ScheduleWakeup or Monitor to keep the machine on.
6. If a PushNotification tool is available, send short notes:
   - setup done, with the ETA;
   - each run finished, with its best val onset F;
   - the verdict;
   - any crash or stall.
7. If the runner dies without DONE_all:
   - find out why in runner.out and the logs;
   - if it is a bug in your code, fix it, commit, and relaunch (it resumes);
   - if the same step fails twice, stop, push what exists, and explain in NOTES.md and HANDOFF_SECTION.md.
   Keep a WATCH pending throughout.

FINISH
When DONE_all exists:
1. Write the reports and NOTES.md, and push them.
2. Write HANDOFF_SECTION.md and push it last.
3. End with a short plain-language summary: the four-column table, the verdict, and anything skipped.
````
