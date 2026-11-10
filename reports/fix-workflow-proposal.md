# Proposal: a workflow for building and testing the fixes one at a time

Status: **proposal, not locked.** Nothing in the code or `config/locked.yaml` changes with this document. Written
10 Nov 2026. It takes the fix table (fixes 00 to 10, ranked by how much they leave the connectome alone) and the specs
in `reports/fix-specs-proposal.md` (branch `claude/friendly-cray-y9m2p8`) and says who builds each fix, in what order,
how each one is tested, and what has to be true before the next one starts.

Setting: Claude Code runs on Andrew's Windows PC, where `.venv`, the Slakh data, the slice and the GPU are. One
session there is the **bench**. Everything below happens on the bench; nothing runs in the cloud. There is one GPU, so
runs are strictly serial, and that is the pace the workflow is built around.

## 1. Roles

Four roles, three of them filled by cheaper models. Each role gets a narrow brief and a narrow file list so that a
weaker model cannot do much harm by misunderstanding.

| Role | Model | Lives where | May edit | Job |
|---|---|---|---|---|
| **Coordinator** | Opus (or the session Andrew is talking to) | The bench session | `reports/*.md`, `reports/fix-results.md` | Holds the queue, dispatches one fix, runs the GPU commands, records results, decides what is next by the rules in section 4 |
| **Builder** | Sonnet | A subagent of the bench session, fresh context per fix | Only the files the spec names | Implements one spec with its tests, runs the suite, hands back a diff |
| **Reviewer** | Haiku | A subagent, fresh context, sees only the diff and the spec | Nothing | Walks the checklist in section 3.3 and returns pass or a numbered list of failures |
| **Reader** | Sonnet | A subagent, fresh context | `reports/fix-results.md` only | Applies the spec's pre-written decision rule to the run's `memorise.json` and `log.jsonl`, writes one table |
| **Andrew** | | | `config/locked.yaml`, `config/CHANGELOG.md`, tags | Approves specs for 08-10 before they are built; decides adoption and locking; starts the full 2.5-hour runs if he wants to watch them |

Out of bounds for every agent, as the specs already say: `config/locked.yaml`, `config/CHANGELOG.md`,
`src/flybeats/scoring/`, and `src/flybeats/connectome/` (except fix 08, whose spec will name its exact lines).

Mechanics on the bench:

- Subagents are started with the Agent tool and a `model` override (`sonnet`, `haiku`). No worktree isolation:
  `config/paths.local.yaml` is not committed, so a worktree would not find the data. Each fix is a branch in the one
  checkout, and because only one fix is open at a time there is nothing to collide with.
- GPU commands are run by the coordinator in a background shell and watched with Monitor, never by a subagent. One
  run at a time, Windows Update paused first (RETROSPECTIVE-v0.0.md). Screens use `--no-chart`; full runs keep the
  chart so Andrew can watch.
- Every command uses `.venv\Scripts\python.exe`, as the README does.

## 2. Order

The table's order is the build order, with three adjustments forced by dependencies. Each stage opens only when the
stage before it has closed (section 4).

| Stage | Fixes | Connectome | GPU needed | Opens when |
|---|---|---|---|---|
| 0. Measure | 00 count dials, 01 calibrate gate, 02 screening run | Untouched | 02 only (three 15-minute validation screens) | Now |
| 1. Training and scoring | 03 split learning rates, 04 sqrt hit weights, 05 minimum gap | Training or scoring only | 03, 04: screen, then a full pair if the screen passes. 05: none (CPU rescoring) | 00 counted; 02 validated or rejected |
| 2. After the motor neurons | 06 refit alone, 07 muscle filter | Readout side only | Same as 03/04 | Stage 1 closed |
| 3. Strength expression | 08 size-matched control groups, 09 loop-strength cap, 10 per-connection dials | Strengths | Same, plus `build_controls.py` for 08 | A spec exists and Andrew has approved it |

Adjustments to the table's order:

1. **02 must be validated before 03 to 07 use it.** Its three validation screens (locked real, locked control 1,
   real with `--history-readout`) cost 45 minutes of GPU and run before anything in stage 1. If 02 fails its own
   test (it does not reproduce both known orderings), stage 1 fixes skip the screen and go straight to full 2.5-hour
   pairs, which is 5 hours a fix instead of 30 minutes plus 5 hours only for the ones that look good.
2. **05 is CPU-only, so it fills GPU gaps.** Its first test is a rescoring of saved weights. It is built in its
   table position but its rescoring can run while a screen is on the GPU.
3. **07 is blocked until Andrew supplies a cited time constant.** The builder is not started until the value is in
   the spec. If none arrives, 07 is skipped and the gap noted.
4. **08 to 10 have no specs.** Stage 3 begins with the coordinator writing each spec in the format of
   `fix-specs-proposal.md`, including the decision rule, and ends with Andrew's approval. Each is also conditional:
   - 08 only if fix 00 shows the controls have materially more dials than the real network (the table's guess is
     about 2.4 times). If the counts are close, 08 has nothing to fix.
   - 09 only if fix 03 did not bring the real network's share of above-clip steps near control 1's.
   - 10 changes `model.weights` in `locked.yaml`, so it is not a trial option but a new preregistration. The
     workflow stops before 10 and hands Andrew the evidence from 00 to 09; building it is his call.

## 3. The loop for one fix

Every fix goes through the same eight steps. A step that fails sends the fix back, not forward.

### 3.1 Open

The coordinator updates `main`, creates `fix/NN-short-name` from it, and starts a builder with the prompt in
section 5.1. The prompt carries the spec text verbatim, the allowed file list, and the test commands. The builder
does not see this document or the other fixes.

### 3.2 Build

The builder implements the spec and its tests, then runs:

```
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m pytest -q -m gpu tests/test_model.py
```

The second line runs the GPU-marked tests in `tests/test_model.py`; they take a few minutes and need the GPU idle.
The builder returns the diff, the test output, and a list of every file it touched. A builder that wants to edit a
file outside its list stops and says so instead.

### 3.3 Review

A reviewer gets only the spec and `git diff main...fix/NN`. It answers each line with pass or fail and a line
number:

1. Every file in the diff is on the spec's allowed list.
2. The new behaviour is behind the spec's option, and the option is off by default.
3. There is a test that the option off gives today's output exactly (bit-identical weights after the spec's number
   of steps, or exact equality of the function's output, as the spec says).
4. Each test the spec lists exists and ran in the builder's output.
5. No change to `locked.yaml`, `CHANGELOG.md`, `scoring/` or `connectome/` (fix 08: only its named lines).
6. Does the diff touch `network.py`, `ear.py`, `loss.py` or `training.py`? Yes or no. (This decides step 3.4; it is
   not a failure.)
7. Nothing in the diff is unrelated to the spec: no reformatting, no renamed functions, no comment rewrites.

Two fails go back to the builder with the reviewer's list. A third round goes to the coordinator, who either fixes
it or closes the fix as "not built" with the reason.

### 3.4 Fingerprint

If the reviewer answered yes to line 6, the recorded training step's fingerprint has changed and
`memorise.py` will refuse the recorded step until the check is rerun. The coordinator runs:

```
.venv\Scripts\python.exe scripts\check_cuda_graph.py
```

About 15 minutes of GPU. It must pass (all three parts) before any run in step 3.5 uses the recorded step. If it
fails, the fix goes back to the builder with the check's output: the fix changed training itself with the option
off, whatever the identity test said. Fix 05 touches `loss.py` only in `peaks`, which is not in the step; it still
changes the fingerprint, so the check runs, but it is expected to pass unchanged.

### 3.5 Run

What runs depends on the fix. The values and the decision rule are in the spec before the run starts; the coordinator
copies them into `reports/fix-results.md` under the fix's heading before pressing go.

| Fix | Run | Cost |
|---|---|---|
| 00 | `scripts\build_controls.py` (count only, if the script gains a `--count` mode; otherwise a full rebuild) | Minutes to an hour, CPU, 8 GB RAM |
| 01 | `scripts\gate_calibration.py` at both sizes | CPU, hours; can share the machine with a GPU run |
| 02 | Three screens: `memorise.py --screen --pair`, then `--screen --history-readout` | 45 minutes GPU |
| 03, 04, 06, 07 | `memorise.py --screen --pair --<option>`; then, if the screen passes, `memorise.py --pair --<option>` | 30 minutes GPU, then 5 hours GPU |
| 05 | The CPU rescoring script on `real` and `control-1` at gap 10 | Minutes, CPU |
| 08 | `build_controls.py` rebuild, then screen and full pair as above | Hours CPU, then as 03 |
| 09 | As 03 | As 03 |

The screen's pass rule for a stage 1 or 2 fix, fixed here so a weaker model can apply it: the real network's mean
hit F1 over kick, snare and closed hi-hat at step 300 with the option on exceeds the locked real screen by at least
0.03, and control 1 with the option on is not below its locked screen by more than 0.03. Both numbers come from the
02 validation screens, so they exist before any fix is screened. A fix that fails the screen does not get a full run;
its branch is still merged (section 3.7) with the option off.

### 3.6 Read

A reader gets the spec's "read the result by" paragraph, the decision rule copied into `fix-results.md`, and the
paths of the run's `memorise.json` and `log.jsonl` (and control 1's). It fills in one table under the fix's heading:
the numbers the rule names, the rule's verdict, and anything the spec lists as a guard (fix 04: a rare drum that
stopped firing). It does not interpret beyond the rule. The coordinator checks the table against the files once,
then adds a two-line note to the session notes.

### 3.7 Close

A fix is closed in one of three states:

- **Merged, option off.** Tests and review passed, the fingerprint check passed if needed, and the run happened.
  The branch merges to `main`. This is the state of `--history-readout` today, and it is the state every built fix
  ends in whatever its result, so the trial options accumulate and can be combined later.
- **Not built.** Review failed three times, or the fix was blocked (07 with no time constant). The reason goes in
  `fix-results.md`.
- **Adopted.** Only Andrew can move a fix here, by locking it with a new tag. Nothing in this workflow does that.

### 3.8 Next

The coordinator reads section 4 and either opens the next fix or stops.

## 4. Gates between stages

- **Stage 0 closes** when 00's counts are in `controls_report.json`, 01 has written `gate_calibration.json` with its
  verdict, and 02 has been validated or rejected. If 01 says the gate is mis-set (neither GRU reaches 0.9), the
  coordinator stops and reports: a mis-set gate changes what every later result means, and that is Andrew's
  decision, not a fix.
- **Stage 1 closes** when 03, 04 and 05 are each merged or not built, with a line in `fix-results.md`.
- **Stage 2 closes** the same way for 06 and 07.
- **Stage 3** opens one fix at a time, each on its own approved spec, with the conditions in section 2. It ends
  before 10.
- **Combination run**, after stage 2 or 3: the coordinator lists the merged fixes whose full runs passed their
  rule, and proposes one combined trial (`memorise.py --pair` with those options together) to Andrew. It is not run
  without his go, because its cost is 5 hours and its result is the one most likely to tempt a change to the locked
  settings.

A hard stop at any point: a run that trips `stop_if` (activity unbounded or not settled), a test suite that fails on
`main`, or a diff that touches an out-of-bounds file. The coordinator reports and waits.

## 5. Prompts

These are the whole brief each subagent gets. They are deliberately short and repetitive, because a weaker model does
best with one job, one file list and one way to say it is done.

### 5.1 Builder

```
You are implementing one change in the flybeats repository on this machine, on branch fix/NN-name, which is
already checked out. Read the spec below and implement exactly it, with the tests it lists. Add tests to
tests/test_model.py (or the test file the spec names).

You may edit only these files: <list from the spec>. If the change needs any other file, stop and report
which file and why. Do not touch config/locked.yaml, config/CHANGELOG.md, src/flybeats/scoring/ or
src/flybeats/connectome/.

The new behaviour must be off by default. With it off, every output and every gradient must equal today's
exactly; write the test the spec describes for that.

When done, run:
  .venv\Scripts\python.exe -m pytest -q
  .venv\Scripts\python.exe -m pytest -q -m gpu tests/test_model.py
and paste both outputs. Do not commit. Report: the files you changed, the tests you added, the two outputs,
and anything in the spec you could not do.

Spec:
<spec text verbatim>
```

### 5.2 Reviewer

```
Review a diff against a spec. Answer each of the seven lines below with PASS or FAIL and, for a FAIL, the
file and line. Do not suggest improvements; only check what is listed.

1. Every file in the diff is on this list: <list>.
2. The new behaviour is behind the option <name> and off by default.
3. A test exists that with the option off the output equals today's exactly, as the spec describes.
4. Each test the spec lists exists, and appears in the test output below.
5. No change to config/locked.yaml, config/CHANGELOG.md, src/flybeats/scoring/, src/flybeats/connectome/.
6. Does the diff touch src/flybeats/model/network.py, ear.py, loss.py or training.py? Answer YES or NO.
7. Nothing in the diff is unrelated to the spec (no reformatting, renames or comment rewrites).

Spec:
<spec text>

Diff:
<git diff main...fix/NN>

Test output:
<builder's pytest output>
```

### 5.3 Reader

```
Fill in one table from two result files. Do not interpret beyond the rule.

Rule: <decision rule copied from reports/fix-results.md>
Read the result by: <the spec's paragraph>
Guards: <the spec's guards, or "none">

Files: <path to memorise.json and log.jsonl for the run; the same for control 1; the locked baselines>

Write, as a markdown table: each number the rule names, for the run and for its baseline; the rule's
verdict (PASS or FAIL); each guard and whether it fired. Then one sentence saying which numbers decided it.
```

## 6. Cost and calendar

GPU hours per fix if 02 validates (screen 0.5 h, full pair 5 h only on a passing screen, fingerprint check 0.25 h
when needed):

| Fix | Build and review | GPU, worst case |
|---|---|---|
| 00, 01 | An hour each | None |
| 02 | An hour | 0.75 h |
| 03, 04 | An hour each | 5.75 h each |
| 05 | An hour | 0.25 h (fingerprint only) |
| 06 | Two hours (refit, Adam state, CUDA-graph part) | 5.75 h |
| 07 | An hour, if unblocked | 5.75 h |
| 08, 09 | Spec, approval, then as 03 | 5.75 h each plus rebuilding controls for 08 |

Stages 0 to 2 fit in about four GPU days if every screen passes, and under two if most fail, since a failed screen
costs half an hour. Without a validated 02 every stage 1 and 2 fix costs 5 hours flat, so the first GPU hour of
this workflow (the 02 validation) is the one that decides its length.

## 7. What this workflow does not do

- It does not adopt anything. Every fix ends merged with its option off; locking is Andrew's step with a tag.
- It does not run two things on the GPU at once, or let a subagent start a run.
- It does not build fix 10. The table ranks it most destructive of the connectome's role in the model, and it
  changes a locked setting, so it is a new preregistration rather than a fix.
- It does not promise the gate. The specs put every single fix under 5 out of 100 to pass 0.9, and fix 01 exists to
  find out whether 0.9 is the right number at all.
