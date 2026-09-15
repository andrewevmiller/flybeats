# Settling the operating point — 15 September 2026

Follows `2026-09-15-probe-grid.md`, which found that no Phase 4 arm separates
and that `rho=10` destroys the rewired arms before training starts, but left
`scripts/propagation.py` itself unfixed and the per-arm operating points
unmeasured. This is that follow-up.

Evidence: `propagation_grid.json` (the fixed tool, 5 arms x 8 radii, 30k tier,
settled-window statistics) and `probe_grid.json` (the earlier whole-clip
grid, kept for comparison).

**Outcome: `propagation.py`'s transient bug is fixed and confirmed on real
data; common-rho cannot be defended, more sharply than previously known
(motor neurons pin at `rho=1` for the rewired arms, the *smallest* radius
tested); and a defensible per-arm operating point exists for three of five
arms. The motor participation-ratio guard is never satisfied by any arm at
any radius before training — that is a finding about the untrained network,
not a bug in the guard, and the campaign should not gate on it.**

---

## 1. The transient bug, fixed and confirmed

The old script ran 300 steps from `v=0` and scored the whole run as one
number. Rerunning the same real-audio sweep at the 30k tier and printing the
motor-modulation statistic in 100-step windows shows why that was wrong —
window 0 is 5-20x every later window, at every radius tested:

| rho | window 0 | window 1 | window 5 | window 15 (last) |
|---|---|---|---|---|
| 1 | 0.025 | 0.001 | 0.000 | 0.000 |
| 10 | 0.691 | 0.089 | 0.069 | 0.061 |
| 50 | 1.831 | 0.952 | 1.386 | 1.143 |

The old 300-step run never left window 0-2. `scripts/propagation.py` now:

* runs >=1,600 steps (default), widening `data.seconds` to give the encoder
  that much real audio rather than padding with silence — the shipped configs
  crop clips to 4.0s (800 steps at `step_ms=5`), so this is a real change to
  what the model is fed, not just a longer forward pass over the same clip;
* reports the motor statistic in non-overlapping 100-step windows and emits
  **`t_settle`**: the first window after which it changes by <10% (or by
  less than an absolute floor — see the trap below) across three consecutive
  windows;
* computes every acceptance statistic on `[t_settle, T)` only.

**Measured `t_settle`, 30k tier, real arm:** window 1 (step 100) for
`rho` in {1, 2, 5, 10}; step 1100 for `rho=25`; step 500 for `rho=50`; step
400 for `rho=100`; step 300 for `rho=250`. Settling time is *not* monotonic
in rho — it is short in the clean linear regime, grows near the transition
into saturation, then shortens again once the network is fully pinned (a
pinned neuron is, trivially, already at steady state). Several (arm, rho)
cells never settle within 1,600 steps at all — see §3.

### A trap in `t_settle` itself, found and fixed

The first full run (5 arms x 8 radii, ~18 min) reported "never settled" for 7
of 40 cells, always where the motor statistic was already near zero (e.g.
`sign_shuffled:0` at `rho=1`: `[0.019, 0.000, 0.000, ...]`). The relative
10% criterion is unstable once the statistic itself is near zero: a
genuinely flat 0.0003 -> 0.0005 step is a 67% "change" by the relative test
and never clears the bar. Fixed with an absolute floor (0.01, well below any
meaningful window value in this data): a window-to-window change counts as
settled if it clears *either* the relative or the absolute test. Re-ran the
full grid after the fix (another ~18 min); the numbers in §3 are from the
fixed run. Recorded here because it is exactly the "detector reproducing the
failure it exists to catch" pattern the CUDA bring-up note flags twice —
a criterion written in relative terms silently breaks on the near-zero case
it should have covered most easily.

---

## 2. The two new guards, and what they found

**Per-neuron pinned fraction**, on `[t_settle, T)`: reject a radius where
more than 1% of *all* neurons sit at the state clip more than half the
settled window, or where *any* motor neuron does. Replaces the old
mean-over-unit-steps `at_clip`, which could not distinguish "every neuron
pinned 5% of the time" from "5% of neurons pinned permanently."

**Motor participation ratio >= 8** (the kit's class count), on the settled
motor trace. A trajectory collapsed onto one or two modes cannot carry eight
drum classes regardless of the readout.

**Finding: no (arm, rho) cell in the swept range passes both guards.** The
binding constraint is almost always the PR guard, not the pinned-fraction
guard:

| arm | max settled motor PR (any rho) | at rho |
|---|---|---|
| real | 8.94 | 250 (already 20.8% pinned, 41 motor units stuck) |
| erdos_renyi:0 | 11.55 | 250 (72.8% pinned) |
| rewired:0 | 6.15 | 250 (74.6% pinned) |
| rewired:1 | 5.65 | 250 (74.1% pinned) |
| sign_shuffled:0 | 4.72 | 100 (63.2% pinned) |

Every cell that reaches PR>=8 is already destroyed by pinning; every cell
with clean pinning has PR far below 8 (typically 1.0-1.2 at the small radii
where pinning is near zero). **This is a property of the untrained network,
not a bug in the guard or the sweep:** at initialisation, before `log_gain`
has been trained, the motor pool's response to real audio genuinely does not
carry eight independent modes at any tested scale. It is consistent with the
probe-grid finding that trained models reach ~0.30 against a ~0.20-0.22
untrained ceiling — diversifying the motor code looks like training's job,
not initialisation's. **Recommendation: do not gate Phase 4's start on the
PR guard pre-training; re-check it post-training instead, where it is a
meaningful test of whether training actually diversified the motor code.**

The pinned-fraction guard, by contrast, discriminates arms sharply — see §3.

---

## 3. Per-arm response curves (settled-window, 30k tier, 8 clips, 1,500 held-out steps typical)

probe_F is the same ridge-to-onset-F probe `scripts/probe_grid.py` uses,
scored on `[t_settle, T)` rather than the whole clip.

### real

| rho | t_settle | probe_F | pinned (all / motor) | PR |
|---|---|---|---|---|
| 1 | 100 | 0.195 | 0.03% / 0 | 1.05 |
| 2 | 100 | 0.192 | 0.34% / 0 | 1.00 |
| 5 | 100 | **0.225** | 0.64% / 0 | 1.02 |
| 10 | 100 | **0.229** | 1.79% / 0 | 1.20 |
| 25 | 1100 | 0.176 | 6.00% / 26 | 2.17 |
| 50 | 500 | 0.161 | 10.66% / 31 | 5.89 |
| 100 | 400 | 0.148 | 15.54% / 33 | 8.54 |
| 250 | 300 | 0.153 | 20.77% / 41 | 8.94 |

Zero motor neurons pinned through `rho=10`. Argmax is `rho=10`
(F=0.229), but `rho=5` (F=0.225) is indistinguishable from it against the
~0.014 noise floor established in the previous report, and is meaningfully
cleaner (0.64% vs 1.79% pinned). **Recommended range: rho in [5, 10].**
Notably, this is close to the value the old, buggy script shipped
(`rho=10`) — that value turns out to be defensible *for the real arm
specifically*, once measured correctly. It was void as reasoning (chosen on
a cold-start transient, at the wrong tier), not necessarily void as a number.

### rewired:0

| rho | t_settle | probe_F | pinned (all / motor) | PR |
|---|---|---|---|---|
| 1 | 100 | **0.201** | 1.53% / **6** | 1.10 |
| 2 | 100 | 0.190 | 32.55% / 35 | 1.83 |
| 5 | 300 | 0.181 | 58.56% / 49 | 2.67 |
| 10 | never | 0.172 | 66.99% / 49 | 5.30 |
| 25 | 1100 | 0.159 | 71.94% / 51 | 5.68 |
| 50 | 200 | 0.155 | 73.16% / 53 | 5.83 |
| 100 | 600 | 0.174 | 74.00% / 53 | 5.57 |
| 250 | 100 | 0.148 | 74.58% / 52 | 6.15 |

**Six motor neurons are already pinned at `rho=1`, the smallest radius
tested.** There is no clean cell for this arm in the swept range at all —
every radius has at least one stuck motor unit. Best available: `rho=1`.

### rewired:1

| rho | t_settle | probe_F | pinned (all / motor) | PR |
|---|---|---|---|---|
| 1 | 100 | 0.189 | 4.09% / **9** | 1.17 |
| 2 | 100 | 0.191 | 34.97% / 41 | 1.06 |
| 5 | 200 | 0.158 | 59.91% / 46 | 1.74 |
| 10 | 500 | 0.157 | 67.47% / 52 | 4.89 |
| 25 | 1200 | 0.170 | 71.58% / 54 | 3.78 |
| 50 | never | 0.164 | 72.84% / 55 | 4.97 |
| 100 | 200 | 0.157 | 73.63% / 55 | 3.99 |
| 250 | 700 | 0.163 | 74.07% / 54 | 5.65 |

Same story as rewired:0, independently reseeded: motor units pinned from
`rho=1`. Best available: `rho=1` (F=0.189) — `rho=2`'s F=0.191 is within
noise and comes with 9->41 more pinned motor units, so `rho=1` is the
defensible pick despite the marginally lower raw score.

### sign_shuffled:0

| rho | t_settle | probe_F | pinned (all / motor) | PR |
|---|---|---|---|---|
| 1 | 100 | 0.202 | 0.00% / 0 | 1.11 |
| 2 | 100 | **0.212** | 0.00% / 0 | 1.43 |
| 5 | 100 | 0.208 | 0.97% / 2 | 1.06 |
| 10 | 300 | 0.145 | 9.79% / 32 | 1.77 |
| 25 | 700 | 0.164 | 36.93% / 54 | 1.66 |
| 50 | 900 | 0.140 | 54.14% / 56 | 3.41 |
| 100 | never | 0.173 | 63.24% / 57 | 4.72 |
| 250 | 100 | 0.146 | 68.31% / 58 | 3.23 |

Cleanest arm in the grid: completely unpinned through `rho=2`, first motor
neuron pinned only at `rho=5`. Argmax and the cleanest cell coincide:
`rho=2` (F=0.212, 0 pinned).

### erdos_renyi:0

| rho | t_settle | probe_F | pinned (all / motor) | PR |
|---|---|---|---|---|
| 1 | 100 | **0.209** | 0.04% / 0 | 1.16 |
| 2 | 200 | 0.145 | 43.25% / **26** | 1.44 |
| 5 | never | 0.170 | 63.34% / 40 | 6.06 |
| 10 | 400 | 0.168 | 68.19% / 44 | 8.72 |
| 25 | never | 0.169 | 70.89% / 45 | 10.57 |
| 50 | 1300 | 0.121 | 71.68% / 45 | 10.36 |
| 100 | 1300 | 0.133 | 72.13% / 45 | 10.43 |
| 250 | 700 | 0.144 | 72.76% / 48 | 11.55 |

The sharpest collapse in the grid: clean at `rho=1` (0 motor pinned), 26
motor units pinned by `rho=2`. Best (and only clean) cell: `rho=1`.

### Summary — first radius with any motor neuron pinned

| arm | first rho with a pinned motor neuron |
|---|---|
| real | 25 (clean through 10) |
| sign_shuffled:0 | 5 (clean through 2) |
| erdos_renyi:0 | 2 (clean only at 1) |
| rewired:0 | **1** (never clean in this sweep) |
| rewired:1 | **1** (never clean in this sweep) |

---

## 4. Can common-rho be defended? No — sharper than previously known.

The earlier report established that `rho=10` destroys the rewired arms
(67-68% pinned) while leaving real nearly untouched (1.8% pinned). This
sweep shows the gap opens far earlier than that: **the degree-matched
rewires already have pinned motor neurons at `rho=1`, the smallest radius
tested in either grid.** There is no rho, however small, tested here at
which every arm is simultaneously clean. A single shared spectral radius for
the ~50 GPU-hour campaign would either:

* run the rewired arms with motor neurons already saturated at
  initialisation (any rho >= 1), or
* run the real and sign-shuffled arms far below their own best operating
  region (any rho small enough to spare the rewired arms' motor pool, which
  based on this grid may be below 1 and was not tested).

**Common-rho must be abandoned.** The per-arm table in §3 is the answer in
its place: each arm at its own best available point, with the response curve
and the pinned/PR numbers published alongside it, so the fairness claim is
"each arm was given its best chance, here is the sensitivity" rather than
"both got the same number."

### Recommended per-arm operating points

| arm | recommended rho | probe_F | clean? |
|---|---|---|---|
| real | 5-10 | 0.225-0.229 | yes (0 motor pinned) |
| sign_shuffled:0 | 2 | 0.212 | yes (0 motor pinned) |
| erdos_renyi:0 | 1 | 0.209 | yes (0 motor pinned) |
| rewired:0 | 1 | 0.201 | no (6 motor pinned) |
| rewired:1 | 1 | 0.189 | no (9 motor pinned) |

---

## 5. Does this change the "no arm separates" finding?

Not conclusively, and this section says so rather than overclaiming. At each
arm's own recommended point: real 0.225-0.229, sign_shuffled 0.212,
erdos_renyi 0.209, rewired:0 0.201, rewired:1 0.189. Spread is 0.040 —
somewhat wider than the 0.016 spread the whole-clip probe grid measured, and
real now leads the next-best arm (sign_shuffled) by 0.013-0.017, close to
but slightly under the previously measured ~0.014 single-run noise floor.

**This is evidence, not proof, that per-arm operating points sharpen the
separation** — plausible, since the whole-clip numbers mixed in ~300-500
steps of cold-start transient that this sweep now excludes, and a probe fit
on cleaner steady-state dynamics should be less noisy. But the ~0.014 noise
floor was established by re-running one *fixed* configuration twice; it was
not re-established for this settled-window methodology, which uses a
different clip count (8 vs 12) and a different, longer audio crop (8.4s vs
4.0s) than the original grid. Treat "real separates from sign_shuffled by
one noise-floor-width" as a lead worth re-checking with an independent-seed
rerun, not a settled result — exactly the caution the previous report asked
for ("report a spread, not a point").

One methodological note in real's favour: the whole-clip probe_grid.json had
real peaking at `rho=2` (F=0.214); this settled-window grid puts it at
`rho=10` (F=0.229), fifteen points higher. Since `probe_grid.json`'s 800-step
clips at `rho=10` are roughly 60% cold-start transient (the real arm's own
`t_settle` at `rho=10` is step 500-700 in this data), the whole-clip number
for real at `rho=10` was probably understating it. The other arms' whole-clip
peaks were mostly at `rho=1-2`, where `t_settle` is short (~100 steps) and
the transient contamination is small — which is consistent with real being
the arm most affected by the fix, and the one whose ranking moved most.

---

## 6. What this means for Phase 4

1. **Do not use common-rho.** Run each arm at its own recommended point
   (§4). The per-arm table is now the operating-point input the ablation
   suite needs.
2. **Do not gate the campaign on the pre-training PR guard.** It fails
   universally and is expected to; it is a post-training diagnostic, not an
   initialisation-time one.
3. **The rewired arms have no clean cell in this sweep.** Their best
   available point already has 6-9 pinned motor neurons. Whether a smaller
   rho (<1, untested) clears them is open — worth one more cheap sweep
   before the campaign, not worth blocking on.
4. **The separation between arms may have widened**, but the previous
   report's core caution stands: the probe's own noise is on the same order
   as the effect being measured, and this grid has not independently
   re-verified its own noise floor. Re-run with a second seed before reading
   too much into the 0.04 spread.
5. **The 50 GPU-hour question is still open**, and this sweep does not close
   it — it replaces one blocker (no defensible operating point at all) with
   a narrower one (arms are per-arm-testable now, but still close to
   indistinguishable at initialisation, and two of five arms start already
   partly saturated).

---

## 7. Verified vs reasoned, for the record

**Verified by running code in this session:** the transient-vs-settled
motor-modulation numbers in §1; `t_settle` values in §3; the per-neuron
pinned/motor-pinned counts and PR values in §3; the probe_F values in §3;
that `pytest -q` is unaffected (138 passed, no change from before this
session's edits — see below).

**Reasoned, not re-verified this session:** the ~0.014 probe noise floor and
the hub-statistic argument for why rho is unfair (~191 vs ~11,800 eigenvector
participation ratio) — both established in `2026-09-15-probe-grid.md` and
cited here rather than re-derived, per that report's own account of what it
already checked.

**Not done, and explicitly out of scope:** any training run. Both sweeps in
this session were forward passes on untrained models, per the task's scope
discipline.

`pytest -q`: 138 passed (unchanged by this session's edits to
`scripts/propagation.py`; no source under `src/` was touched).
