# The probe was scoring different audio every run — 15 September 2026

Follows `2026-09-15-operating-point.md`, which fixed `propagation.py`'s
transient bug and published per-arm operating points. This ran the cheap
follow-ups that note left open — sub-unit radii for the rewired arms, and the
noise floor it declined to re-establish — and the second one turned up a bug
that invalidates the probe_F column of every grid this project has produced.

Evidence: `propagation_rep_cs0/1/2.json` (3 independent clip draws, 5 arms x 6
radii, 30k tier) and `propagation_subunit.json` (the sub-unit sweep).

**Outcome: `probe_F` was never reproducible. Both probe harnesses drew their
clips off torch's *unseeded* global generator, so every invocation scored a
different random crop of the corpus, with the onset targets moving too. The
same cell re-run four times spans 0.069 — five times the 0.014 "noise floor"
the 15 Sep probe-grid note attributed to model init and node ordering. Seeded,
two runs are now bit-identical. With the audio controlled and three
independent draws, no arm separates from any other, the ranking reverses
between draws, and the real arm does not lead. Two claims in the previous note
are retracted (§2); the pinned/PR/`t_settle` findings survive intact.**

---

## 1. The bug

`src/dataset.py:287` takes the training split's crop offset from
`torch.randint(0, span, (1,))` — torch's **global** generator. That is
deliberate and documented: `build_loaders` seeds the DataLoader's per-worker
generators from `train.seed`, so a training run sees reproducible windows
without freezing them.

`scripts/propagation.py` and `scripts/probe_grid.py` never go through the
DataLoader. They index `loader.dataset[i]` directly from the main process,
where torch's global RNG is seeded from entropy at import. The guarantee does
not reach them.

Measured — the same draw code, three separate processes:

| process | `torch.initial_seed()` | rows picked | wav sha1 | target sum |
|---|---|---|---|---|
| 1 | 449929960991100 | 0,120,241,... | `4bb23fb2f2d56c34` | 3867.26 |
| 2 | 449936365686900 | 0,120,241,... | `c20ddc6f4bda7868` | 4044.44 |
| 3 | 449942455109800 | 0,120,241,... | `51f0e5de345516d3` | 3874.56 |

Same eight rows every time, different audio every time — and the targets move
with it, so the probe was being re-fit and re-scored against a different
labelling of a different eight-and-a-bit seconds on each run.

### What it did to the numbers

`rewired:1` at `rho=1.0`, the identical command, four times:

| run | probe_F |
|---|---|
| the committed grid | 0.189 |
| sub-unit sweep | 0.258 |
| repro run 1 | 0.198 |
| repro run 2 | 0.248 |

Mean 0.223, **sd 0.035, range 0.069**. Meanwhile the *dynamics* columns of
those same runs agree to three digits (pinned 0.0409 vs 0.0408, PR 1.164 vs
1.170, `motor_settled` 0.0294 vs 0.0299). That split is the diagnostic: a
statistic averaged over 30,000 neurons and 1,500 steps is insensitive to which
8 seconds it saw; a probe fit on four clips and scored on four more is not.

**Ruled out along the way:** GPU nondeterminism. cuSPARSE spmm uses atomics and
the model runs a 1,600-step recurrence, so chaotic amplification was the
obvious suspect and would have been much worse news. It is not the cause —
with the crop seeded, two separate processes return `F=0.2158` and `F=0.2158`,
with `pinned`, `PR` and `motor_settled` identical to every digit printed. The
forward pass was deterministic all along.

### The fix

Seed the crop immediately before the draw, in both scripts, defaulting to
`train.seed`. `propagation.py` gains `--clip-seed` to vary it deliberately:
that is now the *only* way the audio changes, which makes a replicate a
replicate. `--seed` (model init) was added at the same time and turns out to
matter far less — `calibrate_encoder` pins its own seed from the config and
restores RNG state around itself, so the encoder affine is identical across
init seeds by construction.

---

## 2. What this retracts

The previous two notes' **probe_F numbers are all single uncontrolled draws**,
including `probe_grid.json`'s headline table and its 0.218 no-recurrence
baseline. The 0.014 figure that note called a noise floor and attributed to
"model init and node ordering" was this bug, and it understated it.

Two specific claims from `2026-09-15-operating-point.md` are **withdrawn**:

| claim | status |
|---|---|
| "the degree-matched rewires already have pinned motor neurons at `rho=1`, the smallest radius tested... **there is no rho, however small, tested here at which every arm is simultaneously clean**" | **Wrong, and the sentence says why in its own clause** — nothing below 1 had been tested. Both rewires are completely clean at `rho<=0.5`, as is every other arm (§4). |
| "real now leads the next-best arm by 0.013-0.017", and the §5 reading that per-arm points sharpen the separation | **Not supported.** With controlled audio the lead vanishes and the ranking reverses between draws (§5). |

**What survives unchanged:** everything that does not go through `probe_f`.
The `t_settle` machinery, the per-neuron pinned fractions, the motor
participation ratios, and the finding that no untrained arm reaches PR>=8 at
any radius are all reproducible to the digits printed — the pinned fractions
in this session's grid match the committed grid's to within 0.01 percentage
points at every shared cell. The transient bug was real and its fix is sound.
The operating-point recommendations rest on the pinned-fraction guard, which
is the draw-invariant half of the tool.

---

## 3. The replicated grid

5 arms x 6 radii x **3 independent clip draws**, 30k tier, 8 clips, 1,600
steps, settled-window statistics. `F` is the mean over draws; `range` is
max-min across them. `pinned`/`motor`/`PR` are draw-invariant to the digits
shown, so one value is quoted. Bold marks each arm's best **clean** cell.

| arm | rho | F (mean) | range | pinned | motor pinned | PR |
|---|---|---|---|---|---|---|
| **real** | 0.25 | 0.191 | 0.011 | 0.00% | 0 | 1.00 |
| | 0.5 | 0.193 | 0.027 | 0.00% | 0 | 1.00 |
| | 1 | 0.194 | 0.015 | 0.03% | 0 | 1.05 |
| | 2 | 0.208 | 0.007 | 0.34% | 0 | 1.00 |
| | **5** | **0.217** | **0.005** | 0.64% | **0** | 1.02 |
| | 10 | 0.210 | 0.063 | 1.79% | 0 | 1.18 |
| **rewired:0** | 0.25 | 0.204 | 0.031 | 0.00% | 0 | 1.03 |
| | **0.5** | **0.215** | 0.014 | 0.00% | **0** | 1.09 |
| | 1 | 0.206 | 0.030 | 1.53% | 6 | 1.10 |
| | 2 | 0.215 | 0.021 | 32.55% | 35 | 1.79 |
| | 5 | 0.192 | 0.033 | 58.56% | 49 | 2.73 |
| | 10 | 0.162 | 0.037 | 67.00% | 49 | 4.45 |
| **rewired:1** | 0.25 | 0.204 | 0.021 | 0.00% | 0 | 1.25 |
| | **0.5** | **0.223** | 0.055 | 0.00% | **0** | 1.16 |
| | 1 | 0.215 | 0.013 | 4.09% | 9 | 1.17 |
| | 2 | 0.211 | 0.044 | 34.97% | 41 | 1.06 |
| | 5 | 0.189 | 0.057 | 59.91% | 46 | 1.77 |
| | 10 | 0.151 | 0.009 | 67.47% | 52 | 4.85 |
| **sign_shuffled:0** | 0.25 | 0.181 | 0.100 | 0.00% | 0 | 1.00 |
| | 0.5 | 0.197 | 0.055 | 0.00% | 0 | 1.00 |
| | 1 | 0.200 | 0.028 | 0.00% | 0 | 1.11 |
| | **2** | **0.206** | 0.024 | 0.00% | **0** | 1.43 |
| | 5 | 0.212 | 0.036 | 0.97% | 2 | 1.07 |
| | 10 | 0.188 | 0.009 | 9.79% | 32 | 1.76 |
| **erdos_renyi:0** | 0.25 | 0.222 | 0.049 | 0.00% | 0 | 1.19 |
| | **0.5** | **0.248** | 0.121 | 0.00% | **0** | 1.19 |
| | 1 | 0.219 | 0.054 | 0.04% | 0 | 1.15 |
| | 2 | 0.177 | 0.062 | 43.26% | 26 | 1.43 |
| | 5 | 0.174 | 0.044 | 63.34% | 40 | 5.23 |
| | 10 | 0.146 | 0.014 | 68.13% | 43 | 8.91 |

Median per-cell sd of probe_F across draws: **0.0163** (30 cells). The largest
single-cell range is 0.121 — at the cell with the highest mean.

The sub-unit radii are the new ground here, and they are uniformly clean: not
one motor neuron pinned in any arm at `rho<=0.5`. The previous note's
"the rewired arms have no clean cell in this sweep" is answered — they have
one, it was just below the floor of the sweep.

---

## 4. Can common-rho be defended? Yes on cleanliness, no on fairness.

The previous note's argument — that no common radius leaves every arm clean —
was an artefact of never sweeping below 1. **Two tested radii leave all five
arms completely clean**, counting motor neurons pinned more than half the
settled window, of 66:

| rho | real | rewired:0 | rewired:1 | sign_shuffled:0 | erdos_renyi:0 |
|---|---|---|---|---|---|
| **0.25** | 0 | 0 | 0 | 0 | 0 |
| **0.5** | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 6 | 9 | 0 | 0 |
| 2 | 0 | 35 | 41 | 0 | 26 |
| 5 | 0 | 49 | 46 | 2 | 40 |
| 10 | 0 | 49 | 53 | 32 | 43 |

So common-rho at 0.5 is *available*. It is still the wrong choice, for a
reason the numbers now state precisely: **`rho=0.5` is the best cell for all
three null arms, and is 0.024 below the real arm's own best.**

| arm | own best rho | F there | F at rho=0.5 | cost of the common point |
|---|---|---|---|---|
| real | 5 | 0.217 | 0.193 | **+0.024** (SE 0.010) |
| sign_shuffled:0 | 2 | 0.206 | 0.197 | +0.009 (SE 0.023) |
| rewired:0 | 0.5 | 0.215 | 0.215 | 0 |
| rewired:1 | 0.5 | 0.223 | 0.223 | 0 |
| erdos_renyi:0 | 0.5 | 0.248 | 0.248 | 0 |

A single shared radius that happens to sit at every null's optimum and below
the real arm's is a handicap applied to exactly one arm — the one the
hypothesis is about. The real arm's preference for its own point is also the
one within-arm comparison in this grid that clears its own noise (+0.024
against SE 0.010, paired across draws); every other arm is flat in rho.

**Recommendation: per-arm operating points, chosen by the pinned-fraction
guard** (draw-invariant) **rather than by probe_F** (which cannot resolve
them): real `rho=5`, sign_shuffled `rho=2`, erdos_renyi `rho=1`, both rewires
`rho=0.5`. Publish the §3 curve alongside, so the fairness claim is "each arm
at its own best clean point, and here is the whole response surface."

---

## 5. No arm separates — and real does not lead

probe_F at each arm's own best clean cell, per draw. Because all arms see the
same audio within a draw, the comparison can be paired, which cancels the
dominant noise term:

| arm | rho | cs0 | cs1 | cs2 | mean | median | sd |
|---|---|---|---|---|---|---|---|
| real | 5 | 0.220 | 0.215 | 0.216 | 0.217 | 0.216 | **0.003** |
| rewired:0 | 0.5 | 0.212 | 0.224 | 0.210 | 0.215 | 0.212 | 0.008 |
| rewired:1 | 0.5 | 0.206 | 0.258 | 0.203 | 0.223 | 0.206 | 0.031 |
| sign_shuffled:0 | 2 | 0.206 | 0.194 | 0.218 | 0.206 | 0.206 | 0.012 |
| erdos_renyi:0 | 0.5 | 0.230 | 0.318 | 0.196 | 0.248 | 0.230 | 0.063 |

Paired differences, real minus each null:

| contrast | mean | per-draw | SE |
|---|---|---|---|
| real - rewired:0 | +0.001 | +0.007, -0.009, +0.006 | 0.005 |
| real - rewired:1 | -0.006 | +0.014, -0.044, +0.012 | 0.019 |
| real - sign_shuffled:0 | +0.011 | +0.014, +0.020, -0.002 | 0.007 |
| real - erdos_renyi:0 | **-0.031** | -0.010, -0.103, +0.020 | 0.037 |

And the ranking, within each draw:

* **cs0:** erdos_renyi > **real** > rewired:0 > rewired:1 > sign_shuffled
* **cs1:** erdos_renyi > rewired:1 > rewired:0 > **real** > sign_shuffled
* **cs2:** sign_shuffled > **real** > rewired:0 > rewired:1 > erdos_renyi

`erdos_renyi` — the null that matches nothing, included precisely because it
*should* lose — places first, first, and last. Its lead over real is the
largest gap in the table and is carried by one draw (0.318 at cs1); its sd of
0.063 is twenty times real's. Nothing here is resolved. The honest summary is
that the untrained probe cannot tell any of these five networks apart, and the
one direction it weakly points is not the hypothesised one.

The real arm's redeeming feature in this table is stability, not score: at
`rho=5` it is the most reproducible cell in the grid (sd 0.003 against a
median of 0.016). That is worth noting and is not evidence for the hypothesis.

---

## 6. What this means for Phase 4

1. **The operating point is settled, on the guard that reproduces.** Per-arm
   radii in §4. That deliverable stands.
2. **Do not use probe_F to rank arms.** At 8 clips its per-cell sd is 0.016
   and its worst cell is 0.121 wide, against a between-arm spread of 0.042.
   Any future use needs many more clips, several draws, and paired reporting.
3. **The pre-training evidence for the hypothesis has got weaker, not
   stronger.** The previous note's "real leads by one noise-floor width" was
   an artefact of uncontrolled audio. Corrected, real is mid-table and the
   unstructured null is nominally ahead.
4. **This still does not falsify the project.** The probe bounds a *frozen*
   core under a linear readout; README's trained models reach ~0.30 against
   this ~0.20 ceiling, so training supplies something the probe cannot see.
   What it does say is that the ~50 GPU-hour campaign would be started without
   any pre-training signal that the arms differ, at operating points that are
   defensible but at which every arm scores the same.
5. **The cheap thing worth doing before the campaign** is re-running this grid
   at a much larger clip count, to see whether the arms separate once the
   probe's own variance is beaten down. Note that `propagation.py` keeps every
   neuron's trace to compute the hop statistics — 8 clips x 1,600 steps x 30k
   neurons is already ~1.5 GB — so a high-clip run wants `probe_grid.py`,
   which keeps only the motor columns, not this script.

---

## 7. Verified vs reasoned

**Verified by running code in this session:** the three-process clip hashes in
§1; the four-run spread and the two bit-identical seeded runs in §1; every
number in the §3, §4 and §5 tables (3 full grid runs plus a 15-cell sub-unit
sweep, ~50 min of GPU); that `pytest -q` is **138 passed** both before and
after the edits.

**Reasoned, not re-verified:** that the encoder affine is identical across
`--seed` values — read from `calibrate_encoder`'s seed-and-restore block
rather than measured. The hub-statistic argument for why rho is unfair
(~191 vs ~11,800 eigenvector participation ratio) is still carried over from
`2026-09-15-probe-grid.md` and was not re-derived.

**Tried and abandoned:** an initial `--clip-seed` design that also resampled
*which* corpus rows were drawn, via `rng.choice` instead of the `np.linspace`
spread. Replaced with crop-only variation, so replicates stay matched on
drummer and session and the only thing moving is the window — otherwise a
replicate confounds "different audio" with "different drummers."

**Not done, and out of scope:** any training run. Everything here is forward
passes on untrained models.
