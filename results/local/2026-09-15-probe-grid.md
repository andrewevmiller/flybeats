# The probe grid, and one false alarm — 15 September 2026

What each Phase 4 arm can carry *before* training, at every operating point.
Forward passes only, ~20 minutes of GPU, run to find out whether the design is
testable before ~50 GPU-hours are spent finding out that it is not.

Evidence: `probe_grid.json` (30k, 5 arms x 8 radii) and
`probe_grid_motorfix.json` (the reverted experiment in §3).

**Outcome: the hypothesis is not falsified, but the current Phase 4 design
cannot test it.**

---

## 1. No arm separates from any other

Ridge probe from the wing motor pool to the smoothed onset targets, scored with
the repo's own peak picking and onset F, on held-out clips. Best cell per arm
over radii {1, 2, 5, 10, 25, 50, 100, 250}:

| arm | best probe F | at rho |
|---|---|---|
| **baseline — encoder drive, no recurrence** | **0.218** | — |
| rewired:0 | 0.216 | 1 |
| rewired:1 | 0.215 | 2 |
| erdos_renyi | 0.215 | 1 |
| real | 0.214 | 2 |
| sign_shuffled | 0.200 | 1 |

The whole spread across five arms and eight operating points is ~0.016, and the
best cell in the grid is the one with no connectome in it. The recurrence does
not beat its own input.

**What this does not say.** The probe bounds a *frozen* core under a *linear*
readout. Trained models in README reach ~0.30 against this 0.218 baseline, so
training adds something the probe cannot see. This is not evidence the project
fails. It is evidence that these arms are indistinguishable at initialisation,
which is what the campaign would be trying to separate.

### The probe's own noise floor

The baseline row measured **0.218** in the first grid and **0.204** in the
second. The baseline does not depend on topology at all — it is a ridge from
the encoder drive — so that 0.014 is reproducibility noise from model init and
node ordering. **It is larger than every between-arm difference in the table
above.** Any future use of this probe should report a spread, not a point.

## 2. rho = 10 is not a fair comparison point

At the shipped value, measured as the fraction of neurons pinned at the state
clip for more than half the window:

| arm | pinned at rho=10 |
|---|---|
| real | **1.8%** |
| rewired:0 / rewired:1 | **67.0% / 67.4%** |
| erdos_renyi | 68.1% |
| sign_shuffled | 9.8% |

That is not a comparison; one arm is destroyed before training starts. The
earlier reading of this — that rho is a hub statistic, carried by ~191 neurons
in the real graph and ~11,800 in a rewired draw — is confirmed as behaviour,
not just as an eigenvector argument.

**No radius passes both guards, for any arm.** Below rho ~10 the motor
trajectory collapses to ~1 effective mode against 8 drum classes; above it the
network saturates. There is no window where both hold, so "pick the smallest
radius that passes" has no answer at this tier.

`scripts/propagation.py`'s recommendation should be treated as void until it is
re-run: it chose rho=10 on a cold-start transient, on one arm, at the 10k tier,
and the value was then written into the 30k config.

## 3. A false alarm, and the experiment it caused

**Retracted claim.** An earlier pass reported that the wing motor pool was
starved — 19 of 66 MNs with zero in-degree, median in-degree 1, and the pool
receiving ~1/158th of a typical neuron's input gain.

**All of that was an indexing error.** `ConnectomeRNN.edge_index` is stored
`[post, pre]` — the matrix convention, and `model.py` says so in a comment —
while the constructor argument uses `[pre, post]`. Indexing `[1]` as the
destination measures how much each neuron *sends*. Motor neurons are terminal,
so low out-degree is exactly correct and unremarkable.

Measured on the right axis:

| | claimed | actual |
|---|---|---|
| MNs with zero in-degree | 19 / 66 | **0 / 66** |
| median motor in-degree | 1 | **207** |
| motor input gain vs all-neuron median | 1/158th | **6.6x more** |

The readout is well connected and, if anything, over-driven relative to a
typical neuron. The independent measurement of 2.101 median input gain at
rho=10 was correct; this one was not.

### The change it caused, and why it was reverted

Before the error was found, `_trim` was given an opt-in
`protect_motor_afferents` parameter: protect the motor pool's one-hop afferent
set from the hub ranking, on the theory that the trim was keeping the targets
but not their drivers. It did what it claimed — all 4,895 motor feeders kept
instead of 2,320, median motor in-degree 207 -> 340, at a cost of ~4,900 of the
30,000 node budget.

**It made the probe worse at every operating point:**

| arm | orig subgraph | +motor afferents |
|---|---|---|
| real (best) | 0.214 | **0.172** |
| rewired:0 (best) | 0.216 | **0.159** |

A 0.042 drop on the real arm against a ~0.014 noise floor. Reverted: the knob
was justified by a measurement error and it does not help. Recorded here rather
than silently dropped, because "we checked, and the readout is well connected,
and forcing more afferents in costs 16% of the budget and loses 0.04 F" is
worth knowing before someone proposes it again.

## 4. What this means for the campaign

1. **Do not spend the 50 GPU-hours yet.** Arms that are indistinguishable at
   initialisation, compared at an operating point that saturates 67% of one of
   them, under guards nothing passes, produce an uninterpretable table.
2. **The operating point has to be settled first**, and per-arm rather than
   common-rho, because no single scalar aligns two differently-shaped gain
   distributions.
3. **Report a spread on everything.** The probe's own noise exceeds the effect
   being looked for; the trained comparison has no reason to be kinder.
4. The probe is cheap enough to re-run after any of these changes. That is what
   it is for.
