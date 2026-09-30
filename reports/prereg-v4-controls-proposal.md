# Proposal: prereg-v4 random controls

Status: **accepted on 30 Sep 2026 as method A (strict), with a correction.** `locked.yaml` is prereg-v4 and the
CHANGELOG has its line. Read "Correction and decision" first: parts of the proposal below are superseded.

## Correction and decision (30 Sep 2026)

**Retracted:** "C: 479–515 largest eigenvalue", in the table and draft CHANGELOG line below. The prototype swapped
reciprocal pairs as units, carrying both directions' synapse counts to the new pair. That moves synapse counts
between sending neurons: 58% of neurons' output totals changed (median 5%, up to 478%). That breaks a property locked
since prereg-v1, and it is what kept the loops strong.

With each sender keeping its own counts, the method is **A (strict)**: seeds 1–2 gave the following.

| | Largest eigenvalue | Ear → readout | Median hops to readout (min) | Ear signal at readout after 2 / 3 / 4 steps | Real connections kept |
|---|---|---|---|---|---|
| Real | 731 | 0 | 3 (2) | 0.37 / 4.87 / 2.65 | 100% |
| A: strict (locked) | 265–295 | 0 | 3 (2) | 1.6–1.9 / 17–20 / 27–40 | about 7% |

Two other options were considered and rejected:
- **B, weight-matched pairs** (eigenvalue about 523): 12% of real connections survive and half of the reciprocal
  pairs stay in place, against the locked "under 10%" gate.
- **C, pairs as units** (eigenvalue 502–522): breaks exact output totals.

**Why A:**
- It keeps every earlier commitment.
- Its one addition, no ear-to-motor connections, is justified by anatomy whatever the results.
- B and C match loop strength by copying the heavy reciprocal circuits into the controls. Those circuits are part of
  the wiring the experiment tests, so copying them would bias it towards the real network. The choice was also made
  after seeing control 1 win, so only a correction that is right regardless of outcome is safe.

**Reported, not removed:** with each network scaled to its own critical gain, the real network's connections start
about 2.5× weaker than the controls'.

Written 30 Sep 2026. All numbers come from the slice and control files (CPU only, read-only), and from the prototype
scripts described under "How this was measured".

## The problem with the prereg-v3 controls

The controls are degree-preserving swaps across the whole slice (`random_controls.method: degree_preserving_swaps`).
Each neuron keeps its number of inputs and outputs, but its partners are drawn from the whole slice. Two properties of
the real wiring do not survive this, and both make the controls easier to train than the real network, for reasons
that have nothing to do with how the connectome computes.

**1. Shortcuts from sound to output.** In the real fly, sound reaches a motor neuron through 2–3 layers of
interneurons. The swaps send some ear outputs straight to motor neurons:

| | Real | Controls 1–5 |
|---|---|---|
| Direct connections from an ear to a readout motor neuron | 0 | 561–614 (6,400–7,700 synapses) |
| Readout motor neurons with a direct ear input | 0 of 182 | 125–132 of 182 |
| Median hops from an ear to a readout motor neuron | 3 | 1 |

**2. Loop strength.** 11.4% of real connections are reciprocal (a→b and b→a), against 1.6% in control 1. The
reciprocal pairs make the real network's recurrent loops much stronger: largest eigenvalue 731, against 262–277 in the
controls. The real network with only its reciprocal pairs still has 526; without them it has 277. The model scales
each network so its largest eigenvalue sits at the critical gain. So the real network's connections all start about
2.7× weaker than the controls', and a signal that already has further to travel is scaled down more.

**The effect, already visible:** memorising the same 8 clips over the same number of steps, control 1 is ahead on
every drum. At step 1,500 its F1 was kick 0.78, snare 0.47, closed hi-hat 0.67, against the real network's 0.53,
0.31 and 0.48. In the main comparison (real against 5 controls), the controls would likely win because of the
shortcuts, not because of anything about the wiring.

## Proposed method

**Degree-, class- and reciprocity-preserving swaps.**

1. **Classes.** Every slice neuron gets one class:
   - **ear**: the `^JO-[AB]` neurons (150);
   - **interneuron, layer 1**: neurons 1 hop from an ear in the real slice (763);
   - **interneuron, layer 2+**: neurons 2 or more hops from an ear in the real slice (2,200);
   - **motor**: every motor neuron (189), readout or not.

   The layers come from the real slice, measured once, and are fixed for every control.
2. **One-way connections** (a→b with no b→a) are swapped as now, a→b, c→d into a→d, c→b, but only when a and c have
   the same class and b and d have the same class. A swap is rejected if it would create:
   - a self-connection;
   - a duplicate connection;
   - a new reciprocal pair.
3. **Reciprocal pairs** (a↔b) are swapped only with other reciprocal pairs, as pairs: {a↔b, c↔d} into {a↔d, c↔b},
   with the same class conditions and the same rejections.
4. **Everything else stays as locked:**
   - each connection keeps its synapse count, and its sign travels with it;
   - seeds 1–5;
   - 10 proposed swaps per connection;
   - the rest of the model and training, unchanged.

**What it keeps:**
- every neuron's input and output counts;
- every neuron's input and output counts per class, so ears still never connect directly to motor neurons;
- the number of reciprocal pairs.

**What it still shuffles:** which particular neurons connect. About 93% of real connections are gone in each control.

## Checks each control must pass (added to `check_control`)

Each check failing stops the build.

- Input and output counts per neuron equal the real slice's, as now.
- Output synapses per neuron equal the real slice's, as now.
- Connection counts per (sender class, receiver class) equal the real slice's, per neuron.
- No direct connection from an ear to any motor neuron.
- The same number of reciprocal pairs as the real slice.
- No duplicates or new self-connections, as now.
- Every drum's readout neurons reachable from the ears, as now.

Reported, not pass/fail: the largest eigenvalue, the share of real connections that survive, and hop distances from the
ears to each drum's readout neurons.

## What the prototype gave

Real slice, the current control 1, and three candidate methods, with seeds 1 and 2 for each candidate:

| | Largest eigenvalue | Ear → readout connections | Median hops to readout (min) | Ear signal at readout after 2 / 3 / 4 steps | Real connections kept |
|---|---|---|---|---|---|
| Real | 731 | 0 | 3 (2) | 0.37 / 4.87 / 2.65 | 100% |
| Current control 1 | 276 | 578 | 1 (1) | 19.5 / 18.1 / 23.6 | 3% |
| A: ear / interneuron / motor | 230–242 | 0 | 3 (2) | 2.4–2.9 / 77–83 / 89–94 | 5% |
| B: A with interneurons split by layer | 250–252 | 0 | 3 (2) | 2.2–2.3 / 21–30 / 46–47 | 7% |
| **C: B plus reciprocity (proposed)** | **479–515** | **0** | **3 (2)** | **0.55–0.59 / 2.8–3.1 / 2.4–3.8** | **7%** |

The "ear signal" is a unit input on every ear, passed 2–4 steps through the signed synapse matrix scaled by its own
largest eigenvalue, as the model scales it. The column sums the absolute signal over the 182 readout motor neurons.

- **A and B** remove the shortcuts, but the weak loops remain. After scaling, sound still reaches the readout 5–35×
  more strongly than in the real network.
- **C** matches the real network on everything measured except a smaller gap in loop strength.

## What this does not fix

- **The remaining loop gap:** 479–515 against 731, so the real network's connections still start about 1.5× weaker.
  Some of the real network's loop strength sits in a small circuit around wing motor neurons. Twenty neurons hold 47%
  of its leading eigenvector, several of them readout motor neurons. That circuit is a genuine feature of the wiring,
  so it is fair for the experiment to test it. Report it with the results, don't remove it.
- **The real network's failure to memorise.** At step 3,000 its F1 was kick 0.56, snare 0.38, closed hi-hat 0.53.
  The diagnosis of 30 Sep found separate problems:
  - a fixed learning rate where training uses a cosine schedule;
  - readout scaling fitted once, to activity that later tripled;
  - thin readouts for snare and closed hi-hat.

  These need their own decisions. New controls change the opponents, not the real network.

## Alternatives considered

- **Reject only swaps that create an ear→motor connection.** This is the smallest change and removes the shortcuts,
  but it leaves the rest of the layering and all of the loop gap.
- **Scale every network by the real network's gain instead of each by its own.** This would equalise per-synapse
  weights, but the controls would start far below critical (about 0.35×), which would give them a different problem.
- **Keep the whole type-to-type structure** (a stochastic block model by cell type). This keeps so much of the real
  wiring that the controls stop being meaningfully random. It would test only fine detail within each type pair.

## If accepted: the work, in order

1. **`locked.yaml`:** set `random_controls.method` to `degree_class_reciprocity_preserving_swaps`, with the class
   definition above. New tag `prereg-v4`, and a CHANGELOG line before any run that uses it.
2. **Code:** `connectome/controls.py` gets the new swaps and checks. `build_controls.py` rebuilds seeds 1–5 (CPU, a
   few minutes each).
3. **Tests:** degree, class-block and reciprocity counts preserved; no ear→motor connections.
4. **Memorisation for the controls** must be rerun on the new controls. The real network's run is unaffected.
5. **The CUDA-graph check** is unaffected: it uses the real network. `memorise.py` and `train.py` pick up the new
   control files as before.

### Draft CHANGELOG line

> 2026-09-30 - prereg-v4: random controls keep each neuron's connection counts per class (ears; interneurons 1 hop
> from an ear; interneurons 2+ hops; motor neurons, all from the real slice) and the number of reciprocal pairs, before
> any real training run. The prereg-v3 swaps gave every control 561–614 direct ear→motor-neuron connections, which the
> real slice does not have (0). They also cut reciprocal connections from 11.4% to 1.6%, lowering the largest
> eigenvalue from 731 to about 270. Both made the controls easier to train for reasons unrelated to the wiring.
> Control 1's memorisation run showed it: F1 at step 1,500 was kick 0.78 against 0.53. With the new swaps, the
> controls have no ear→motor connections, the same 2–3 hop layering as the real slice, and eigenvalues of 479–515.
> About 93% of real connections are still replaced.

## How this was measured

Scratch scripts, not part of the repo. They would become `controls.py` and a test if accepted.

- `wiring_compare.py`: hop distances, ear→readout reach, signal propagation and eigenvalues for the real slice and
  control 1.
- `class_controls.py`: schemes A and B, seeds 1–2.
- `class_recip_controls.py`: scheme C, seeds 1–2. It checks that input and output counts per neuron are kept, and
  that reciprocity stays at 11.4%.
