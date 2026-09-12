# FlyBeats / FlyDrums

Connectome-constrained drum performance from audio. A sparse recurrent network
whose **topology is fixed by the real MaleCNS v1.0 wiring diagram** and whose
**per-connection gains are learned**. Gradient descent can change what a
connection is worth; it cannot create a connection the animal does not have.

See [PLAN.md](PLAN.md) for the design this implements.

> **Status.** The pipeline is complete and runs end to end: Phase 0 gate through
> Phase 5 streaming playback, with the full Phase 4 ablation harness. What has
> **not** happened is a trained run at a scale that would let anyone answer the
> project's central question. Read [Results](#results-and-what-they-are-not)
> before quoting a number from this repo.

---

## Quick start

```bash
pip install -r requirements.txt

python scripts/fetch_data.py          # MaleCNS v1.0 flat connectome (~1.1 GB)
python scripts/verify_types.py        # Phase 0 gate -> data/verified_types.json
python scripts/fetch_egmd.py          # drum corpus (~5.4 GB)

python src/subgraph.py                # Phase 1 -> data/cache/subgraph.npz
python src/train.py   --config configs/v1_8piece.yaml
python src/ablations.py --config configs/v1_8piece.yaml --lesion

# size a subgraph against the latency budget before training it
python src/realtime.py --config configs/v1_8piece.yaml --benchmark
```

---

## Phase 0: what the gate actually found

PLAN.md insists on confirming cell-type strings against the dataset before any
code assumes them. That gate earned its place — **almost every name in the plan
differs from the real v1.0 type string**, and two of them collide with unrelated
neurons.

`neuprint.janelia.org` is blocked from this environment, so the gate reads the
public `body-annotations` feather instead: same v1.0 release, CC-BY, and the
table the connectome weights are keyed against. `scripts/verify_types.py --neuprint` cross-checks the resolved
types against the live server if you have a token — though that path could not
be exercised here, so treat it as unverified until it has run somewhere.

| PLAN.md said | v1.0 reality |
|---|---|
| `JO-A`, `JO-B`, `JO-E` | Zone prefixes, not types. Real: `JO-A1..A4`, `JO-B1_a..B4_b`, `JO-ED*`/`JO-EV*`, plus `*-unclear` truncation fragments |
| AMMC `B1` interneuron | **No type, alias or synonym in v1.0.** AMMC interneurons are numbered `AMMC001`–`AMMC038`. Left deliberately unconfirmed rather than mapped onto a guess |
| `aPN1` | Synonym only → `SAD051_a/b`, `CB1078`, `CB1542` |
| `vPN1` | Synonym only → `AVLP761m`, `AVLP762m`, `AVLP763m`. `VP1m+VP2_lvPN1` matches the string but is an **olfactory** projection neuron — a pure name collision, excluded explicitly |
| `pC2` | Not a type either; maps via synonyms onto 30 `AVLP`/`LAL`/`SIP`/`VES`/`PVLP` "m" types |
| `pIP10` | Confirmed exactly. One bilateral pair |
| `iii1`–`iii4` | **`iii2` and `iii4` do not exist.** Only `iii1`, `iii3` |
| (not listed) | **`ps2` does exist** and the plan omitted it |
| `OA-VPM1`, `OA-VPM2` | Synonyms of `DNg34` / `DNg104`, not types. Real: `OA-VUMa1..a8`, `OA-VPM3`, `OA-VPM4` |

Type names also carry literal spaces and commas — `b1 MN`, `DLMn a, b` — which
is exactly the kind of detail that silently produces an empty population.

**Open question 1 — does the confirmed wing-MN count support an 8-piece kit
cleanly?** Yes, with room to spare. 15 confirmed steering types as bilateral
pairs (30 bodies), and 25 types / 66 bodies in the dataset's own `wm` (wing
motor) subclass. Leg MNs are not needed until the 20–30 piece tier.

---

## The graph

The raw weights table is segment-to-segment over the whole segmentation:
**151.8M rows spanning ~87M post-synaptic fragments**, most of them unannotated
debris. Reduced to the neuron-level graph — both endpoints annotated, traced and
typed — it is **162,517 neurons, 25.1M edges, 122.2M synapses**.

Signs come from the presynaptic neuron's transmitter and are frozen (ACh `+1`;
GABA, glutamate and histamine `−1` — GluCl-α makes glutamate inhibitory in the
fly, unlike vertebrates). Only 2,202 of 162k neurons needed the excitatory
default, and those are flagged so the sign-shuffle ablation can price the guess.

**A finding worth recording:** 3 hops forward from the JO afferents already
reaches 154,853 of 162,517 neurons, and 3 hops backward from the wing motor pool
reaches 130,934 — an intersection of 128,433. *k*-hop seeding barely constrains
anything in a brain this small-world. What actually selects the circuit is the
pathway-strength trim, not the hop count. Anyone reading "k-hop subgraph" as a
meaningful anatomical restriction here would be mistaken.

---

## Results, and what they are not

**No trained result in this repo answers the project's central claim.** The
environment has no GPU, and the arms have only been run to prove the harness
works. The ablation table below is real output from a real run of all five arms
— and it is **1 epoch on 8 synthetic clips**, which is worth nothing as evidence.
It is shown to demonstrate the harness, not the hypothesis.

```
arm                   params   onset F   groove   beat ms   dev ms
------------------------------------------------------------------
real                 725,272    0.1785    0.428      29.8    -20.8
rewired              725,272    0.5426    0.402      36.6      4.8
sign_shuffled        725,272    0.1561    0.333      29.2    -28.3
gru                  724,379    0.2521    0.299      32.6    -13.0
shortcut              79,499    0.2683    0.446      39.8    -22.0
```

At one epoch the simplest arms lead, which is what one epoch measures. Drawing
"the rewired graph beats the connectome" from this would be wrong.

Running the suite properly needs a GPU and the real corpus. `--seeds` matters
here: `rewired` and `sign_shuffled` each draw *one* random topology, so a single
run cannot tell "random topologies do worse" from "this draw was unlucky". A gap
smaller than the spread across seeds is not a result.

```bash
python src/ablations.py --config configs/v1_8piece.yaml --lesion --epochs 40 --seeds 5
```

**Open question 3 — does the rate model converge before anyone builds the
spiking version?** Yes. On the 3-piece sanity task (10k-neuron subgraph,
synthetic clips, 12 epochs) training loss falls monotonically from 1.399 to
1.121 and groove similarity rises from 0.395 to 0.441
(`runs/sanity_3piece_run/history.json`). The surrogate-gradient spiking version
is unblocked; it is still unwritten.

That run is also where the threshold problem surfaced. Onset F at a *fixed* 0.3
threshold swung between 0.41 and 0.65 across those same epochs while the loss
fell smoothly — the metric was tracking output
scale, not timing. Hence the per-model threshold sweep. Treat this as a
convergence check, not a performance claim: the data is synthetic and the kit
is 3 pieces.

**Open question 2 — is octopaminergic tonic drive numerically stable as a bias
current?** Not unbounded. The OA pool is 25 neurons feeding high-gain drive
populations, and an unconstrained embedding saturates pC1/pIP10 within a few
steps. Bounding it through `tanh × max_current` keeps genres separable and
interpolation smooth; a test pushes the embedding 1000× out of distribution and
asserts the injected current stays inside its envelope.

### Bugs that would have produced plausible, wrong numbers

Most of the work in this session was finding these. Each one is the kind that
trains, evaluates, prints a number, and is silently meaningless.

1. **The ablation arms started at incomparable operating points.** The real
   subgraph's recurrent operator has |λ_max| ≈ 4,019; a degree-matched rewiring
   of the *same* edges and weights has ≈ 404. Under one shared gain constant the
   first saturates and the second goes near-silent, so the comparison measured
   which topology happened to land in range. Before normalising, the rewired arm
   diverged to loss 24 and onset F 0.0 — it would have been easy, and wrong, to
   report that as the connectome beating a random graph. Every arm is now scaled
   to a common target spectral radius; within-graph weight ratios are untouched.

2. **The model did not start where PLAN.md says it starts.** With
   `softplus(log_gain)` and `log_gain = log(synapse_count)`, the initial weight
   is `log(1 + synapse_count)`: a 2,591-synapse connection and a 26-synapse one
   collapse from a ratio of ~100 to ~2.4. Switched to `exp(log_gain)`, so the
   initial weight *is* the synapse count.

3. **The genre embedding was conditioned on a scrambled mapping.** The style→id
   map was built per split, so id 2 could mean "jazz" in train and "latin" in
   validation. It surfaced only as an out-of-range crash, and only because the
   validation split happened to contain a style the truncated training split
   lacked. With matching style counts it would have trained and evaluated
   cleanly on noise. The vocabulary now comes from the whole corpus.

4. **A missing corpus silently became synthetic data.** A run whose config named
   `data/egmd/groove` would fall back to click tracks and report metrics, with
   nothing in the logs or the numbers to distinguish it from a real run. Hit this
   live. Now a hard error; synthetic data is used only when explicitly requested.

5. **One subgraph cache served several configs.** Configs inherit their cache
   path, so `sanity_3piece` (10k nodes, min_weight 5) and `v1_8piece` (30k, 3)
   resolved to the same file and whichever ran last won. Cached metadata is now
   checked against the request and rebuilt on mismatch.

6. **The envelope filter aliased.** It smoothed at the audio rate then sampled
   every hop-th output, so whether an onset registered depended on its phase
   within the hop window. Now the sub-bands are decimated by a box mean first —
   which also happens to be where the 54× speedup came from.

7. **Onset F was scored at one fixed threshold.** Training loss fell smoothly
   while F bounced between 0.41 and 0.65; the metric was tracking output scale.
   Across ablation arms that is not just noisy but unfair. Now swept per model,
   with the chosen threshold reported.

Two more, smaller: torch's CSR autograd returns a gradient sized to the
*deduplicated* values when edges repeat, which the rewiring ablation can
produce — so the backward is written out explicitly and checked against a dense
reference. And the streaming peak picker skipped its `prev` update on a rising
edge, leaving stale state exactly at block boundaries.

---

## Real-time

Benchmark the streaming path before trusting it. The first measurement missed
PLAN.md's budget by 15× — **295 ms of inference per 20 ms block** — and the
blame was not the connectome: 10k and 30k node subgraphs cost nearly the same
(295 vs 309 ms), because 140 of the 173 ms encoder cost was a 15 ms one-pole
envelope expanded into a 1,323-tap FIR run on every audio sample.

| tier | nodes | edges | inference / 20 ms block | headroom |
|---|---|---|---|---|
| live (small) | 10,000 | 643k | 5.4 ms mean, 5.9 ms p95 | 3.7× realtime |
| live (default) | 30,000 | 2.94M | 14.2 ms mean, 16.8 ms p95 | 1.4× realtime |

Measured on 4 CPU cores, no GPU.

The encoder is shared between training and live playback, so a drift between the
windowed and whole-clip paths would train on one signal and play back on another
with nothing downstream to reveal it. `tests/test_encoder.py` pins them to agree
exactly, block by block.

---

## Layout

```
data/verified_types.json     Phase 0 gate output — the source of truth for type names
src/connectome.py            151.8M-row table -> cached neuron-level graph
src/subgraph.py              Phase 1: k-hop sweeps + pathway-strength trim, frozen signs
src/encoder.py               Phase 2: fixed gammatone DSP + one trainable linear layer
src/decoder.py               Phase 2: one linear layer, bilateral hemisphere masking
src/model.py                 ConnectomeRNN, genre modulation, sliders, lesion gates
src/train.py                 Phase 3: truncated BPTT
src/ablations.py             Phase 4: rewire / sign-shuffle / GRU / shortcut + lesion sweep
src/metrics.py               onset F, beat alignment, groove similarity
src/feel.py                  measured swing and timing offsets — never imposed
src/realtime.py              Phase 5: streaming inference, MIDI out, latency benchmark
scripts/verify_types.py      Phase 0 gate
scripts/render_full_graph.py offline pass over all 162k neurons
tests/                       54 tests: exact gradients, frozen signs and topology,
                             ablation invariants, encoder window/full equivalence,
                             checkpointing transparency, streaming peak state
```

No module hardcodes a cell-type string. Every population is read from
`data/verified_types.json`.

---

## Sliders and lesion mode

Each slider maps to a Phase-0-confirmed population:

| slider | population | effect |
|---|---|---|
| `drive` | pC1 (155 neurons) | fill density, intensity |
| `tightness` | inhibitory neurons (GABA/Glu/His) | tight vs. sloppy timing |
| `pocket` | membrane time constant | slower τ drags the beat |
| `gate` | pIP10 (2 neurons) | song on/off, near-binary |

```bash
python src/realtime.py --checkpoint runs/v1_8piece/best.pt \
    --slider drive 1.5 --slider pocket 1.2 --lesion pIP10
```

Lesion mode reuses the Phase 4 ablation machinery at inference, so muting a cell
type mid-performance costs nothing extra to ship.

---

## Data

- **Connectome** — MaleCNS v1.0, Google Research / HHMI Janelia, CC-BY, from
  `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`.
- **Drums** — Magenta **GMD** (`groove-v1.0.0`, 5.4 GB): 1,150 clips,
  897/124/129 train/val/test, 18 styles. PLAN.md names E-GMD, whose audio
  archive is 96 GB; GMD is the same Roland TD-11 recordings and style
  vocabulary. `scripts/fetch_egmd.py --corpus egmd` fetches E-GMD if you have
  the disk.

A synthetic click-track dataset ships alongside so the pipeline and tests run
with no corpus at all, and so a model failure stays distinguishable from a
data-path failure.

---

## What is not done

- **No meaningful trained run.** Needs a GPU. This is the gap that matters.
- **Spiking model** — the rate relaxation converges, so the surrogate-gradient
  LIF version is now unblocked, but unwritten.
- **v2 modes** — continue, call-and-response, accompany; leg mode; 8-limb kit;
  the dopaminergic reward channel.
- **`--full-graph` render tier** is implemented but has not been run end to end
  at 162k nodes.
