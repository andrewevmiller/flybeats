# flybeats

Connectome-constrained drum performance from audio. A sparse recurrent network
whose **topology is fixed by the real MaleCNS v1.0 wiring diagram** and whose
**per-connection gains are learned**. Gradient descent can change what a
connection is worth; it cannot create a connection the animal does not have.

See [PLAN.md](PLAN.md) for the design this implements.

> **Status.** The pipeline is complete and runs end to end: Phase 0 gate through
> Phase 5 streaming playback, with the full Phase 4 ablation harness. What has
> **not** happened is a trained run that answers the project's central question.
>
> Three faults blocked it, and all three are now fixed. The rate regulariser no
> longer rewards silence; the encoder no longer cancels its own input; and the
> global gain normalisation no longer pins the network at an operating point
> where **the drive never reaches the wing motor pool at all** — it was losing
> 28× at the first synapse, which no amount of training could recover. The
> third was hiding behind the first two; see
> [Where the signal stops](#where-the-signal-stops).
>
> With `spectral_radius: 10.0`, a 25-epoch CPU run on real GMD audio beats the
> best constant predictor for the first time (`gain over constant` +0.0128,
> `|corr(pred, target)|` 0.134, onset F 0.31) and the diagnostic now calls the
> model undertrained rather than broken. That is a working pipeline, **not** a
> result about the connectome: the Phase 4 arms still need a GPU.
>
> Resuming work: **[Picking this up again](#picking-this-up-again)**.
> Read [Results](#results-and-what-they-are-not) before quoting any number here.

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

### The real-corpus run, and the faults it exposed

A short run on real GMD audio (10k-neuron subgraph, 8-piece kit, 64 clips,
6 epochs, CPU) is in `runs/v1_8piece_cpu/`. It is not a result either — it is
the run that found the problems below. These were its numbers:

```
ep 0  loss 0.7391   onset F 0.2132   groove 0.275
ep 5  loss 0.6561   onset F 0.2147   groove 0.259
```

**Loss falls; onset F and groove do not.** `scripts/diagnose.py` separates the
causes by layer, and the answer was a design fault, not undertraining:

- The model was **worse than the best constant predictor** (BCE 0.6588 vs
  0.6344), and its per-class outputs sat on the closed-form weighted optimum —
  kick predicting 0.313 against an optimum of 0.316, snare 0.568 against 0.555.
- Mean `|corr(prediction, target)|` was **0.029**. No timing in the output at all.
- Encoder drive varied strongly (0.47 relative); motor rates barely moved
  (0.0056 relative). The time-varying signal was washed out downstream.
- **96% of the encoder's `to_jo` weights were negative**, having been
  initialised non-negative from the JO zone prior, leaving the drive as mostly a
  DC offset. The encoder had learned to suppress its own sensory input.

Two causes, both fixed:

1. **The rate regulariser rewarded silence.** `target_rate_hz: 5.0` became a
   target *activation* of `5 × 5ms/1000 = 0.025`, compared against
   `softplus(v − θ)` — a dimensionless activation with no Hz interpretation,
   sitting at 0.74. The units were meaningless and the penalty was a constant
   ~30× downward pressure on all activity, so silencing the sensory input was
   the cheapest way to satisfy it. It is now **one-sided** — nothing below the
   ceiling is penalised — and the ceiling is measured from each model's own
   activity at initialisation, in the units the network actually has. A config
   still carrying `target_rate_hz` is refused rather than reinterpreted.
2. **The encoder was fighting its own DC.** `log1p` band energy is a large
   positive constant plus a small fluctuation, and cancelling that constant with
   negative weights cancels the signal with it. Features are now standardised
   per channel by a **fixed affine**, calibrated once on training audio and
   saved with the weights — fixed rather than per-clip, so the streaming path
   still matches the training path exactly (`tests/test_encoder.py` pins it).
   `to_jo` is constrained **non-negative** by projection after each optimiser
   step: an onset function driving JO afferents should excite them, and the DC
   offset is the bias's job.

Both worked, at their own layer. Same config, same corpus, same 6 epochs:

| | before | after |
|---|---|---|
| relative temporal variation of the drive | 0.47 | **2.91** |
| `to_jo` weights negative | 96% | **0%** |
| drive DC / temporal sd | ≫1 | **0.34** |

What they did *not* do is move onset F (0.215 → 0.200) or `|corr(pred, target)|`
(0.029 → 0.019). That is not the fixes failing. It is the fault they were
hiding, which the next section is about — and once that one was fixed too, the
same two encoder numbers went on to carry a drive that finally arrives.

At one epoch the simplest arms lead, which is what one epoch measures. Drawing
"the rewired graph beats the connectome" from this would be wrong.

### Where the signal stops

The drive now varies (2.91 relative) and the wing motor pool still does not
(0.0064). "Washed out in the recurrence" is a guess; `scripts/diagnose.py`
section 6 measures it, as relative temporal variation of the firing rate by hop
distance from the JO afferents:

```
hop     neurons  mean rate  sd over time   relative
---------------------------------------------------
0           405     1.1403      0.695646    0.55147
1           376     0.7203      0.014983    0.01945   /28
2          4969     0.7471      0.007273    0.00609   /3
3          4209     0.7024      0.003086    0.00390   /2
4            41     0.7178      0.003523    0.00396   /1
motor pool sits at hops {2: 62, 3: 4}
```

**The loss is not spread through the depth of the connectome. It is a 28× drop
at the first synapse**, after which each hop costs a factor of 2–3. And the
giveaway is the mean rate: every neuron past hop 0 sits at ~0.72, which is
`softplus(0)` — they are receiving essentially nothing.

The suspect is not the topology but the operating point. `gain_scale: auto`
normalises the recurrent operator to `spectral_radius: 0.9`, which is what makes
the Phase 4 arms comparable — but ρ is carried by a small, strongly connected
hub subnetwork, so pinning it at 0.9 divides every weight by that hub's gain and
leaves the typical neuron's synaptic input far below its own resting activation.
`scripts/propagation.py` sweeps it on an **untrained** model over real audio,
forward passes only:

```
  radius     hop 0     hop 1     hop 2     hop 3     motor  mean rate  at clip
      0.90   0.55564   0.01608   0.00788   0.00497   0.00760     0.7429     0.0%
      2.00   0.55630   0.03935   0.02794   0.02475   0.01766     1.0477     0.9%
      5.00   0.55817   0.16779   0.15022   0.11071   0.06244     1.2863     1.7%
     10.00   0.56059   0.72660   1.06303   0.76519   1.00240     1.8206     3.6%
     25.00   0.56473   4.13316   5.06504   4.19136   3.84961     3.1754     9.4%
```

So the architecture *can* carry the signal; at the configured radius it does
not, and it is not something training can recover — no gradient on a
connection's gain can create modulation that never arrives. Between ρ 5 and 10
the motor pool goes from 0.06 to 1.00, at the cost of 3.6% of unit-steps pinned
against the state clip. ρ = 25 is louder and a tenth saturated, which is the
same failure from the other side.

`spectral_radius` is therefore **10.0**, not 0.9. This does not touch the
Phase 4 fairness argument: every arm is still normalised to one common radius,
and only the value of that constant changes.

### The first run that learns anything

Same CPU config, ρ = 10, 25 epochs on the same 64 GMD clips
(`runs/rho10_long/`). Still not a result — 64 clips, a 10k-neuron subgraph and
no GPU — but it is the first run in this repo whose diagnostic says the model is
*undertrained* rather than broken:

```
ep  0  loss 0.7484   onset F 0.186   groove 0.304
ep  8  loss 0.6039   onset F 0.297   groove 0.328
ep 16  loss 0.5809   onset F 0.295   groove 0.347
ep 24  loss 0.5658   onset F 0.292   groove 0.352      best F 0.3118
```

| `scripts/diagnose.py` | ρ = 0.9, 6 ep | ρ = 10, 6 ep | ρ = 10, 25 ep |
|---|---|---|---|
| gain over best constant | −0.0170 | −0.0172 | **+0.0128** |
| mean \|corr(pred, target)\| | 0.019 | 0.069 | **0.134** |
| motor relative modulation | 0.0064 | 0.189 | **0.450** |
| prediction relative variation | 0.0070 | 0.055 | **0.413** |
| verdict | learned the base rate | learned the base rate | **tracks the target** |

The per-hop table flattens out completely — 1.04 / 1.29 / 1.45 / 1.05 from the
JO afferents to hop 3 — so nothing is being lost on the way to the wings any
more. Per-class prediction sd goes from ~0.003 (a constant with noise on it) to
0.10–0.19, and every class that has onsets now correlates positively with its
target.

What this does **not** show: that the connectome is doing the work. That is what
the Phase 4 arms are for, and they need a GPU and the full corpus.

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

8. **The rate regulariser's units were meaningless** — a target in Hz compared
   against a dimensionless activation, which made it a constant downward pull on
   all activity and taught the encoder to go quiet. Detailed
   [above](#the-real-corpus-run-and-the-faults-it-exposed); it trained, it
   evaluated, and its loss curve fell the whole way down.

9. **The encoder's own DC gave it a way to cancel its input.** Same section. The
   symptom was a metric that would not move while everything upstream looked
   healthy.

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
scripts/diagnose.py          why a checkpoint is not learning, separated by layer
scripts/propagation.py       what the subgraph carries, per hop, before training
scripts/render_full_graph.py offline pass over all 162k neurons
tests/                       62 tests: exact gradients, frozen signs and topology,
                             ablation invariants, encoder window/full equivalence
                             (calibrated and not), the non-negativity constraint,
                             one-sided rate penalty, hop distances, checkpointing
                             transparency, streaming peak state
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

- **No meaningful trained run.** Needs a GPU and the full corpus. This is the gap
  that matters. The model now learns *something* (see
  [the first run that learns anything](#the-first-run-that-learns-anything)), but
  64 clips on a 10k-neuron subgraph settles nothing about the connectome.
- **The radius is calibrated for the 10k tier only.** Re-run
  `scripts/propagation.py` before training the 30k one.
- **Spiking model** — the rate relaxation converges, so the surrogate-gradient
  LIF version is now unblocked, but unwritten.
- **v2 modes** — continue, call-and-response, accompany; leg mode; 8-limb kit;
  the dopaminergic reward channel.
- **`--full-graph` render tier** is implemented but has not been run end to end
  at 162k nodes.

---

## Picking this up again

Everything is committed and pushed on `claude/resume-previous-session-8r6f7g`;
62 tests pass; the connectome cache, subgraph cache and GMD corpus are rebuilt by
the `scripts/fetch_*.py` commands in [Quick start](#quick-start) (they are
gitignored, not in the repo).

The three faults the last session's diagnostic found are fixed and verified: the
rate regulariser is one-sided against a measured ceiling, the encoder is
standardised and its map constrained non-negative, and `spectral_radius` is 10
rather than 0.9. `scripts/diagnose.py` now reads *"output varies and tracks the
target; likely undertrained rather than structurally broken"* — which it had
never said before.

**A GPU ablation run is no longer blocked on a known fault.** It is still worth
one more CPU hour first, because nothing here has been trained long enough to
know where it plateaus.

### First, on CPU

1. **Train longer at ρ = 10 and find the plateau.** 25 epochs on 64 clips ends
   with the loss still falling (0.5658) and onset F flat around 0.29–0.31, which
   is the signature of a model limited by data, not by epochs. Raise
   `data.max_files` before raising `train.epochs`.

   ```bash
   python src/train.py --config configs/v1_8piece_cpu.yaml --epochs 40
   python scripts/diagnose.py --checkpoint runs/v1_8piece_cpu/best.pt
   ```

2. **Re-run the radius sweep for the 30k tier.** The 10 was chosen on the
   10k-node subgraph. A different tier has a different hub structure, so the
   number does not transfer by assumption:

   ```bash
   python scripts/propagation.py --config configs/v1_8piece.yaml
   ```

   It prints a recommended radius under a stated rule (motor modulation at least
   half the JO afferents' own, under 5% of unit-steps at the state clip).

3. **Watch the saturation end.** ρ = 10 pins 3.6% of unit-steps against
   `state_clip` at initialisation and the trained model's hop-4 population sits
   at a mean rate of 4.2. If a longer run pushes that up, the one-sided rate
   penalty is the knob (`rate_headroom`), and `scripts/diagnose.py` section 6
   will show it as a *rising* mean rate with falling relative modulation.

### Then, in order

- **Run the real experiment** (needs a GPU):
  `python src/ablations.py --config configs/v1_8piece.yaml --lesion --epochs 40 --seeds 5`.
  `--seeds` is not optional: a gap smaller than the across-seed spread is not a
  result. Every arm is normalised to the same radius, so the comparison is still
  about topology — but check `propagation.py` for the 30k tier first, or all five
  arms may sit at an operating point that carries nothing.
- **Fix the weakest claim in Phase 1.** 3 hops from JO reaches 154,853 of 162,517
  neurons, so the subgraph is selected by `_trim`, not by anatomy — the code now
  says so plainly, which is the honest half of the fix. The other half is a
  path-based criterion to replace the two-hop heuristic.
- **Run the render tier** — `scripts/render_full_graph.py` has never been executed
  at 162k nodes.
- **Deferred:** spiking model (the rate model now does something, so this is
  genuinely next), v2 modes, leg/8-limb tiers, live playback on real hardware,
  and the sound bank in [SOUNDBANK_PLAN.md](SOUNDBANK_PLAN.md).

### Things that will bite

- `neuprint.janelia.org` is blocked from this environment; the Phase 0 gate reads
  the flat-connectome feather instead. `--neuprint` is implemented but has never
  been exercised.
- E-GMD's audio archive is 96 GB. The default corpus is GMD (5.4 GB), same
  recordings and style vocabulary.
- No GPU here, so every number in this repo is CPU-scale.
