# flybeats: Connectome-Constrained Drum Performance from Audio

## What this actually is

MaleCNS v1.0 (Google Research / HHMI Janelia, published Sept 2026, 166,691
neurons, 125M synapses, CC-BY) is a **wiring diagram**, not a set of trained
weights. We are not "training the fly brain to play drums." We are building a
sparse recurrent network whose **topology is fixed by the real connectome**
and whose **per-connection gains are learned**, following the pattern
established by tools like ConnecTorch: gradient descent can change what a
connection is worth, it cannot create a connection the animal doesn't have.

The task is a natural fit: flies hear via Johnston's Organ and produce
rhythmic courtship song through a well-mapped descending pathway
(JO afferents → AMMC/WED interneurons → pIP10 → wing motor neurons). We are
pointing that circuit at a drum kit instead of a wing.

**Core claim to prove, not assume:** real connectome topology is a useful
inductive bias for this task, compared to degree-matched random graphs and
conventional baselines. If it isn't, that's the result. Report it either way.

---

## Phase 0 — Data acquisition + verification gate

- `pip install neuprint-python connectorch`
- Create a neuPrint account at neuprint.janelia.org, get a token, `export
  NEUPRINT_TOKEN=...`
- Bulk download from `gs://flyem-male-cns/v1.0/connectome-data/flat-connectome/`:
  - `connectome-weights-male-cns-v1.0-minconf-0.5.feather` (segment-to-segment graph)
  - `body-annotations-male-cns-v1.0-minconf-0.5.feather` (types, classes, sides)
  - neurotransmitter predictions table

**Gate — do not skip:** query neuPrint directly for the real cell-type
strings before writing code that assumes any of them. Sensory afferents are
frequently truncated fragments in EM reconstructions, and type names drift
between dataset versions. Confirm before coding:

- JO-A, JO-B, JO-E (auditory afferents)
- AMMC / WED interneurons, including B1
- aPN1, vPN1
- pC1, pC2 (persistent internal-state / courtship drive)
- pIP10 (descending song command neuron)
- Wing steering motor neurons: b1, b2, b3, hg1–hg4, i1, i2, iii1–iii4, ps1, tp1, tp2
- DLM / DVM power muscles
- Octopaminergic neuromodulatory types (OA-VPM, VUM), for genre injection later

Log the confirmed type list to `data/verified_types.json`. Every later phase
reads from this file — never hardcode a type name that wasn't confirmed here.

---

## Phase 1 — Subgraph extraction

Full-graph backprop is not viable for sequence lengths we need (a 4-bar loop
at 5 ms resolution is ~1,600 steps; ConnecTorch reports ~1.1s and ~7.8 GiB
VRAM per forward+backward at batch 4 / 8 steps on the full graph).

- Seed a k-hop subgraph on JO afferents, intersected with the ancestor set of
  the wing (and optionally leg) motor neurons confirmed in Phase 0.
- Target 10k–30k neurons for the trainable "live" graph.
- Keep a `--full-graph` flag for an offline, single-batch "render mode" run
  on the complete 164k-node graph, for demo/comparison purposes only.
- Assign edge signs from neurotransmitter predictions (ACh → excitatory,
  GABA/Glu → inhibitory) and **freeze signs**. Only magnitudes are trainable.

---

## Phase 2 — Encoder and decoder

Keep both thin. If either is a deep net, it will learn the task itself and
the connectome becomes decoration rather than the mechanism being tested.

**Encoder (audio → JO input current):**
- 64-band gammatone filterbank → half-wave rectify → per-band envelope →
  spectral flux onset function
- Map bands to JO channels by frequency; weight JO-B toward 100–500 Hz
  (fly song range)
- One small trainable linear layer on top of the fixed DSP block, nothing deeper

**Decoder (motor neuron rates → drum triggers):**
- Single linear layer, motor neuron firing rate → per-class drum velocity
- No hidden layers

---

## Phase 3 — Training

- **Dataset:** Magenta E-GMD (Expanded Groove MIDI Dataset) — real drum audio
  with sample-aligned MIDI, includes style labels (funk, rock, jazz, latin,
  hiphop, afrobeat, etc.). Start with the 10 most common velocity-collapsed
  classes.
- **Loss:** per-class BCE against Gaussian-smoothed onset targets (σ ≈ 20 ms)
  + firing-rate regularizer to prevent saturation
- **Model:** start with a rate-based relaxation of the LIF dynamics —
  converges far more reliably than surrogate-gradient spiking. Move to
  surrogate-gradient LIF only once the rate model is solid.
- Truncated BPTT, 100–200 steps, gradient checkpointing on, `bfloat16`
- Learnable: per-edge log-gain (init at `log(synapse_count)`), per-neuron
  time constant, per-neuron threshold

---

## Phase 4 — Ablation controls (required, not optional)

Without these there's no evidence the real connectome contributed anything.
Run all four on identical data/hyperparameters and report onset F-measure for
each:

1. Real MaleCNS subgraph (the actual experiment)
2. Degree-matched random rewiring of the same subgraph
3. Real topology, shuffled neurotransmitter signs
4. Dense GRU, matched parameter count, no connectome structure at all

Also keep an "encoder → decoder direct shortcut" ablation (connectome fully
bypassed) as a sanity floor.

---

## Phase 5 — Real-time playback

- Ring-buffer audio input (`sounddevice`), stateful streaming inference
- MIDI out via `mido` / `python-rtmidi` to a drum sampler
- Per-class refractory window; velocity from readout magnitude
- Latency budget: ~20 ms buffer + ~10 ms inference; add a short lookahead
  buffer if onset-anticipation becomes the dominant error mode

---

## Feature / config surface

Expose these as CLI flags / a single YAML config (`configs/*.yaml`), not
scattered constants.

### Kit size — tied to real motor populations, not picked arbitrarily
| Tier | Source | Notes |
|---|---|---|
| 3-piece | subset of wing MNs | training sanity check |
| 8-piece (default) | GM-style map from wing steering MNs | ship default |
| Articulated | wing MNs, finer readout | hat open/closed/pedal, rimshot, sidestick, ghost notes |
| Leg mode | ~70 MNs/leg in VNC | large kit (20–30 piece) option |
| Bilateral split | left/right hemisphere MNs | left MNs → left-hand drums, right → right-hand; free limb-independence constraint |
| 8-limb | 6 legs + 2 wings | demo-only, physically superhuman kit |

Cap kit size at the **confirmed** motor-neuron-type count from Phase 0 — don't
invent a number before that gate returns real types.

### Genre — injected as neuromodulatory bias, not separate checkpoints
- Learn a small embedding per E-GMD style label
- Feed as tonic bias current into octopaminergic populations (OA-VPM/VUM)
- One trained model, swap the state vector at inference → different groove
- Supports interpolation between genres (not possible with per-genre checkpoints)

### Biological control sliders (each maps to a real, confirmed population)
| Slider | Population | Expected effect |
|---|---|---|
| Drive / density | pC1 / P1 | fill density, intensity |
| Tightness | global GABA gain | tight vs. sloppy timing |
| Pocket | membrane time constant scale | slower τ drags the beat |
| Master rhythm gate | pIP10 gain | song on/off, near-binary |
| (v2) reward channel | PPL1/PAM dopaminergic drive | hook for online adaptation |

### Performance modes
- **Transcribe** (v1 default / training objective): audio in → matching drums out
- **Lesion mode** (cheap, ship in v1): mute a cell type mid-performance via
  the same ablation machinery from Phase 4; audible beat degradation is the
  best demo this project produces
- **Continue** (v2): audio cuts, network keeps time from its own recurrent
  state — a real result if tempo holds, not just a feature
- **Call and response** (v2): listen N bars, play N bars
- **Accompany** (v2): non-drum audio in (bassline, guitar loop) → invented
  part; needs different training pairs, scope separately

### Feel — measure, don't impose
No hand-authored swing parameter. Instead compute, per class, against the
grid: mean onset deviation, swing ratio, hat-vs-kick timing offset. "The
model rushes closed hats by 8 ms" is a finding; a swing slider is a guess.

### Compute tiers
- **Live**: 10k–30k node subgraph (Phase 1), real-time capable
- **Render**: full 164k-node graph, offline, single batch — same trained
  weights, wider graph scope, used for final demo renders and to check
  whether the larger graph changes anything

---

## Metrics
- Onset F-measure per class, 50 ms tolerance
- Beat-alignment error vs. ground-truth tempo
- Groove-similarity score vs. held-out human performances (E-GMD test split)
- Ablation table (Phase 4) reported alongside every headline result

---

## v1 scope (recommended cut line)
8-piece kit · bilateral split · genre embedding via octopaminergic drive ·
three sliders (drive, tightness, pocket) · transcribe mode · lesion mode ·
full Phase 4 ablation suite.

Everything else (leg mode, 8-limb, continue/call-response/accompany, reward
channel) is v2.

---

## Suggested repo structure
```
flybeats/
  data/
    verified_types.json        # Phase 0 gate output — source of truth for type names
    raw/                        # downloaded feather files (gitignored)
  src/
    subgraph.py                 # Phase 1: k-hop extraction
    encoder.py                  # Phase 2: audio -> JO current
    decoder.py                  # Phase 2: MN rate -> drum velocity
    model.py                    # ConnecTorch-based ConnectomeRNN wrapper
    train.py                    # Phase 3
    ablations.py                # Phase 4: random-rewire, sign-shuffle, GRU baseline, shortcut
    realtime.py                 # Phase 5: streaming inference + MIDI out
  configs/
    v1_8piece.yaml
    ablation_*.yaml
  scripts/
    verify_types.py             # run Phase 0 gate, writes verified_types.json
    render_full_graph.py        # offline full-164k-node demo pass
  notebooks/
    feel_analysis.ipynb         # measured swing/pocket/timing-offset stats
```

## Open questions to resolve early with Claude Code
1. Does the confirmed wing-MN type count (Phase 0) support an 8-piece kit
   cleanly, or does the mapping need to fold in leg MNs to reach 8?
2. Is octopaminergic tonic drive numerically stable as a bias current, or
   does it need to be gated/normalized per-genre to avoid saturating pC1/pIP10?
3. Rate-based vs. surrogate-gradient LIF: confirm the rate model converges on
   a 3-piece sanity task before investing in the spiking version.
