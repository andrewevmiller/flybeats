# flybeats

**Can the real fruit-fly wiring diagram help a neural network play drums to music?**

flybeats trains a network whose connections are fixed by the *Drosophila* male central nervous system connectome (MaleCNS v1.0). It listens to audio through the fly's ear neurons (Johnston's organ) and plays a drum kit through the wing and hind-leg motor neurons. Only connection strengths are learned; which neurons connect, and whether each connection excites or inhibits, comes from the connectome.

The project is a pre-registered experiment. The real connectome has to beat five degree-preserving random rewirings on beat alignment by more than the seed-to-seed spread. If it does not, that is the result.

> **Status: v0.1, pre-training.** The connectome slice, the data pipeline and the model are built and their gates pass. The final full-set training and the control comparison have **not** been run, so there is no result yet. `RETROSPECTIVE-v0.0.md` explains why v0.0 was abandoned. Claude writes the code, and the author verifies and runs each step by hand.

## How it works

```
Slakh mix (mono, 16 kHz)
  -> ear: 16 log-spaced bands, one frame per 5 ms
  -> 150 Johnston's-organ ear neurons (one fixed band each)
  -> 3,302-neuron fly network, wiring fixed by the connectome
  -> 182 wing / hind-leg motor neurons, split into 8 disjoint groups
  -> per drum: hit probability + velocity, every 5 ms
```

### The connectome slice (`src/flybeats/connectome/`)

`build_slice.py` turns three MaleCNS v1.0 files (annotations, neurotransmitter predictions, and about 152 million synapse-weight rows) into one saved slice. Everything is sorted, so the same `locked.yaml` always gives byte-identical output.

| Step | Rule | Result |
| --- | --- | --- |
| Neurons | Annotated neurons that have a superclass | 166,700 |
| Connections | Neuron pairs with at least 5 synapses (confidence at least 0.5) | 6,242,118 |
| Ears | Johnston's organ neurons of type `JO-A*` / `JO-B*` with at least one outgoing connection that passes the filter | 93 usable (75 left, 18 right) |
| Mirroring | The 75 usable left ears are copied to the right side and all real right ears are dropped, so the ears are symmetric. Targets of a mirrored ear are the same-type neurons on the opposite side (midline targets keep the same neuron and the synapses are split evenly) | 150 ears; 0.5% of synapses lost |
| Motor targets | MaleCNS subclasses `wm` (wing) and `hl` (hind leg) | 67 wing, 120 hind-leg |
| Cut | Keep every neuron on some ear-to-motor path of at most 3 hops (ear distance + distance back from the motor group) | **3,302 neurons, 108,593 connections** |

A few things about the slice:

- **Sides:** 1,595 left, 1,645 right and 62 midline neurons.
- **Signs:** acetylcholine is excitatory, and GABA and glutamate are inhibitory. Motor neurons are set to glutamate because the predictor is unreliable on them. Modulators (consensus serotonin, octopamine or dopamine) and neurons with an unknown transmitter stay in the graph but are silenced.
- **Roles in the slice:** 1,863 excitatory, 1,228 inhibitory, 189 motor, 8 modulatory and 14 unknown.
- **Cell types:** 1,224. Each of the 22 untyped neurons is its own type. This gives **41,186 (sending type, receiving type) pairs**.
- **Leftover motor neurons:** 2 middle-leg motor neurons are in the slice but are not read out. One wing and four hind-leg motor neurons have no type and are left out of the readout.
- **Scale:** the largest eigenvalue of the signed weight matrix is 730.6. The model divides its weights by it, so at the initial gain of 1.0 the network starts at the edge of stability.

**Random controls.** `controls.py` makes five rewired copies (seeds 1 to 5) by degree-preserving edge swaps, about 10 attempted swaps per edge and about 1.0 million accepted. Each neuron keeps its in and out degree and its sign, and the synapse counts travel with the edges. Only about 3.4% of the real connections survive in a control. Every drum's motor neurons are still reachable from the ears in each control, but at shorter distances (median 1 to 2 hops, against 2 to 3 in the real slice), which makes the real wiring's longer paths part of what is being tested.

### The model (`src/flybeats/model/`)

**Ear (`ear.py`).** This is fixed and causal: each frame's window ends at the frame, so it never sees the future.

- Front end: 512-point FFT with a Hann window and a hop of 80 samples (5 ms).
- Bands: energy is summed into 16 log-spaced bands from 40 Hz to 4 kHz, then passed through `log(1 + energy)` and standardised per band with fixed statistics from a random sample of training clips.
- Ear neurons: `JO-B` neurons take the 8 low bands and `JO-A` neurons the 8 high bands, round-robin by body ID. A mirrored ear takes its original's band. Each ear neuron's input current is its band's value, times a learned gain and plus a learned offset.

**Dynamics (`network.py`).** This is a rate network in 5 ms steps:

```
v[t+1] = v[t] + (dt / tau_type) * ( -v[t] + W r[t] + bias_type + ear_drive[t] )
r      = relu(v), zero for silenced neurons
W[post, pre] = sign[pre] * synapses * (gain / largest_eigenvalue) * softplus(theta[type_pre, type_post])
```

The wiring (which neurons connect), the signs and the synapse counts are fixed. Only the parameters below are trained. The real slice and each random control use the same class, and only the edge table differs.

**Readout.** Each of the 8 drums reads only its own motor neurons, and the groups do not overlap. Each readout neuron's rate is standardised first, with a mean and spread measured once on 16 training clips through the untrained network. Then, per drum, the hit logit is a weighted sum of these neurons plus a bias, and the velocity is a sigmoid of a second weighted sum plus a bias.

| Drum | Motor neurons | Types | Limb and muscle family |
| --- | --- | --- | --- |
| kick | 59 | 26 | right hind leg, all |
| hihat_pedal | 57 | 24 | left hind leg, all |
| snare | 12 | 5 | left wing, power |
| hihat_closed | 12 | 5 | right wing, power |
| tom | 13 | 13 | left wing, steering |
| ride | 13 | 13 | right wing, steering |
| crash | 8 | 7 | left wing, tergal / pleural / other |
| hihat_open | 8 | 7 | right wing, tergal / pleural / other |

The wing families are defined by MaleCNS type strings in `locked.yaml`. Power muscles are the DLMn and DVMn types, steering muscles are the b, i, iii, hg and ps types, and tergal / pleural / other are the tp, TTMn and STTMm types.

**Learned parameters.** About 44,000 in total. The counts below come from the slice report, not from instantiating the model.

| Parameter | Count | Initial value / range |
| --- | --- | --- |
| Strength dial `theta`, one per type pair | 41,186 | `softplus(theta) = 1`, so untrained dials change nothing |
| Time constant `tau`, one per cell type | 1,224 | 20 ms, clamped to 10 to 200 ms |
| Bias, one per cell type | 1,224 | 0 |
| Ear band gain and offset | 16 + 16 | 1 and 0 |
| Hit and velocity weights, per readout neuron | 182 + 182 | random, std 0.01 (zero would block every gradient) |
| Hit bias and velocity bias, per drum | 8 + 8 | -4 (each drum starts near "no hit") and 0 |

### The data (`src/flybeats/data/`)

Slakh2100 (redux, 16 kHz) is a set of synthesised multitrack songs with aligned MIDI. `BabySlakh` is a small subset used for gates and quick checks.

- **Splits:** the official Slakh redux splits, by song: 1,162 train (79.9 h), 270 validation (18.3 h) and 151 test (11.0 h), which is 1,583 songs. The training songs exclude 127 listed in `config/excluded_songs.txt`, either because their drum part is note-for-note identical to a validation or test song's or because they have only auxiliary percussion. Eight drum parts shared between validation and test are left alone.
- **Input:** the mono sum of the song's non-drum stems, with the drums left out. Each stem is dropped with probability 0.2, keeping at least one.
- **Targets:** MIDI drum notes are mapped to 8 kit pieces (General MIDI notes in `locked.yaml`; note 39, hand clap, counts as snare) on a 5 ms grid, with the velocity divided by 127. Auxiliary percussion (tambourine, cabasa, maracas, congas and the like) is dropped, which is 28.4% of all drum notes.
- **Hits in the full set:** hihat_closed 846,180; kick 492,966; snare 370,490; ride 119,171; hihat_open 97,191; hihat_pedal 92,549; tom 64,305; crash 45,286.
- **Timing:** 69.3% of hits fall on the 16th-note grid within 1 ms, and 468 of 1,583 songs have tempo changes. Beat grids come from the MIDI and are used for the tempo-scaled tolerance in the score.
- **Clips:** each training batch is 8 clips of 16 s (3,200 frames), each from a different song, chosen with probability proportional to duration, at a random start. The first 2 s of every clip is a warm-up and is not scored. Validation uses fixed clips (4 per song, seed 1234).

### Loss and training

- **Hit loss:** binary cross-entropy per frame. The target is 1 on a hit frame and 0.5 on the frame either side. Frames with a target above zero are weighted per drum by (non-hit frames / hit frames) over the training split, capped at 50.
- **Velocity loss:** squared error on hit frames only, at weight 0.5.
- **Optimiser:** Adam at 1e-3, cosine-decayed to zero over the run, gradient norm clipped at 1.0. Backpropagation runs through the whole 16 s clip, with no truncation.
- **Steps:** 1,000 on BabySlakh. The full-set count is open until it is set from the measured cost (about 2.2 h per 1,000 batches on the development GPU).
- **Validation:** every 500 steps, with the checkpoint kept on the best mean hit F1. A hit is a local peak of hit probability at or above 0.5. It counts as correct within 10% of the local beat (50 ms at 120 BPM).
- **Stability:** a silence test runs music followed by 4 s of silence. Activity must stay finite and below 50 times the music level, and must fall below 5% of it within 1 s. At least half the readout motor neurons must respond, and every drum must have one. Training stops if activity is unbounded or does not settle.
- **Same for all arms:** the real network and every control train with identical settings, and there are no command-line overrides.

## Pre-registration

All design choices live in `config/locked.yaml` (currently `prereg-v3`) and are recorded in `config/CHANGELOG.md`. The file covers the slice rule, model, loss, training, kit mapping, scorecard and win rule. Every change gets a new tag and a dated changelog line before the run that uses it. The code refuses to run if the ear settings differ from the locked ones.

- **Main win rule:** the real network beats all 5 random controls on beat alignment by more than the seed spread.
- **Supporting rule:** it beats the random median on 4 of 6 other scorecard facets.
- **Evaluation:** on the test split, once.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/flybeats/` | Library: `config`, `connectome/`, `data/`, `model/` |
| `scripts/` | One script per phase step: `unpack_slakh`, `build_manifest`, `build_slice`, `build_controls`, `band_stats`, `check_*_gate`, `stability`, `measure_cost`, `memorise`, `train`, `check_cuda_graph`. Also the live training chart (`live_chart.*`, `machine.py`) |
| `scripts/phase0/` | PowerShell setup, data verification and tagging (see its README) |
| `config/` | `locked.yaml`, `CHANGELOG.md`, `excluded_songs.txt` |
| `reports/` | JSON and text output from the gates, cost, stability and memorisation checks, plus session notes |
| `tests/` | pytest tests for the slice and the model |
| `SOURCES.md` | Dataset versions, URLs, checksums and licences |

## Getting started

The project targets Windows with an NVIDIA GPU (developed on an RTX 2060 Max-Q with 6 GB). `requirements.lock.txt` pins the environment, including `torch 2.14+cu126`. The whole set of steps below needs about 50 GB of disk for Slakh, and about 8 GB of RAM to build the slice.

```powershell
# Phase 0: environment and data checks (see scripts/phase0/README.md)
scripts\phase0\2_build_env.ps1
scripts\phase0\3_verify_data.ps1

# Phase 1: connectome slice and controls
.venv\Scripts\python.exe scripts\build_slice.py
.venv\Scripts\python.exe scripts\build_controls.py
.venv\Scripts\python.exe scripts\check_slice_gate.py

# Phase 2: data
.venv\Scripts\python.exe scripts\unpack_slakh.py --set baby
.venv\Scripts\python.exe scripts\build_manifest.py --set baby
.venv\Scripts\python.exe scripts\check_data_gate.py --set baby

# Phase 3: model
.venv\Scripts\python.exe scripts\band_stats.py --set baby
.venv\Scripts\python.exe scripts\stability.py --set baby
.venv\Scripts\python.exe scripts\memorise.py --pair
.venv\Scripts\python.exe scripts\train.py --set baby --run baby_real

# Tests
.venv\Scripts\python.exe -m pytest
```

Data paths are set in `config/paths.local.yaml`, which is not committed. Every training setting comes from `locked.yaml`, so there are no command-line overrides. The real network and each control train identically.

## Current state (29 Sep 2026)

- Phase 1 (slice and 5 controls) and Phase 2 (BabySlakh and full-set data gates) are done.
- Phase 3 stability passes at gain 1.0. Cost is about 7.9 s per batch ordinary and 5.4 s recorded, with a 1.3 GB peak.
- The memorisation check is the open item. An earlier attempt plateaued at the audio-blind loss, which led to the readout normalisation added in `prereg-v3`. The rerun's loss was below the audio-blind floor at step 171, and the check is still running.
- Not started: full-set training, the scorecard (Phase 4) and the control comparison.

## Licence

Copyright (c) 2026 Andrew Miller. **All rights reserved.** No permission is granted to use, copy, modify or distribute this code except with the copyright holder's prior written permission. See `LICENSE`.

The datasets are not part of this repository and keep their own licences (below).

## Credits and data licences

- **MaleCNS v1.0**: Janelia FlyEM, CC BY 4.0. Berg et al., "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system", Cell (2026).
- **Slakh2100 / BabySlakh**: Manilow, Wichern, Seetharaman and Le Roux, WASPAA 2019, CC BY 4.0.

Full sources and checksums are in `SOURCES.md`.
