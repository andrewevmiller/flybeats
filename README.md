# flybeats

**Can the real fruit-fly wiring diagram help a neural network play drums to music?**

flybeats trains a network whose connections are fixed by the *Drosophila* male central nervous system connectome (MaleCNS v1.0). It listens to audio through the fly's ear neurons (Johnston's organ) and plays a drum kit through the wing and hind-leg motor neurons. Only connection strengths are learned; which neurons connect, and whether each connection excites or inhibits, comes from the connectome.

The project is a pre-registered experiment. The real connectome has to beat five degree-preserving random rewirings on beat alignment by more than the seed-to-seed spread. If it does not, that is the result.

> **Status: v0.1, pre-training.** The connectome slice, the data pipeline and the model are built and their gates pass. The final full-set training and the control comparison have **not** been run, so there is no result yet. `RETROSPECTIVE-v0.0.md` explains why v0.0 was abandoned. Claude writes the code, and the author verifies and runs each step by hand.

## How it works

1. **Connectome slice** (`src/flybeats/connectome/`). Builds one graph from three MaleCNS files: annotations, neurotransmitters and synapse weights.
   - It keeps neurons on any ear-to-motor path of at most 3 hops.
   - The ears are the usable JO-A/B neurons, with left ears mirrored to the right.
   - The motor targets are wing and hind-leg motor neurons.
   - Signs come from neurotransmitter: acetylcholine excites, GABA and glutamate inhibit, and modulators are silenced.
   - `controls.py` makes five random rewirings. Each neuron keeps its in and out degree, sign and synapse counts.
2. **Data** (`src/flybeats/data/`). Slakh2100 (redux, 16 kHz) gives a manifest, drum events, beat grids and a clip sampler.
   - Splits are per song.
   - Stem dropout is applied.
   - 127 training songs are excluded, either because their drum part duplicates a validation or test song's or because they have no kit hits. The list is in `config/excluded_songs.txt`.
3. **Model** (`src/flybeats/model/`).
   - `ear.py` is a fixed causal front end: 16 log-spaced bands from 40 Hz to 4 kHz, one frame per 5 ms.
   - `network.py` holds a per-type time constant and bias, one strength dial per (sending type, receiving type) pair (about 41,000 in total), and a hit and loudness readout per motor neuron.
   - Each of the 8 kit pieces reads only its own disjoint motor-neuron group: kick, hi-hat (closed, open and pedal), snare, tom, crash and ride.
   - `loss.py` is a weighted BCE for hits plus a velocity loss, with a hit F1 used for checkpoints.
   - `training.py` runs the training step, validation, the silence and stability test, and an optional recorded CUDA-graph step (1.45x faster, verified identical).

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

## Credits and licences

- **MaleCNS v1.0**: Janelia FlyEM, CC BY 4.0. Berg et al., "Sexual dimorphism in the complete connectome of the Drosophila male central nervous system", Cell (2026).
- **Slakh2100 / BabySlakh**: Manilow, Wichern, Seetharaman and Le Roux, WASPAA 2019, CC BY 4.0.

Full sources and checksums are in `SOURCES.md`. No licence has been set for this repository's own code yet.
