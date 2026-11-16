# flybeats plugin: Fly Lab

The VST3 plugin that runs an exported fly network in a DAW. Status: **engine phases P0 to P2 (this directory); no
plugin shell yet; nothing has run in FL Studio.** The plan is the VST plugin plan of 15/16 Nov 2026 (owner's copy);
the decisions it needed are in `docs/P0-decisions.md` and the gate memo in `docs/gate-P2-memo.md`.

| Path | What |
|---|---|
| `docs/` | P0 decision records (ADRs, parity tolerances, perturbation tests, slot-in bar), gate memos |
| `export/flybrain/` | Python: the `.flybrain` file format, the exporter (checkpoint + slice -> brain file), the numpy reference engine with every Lab perturbation, the parity harness against PyTorch, the drift test, the laptop CLI |
| `export/tests/` | pytest: `python -m pytest plugin/export/tests` from the repo root |
| `engine/` | C++17 engine library (no JUCE, no SDK): brain file reader, resampler, ear, sparse core, readout, decoder, snapshot tap, control bus, runaway guard, recorder; command-line render and bench tools; ctest tests |
| `s0/` | The S0 FL Studio test build (two trivial JUCE VST3 plugins) and its observation sheet |
| `reports/` | Measured results from this session (drift, engine report, S0 build report) |

## Exporting a real checkpoint (laptop)

```
.venv\Scripts\python.exe -I -m flybrain.cli --set baby --checkpoint runs\baby_real\best.pt --out brains\baby_real.flybrain --name "real fly, baby-trained"
```
from `plugin\export` (or with it on `PYTHONPATH`). It rebuilds the network from the slice, loads the checkpoint
strictly, calibrates the cliff with the repo's silence test, writes provenance hashes, and runs the parity harness
against PyTorch on training clips. Every brain exported this way is `observation_only`; `drum_capable` is set only
through Gate B.

## Honesty

No network has learned timing (v0.1 results). Nothing here claims otherwise: the Scoreboard payload of a brain
exported without a v0.15 verdict says "none".
