# Installing FlyBeats on Windows

A step-by-step install for a Windows 10/11 laptop, with the cost and the
expected output of every step.

**How this was verified.** Every command, size, timing and memory figure below
was produced by running the step in a clean checkout on 4 CPU cores / 15 GB RAM
/ Python 3.11 — on **Linux**, not Windows. The pipeline steps are
platform-independent (pathlib throughout, no shell-outs), so the numbers
transfer; the Windows-specific parts — the `py` launcher, PowerShell activation,
wheel availability — were checked against PyPI and the Python docs rather than
executed on a Windows machine. Anything not verified either way is marked.

**Time budget:** about 30 minutes of attention, plus two downloads (1.11 GB and
5.11 GB) that run unattended.

---

## Two ways in

This guide installs the **build** path: the connectome, the corpus, and
everything needed to train a model. There is a much smaller **play** path — a
trained model exports as a self-contained ~10 MB bundle that needs no
connectome, no corpus and no `data/` directory at all:

```powershell
pip install -r requirements.txt
python src/realtime.py --bundle flybeats-8piece.fb --render song.wav `
    --sound-source samples --out drums.wav
```

That is roughly 1 GB and five minutes. **But no bundle is published yet** — the
repo ships no `.fb` file, so today the only way to get one is to train it via
the build path below and run `scripts/export_bundle.py`. Shipping a bundle is a
release task, not an install step.

## Contents

1. [Do this part first](#1-do-this-part-first) — what's worth installing now
2. [Prerequisites](#2-prerequisites) — hardware, Python version, disk
3. [Install Python and Git](#3-install-python-and-git)
4. [Clone and create the environment](#4-clone-and-create-the-environment)
5. [Install the dependencies](#5-install-the-dependencies)
6. [Download the connectome](#6-download-the-connectome)
7. [Run the Phase 0 gate](#7-run-the-phase-0-gate)
8. [Run the tests](#8-run-the-tests)
9. [Build the subgraph](#9-build-the-subgraph)
10. [Prove the pipeline runs](#10-prove-the-pipeline-runs)
11. [Optional: the drum corpus](#11-optional-the-drum-corpus)
12. [Optional: realtime MIDI](#12-optional-realtime-midi)
13. [Verification checklist](#13-verification-checklist)
14. [Troubleshooting](#14-troubleshooting)
15. [What lives where, and how to remove it](#15-what-lives-where-and-how-to-remove-it)
16. [When the training fixes land](#16-when-the-training-fixes-land)

---

## 1. Do this part first

Work is still landing on the features, so part of this install will be thrown
away and part of it will not. The slow half is the half that survives.

> **Changed since this guide was written.** The three faults that had the model
> converging to a constant predictor are fixed — the rate regulariser, the
> encoder's DC collapse, and a global gain losing 28× at the first synapse — and
> a CPU run now beats the best constant predictor. A velocity head has landed
> since, so the decoder's velocity is trained dynamics rather than detection
> confidence. What is still in flight is below.

**Stable — nothing pending touches these. Do them now.**

| step | why it survives |
|---|---|
| Python, venv, `pip install` | The dependency set has held across every change so far. Later work can *add* to `requirements.txt`, and re-running the same command picks that up. |
| `scripts/fetch_data.py` | MaleCNS v1.0 is a frozen public release. 1.11 GB, downloaded once, never invalidated by anything in this repo. |
| `scripts/fetch_egmd.py` | Magenta GMD `groove-v1.0.0` is likewise frozen, and the README pins the corpus to GMD rather than E-GMD. 5.11 GB, once. |
| `scripts/verify_types.py` | The Phase 0 gate. Its output is already committed to the repo and regenerating it takes 4 seconds, so it costs nothing either way. |

**Not stable — cheap to redo, so don't bother yet.**

| step | what invalidates it | cost to redo |
|---|---|---|
| `data/cache/subgraph*.npz` | The Phase 1 to-do is to *replace the pathway-strength trim* with a path-based criterion, which changes which neurons the trim selects. | 8 s |
| anything in `runs/` | `runs/` is gitignored, so a checkpoint is not durable regardless — **export a bundle or lose it**. The velocity head also changed the loss, so anything trained before it is not comparable. | minutes to hours |
| edits to `configs/*.yaml` | The loss keys are still moving; `velocity_weight` is the newest. | — |
| `sounddevice` / `mido` / `python-rtmidi` | Commented out of `requirements.txt`, the only install here that can need a C++ compiler, and needed only for *live* input — the offline render and the MIDI file path work without them. | — |

Steps 3 through 10 below cover the stable half plus a one-off proof that the
pipeline runs. Steps 11 and 12 are optional.

---

## 2. Prerequisites

### Hardware

| | requirement | why |
|---|---|---|
| **CPU** | x64 (Intel or AMD) | PyTorch publishes **no `win_arm64` wheel**. On a Snapdragon X / Copilot+ PC, `pip install torch` fails outright. Checked against PyPI: torch 2.14.0 ships `win_amd64` only. |
| **RAM** | 16 GB | The neuron-graph build peaks at **5.74 GB resident** while reducing 151.8M edge rows. 8 GB will swap hard or die. |
| **Disk** | 20 GB free | See the table below. The transient peak during GMD extraction is the binding constraint. |
| **GPU** | not needed | Every step here is CPU-only. A GPU matters for the real ablation run, which is not ready to start. |

### Disk, exactly

| item | size | notes |
|---|---|---|
| venv with CPU torch | ~1 GB *(estimated)* | The Windows torch wheel is 124 MB compressed. Measured here on Linux the venv was 5.8 GB, but that is CUDA wheels — Windows does not get those by default. |
| `data/raw/` connectome | 1.11 GB | 3 feather files; the weights table alone is 1.05 GB |
| `data/cache/` | 94 MB | 82 MB neuron graph + 12 MB subgraph |
| `data/egmd/` GMD | 6.60 GB extracted | from a 5.11 GB zip, deleted after extraction |
| **transient peak** | **11.71 GB** | zip and extracted tree coexist briefly |

### Python version

Checked against PyPI for `win_amd64` wheels on the current release of each
dependency:

| Python | torch | numpy / scipy | pandas / pyarrow | soundfile / pretty_midi | python-rtmidi *(optional)* |
|---|---|---|---|---|---|
| 3.11 | yes | older releases only | yes | yes | yes |
| **3.12 — recommended** | yes | yes | yes | yes | yes |
| 3.13 | yes | yes | yes | yes | **no wheel** |
| 3.14 | yes | yes | yes | yes | **no wheel** |

**Use Python 3.12.** It is the only version with a binary wheel for every
package including the optional MIDI one. 3.11 works — pip just resolves numpy
and scipy back a release or two, which is what ran here. On 3.13+ everything in
`requirements.txt` installs fine and only the deferred realtime extra would need
a compiler.

`soundfile` and `sounddevice` bundle their native libraries in the Windows
wheel, so there is no libsndfile or PortAudio to install separately. That is a
real difference from macOS and Linux.

---

## 3. Install Python and Git

Using winget, from PowerShell:

```powershell
winget install Python.Python.3.12
winget install Git.Git
```

Or download from [python.org](https://www.python.org/downloads/windows/) and
tick **"Add python.exe to PATH"** during the install.

Close and reopen PowerShell, then confirm:

```powershell
py -0p        # lists installed Pythons and their paths
git --version
```

`py -0p` should show a `3.12` entry. If typing `python` opens the Microsoft
Store instead of Python, turn off the alias: **Settings → Apps → Advanced app
settings → App execution aliases**, switch off `python.exe` and `python3.exe`.
Using the `py` launcher, as below, avoids the problem entirely.

---

## 4. Clone and create the environment

```powershell
git clone https://github.com/andrewevmiller/flybeats.git C:\dev\flybeats
cd C:\dev\flybeats
git checkout claude/quirky-turing-e735dk
```

> The `git checkout` is only needed until that branch merges — this guide and
> the test fix it describes live there. After it merges, `main` is enough.

A short clone path is tidy but **not** required: the longest path inside the GMD
archive is 71 characters (`groove/drummer5/session1/10_latin-brazilian-sambareggae_96_beat_4-4.wav`),
so Windows' 260-character limit only bites if you clone somewhere pathological.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Your prompt should now start with `(.venv)`. If PowerShell refuses with
*"running scripts is disabled on this system"*:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

That lasts for the current window only. From `cmd` instead of PowerShell, use
`.\.venv\Scripts\activate.bat`, which needs no policy change.

**Re-activate the venv in every new terminal.** Nothing below works without it.

---

## 5. Install the dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements-dev.txt
```

`requirements-dev.txt` pulls in `requirements.txt` plus pytest. Installing plain
`requirements.txt` also works, but then step 8 fails with
`No module named pytest` — pytest is not a runtime dependency.

Confirm:

```powershell
python -c "import torch, numpy, pandas, pyarrow, scipy, yaml, soundfile, pretty_midi; print(torch.__version__)"
```

### If you have an NVIDIA GPU

`pip install torch` on Windows installs the **CPU-only** build. This differs
from Linux, where the same command pulls CUDA wheels — which is why the venv
measured 5.8 GB here and will be far smaller for you.

Nothing in this guide needs CUDA. If you want it anyway, install torch first
from the CUDA index URL that
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) currently
shows for your CUDA version, then run the `pip install -r` above; it will leave
the CUDA build in place. Check it took:

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

`configs/v1_8piece.yaml` sets `device: auto`, which resolves to CUDA when it is
available and CPU otherwise, so nothing needs configuring either way.

---

## 6. Download the connectome

```powershell
python scripts/fetch_data.py
```

1.11 GB into `data\raw\`, as three files:

```
     14.5 MB  body-annotations-male-cns-v1.0-minconf-0.5.feather
     43.3 MB  body-neurotransmitters-male-cns-v1.0.feather
   1051.2 MB  connectome-weights-male-cns-v1.0-minconf-0.5.feather
```

This is **mandatory**, not optional. The synthetic dataset removes the need for
the drum corpus, but the network topology comes from the real wiring diagram, so
nothing past the unit tests runs without these files.

The script skips files already downloaded, but it does **not** resume a partial
one. If it dies mid-transfer, delete the `.part` file in `data\raw\` and re-run.

---

## 7. Run the Phase 0 gate

```powershell
python scripts/verify_types.py
```

**3.9 seconds, 0.65 GB peak.** It resolves every cell-type name the project
depends on against the real v1.0 annotation table and writes
`data/verified_types.json`. The tail of the output:

```
[OK  ] pC1                      48 types     155 bodies   e.g. pC1_16b, pC1_19, pC1_1a
[OK  ] pIP10                     1 types       2 bodies   e.g. pIP10
[OK  ] wing_steering_MN         15 types      30 bodies   e.g. b1 MN, b2 MN, b3 MN, hg1 MN
[OK  ] wing_motor_all           25 types      66 bodies   e.g. DLMn c-f, DVMn 1a-c
```

Every line should say `[OK ]`. The file it writes is already committed to the
repo, so `git status` will show it as modified afterwards — the only differences
are the `generated` date and a `neuprint_crosscheck` block. Discard it with
`git checkout data/verified_types.json` to keep your tree clean.

---

## 8. Run the tests

```powershell
pytest -q
```

**11 seconds** on a warm cache. The first run after step 6 takes a minute or
two longer, because two of the tests build small subgraphs of their own and
cache them. Expected, with the connectome present and no drum corpus:

```
53 passed, 1 skipped
```

The skip is the GMD-dependent pipeline test; it runs once step 11 is done. If
you run the tests before step 6, you get **43 passed, 11 skipped** — also
correct. Both counts were verified.

A failure here means something is wrong with the install, not with the project.

---

## 9. Build the subgraph

```powershell
python src/subgraph.py
```

**52.7 seconds, 5.74 GB peak RAM** on the first run, because it reduces the
151.8M-row weights table to the neuron-level graph and caches that. Close your
browser first on a 16 GB machine. Later runs reuse the cached graph and take
**8.2 seconds at 0.75 GB**.

Expected output:

```
Graph: 162,517 neurons, 25,120,209 edges, 122,181,879 synapses | signs +104,778 / -56,871 / mod 868 | 2,202 inferred
  forward 3-hop: 154,853 | backward 3-hop: 130,934 | intersection: 128,433
  trimmed to 30,000 (cap 30,000)
SubGraph: 30,000 neurons, 2,942,102 edges (density 3.27e-03, mean in-degree 98.1)
  roles: aPN1=24, motor=66, octopaminergic=25, pC1=155, pC2=140, pIP10=2, sensory=405, vPN1=11
```

Those numbers are deterministic — if yours differ, the connectome download is
incomplete.

---

## 10. Prove the pipeline runs

The unit tests exercise components. This trains a real model end to end —
encoder, subgraph, recurrence, decoder, metrics — on the built-in synthetic
click track, so it needs no drum corpus:

```powershell
python src/train.py --config configs/sanity_3piece_run.yaml
```

**~15 minutes**: 12 epochs at ~73 s each on 4 CPU cores. Expect a line per
epoch and this at the end:

```
best val onset F: 0.6733 -> C:\dev\flybeats\runs\sanity_3piece_run
```

That 0.67 is **not a result**. It is a click track with known onsets, and it
says only that the machinery works. Delete `runs\sanity_3piece_run` afterwards
if you like; the checkpoint has no lasting value.

Your install is now complete. Everything below is optional.

---

## 11. Optional: the drum corpus

```powershell
python scripts/fetch_egmd.py
```

5.11 GB zip → 6.60 GB extracted into `data\egmd\`, with the zip deleted
afterwards, so you need **11.71 GB free** while it runs. This fetch resumes
after an interruption, unlike step 6 — re-run the same command and it picks up
where it stopped.

The default corpus is Magenta **GMD** (`groove-v1.0.0`): 1,150 clips, 18 styles,
audio plus sample-aligned MIDI. `--corpus egmd` fetches E-GMD instead, whose
audio archive is 96 GB — the README explains why GMD is the default.

With the corpus present, `pytest -q` becomes **54 passed**, and the real-corpus
CPU run becomes available:

```powershell
python src/train.py --config configs/v1_8piece_cpu.yaml
```

Read [Results, and what they are not](README.md#results-and-what-they-are-not)
before quoting anything that run prints.

---

## 12. Optional: realtime MIDI

Deferred in section 1, and here is the detail. The realtime extras are commented
out of `requirements.txt`:

```powershell
pip install sounddevice mido python-rtmidi
```

`sounddevice` and `mido` are trivial — the Windows wheel bundles PortAudio.
`python-rtmidi` ships `win_amd64` wheels for **Python 3.11 and 3.12 only**. On
3.13+ pip falls back to building from source, which needs the
[MSVC Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
(the "Desktop development with C++" workload, several GB).

This is the other reason to prefer 3.12. The latency benchmark itself needs none
of it:

```powershell
python src/realtime.py --config configs/v1_8piece.yaml --benchmark
```

---

## 13. Verification checklist

| # | check | expected |
|---|---|---|
| 1 | `py -0p` | a 3.12 entry |
| 2 | prompt shows `(.venv)` | after `Activate.ps1` |
| 3 | `python -c "import torch; print(torch.__version__)"` | a 2.x version |
| 4 | `dir data\raw` | 3 files, ~1.11 GB total |
| 5 | `python scripts/verify_types.py` | every line `[OK ]` |
| 6 | `pytest -q` | `53 passed, 1 skipped` |
| 7 | `python src/subgraph.py` | `SubGraph: 30,000 neurons, 2,942,102 edges` |
| 8 | `python src/train.py --config configs/sanity_3piece_run.yaml` | `best val onset F: 0.67` |

---

## 14. Troubleshooting

| symptom | cause | fix |
|---|---|---|
| `running scripts is disabled on this system` | PowerShell execution policy blocks `Activate.ps1` | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use `activate.bat` from `cmd` |
| typing `python` opens the Microsoft Store | Windows app execution alias | turn off the `python.exe` alias in Settings, or use `py -3.12` |
| `ERROR: Could not find a version that satisfies the requirement torch` | ARM64 Windows — no `win_arm64` torch wheel exists | no fix; this project needs an x64 machine |
| `No module named pytest` | pytest is not in `requirements.txt` | `pip install -r requirements-dev.txt` |
| `ModuleNotFoundError: No module named 'build'` (or `encoder`, `model`) | The modules in `src/` import each other by bare name, which relies on `src/` being the script's own directory | run `python src/train.py` from the repo root — **not** `python -m src.train`, which does not work |
| `FileNotFoundError: ...body-annotations-...feather` | step 6 not done, or a partial download | delete any `.part` in `data\raw\`, re-run `python scripts/fetch_data.py` |
| the machine swaps or the build is killed during `src/subgraph.py` | the 5.74 GB peak | close other applications; 8 GB of RAM is genuinely marginal here |
| `fetch_data.py` stalls partway | no resume support in that script | delete the `.part` file and re-run |
| `fetch_egmd.py` stalls partway | — | just re-run it; it resumes by design |
| a DataLoader worker crashes on startup | `configs/v1_8piece.yaml` sets `workers: 2`, and Windows spawns rather than forks, so workers re-import the module | `src/train.py` has the `if __name__ == "__main__":` guard this needs, so it should work — if it does not, set `workers: 0` in the config |
| `torch.cuda.is_available()` is `False` with an NVIDIA GPU present | the default PyPI wheel on Windows is CPU-only | reinstall torch from the CUDA index URL on pytorch.org |
| `python-rtmidi` fails to build | no wheel for your Python version | use Python 3.12, or install the MSVC Build Tools |
| `git status` shows `data/verified_types.json` modified | the gate rewrites its own committed output with a new date | `git checkout data/verified_types.json` |

---

## 15. What lives where, and how to remove it

Everything the install writes is inside the clone, and all of it is gitignored:

```
C:\dev\flybeats\
  .venv\                 the environment            ~1 GB
  data\raw\              connectome feathers        1.11 GB
  data\cache\            neuron graph + subgraph    94 MB
  data\egmd\             GMD corpus (optional)      6.60 GB
  runs\                  checkpoints and history    small
```

Nothing is installed outside the clone except Python and Git themselves, and
pip's download cache in `%LOCALAPPDATA%\pip\Cache`. To uninstall, delete the
folder. To free space without starting over, delete `runs\` and `data\cache\`;
both rebuild from what remains.

---

## 16. Keeping up with the repo

```powershell
cd C:\dev\flybeats
git pull
pip install -r requirements-dev.txt   # in case dependencies moved
python src/subgraph.py                # rebuild the cache, 8 s
python src/train.py --config configs/v1_8piece_cpu.yaml
python scripts/diagnose.py --checkpoint runs/v1_8piece_cpu/best.pt
```

No re-downloading: `data\raw\` and `data\egmd\` are frozen public datasets.

Read [Picking this up again](README.md#picking-this-up-again) first, and
**export a bundle from any run worth keeping** — `runs/` is gitignored, so a
bundle is the only durable form a trained model has:

```powershell
python scripts/export_bundle.py --checkpoint runs/v1_8piece_cpu/best.pt
```

`scripts/diagnose.py` is how you tell a model that is learning from one that is
not: watch `gain over constant` staying positive and `|corr(pred, target)|`
well off ~0.03. Per-epoch, `velocity_r` is the one to watch for the velocity
head — a head that has collapsed to a single constant velocity still scores a
respectable MAE, because drummers are not that dynamic.
