# Setting up on a laptop (Windows)

Everything below was run end to end in a clean checkout before it was written,
on 4 CPU cores / 15 GB RAM / Python 3.11 — Linux, not Windows, so the timings
and sizes are real but the Windows-specific notes are flagged as such where they
have not been executed on Windows.

## What is stable and what will be replaced

Work is still landing on the training loop and the features. The setup splits
cleanly into a part that survives that work and a part that does not, so do the
slow half now and leave the fast half until the code settles.

**Stable — none of the pending work touches these:**

| step | why it survives |
|---|---|
| Python + venv + `pip install -r requirements.txt` | The deps are the core scientific stack. The two pending fixes are in `src/train.py` and `src/encoder.py`; neither adds a dependency. Later work (the spiking model, realtime on hardware) can *add* to `requirements.txt`, and re-running the same command picks that up. |
| `scripts/fetch_data.py` → `data/raw/` | MaleCNS v1.0 is a frozen public release. 1.11 GB, downloaded once, never invalidated by anything in this repo. |
| `scripts/fetch_egmd.py` → `data/egmd/` | Magenta GMD `groove-v1.0.0` is likewise frozen, and README pins the corpus to GMD rather than E-GMD. 5.11 GB, downloaded once. |
| `scripts/verify_types.py` → `data/verified_types.json` | The Phase 0 gate. Its output is already committed to the repo, and regenerating it takes 4 seconds, so it costs nothing either way. |

**Not stable — cheap to redo, so wait:**

| step | what invalidates it | cost to redo |
|---|---|---|
| `data/cache/subgraph*.npz` | The Phase 1 to-do is to *replace the pathway-strength trim* with a path-based criterion, which changes the subgraph the trim selects. | 8 s, once the neuron-graph cache exists |
| `runs/**/best.pt` and any training run | Both pending fixes change what the model converges to. Every checkpoint made before them is a constant predictor. | minutes on CPU |
| `configs/*.yaml` edits | `target_rate_hz` and the rate term are exactly what is being reworked. | — |
| `sounddevice` / `mido` / `python-rtmidi` | Commented out of `requirements.txt`, the hardest install on Windows, and useless until there is a checkpoint worth playing back. | — |

So: install the deps, pull both downloads, run the gate, run the tests. Stop
there. When the training fixes land, `git pull`, rebuild the subgraph cache, and
train — no re-downloading.

## Requirements

- **Python 3.11 or 3.12.** `torch>=2.2` and the audio deps have solid wheels for
  both. Avoid 3.13 unless you want to debug wheel availability.
- **~20 GB free disk.** Measured: venv 5.8 GB (Linux, where PyPI ships CUDA
  wheels — the default Windows wheel is CPU-only and much smaller), connectome
  1.11 GB, GMD 5.11 GB zip which needs roughly double that transiently while it
  extracts.
- **16 GB RAM.** The neuron-graph build peaks at **5.74 GB** resident, reducing
  151.8M edge rows. 8 GB will swap or die.
- No GPU needed for any of the steps below. A GPU only matters for the real
  ablation run, which is not ready to start.

## Steps

Clone somewhere with a short path — `C:\dev\flybeats`, not a deep folder under
`Documents`. GMD extracts to long nested filenames and Windows' 260-character
path limit is a real hazard there.

```powershell
git clone https://github.com/andrewevmiller/flybeats.git C:\dev\flybeats
cd C:\dev\flybeats

py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell refuses to run the activation script:
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or use
`.\.venv\Scripts\activate.bat` from `cmd`.

`pip install torch` on Windows gives the **CPU-only** build — unlike Linux,
where the same command pulls the CUDA wheels. If your laptop has an NVIDIA GPU
and you want to use it, install torch first from the CUDA index URL that
[pytorch.org/get-started](https://pytorch.org/get-started/locally/) currently
shows, then `pip install -r requirements.txt`.

Then the two downloads and the gate:

```powershell
python scripts/fetch_data.py     # 1.11 GB, MaleCNS v1.0 flat connectome
python scripts/verify_types.py   # 4 s -> data/verified_types.json
python scripts/fetch_egmd.py     # 5.11 GB, Magenta GMD
```

`fetch_egmd.py` resumes a cut transfer by itself. `fetch_data.py` does **not** —
if it dies partway, delete the `.part` file in `data\raw\` and re-run it.

Verify:

```powershell
pytest -q
```

Expected on a checkout with the connectome present and no drum corpus:
**53 passed, 1 skipped** (the skip is the GMD-dependent pipeline test; it passes
once `fetch_egmd.py` has run). With no downloads at all, 43 pass and 11 skip.

To prove the whole pipeline runs on your machine and not just the unit tests,
train the synthetic sanity config — no drum corpus needed, and it exercises
encoder, subgraph, recurrence, decoder and metrics end to end:

```powershell
python src/subgraph.py                                  # builds both caches
python src/train.py --config configs/sanity_3piece_run.yaml
```

12 epochs at ~73 s each on 4 CPU cores, so budget **~15 minutes**. It ends with
`best val onset F: 0.67`. That number is not a result — it is a click track with
known onsets, and it says only that the machinery works.

## When the training work lands

```powershell
git pull
pip install -r requirements.txt        # in case deps moved
python src/subgraph.py                 # rebuild the cache; 8 s
python src/train.py --config configs/v1_8piece_cpu.yaml
python scripts/diagnose.py --checkpoint runs/v1_8piece_cpu/best.pt
```

Read [Picking this up again](README.md#picking-this-up-again) first — the model
converges to a constant predictor today, and the diagnostic is how you tell
whether that has been fixed.

## Measured costs

| step | wall | peak RAM | on disk |
|---|---|---|---|
| `pip install -r requirements.txt` | — | — | 5.8 GB (Linux/CUDA wheels; less on Windows) |
| `scripts/fetch_data.py` | network-bound | — | 1.11 GB |
| `scripts/verify_types.py` | 3.9 s | 0.65 GB | 227 KB |
| `src/subgraph.py` (first run, builds both caches) | 52.7 s | 5.74 GB | 82 MB graph + 12 MB subgraph |
| `src/subgraph.py` (graph cache already built) | 8.2 s | 0.75 GB | 12 MB |
| `src/train.py --config configs/sanity_3piece_run.yaml` | 15 min (12 × 73 s) | — | — |
| `pytest -q` | 12 s | — | — |
| `scripts/fetch_egmd.py` | network-bound | — | 5.11 GB |

## Windows-specific things that will bite

- **Long paths.** Clone shallow, and if GMD extraction fails on a filename,
  enable long paths:
  `Set-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem' -Name LongPathsEnabled -Value 1`
  (needs admin and a reboot). Not exercised here.
- **DataLoader workers.** `configs/v1_8piece.yaml` sets `workers: 2`, and
  Windows spawns rather than forks, so the worker processes re-import the
  module. `src/train.py` has the `if __name__ == "__main__":` guard this needs;
  keep it in anything you add. Set `workers: 0` if you hit a spawn error.
- **`python-rtmidi`** builds from source when no wheel matches your Python
  version, which pulls in the MSVC Build Tools. Deferred above for that reason.
- **`neuprint.janelia.org`** is only needed for `verify_types.py --neuprint`,
  which has never been run anywhere. The default path uses the downloaded
  feathers and needs no token.
