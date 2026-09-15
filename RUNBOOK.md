# Running flybeats on your own machine

For the machine with the NVIDIA card. Everything here is blocked, slow or
impossible in a cloud session, and most of it is an evening locally.

## Why this document exists

The cloud container **only runs while a session is active**. It starts when
something wakes the session and stops shortly after the turn ends. Measured on
14 September: training attempts began at 12:40:51, 12:51:31 and 13:14:15, each
within seconds of a wake, with `uptime` reading 0 min at every check. An epoch
takes about 7 minutes there, so a turn that starts training and returns gives
it roughly one. Between 08:23 and 12:24 — four hours, with a queue nominally
"running" — **not one epoch finished**.

A background job in that environment is not a background job. It is a
foreground job that dies with the turn. That is the whole argument for doing
the long work here.

Three things follow from having real hardware, in the order they pay off:

1. **Phase B′ can finish** — the full-corpus run that produces the first model
   worth shipping.
2. **Phase D becomes reachable** — the ablation campaign that answers whether
   the connectome's topology earns its place. It is the project's central
   question and it has never been run.
3. **The untested paths get tested** — live audio in, MIDI out, and SETUP.md on
   an actual Windows machine.

---

## Start here: one command

From the clone — including the one at
`C:\Users\ricos\Documents\AI Databases\flybeats\git`, which may be behind:

```powershell
cd "C:\Users\ricos\Documents\AI Databases\flybeats\git"
git pull origin main
py -3.12 scripts\bootstrap_local.py
```

It reports the machine, finds the card via `nvidia-smi`, checks whether the
clone is behind `origin/main`, creates the venv, installs torch and the
requirements, runs the suite, runs the CUDA smoke test, measures thread scaling,
and writes a JSON report to `results/local/`. `--check` does the inspection
without installing anything; `--pull` updates the clone first.

**Commit the report.** Every number in this repository came off four shared
cloud cores. There is no record of what real hardware does with any of it, and
that is the first gap this machine closes.

If the bootstrap reports `cuda_available=False` while `nvidia-smi` sees the
card, it is the wheel, and nothing after that point is worth reading until it
is fixed. Re-run it with the index for a CUDA build your driver supports
(pytorch.org's selector names the current one):

```powershell
py -3.12 scripts\bootstrap_local.py --index-url https://download.pytorch.org/whl/cu124
```

Given `--index-url`, the script forces the reinstall. It has to: the CUDA wheel
is version `2.x.y+cuNNN` and the CPU one is plain `2.x.y`, so a bare `torch`
requirement reads as already satisfied and a plain re-run would leave the CPU
wheel exactly where it was.

---

## What testing looks like once it lives here

| command | what it covers | where it can run |
|---|---|---|
| `pytest -q` | everything: 134 CPU tests + 4 GPU tests | this machine only |
| `pytest -q -m "not gpu"` | exactly what CI runs | anywhere |
| `pytest -q -m gpu` | the four that need a card | this machine only |
| `python scripts/cuda_smoke.py` | the same GPU checks plus timings, VRAM and thread scaling | this machine only |

Counts, so a surprise is legible: **133 passed, 5 skipped** on a machine with
no card and no corpus; **137 passed, 1 skipped** with a working CUDA install
(the remaining skip needs GMD); **138 passed** once the corpus is downloaded.
If the GPU tests skip on the machine that has the card, the wheel is wrong.

The GPU tests (`tests/test_cuda.py`) skip themselves without a card, so CI and
cloud sessions stay green while simply never exercising them. That asymmetry is
the point: **this machine is now the only place the full suite exists.** A
change that breaks the CUDA path will pass CI and fail here, which is why the
suite wants running here before anything long starts.

What they check, and why each one is worth a test rather than trust:

- **The sparse backward, with duplicate edges present on purpose.** It is
  hand-written precisely because torch's CSR autograd returns a gradient sized
  to the *deduplicated* values when an edge list repeats a pair — which is what
  the Phase 4 rewiring arm emits. A wrong answer here trains, evaluates, and
  reports a plausible onset F.
- **bf16 gradients** finite, non-zero, and within a sane band of fp32.
  `train.bf16` is honoured only on CUDA, so no run in this repo has ever had it
  on.
- **Gradient checkpointing** changing nothing. It is a memory optimisation; if
  it moves the gradient, every result downstream is quietly wrong.
- **GPU and CPU forward passes agreeing** past float32 noise. Not bitwise —
  different kernels reduce in different orders — but a real disagreement means
  the device path computes something else entirely.

---

## Setup, the manual version

What the bootstrap does, in case you would rather do it yourself or it fails
partway. [SETUP.md](SETUP.md) is the fuller step-by-step, with disk, RAM and
timings.

```bash
git clone https://github.com/andrewevmiller/flybeats && cd flybeats
python -m venv .venv && . .venv/bin/activate        # Windows: .venv\Scripts\Activate.ps1
```

**Get a CUDA torch wheel.** The default differs by platform, and the failure is
silent — a CPU wheel installs cleanly and then `torch.cuda.is_available()` is
just `False`:

```bash
nvidia-smi                      # driver version and the CUDA it supports
# The index URL has to match a CUDA build your driver supports -- pytorch.org's
# selector gives the current one. cu124 is an example, not a recommendation:
pip install torch --index-url https://download.pytorch.org/whl/cu124 --force-reinstall
pip install -r requirements-dev.txt
```

On Linux the plain `pip install torch` gives you CUDA; on Windows it gives you
a CPU build, so the index URL is not optional there.

```bash
pytest -q                       # needs no downloads; see the counts below
```

That suite runs off committed fixtures, so a failure here is the install, not
the project. Then the data, once:

```bash
python scripts/fetch_data.py    # MaleCNS v1.0, 1.11 GB
python scripts/verify_types.py  # Phase 0 gate, ~4 s
python scripts/fetch_egmd.py    # GMD, 5.11 GB zip -> 6.60 GB extracted
python src/subgraph.py          # Phase 1 cache
```

---

## Step 0 — before booking any GPU time

```bash
python scripts/cuda_smoke.py
```

**Nothing in this project has ever executed on a GPU.** Every number in the
repository came off 4 CPU cores, and three pieces of the training path are only
*reached* on CUDA:

| check | why it could be wrong, silently |
|---|---|
| `SparseSpMM` forward + backward | The backward is hand-written, because torch's CSR autograd returns a gradient sized to the *deduplicated* values when an edge list repeats a (row, col) pair — which the Phase 4 rewiring arm produces. It has only ever been checked against a dense reference on CPU. |
| bfloat16 autocast | `train.bf16` is honoured on CUDA and ignored everywhere else, so it has never been on. |
| gradient checkpointing | The recomputed forward has to reproduce the first one closely enough to leave the gradient unchanged. |

Each of these fails by producing a number rather than an error: a mis-sized or
mis-scaled gradient still trains, still evaluates, and still prints a plausible
onset F. Forty hours into an ablation campaign is a bad place to discover one.
The smoke test is minutes, and exits non-zero if anything fails.

It also runs one real training epoch and one streaming benchmark, reporting
per-epoch time and peak VRAM — which is the number you want before sizing
anything below. Point it at the tier you actually intend to train:

```bash
python scripts/cuda_smoke.py --config configs/v1_8piece.yaml
```

### Measure your thread count; do not copy anyone else's

The same 10k model and the same 20 ms block, benchmarked on one cloud container
under two conditions:

| | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| container under load | 12.2 ms | — | 160.9 ms |
| same container, idle | 9.0 ms | 6.6 ms | 4.5 ms |

Under load, more threads cost 13×. Idle, they behave exactly as you would
expect and the 20 ms budget is met with room to spare. **The first reading was
contention, not a property of the code** — I recorded it as a finding before
re-measuring on an idle box, which was wrong.

What survives is narrower and more useful: this benchmark is extremely
sensitive to whatever else is running, so a number from someone else's machine
predicts nothing about yours. `scripts/cuda_smoke.py` measures it on yours, and
re-measures at one thread if the budget is missed — if that helps on your box,
`OMP_NUM_THREADS=1` (PowerShell: `$env:OMP_NUM_THREADS=1`) is the fix for the
live path; if it does not, leave the threads alone.

**Never run two training jobs at once on one box.** Sequential, always — and
the numbers above are the reason: the second job does not halve throughput, it
collapses it.

---

## Step 1 — Phase B′, the run that produces something shippable

Every training clip the corpus can supply, which is **846** — not the 897 that
ROADMAP.md still quotes. GMD indexes 897 train rows but ships 51 of them
MIDI-only, with `audio_filename` blank, and the loader drops those. (Checked
against the corpus on a sibling branch; no *named* audio file is missing from
disk in any split, so the shortfall is the dataset's rather than a broken
download.)

```bash
python src/train.py --config configs/v1_8piece.yaml
python scripts/diagnose.py --checkpoint runs/v1_8piece/best.pt
python scripts/export_bundle.py --checkpoint runs/v1_8piece/best.pt
```

Cost on CPU is 15–20 min/epoch, 3–4 h for 12 — an extrapolation from Phase A′
runs, not a measurement. On a GPU, time the first epoch and scale from that;
do not trust any projection in this repository for your hardware.

Note this config is the **30k-node tier, which has never been trained at any
size**. Every run so far was at 10k. If it will not fit, `configs/v1_8piece_cpu.yaml`
is the 10k version — raise `data.max_files` (or remove it) to use the whole
corpus.

**Export the bundle the moment it finishes.** `runs/` is gitignored and a
checkpoint is not durable; a bundle is ~10 MB, takes seconds, and
`export_bundle.py` verifies it reproduces the checkpoint's output exactly
before writing. Losing one costs the downloads, the graph build and the hours.

**Done when** onset F has either moved off 0.30 or provably stopped moving with
the data limit lifted. Either answer is worth having — one says the model was
data-starved, the other says 0.30 is the ceiling at this scale, and the second
would change what v0.1 promises.

---

## Step 2 — Phase D, the actual question

```bash
python src/ablations.py --config configs/v1_8piece.yaml --lesion --epochs 40 --seeds 5
```

That is **13 training runs**: one `real`, five `rewired`, five `sign_shuffled`,
one `gru`, one `shortcut`. Only the random-topology arms repeat, because only
they draw a topology — the others are the same model every time given the seed.

`--seeds` is not optional. `rewired` and `sign_shuffled` each draw *one* random
graph, so a single run cannot separate "random topologies do worse" from "this
draw was unlucky". **A gap smaller than the across-seed spread is not a
result** — and the project has already been burned by exactly this: a seed
change moved one class's velocity correlation from −0.28 to +0.32, with
non-overlapping intervals, on identical data and hyperparameters.

What the arms now share, which they did not before this week: identical batches
in identical order, identical training windows, fixed validation windows, and
one shared encoder calibration. The difference between two arms is topology.

Watch VRAM. 30k nodes is 2.94M edges at `tbptt_steps: 150`; if it will not fit,
`train.grad_checkpoint: true` trades about 30% more compute for roughly half
the per-chunk activation memory, and the smoke test has already proved it does
not change the gradient on your card.

---

## Step 3 — what only hardware can answer

None of this needs the GPU, and none of it can be done in a cloud session.

- **Live audio.** `sounddevice` in, `mido`/`python-rtmidi` out. Written,
  unexercised, never run against a real device. Start with
  `python src/realtime.py --bundle flybeats-8piece.fb --benchmark`, then live
  input with `OMP_NUM_THREADS=1`.
- **The lesion demo**, which is the best thing this project does:
  `--lesion pIP10` mid-performance, audibly.
- **SETUP.md on Windows.** Every figure in it was measured on Linux and the
  Windows-specific parts were checked against PyPI rather than executed.
- **`SoundFontBank`**, which needs FluidSynth and a sound card.

---

## What to keep

| artefact | why |
|---|---|
| **bundles** (`*.fb`) | The only durable form of a trained model. `runs/` is gitignored. |
| `runs/*/history.json` | Per-epoch loss and metrics; small, and the evidence behind any claim. |
| probe and diagnose output | Commit it, the way `scripts/run_velocity_queue.sh` already does — a number in a commit message with no output behind it is not evidence. |

Publishing a bundle as a GitHub release is what makes the README's quick start
followable by anyone. It is the cheapest thing on the roadmap that changes
whether the project is usable, and it needs a trained model and about an hour.

---

## When something goes wrong

| symptom | what it usually is |
|---|---|
| `torch.cuda.is_available()` is False | A CPU wheel. Reinstall from the CUDA index; `nvidia-smi` confirms the driver is fine. |
| CUDA out of memory | Lower `train.batch_size`, then `train.tbptt_steps`, then set `train.grad_checkpoint: true`. Drop to the 10k tier before giving up. |
| bf16 errors or looks wrong | `torch.cuda.is_bf16_supported()` is False on older cards. Set `train.bf16: false`; it is a speed option, not a correctness one. |
| Live path misses the 20 ms budget | Threads, nearly always. `OMP_NUM_THREADS=1` first, `--speed` second. Offline `--render` has no budget and is unaffected. |
| An epoch is 20× slower than expected | Two jobs on one box, or thread oversubscription. Never run two. |
| A metric moved and nothing else did | Check the seed before the science. `train.seed` now pins batch order, training windows and calibration; validation windows are fixed regardless. |
