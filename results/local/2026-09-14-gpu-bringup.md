# GPU bring-up on the local machine — 14 September 2026

Record of the first attempt to move GPU testing off shared cloud cores and onto
real hardware. Covers the machine that turned up, a verification of what its
card can and cannot do, and what the first bootstrap report actually said.

Companion evidence: `bootstrap-LAPTOP-I9B30DJE-20260914-2359.json` in this
directory. Where the two disagree, the JSON is the measurement and this is the
reading of it.

**Outcome: not yet unblocked.** The bring-up ran end to end and passed, but on a
CPU-only torch wheel, so the card was never exercised. One command remains.

---

## 1. The machine

| | |
|---|---|
| OS | Windows 11, AMD64 |
| CPU | AMD Ryzen mobile (Family 23 Model 96), 16 logical cores |
| RAM | 42.3 GB |
| Python | 3.12.10 |
| GPU | **NVIDIA GeForce RTX 2060 with Max-Q Design**, 6144 MiB |
| Driver | 616.56 |

Every Windows-specific path in `scripts/bootstrap_local.py` had been written
without a Windows machine to run it on — `Scripts\python.exe`, the wmic /
PowerShell RAM probe, the `nvidia-smi` CSV parse. **All of them worked.** That
caveat is discharged.

Three properties of this machine matter more than the rest:

- **6 GB of VRAM** is the binding constraint, not compute. `v1_8piece.yaml` is
  the 30k-node tier, which has never been trained at any size, and it ships
  `grad_checkpoint: false`.
- **Turing**, which decides whether `train.bf16: true` is worth anything —
  see §2.
- **16 logical cores and 42 GB**, four times the cloud container, which makes
  the thread-scaling measurement meaningful for the first time.

---

## 2. Verified: Max-Q is a power envelope, not an architecture

The question was whether "RTX 2060 Max-Q" differs architecturally from a
desktop RTX 2060. It does not:

- Same **TU106** silicon as the desktop RTX 2060, at lower clocks and TGP for
  thermals; same 6 GB GDDR6.
- All Turing GeForce parts sit at **CUDA compute capability 7.5**, whichever
  die (TU102 / TU104 / TU106) and whatever the board is called, laptop and
  Max-Q variants included.

**Verification caveat.** This container's egress proxy blocks
`developer.nvidia.com`, `docs.nvidia.com`, TechPowerUp and Wikipedia, so
NVIDIA's primary compute-capability table was not reachable. The above rests on
consistent secondary sources. The authoritative answer for this machine is
`torch.capability` in the bootstrap report — which cannot be read yet, because
torch has no CUDA (§3).

### The consequence: `bf16` is probably worthless here

**bfloat16 has no hardware support below compute capability 8.0 (Ampere).**
Turing has fp16 tensor cores and no bf16 ones. `configs/v1_8piece.yaml` ships
`train.bf16: true`.

The trap worth knowing before reading any future report: **`torch.cuda.is_bf16_supported()`
can still return `True` on Turing.** Recent PyTorch falls back to checking
whether a bf16 tensor can be created and operated on, which succeeds by
emulation. `bootstrap_local.py` prints that flag as `bf16 yes`. That line means
torch will *execute* bf16, not that the card *accelerates* it.

Recorded in `RUNBOOK.md` and on the config's `bf16` line
([`79aabe8`](https://github.com/andrewevmiller/flybeats/commit/79aabe8)). The
resolution is a measurement, not an argument: time one epoch each way below 8.0,
and set `bf16: false` if it is not faster.

**Not built:** capturing Turing's fp16 tensor cores would need a real
`precision:` setting and a `GradScaler`, because `train.bf16` is a bool and
`run_epoch` reads it as bfloat16 or nothing. Proposed only, pending a
measurement that says the throughput is there to win.

---

## 3. The first report: a CPU wheel

```
torch          2.14.0+cpu
cuda_build     null
cuda_available false
```

The install ran from the default index (112 s), which on Windows is CPU-only.
The card was never touched. So:

- every timing in the report is a **CPU timing**;
- the four GPU tests were **deselected, not run**;
- the bf16 check reports `SKIP — CUDA only`.

This was the predicted failure, and `bootstrap_local.py` exists specifically to
catch it. It caught it, and then undercut itself — see §4.

### What the report does establish

**Windows parity.** `133 passed, 1 skipped, 4 deselected` in 74.55 s —
identical to CI and to the cloud container. The platform breaks nothing.

**The smoke test passes on CPU.** Sparse SpMM forward agreed with dense to
1.55e-15 with duplicate edges forced; gradient checkpointing was transparent to
0.00e+00 across 13 tensors; one training epoch ran end to end (loss 1.7484,
0.8 s for 4 clips).

### Reading the two thread figures correctly

They are **different models**, not a contradiction:

| source | model | result |
|---|---|---|
| `cuda_smoke` streaming benchmark | 2,000-neuron fixture | 4.4 ms mean / 5.5 ms p95, 4.6× realtime, 8 threads |
| `thread_timing` | 10,000-neuron fixture | 17.5 ms (1 thread) → 11.5 ms (16 threads) |

Neither is the 30k tier that would ship.

On the 10k model, **16 threads bought only 1.5×** over one thread, and 11.5 ms
is 1.74× realtime — inside the 20 ms budget, but with less headroom than is
comfortable. For contrast, this idle cloud container did the same model in
9.0 ms at one thread and 4.5 ms at four. **The mobile Ryzen is roughly 2× slower
per thread.** That is a real finding, and it cuts against the live-tier latency
claim at 30k, which still has to be measured on this box.

---

## 4. Bugs this exercise found

### `exit_code: 0` with the GPU step `FAILED`

The report contains `"torch sees the GPU": "FAILED"` and `"exit_code": 0` in the
same file. The exit code was computed from the test suite and the smoke test
only, so a card present but invisible to torch — *the exact silent failure the
script was written to catch* — exited clean.

Reproducing the failure inside the detector is worse than not having the
detector. Fixed in
[`54be91c`](https://github.com/andrewevmiller/flybeats/commit/54be91c) and
verified across all three cases:

| GPU present | torch sees it | exit |
|---|---|---|
| yes | no | **1** |
| yes | yes | 0 |
| no | — | 0 (CPU-only is supported, not a fault) |

### `--index-url` silently did nothing

`pip install torch --index-url <cuda>` is a no-op when a CPU torch is already
installed: the CUDA wheel is `2.x.y+cuNNN`, the CPU one is plain `2.x.y`, so pip
reads the bare `torch` requirement as satisfied and changes nothing. The user
asks for CUDA, pip reports success, `cuda_available` stays `False`.

The script's own remedy reproduced the failure it was written to prevent, and it
was reachable in practice because the script reuses an existing `.venv`. Fixed
with `--force-reinstall --no-cache-dir` on the explicit-index path
([`bfe0f17`](https://github.com/andrewevmiller/flybeats/commit/bfe0f17)).

### Two reporting gaps

- `cuda_smoke.py` recorded a bare `ok` after a CPU-only run, in which the bf16
  check does not execute at all. It now says *"ran on CPU — the CUDA path is
  still unverified"*, so the report cannot be read as a GPU clean bill.
- `thread_timing` measured 1 and `cpu_count`, which on this 16-thread SMT
  machine meant 1 and 16. **torch defaulted to 8, and 8 was never measured** —
  the one number you get by setting nothing was the one number missing. It now
  measures torch's default too and names the fastest count.

---

## 5. Repository state

`main` at [`bbe377d`](https://github.com/andrewevmiller/flybeats/commit/bbe377d).

| commit | |
|---|---|
| [`1a7888c`](https://github.com/andrewevmiller/flybeats/commit/1a7888c) | Moved GPU-dependent testing to the machine with a GPU (prior work) |
| [`bfe0f17`](https://github.com/andrewevmiller/flybeats/commit/bfe0f17) | Force the torch reinstall when `--index-url` is given |
| [`79aabe8`](https://github.com/andrewevmiller/flybeats/commit/79aabe8) | Record what compute capability 7.5 means for `train.bf16` |
| [`c160526`](https://github.com/andrewevmiller/flybeats/commit/c160526) | The hardware report itself, committed from the Windows machine |
| [`54be91c`](https://github.com/andrewevmiller/flybeats/commit/54be91c) | Exit non-zero when a card is present and torch cannot see it |
| [`bbe377d`](https://github.com/andrewevmiller/flybeats/commit/bbe377d) | Merge |

The report was committed from Windows and from the cloud session simultaneously,
producing an add/add conflict that was **CRLF vs LF only** — content identical.
Resolved keeping the Windows commit, with a scoped `.gitattributes`
(`results/local/*.json text eol=lf`) so it cannot recur.

---

## 6. Next action

```powershell
cd "C:\Users\ricos\Documents\AI Databases\flybeats\git"
git checkout main
git pull origin main
py -3.12 scripts\bootstrap_local.py --index-url <url from pytorch.org's selector>
```

Pull first: without `bfe0f17` the reinstall no-ops on the existing venv.

Take the index URL from pytorch.org's selector rather than a hardcoded one —
which `cuNNN` builds torch 2.14 ships could not be verified from here
(`pytorch.org` and `download.pytorch.org` are both blocked by this container's
proxy). The driver is not the constraint: 616.56 and compute capability 7.5 are
supported by every current CUDA build, so whichever option the selector offers
will work.

---

## 7. Still open

1. **The CUDA path is entirely unverified.** Three of the four reasons
   `tests/test_cuda.py` exists remain untested on a real device: the
   hand-written `SparseSpMM` backward, bf16 autocast, and GPU/CPU forward
   agreement. Each fails by producing a plausible number rather than an error.
2. **Peak VRAM at the 30k tier against 6 GB.** The most likely thing to force a
   config change. `cuda_smoke.py`'s default run is a 2,000-neuron correctness
   sizing and does not answer this; `--config configs/v1_8piece.yaml` does, and
   needs the subgraph cache and corpus on disk. If it will not fit, the order is
   `grad_checkpoint: true` → lower `tbptt_steps` → `v1_8piece_cpu.yaml` at 10k.
3. **Whether `bf16` earns its place on Turing** (§2) — one epoch timed each way.
4. **Live-tier latency at 30k on this machine.** The 10k figures suggest less
   headroom here than the cloud numbers implied.

Items 1–3 unblock the moment a CUDA wheel lands.
