# The CUDA path, verified — 15 September 2026

Follows `2026-09-14-gpu-bringup.md`, which ended with one command left to run.
It ran. This is what the card said.

Companion evidence: `bootstrap-LAPTOP-I9B30DJE-20260915-0030.json` in this
directory. Where the two disagree, the JSON is the measurement.

**Outcome: unblocked.** `137 passed, 1 skipped` — the number RUNBOOK.md predicts
for a working CUDA install, reached for the first time. Three of the four
reasons `tests/test_cuda.py` exists are now verified on real silicon; the
fourth was a genuine bug and is fixed. The smoke test still has two failures,
both newly reachable and neither caused by these changes (§6).

---

## 1. The wheel

`torch 2.14.0+cu126`, installed in 245 s. Chosen, not guessed:

- The 14 Sep note could not reach `pytorch.org` from the cloud container. From
  this machine it resolves fine — that block was an artifact of the container,
  not a property of the network.
- The selector offers **cu126 / cu130 / cu132** for Windows, but only cu126 and
  cu130 carry a `torch-2.14.0-cp312-win_amd64` wheel at all.
- cu126 over cu130 because CUDA 13.x dropped older architectures and Turing sits
  at that edge. Confirmed after the fact: `get_arch_list()` is
  `['sm_50','sm_60','sm_61','sm_70','sm_75','sm_80','sm_86','sm_90']`. **`sm_75`
  is compiled in.** This was never a missing-kernel problem.

`torch.capability` reads **7.5**, from the device. §2 of the 14 Sep note inferred
this from secondary sources because NVIDIA's table was unreachable. The
inference was correct.

---

## 2. What broke, and why

The bring-up failed on one test:

```
RuntimeError: Sparse operations with CUDA tensors of BFloat16 type are not
supported on GPUs with compute capability < 8.0 (current: 7.5)
```

Three independent things had to line up, and no single one of them is the bug:

```
configs/v1_8piece.yaml   train.bf16: true
  -> src/train.py        torch.autocast("cuda", dtype=torch.bfloat16)
  -> src/model.py:53     torch.sparse.mm(csr, r.t())   <- operands are fp32
  -> aten::_sparse_mm    no autocast registration
  -> aten::_sparse_addmm no autocast registration
  -> aten::addmm         AutocastCUDA = TRUE, lower_precision_fp
  -> casts BOTH operands to bf16, not knowing one is a sparse CSR
  -> bf16 CSR kernel hits the hard cc < 8.0 gate
```

**Nothing in this repository ever asks for bf16.** Both operands arrive fp32, and
`model.py` already casts the encoder drive back to `v.dtype` on the way in. The
cast happens inside the C++ dispatch: `_sparse_addmm` forwards to `at::addmm`,
which re-enters the dispatcher and meets an autocast policy that has no concept
of sparsity. The clincher is that the same fp32 inputs under *fp16* autocast
return a `float16` result — autocast, and only autocast, produced that dtype.

### Why it did not skip

`tests/test_cuda.py` guards with `torch.cuda.is_bf16_supported()`. The 14 Sep
note predicted this returns `True` on Turing. It does, and the mechanism is now
exact:

```python
def is_bf16_supported(including_emulation: bool = True):
    ...
    if torch.cuda.get_device_properties(device).major >= 8:
        return True
    if not including_emulation:
        return False
    return _check_bf16_tensor_supported(device)   # can a bf16 tensor be ALLOCATED
```

The default takes the emulation branch, where the entire test is whether a
one-element bf16 tensor can be allocated. It can. The function answers "can I
allocate this"; the guard read it as "can I compute with this".

| call | result |
|---|---|
| `is_bf16_supported()` | `True` |
| `is_bf16_supported(including_emulation=False)` | `False` |

---

## 3. What the hardware actually says

Measured on this card, not argued.

### bf16 is not slow here — it is *absent*, and only for sparse

| dtype | dense `mm` | sparse CSR | sparse COO |
|---|---|---|---|
| fp32 | ok | ok | ok |
| bf16 | **ok** | gated, cc < 8.0 | not implemented |
| fp16 | ok | **ok** | not implemented |

This narrows the 14 Sep claim. bf16 is *not* dead on Turing — dense bf16 runs
forward and backward with finite grads. It is unusable **for this model**,
because this model's core operation is sparse. Sparse bf16 is blocked twice
over: a hard capability gate in CSR, and a plain dtype-coverage hole in COO.

The gate is on the operation, and autocast cannot dodge it. `sparse fp32 @ dense
bf16` and `sparse bf16 @ dense fp32` both fail, so keeping the sparse matrix in
fp32 while activations go bf16 does not help — and that mixed shape is exactly
what a backward pass presents.

### The throughput numbers that settle it

2048^3 GEMM, TF32 off:

| | time | throughput |
|---|---|---|
| fp32 | 4.626 ms | 3.71 TFLOP/s |
| **fp16** | **0.674 ms** | **25.48 TFLOP/s** |
| bf16 | 6.494 ms | 2.65 TFLOP/s |

**bf16 is 1.4x slower than fp32 on this card**, not merely unaccelerated. End to
end agrees: 0.458 s per epoch fp32 against 0.519 s bf16, ~13% worse. §2 of the
14 Sep note proposed settling this by timing an epoch each way. That is now
moot twice: it crashed, and forced to run it loses.

### The result that changes the roadmap

Sparse CSR **fp16 works**, and is numerically sound — against an fp32 reference,
forward rel-err 3.6e-4, backward 3.1e-4, all finite.

The 14 Sep note filed a real `precision:` setting plus a `GradScaler` under
*"Proposed only, pending a measurement that says the throughput is there to
win."* That measurement now points yes: the sparse path executes in fp16 on a
card with genuine fp16 tensor cores, at 6.9x the fp32 GEMM rate.

**The honest caveat:** 6.9x is dense GEMM at 2048^3, not this model end to end.
The model is sparse, small (2,000 neurons in the fixture), and bounded by a
1,600-step recurrence. Nobody should bank a number until it is measured on the
real thing. What has changed is that the *blocker* is gone — it is an
engineering question now, not a hardware one.

Two constraints on building it: **CSR throughout** (COO fp16 is unimplemented),
and **both operands fp16** at the sparse op — `sparse fp16 @ dense fp32` throws
a cuSPARSE type-combination error. `autocast(cuda, float16)` handles this.

---

## 4. The fixes

### Calibration accumulators were allocated on the wrong device

`encoder.calibrate` moved its *input* to the buffers' device and then summed it
into accumulators built with a bare `torch.zeros(...)`, i.e. on CPU. Identical
on CPU, fatal on GPU. This blocked three of the four GPU tests before any of
them reached the code they were written to exercise.

### `SparseSpMM` now pins its own operands to fp32

`@torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)` and the
matching `custom_bwd`. This is the sanctioned mechanism for a custom
`autograd.Function` under autocast, and `SparseSpMM` previously had no autocast
annotation at all. The dense half of the model still autocasts; only the sparse
island is pinned.

Verified: gradients finite, non-zero, worst relative deviation from fp32 of
**0.0062** against the test's 0.25 band — and bit-identical to the old code when
autocast is off, so it is a true no-op in fp32.

It buys no speed. What it buys is that `train.bf16: true` degrades to "runs
without benefit" instead of "hard crash on every pre-Ampere card", and that the
bf16 path can be *tested* on this machine rather than skipped.

### `train.bf16` accepts `auto`, and `v1_8piece.yaml` now uses it

`auto` resolves against compute capability at load: bf16 only at 8.0 and up.
`true` and `false` still force it.

Plain `false` was the wrong fix. It is right for this card and wrong for the
next one — an Ampere box deserves bf16 from the same config. `auto` is also not
the timid option: given that emulated bf16 is *slower* than fp32 here, it is the
faster one on both machines.

### The test guard was left alone, deliberately

`including_emulation=False` is a correct one-line change and it is the wrong
one. With `SparseSpMM` pinned, the bf16 path is genuinely executable on this
card — the test now runs and passes. Tightening the guard would skip it on
every pre-Ampere card and discard the only bf16 coverage this project has ever
had. The guard now carries a comment saying so, because the "fix" is otherwise
an obvious-looking improvement someone will make later.

---

## 5. Results

| | 14 Sep (CPU wheel) | 15 Sep (cu126) |
|---|---|---|
| `pytest -q` | 133 passed, 1 skipped, 4 deselected | **137 passed, 1 skipped** |
| `pytest -q -m gpu` | 4 deselected | **4 passed** |
| bf16 autocast | `SKIP — CUDA only` | passes on a 7.5 card |

Thread timing moved, and the shape is worth keeping:

| threads | per 20 ms block | realtime |
|---|---|---|
| 1 | 14.8 ms | 1.35x |
| 8 (torch's default) | 17.2 ms | 1.16x |
| 16 | 12.4 ms | 1.61x |

**torch's default of 8 is the worst of the three measured** — slower than a
single thread. This is the contention inversion RUNBOOK.md describes, and it is
the exact number the 14 Sep reporting-gap fix was written to expose. Set
`OMP_NUM_THREADS=16` for the live path.

---

## 6. Bugs this exercise found and did *not* fix

All three are newly reachable — the CUDA path had never executed. None is caused
by the changes in §4.

### `cuda_smoke.check_bf16` compares two runs over different data

It reports `worst relative drift 1.912` and fails its own 0.25 band. That number
is not a bf16 measurement. Running the **same config twice in fp32** under the
same method gives a worst relative difference of **1.533** — the check cannot
distinguish dtype error from run-to-run data variation.

Cause: `tests/test_cuda.py::_grads` calls `T.reseed_loader(loader, 0)` before
each run; `cuda_smoke.py`'s `one_step` does not, so the two runs see different
synthetic clips. The remedy is the one line the test suite already has.

This is the §4 pattern from the 14 Sep note for a third time: **the detector
reproducing the failure it exists to catch.** Its own summary line — "each one
is a wrong number that still looks like a right number" — describes itself here.

### The streaming path has never run on GPU

```
RuntimeError: Expected all tensors to be on the same device, but got weight is
on cuda:0, different from other tensors on cpu
```

`realtime.benchmark` builds numpy blocks and pushes them into a model that is on
`cuda`; `StreamingDrummer` never learns the model's device. Same class as the
`encoder.calibrate` bug in §4, different code path. It means **no live-path
latency number on GPU exists yet**, which is the measurement open item 2 wants.

### `torch.sparse.mm` is autocast-cast transitively

`aten::_sparse_mm` and `aten::_sparse_addmm` carry no autocast registration; they
are caught through `aten::addmm`. Arguably a torch bug — the autocast policy
cannot see that an operand is sparse. Recorded because it is the root cause in
§2 and because it will behave this way again on the next sparse op added.

### A methodological trap, recorded so it is not repeated

A `TorchDispatchMode` **cannot** be used to observe autocast casting. The Python
dispatch key sits above `AutocastCUDA`, so re-invoking the op from inside the
mode dispatches *beneath* autocast and skips it. A first probe using one
reported fp32 operands and no error under bf16 autocast — a clean, plausible,
entirely wrong result. Exactly the failure mode open item 1 of the 14 Sep note
warns about.

---

## 7. Still open

1. **Peak VRAM at the 30k tier against 6 GB.** `bf16: auto` resolves to fp32
   here, so the run no longer crashes on precision — but it is still blocked,
   on data. **Correction to an earlier reading of this line:** the subgraph
   cache is on disk (`data/cache/subgraph.npz`), the corpus is *not*.
   `data/egmd/groove/` holds 136 KB — seven MIDI files, no `info.csv`, no
   audio — and the 4.8 GB is `groove-v1.0.0.zip`, unextracted. `build_dataset`
   raises without `info.csv`. Extract it before anything here is runnable.
   Order if it does not fit is unchanged: `grad_checkpoint: true` -> lower
   `tbptt_steps` -> `v1_8piece_cpu.yaml` at 10k.

   Separately, VRAM is very unlikely to be the constraint: peak measured at the
   30k tier is 885 MB of 6 GB at B=4. Wall clock is (see §8 of the plan).
2. **Live-tier latency at 30k.** Blocked on the streaming device bug in §6 for
   the GPU figure. The CPU figures above still stand.
3. **Whether an fp16 `precision:` path earns its build.** The hardware answer is
   in (§3); the model-level answer is not.
4. **The two smoke-test failures in §6.** One is a one-line fix in the detector;
   the other is a real bug in the live path.

Item 3 of the 14 Sep list — *whether bf16 earns its place on Turing* — is
**closed. It does not.** It crashes, and when made to run it is slower than fp32.
