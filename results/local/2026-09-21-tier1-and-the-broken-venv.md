# The venv that stopped matching itself, and the Tier 1 gate — 21 September 2026

Follows `2026-09-16-threshold-response.md`, whose code changes were still
uncommitted on this machine, and answers items 2 and 3 of
`handoff-2026-09-17.md`'s task order. No training was run: the card was left
alone by instruction, and everything here is CPU or code.

Evidence: `pytest -q -m "not gpu"` before and after, the DLL and wheel-RECORD
checks in §1, and `tests/` itself, which is where most of this session's work
ended up.

**Outcome: the project environment had been silently broken for five days, by
something that is not a flybeats bug at all and that no check in the repo
could see. Fixing it took ten minutes and finding it took the rest of the
hour, which is the argument for the guard now in `bootstrap_local.py`. Then
the Tier 1 test gate — the block `RELEASE_ROADMAP.md` calls the highest-value
thing on its page, and the one that lands before any GPU time — went in, and
its first test found a live bug: two of the five ablation arms could not do a
forward pass. The same failure, on a different keyword, as the one that caused
CI to be added in the first place.**

---

## 1. `import torch` failed, and the message was about torch

Every command in this repo runs through `.venv\Scripts\python.exe`. On
21 September it could not import torch:

```
ImportError: Failed to load PyTorch C extensions:
    It appears that PyTorch has loaded the `torch/_C` folder
    of the PyTorch repository rather than the C extensions ...
```

That message names a PyTorch source checkout, which this is not. What it
actually means is that loading `torch._C` raised, and torch guessed at why.

The elimination, in order:

| check | result |
|---|---|
| `torch/lib/*.dll` present | all 55, including `torch_python.dll` |
| sha256 vs the wheel's `RECORD` | `torch_python.dll`, `torch_cpu.dll`, `_C.cp312-win_amd64.pyd` all **match** |
| `ctypes.CDLL` each dependency | `c10`, `torch_cpu`, `c10_cuda`, `torch_cuda`, `shm`, `cudnn64_9`, MSVCP140* all **load** |
| `ctypes.CDLL('torch_python.dll')` | **fails**, ERROR_MOD_NOT_FOUND |
| its PE import table | 24 DLLs, all loadable — except `python312.dll` |
| `sys.version` of the venv | **3.14.7** |

`.venv/pyvenv.cfg` had been rewritten on 16 September at 11:00 to
`home = ...\pythoncore-3.14-64`, `version = 3.14.7`, against a `site-packages`
full of `cp312` wheels. `python -m venv` over an existing directory rewrites
the launcher and the config and leaves `site-packages` untouched, so any
second Python on the machine can do this. The 16 September note recorded the
inverse trap — bare `python` is 3.14, the project's is 3.12 — one day before
the venv became the 3.14 one.

**Fix:** `...\Python312\python.exe -m venv .venv`, which repoints it and keeps
the packages. Then torch 2.14.0+cu126, `cuda_available True`, RTX 2060 Max-Q,
capability 7.5 — the environment the 15–16 September runs used.

**Guard:** `bootstrap_local.ensure_venv` said "reusing" for a venv it had not
looked inside. It now compares the interpreter's version against the ABI tags
of the compiled extensions installed beside it (`_C.cp312-win_amd64.pyd`,
`_yaml.cpython-312-...so`) and refuses, naming both versions and the way out.
Pure-Python environments have no tags and are coherent anywhere, so they stay
silent. `scripts/run_velocity_queue.ps1` calls it before committing a night to
a queue.

## 2. Two arms could not do a forward pass, again

TEST_PLAN T1.1 exists because of a recorded failure: "two arms that could not
do a forward pass, because a signature changed in `model.py` and the
replacement cores in `ablations.py` did not. A test existed for neither." CI
was the response. CI ran a suite that still never built an arm, and the same
drift recurred:

```
gru            FORWARD FAILS: TypeError: GRUCore.forward() got an unexpected keyword argument 'substeps'
shortcut       FORWARD FAILS: TypeError: ShortcutCore.forward() got an unexpected keyword argument 'substeps'
```

`substeps` — the speed dial — went into `ConnectomeRNN.forward` and
`FlyBeats.forward`; the two replacement cores did not follow. Training never
noticed, because `train.run_epoch` calls `model.rnn` directly and never passes
it. What was broken is everything that calls the model: playback, transcribe,
bundle export, and two of the five Phase D columns.

The fix is one definition of the schedule (`model.substep_schedule`) used by
the real core and by the new `ablations.hold_drive`, which feeds a replacement
core the held frame k times — the same statement about time, and the same
`sum(schedule)` row count. The row count is not cosmetic: `StreamingDrummer`
reads timestamps off it, so a core that accepted `substeps` and ignored it
would put every hit at the wrong moment instead of failing.

## 3. The Tier 1 gate, in full

| | what it pins | found |
|---|---|---|
| T1.1 | every arm builds, forward-passes, backprops to finite gradients, honours the speed dial and rejects a bad schedule | **the bug in §2** |
| T1.2 | `deep_merge`, plus a resolution table for all ten shipped configs | the `_cpu` family resolved to `device: auto` |
| T1.3 | `train.evaluate`: perfect copy 1.0, silence 0.0, headline = curve argmax, curve = independent sweep, clips averaged not pooled, per-class velocity r | held |
| T1.4 | same seed rebuilds an arm; an arm moves with the seed **iff** it is in `STOCHASTIC_ARMS`; a null draw cannot reach the real arm | held |

Suite: **137 CPU tests before, 192 after** (4 GPU tests deselected, not run).

T1.2's resolution table is the durable half. `v1_8piece.yaml` sets
`device: auto`, `device_of` resolves that to CUDA wherever a card is visible,
and all five `_cpu` configs inherited it — so the Phase A′ velocity queue would
have moved to GPU the moment a CUDA wheel landed, with nothing in the filename
or the config saying so, and A′1's CPU intervals would have been compared
against arms measured elsewhere. The pin goes in `v1_8piece_cpu.yaml`, the root
of the family. The table now states `device`, `bf16`, `max_nodes`,
`min_weight` and `max_files` for every config, and a config with no row fails.

## 4. A confound in the Phase D arms, found in passing, not fixed

`GRUCore` and `ShortcutCore` accept `tonic` and discard it. Measured on the
2k fixture, switching `style_id` between two genres:

| arm | max change in output |
|---|---|
| real | 2.75e-04 |
| rewired | 5.68e-04 |
| sign_shuffled | 4.7e-05 |
| **gru** | **0.0** |
| **shortcut** | **0.0** |

So those two arms train and are scored with no genre information at all, while
the three connectome arms receive octopaminergic drive. They differ from the
real arm in two ways, not one, and any deficit they show is partly the missing
conditioning.

It is arguable either way — the genre bias is injected into a named
population, and a GRU has no pC1 — but it is not arguable that this should be
silent. Changing it redefines an arm, which is not a thing to do quietly
between campaigns, so it is recorded here and left. **It wants a decision
before Phase D is funded**, alongside §8.1 of the 16 September note, which is
still open.

## 5. What is now unblocked, and what is not

Done here, from `handoff-2026-09-17.md`: item 2 (the device pin, at the root of
the family rather than in one config) and item 3 (the queue ported to
PowerShell, with preflight). Item 1 was already done on 15 September and its
report is in this directory.

Still needing the card, and none of it started:

- **Phase A′2/A′3** — the queue is ready to run and pinned to CPU, so it needs
  the machine rather than the card. ~4 h per arm here.
- **Phase B′** — the full-corpus run. At the measured B=16 rate (10.5 min per
  epoch at the 30k tier, 897 clips) 12 epochs is ~2.1 h on this card, against
  the ~6 h `ROADMAP.md` projects from CPU.
- **R0.3, cut a bundle** — still the errand only this machine can run, and
  still blocked on there being a checkpoint worth shipping. The three in
  `runs/` are 5-epoch B=4 pilots.

---

## 6. Verified vs reasoned, for the record

**Verified by running code this session:** every row of §1's elimination table,
the arm failures and their fix in §2, all test counts in §3, and the genre
table in §4 — measured on the committed 2k fixture at `device: cpu`.

**Reasoned, not measured:** the ~2.1 h estimate for Phase B′ in §5 is the
15 September throughput table times 12 epochs, not a run. And the account of
*how* the venv came to be rebuilt is inference from timestamps: `pyvenv.cfg`,
`Scripts\Activate.ps1` and the launchers all carry 16 September 11:00–11:01,
and `site-packages` does not.

**Not done:** no training, no GPU work of any kind, and no decision taken on
§8.1 of the 16 September note or on §4 above. Both are recommendations
awaiting one.

`pytest -q -m "not gpu"` in `.venv`: **192 passed, 4 deselected**.
