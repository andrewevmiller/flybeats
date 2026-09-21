# Local hardware reports

`scripts/bootstrap_local.py` writes one JSON here per run. They exist because
every performance number elsewhere in this repository was measured on four
shared cloud cores, which predicts nothing about anyone's actual machine.

Each report records the machine, the GPU and driver, the torch build, the test
suite result, the smoke-test checks and the thread-count timing. Commit them;
a measurement nobody can find again is not a measurement.

## What is here

`2026-09-14-gpu-bringup.md` is the narrative for the first run: what the
machine turned out to be, what its card can and cannot do, what the report
below actually says, and what is still unverified.

`handoff-2026-09-21.md` is where to start. It is the current briefing for
whoever picks the project up next: the two decisions that should be taken
before anything expensive runs, what is done, what is ready to run and has
not been, and the traps specific to this machine.

`handoff-2026-09-17.md` is the one it supersedes, written in a cloud session:
two configuration traps that bite the moment a CUDA wheel lands, and the order
that evening's work wanted to happen in. Its items 1-3 are done.

The dated notes are the narrative record, newest last:
`2026-09-14-gpu-bringup.md`, the five `2026-09-15-*.md` reports,
`2026-09-16-threshold-response.md` (why the headline metric was choosing the
winner) and `2026-09-21-tier1-and-the-broken-venv.md` (the environment
failure, and the Tier 1 test gate).

### `bootstrap-LAPTOP-I9B30DJE-20260914-2359.json`

The first run on real hardware. Ryzen mobile, 16 logical cores, 42.3 GB RAM,
Windows 11, **RTX 2060 Max-Q (6 GB)**.

Read it with three caveats:

- **`torch` is `2.14.0+cpu` and `cuda_available` is `false`.** The run installed
  from the default index, which on Windows is CPU-only. So every timing in this
  file is a CPU timing, the four GPU tests were deselected rather than run, and
  the bf16 check reports `SKIP -- CUDA only`. Nothing here says anything about
  the CUDA path.
- **`exit_code` is `0` and should not be.** The GPU step is `FAILED` in the same
  file. The exit code was computed from the suite and the smoke test only, so a
  card present but invisible to torch — the exact silent failure this script was
  written to catch — exited clean. Fixed in the commit that added this file; a
  later report with a `FAILED` GPU step will exit `1`.
- **The two thread timings are different models.** `cuda_smoke`'s 4.4 ms is the
  2,000-neuron fixture it defaults to; the 17.5 ms / 11.5 ms pair under
  `threads` is the 10,000-neuron one. They are not in conflict and neither is
  the 30k tier.

Useful as it stands: the suite passes identically to CI on Windows (133 passed,
1 skipped, 4 deselected), and `torch.threads` shows torch picking 8 by default
while this run measured only 1 and 16 — which is why `thread_timing` now
measures torch's default as well.
