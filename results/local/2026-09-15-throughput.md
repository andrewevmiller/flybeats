# What a training batch actually costs — 15 September 2026

Measured before committing to the Phase 4 campaign, because the whole schedule
rested on a number with no evidence behind it. Companion to
`2026-09-15-probe-reproducibility.md`, which settled the operating point.

Evidence: a bounded batch-timing harness driving `train.run_epoch`'s real path
(TBPTT at 150, velocity head, grad clip, AdamW step, `encoder.project_()`),
real eGMD audio, 30k subgraph (2,942,102 edges), 846 train clips, 800
steps/clip. Three warmup batches discarded, `torch.cuda.synchronize()` around
every timing, peak memory reset after warmup.

**Outcome: the campaign is cheaper than budgeted and the batch-size decision
that was already made is wrong. `B=4` costs 5.01 s/batch, not the 21.7 s the
config asserts — 4.3x faster, and the old figure has no committed measurement
behind it. `B=16` buys 1.68x, not the "close to 4x" it was adopted for, while
still costing 4x the optimiser steps. The recurrence is much less
launch-bound than assumed: cost is near-linear in batch size.**

---

## 1. The measurement

RTX 2060 Max-Q, 6 GB, cc 7.5, torch 2.14.0+cu126. `bf16: auto` resolved to
`False`, autocast off, all parameters `torch.float32` — as expected on a 7.5
card, and confirmed rather than assumed.

| B | s/batch | spread | peak alloc | batches/epoch | min/epoch | h per 20-epoch run | h for 9 runs |
|---|---|---|---|---|---|---|---|
| 4 | **5.005** | sd 0.011 (4.99-5.03) | 920 MB | 211 | 17.6 | 5.87 | **52.8** |
| 8 | 7.414 | sd 0.036 (7.38-7.47) | 1571 MB | 105 | 13.0 | 4.32 | 38.9 |
| 16 | **12.11** | sd 0.11 (11.97-12.25) | 2869 MB | 52 | 10.5 | 3.50 | **31.5** |

No OOM anywhere; B=16 reserved 3567 MB of 6 GB. A repeat of the B=16 cell came
back at 12.34 s, +2%. Data loading is not the constraint — streaming off the
live DataLoader added a one-time ~11-13 s worker spawn and under 0.3 s/batch
amortised.

Both quantities are near-linear in B:

* time: `t(B) = 2.66 s + 0.592 s * B`, residuals within +/-0.03 s
* memory: `271 MB + 162 MB * B`

Only **53%** of a B=4 batch is batch-independent. That is the whole finding:
the timestep loop was expected to be dominated by kernel-launch overhead, in
which case a wider batch rides along for free. It is not. An infinitely wide
batch bottoms out at **2.1x** over B=4, and the memory line puts B=32 at
~5.5 GB with no headroom, so B=16 is near the practical maximum anyway.

## 2. The 21.7 s/batch figure is withdrawn

`configs/v1_8piece.yaml` carried "B=4 costs 21.7 s/batch -> 65-80 min/epoch ->
~24 h per 20-epoch run", and commit `6402cbc` widened the batch to 16 on the
strength of it. That commit **changed only the config — eight lines, no
benchmark, no results note.** Nothing reproducible was ever committed.

The re-measurement is 4.3x faster. The companion memory figure from the same
commit *does* reproduce (885 MB claimed, 920 MB measured, +4%), so the tier and
the model shape were right; only the wall clock is out. Whatever produced
21.7 s was not this machine in this state, and there is no way to find out what
it was, because nothing was written down.

Recorded at length because this is the second time in two days that an
unevidenced number shaped a decision: the probe's "noise floor" was the other.
A measurement that changes a config should leave a file behind.

## 3. What this does to the batch-size decision

Commit `6402cbc` attached its own caveat to the widening:

> this is 4x fewer optimiser steps per epoch. At lr 3e-3 one Adam step already
> moves `log_gain` by only ~0.15% of its median magnitude, so taking a quarter
> as many steps makes those parameters slower still. [...] do not spend the
> campaign's seeds before that is settled.

That trade was accepted when the saving looked like 4x. At the real 1.68x it
looks materially worse:

| B | h for 9 runs | saved vs B=4 | optimiser steps per 20 epochs |
|---|---|---|---|
| 4 | 52.8 | — | **4,220** |
| 8 | 38.9 | 13.9 h | 2,100 |
| 16 | 31.5 | 21.3 h | **1,040** |

`log_gain` is 97% of the trainable parameters, so step count is not a detail.
B=16 spends three quarters of the optimiser steps to save 40% of the wall
clock, on a model whose headline parameters were already described as moving
slowly per step.

**This is not settled here, and should not be settled by argument.** It is now
a question about optimisation quality per wall-clock hour, which the Stage 2
pilot can measure directly by running the same wall-clock budget at B=4 and
B=16 and comparing validation onset F. `batch_size` is left at 16 with the
config comment rewritten to say it is under review, rather than flipped on
reasoning alone.

## 4. Verified vs reasoned

**Verified by running code:** every number in §1 — three batch sizes, timed
through `run_epoch`'s real code path on real audio at the 30k tier, with a
repeat on the B=16 cell; that `bf16: auto` resolves to fp32 here; that no
configuration OOMs.

**Reasoned, not measured:** that `gru` and `shortcut` cost the same as `real`.
All timings are on the `real` arm. `GRUCore` and `ShortcutCore` replace the
recurrent core outright, so their per-batch cost could differ in either
direction and the campaign total in §1 assumes it does not.

**Not done:** the fp16 sparse path. `2026-09-15-cuda-path-verified.md` measured
sparse CSR fp16 as working and 6.9x on dense GEMM, and it remains the only
large lever left. At a 31-53 h campaign it is not required; at the four weeks
the old numbers implied it would have been.
