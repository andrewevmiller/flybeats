# The threshold decides the winner — 16 September 2026

Follows `2026-09-15-operating-point.md`, which settled per-arm spectral radii
and found that no arm separates *before* training. This note covers the first
trained comparison: the B=4 pilots, the three-draw rewired spread run, and a
full threshold sweep over the resulting checkpoints.

Evidence: `runs/pilot_b4/ablation_results.json` (real, 1 run),
`runs/pilot_b4_rewired/ablation_results.json` (rewired, 1 run),
`runs/pilot_b4_spread/ablation_results.json` (rewired, 3 topology draws),
and `runs/threshold_response.json` (5 checkpoints x 19 thresholds, written by
the new `scripts/threshold_response.py`). All at 5 epochs, B=4, 120 val clips,
3,085,788 trainable parameters per arm.

**Outcome: training clears the frozen-core ceiling, so the campaign will not
produce a flat table. But the headline metric is not measuring what it claims.
`evaluate` picks onset F at the best of several thresholds *chosen on the split
it scores*, while computing every timing metric at a hard-coded 0.3 — and the
two arms put their probability mass in different places, so that pair of
choices decides the ranking. Scored at each arm's own operating point, real's
beat-alignment and groove advantages vanish entirely (z = -0.04 and +0.22,
both inside the null range) and rewired's "8 ms rush" reverses sign. What
survives is not accuracy but stability: real holds a 2x wider usable operating
range and 2.5-3x tighter timing across it.**

---

## 1. Training clears the ceiling

The frozen-core linear probe topped out around F=0.20-0.23
(`2026-09-15-operating-point.md` §3). Both trained pilots land near 0.30:

| arm | rho | onset F (best thr) | thr | F @ fixed 0.3 | groove | beat ms | dev ms |
|---|---|---|---|---|---|---|---|
| real | 5.0 | 0.2993 | 0.4 | 0.2918 | 0.409 | 36.26 | +0.05 |
| rewired | 0.5 | 0.3152 | 0.5 | 0.2497 | 0.431 | 38.45 | -6.73 |

Pilot question 1 is answered yes. The caveat stands that ~0.20 is a ridge
probe on 8 clips and ~0.30 is a trained decoder on 120 — related quantities,
not the same measurement.

## 2. The spread run measures topology draw, not seed

`ablations.py` pins `torch.manual_seed(base_seed)` and the loader reset to
`base_seed` for every rep, varying only the argument to
`degree_matched_rewire`. So `--seeds 3` varies the topology draw and nothing
else. `pilot_b4.yaml`'s header calls this "seed-to-seed spread of trained val
F"; it is not that, and the real arm — deterministic topology, `n_reps=1` —
gets no error bar of this kind at all.

That is the right design for a null-model test, just mislabelled: one observed
topology against N draws from the null. Three draws:

| | onset F | F @ 0.3 | groove | beat ms | dev ms |
|---|---|---|---|---|---|
| seed 0 | 0.3152 | 0.2497 | 0.4307 | 38.45 | -6.73 |
| seed 1 | 0.3381 | 0.2513 | 0.4523 | 38.43 | -8.41 |
| seed 2 | 0.3668 | 0.2392 | 0.4477 | 38.60 | -9.47 |
| mean +/- sd | 0.3400 +/- 0.0211 | 0.2467 +/- 0.0054 | 0.4436 +/- 0.0093 | 38.49 +/- 0.079 | -8.20 +/- 1.13 |

**Reproducibility, free:** `rewired_seed0` reuses seed 0, so it re-runs the
already-finished `rewired.pt`. It agrees to 8.7e-06 on onset F — not bitwise
(the `SparseSpMM` backward uses atomics), but three orders of magnitude below
the 0.021 topology spread. Re-scored across all 19 thresholds in §4 the worst
disagreement is 6.0e-04. **The spread in this table is topology, not numerics.**

## 3. The metric was choosing the winner

At the recorded operating points the two arms disagree about who wins,
depending on which number is read:

| metric | real | null [min, max] | z |
|---|---|---|---|
| onset F (best thr) | 0.2993 | [0.3152, 0.3668] | -1.93 |
| onset F @ fixed 0.3 | 0.2918 | [0.2392, 0.2513] | **+8.38** |

The ordering flips, decisively in both directions. The cause is the selection
itself. Gain from being scored at one's own best threshold:

* real: **+0.0075**
* rewired: +0.0655, +0.0868, +0.1276 — mean **+0.0933, 12.4x real's**

Rewired's entire apparent lead is smaller than what the selection hands it.
Meanwhile `beat_align_ms`, `groove_sim` and `mean_dev_ms` are computed at a
hard-coded 0.3 (`peak_pick`'s default, `src/metrics.py`), which is nowhere near
rewired's own 0.5-0.6 optimum — the mirror-image unfairness, pointing the other
way. Neither family of metric was a fair comparison, and they were unfair in
opposite directions.

## 4. Sweeping every metric, not just F

`scripts/threshold_response.py` takes one forward pass per checkpoint, caches
the probabilities, and sweeps all four metrics over 19 thresholds on the cached
output. It reproduces `evaluate`'s own recorded numbers exactly (real: argmax
0.2993 @ 0.4, F@0.3 0.2918 on the old grid), which is what validates the
checkpoint reload, the topology reconstruction from seed, and the encoder
calibration restore.

Onset F against threshold:

| thr | real | rw0 | rw1 | rw2 |
|---|---|---|---|---|
| 0.20 | 0.2606 | 0.2074 | 0.2119 | 0.1970 |
| 0.25 | 0.2788 | 0.2322 | 0.2312 | 0.2228 |
| 0.30 | 0.2918 | 0.2497 | 0.2513 | 0.2392 |
| 0.35 | **0.3004** | 0.2657 | 0.2703 | 0.2535 |
| 0.40 | 0.2993 | 0.2854 | 0.2874 | 0.2718 |
| 0.45 | 0.2952 | 0.3047 | 0.3096 | 0.2922 |
| 0.50 | 0.2733 | **0.3152** | 0.3346 | 0.3212 |
| 0.55 | 0.2094 | 0.3058 | **0.3542** | 0.3478 |
| 0.60 | 0.1137 | 0.2597 | 0.3381 | **0.3668** |
| 0.65 | 0.0151 | 0.1829 | 0.2910 | 0.3485 |
| 0.70 | 0.0006 | 0.1280 | 0.2144 | 0.2919 |
| 0.80 | 0.0000 | 0.0965 | 0.0903 | 0.1396 |

**Real beats all three draws at every threshold <= 0.40; from 0.50 up all
three beat real.** 0.45 is the crossover, and it is not clean — two of the
three draws pass real there (0.3047 and 0.3096 against 0.2952) while the third
is still behind (0.2922). Real's outputs run cooler — its F is zero by 0.75,
where rewired still scores 0.11-0.21. Whoever picks the threshold picks the
winner.

### The grid extension earned its keep

The old sweep stopped at 0.6, and two of three draws had landed there.

| arm | coarse grid | fine grid | under-reported by |
|---|---|---|---|
| real | 0.2993 @ 0.4 | 0.3004 @ 0.35 | +0.0011 |
| rewired s0 | 0.3152 @ 0.5 | 0.3152 @ 0.5 | 0 |
| rewired s1 | 0.3381 @ 0.6 | **0.3542 @ 0.55** | **+0.0160** |
| rewired s2 | 0.3668 @ 0.6 | 0.3668 @ 0.6 | 0 |

Seed 2's 0.6 was a genuine peak, not a truncation — the edge worry was half
right. But seed 1's peak sits at 0.55, which the old grid did not contain, so
its F was under-reported by more than the whole real-vs-rewired gap.

## 5. Scored fairly, the timing advantage is gone

Each arm at its own F-optimum:

| arm | thr* | F | beat ms | groove | dev ms |
|---|---|---|---|---|---|
| real | 0.35 | 0.3004 | 36.13 | 0.4020 | -1.37 |
| rewired s0 | 0.50 | 0.3152 | 37.42 | 0.3841 | +3.81 |
| rewired s1 | 0.55 | 0.3542 | 35.79 | 0.4083 | +7.65 |
| rewired s2 | 0.60 | 0.3668 | 35.29 | 0.4064 | +8.45 |

Real against the null, both at their own operating points:

| metric | real | null [min, max] | z | inside null? |
|---|---|---|---|---|
| onset F | 0.3004 | [0.3152, 0.3668] | -2.05 | no — loses to all 3 |
| beat_align_ms | 36.13 | [35.29, 37.42] | **-0.04** | **yes, dead centre** |
| groove_sim | 0.4020 | [0.3841, 0.4083] | **+0.22** | **yes** |
| mean_dev_ms | -1.37 | [+3.81, +8.45] | -3.95 | no — closest to zero |

Three corrections to the reading taken from the fixed-0.3 numbers:

1. **Real's 2.2 ms beat-alignment win does not exist.** Two of three rewired
   draws beat real once each is scored where it lives.
2. **Real's groove deficit does not exist either.** It was below all three
   draws at 0.3 (z = -3.68); it is mid-pack at own-optimum.
3. **"Rewired rushes by 8 ms" was wrong in sign.** At 0.3 rewired reads -6.7
   to -9.5 ms; at its own optimum it reads +3.8 to +8.5 ms. The direction of
   the bias was an artifact of the scoring point.

## 6. What does survive: stability, not accuracy

Restricting to each arm's usable range — thresholds where F is at least 90% of
that arm's own peak, which avoids the region where F collapses and the timing
metrics run on a handful of surviving peaks:

| arm | plateau | width | dev ms swing | beat ms swing |
|---|---|---|---|---|
| real | 0.25-0.50 | **0.25** | **3.06** | **0.76** |
| rewired s0 | 0.40-0.55 | 0.15 | 9.80 | 1.97 |
| rewired s1 | 0.50-0.60 | 0.10 | 7.75 | 2.36 |
| rewired s2 | 0.55-0.65 | 0.10 | 7.43 | 2.79 |

Real holds a roughly 2x wider usable operating range and 2.5-3x tighter timing
across it. Its mean deviation stays inside +/-3.1 ms across that range, against
7.4-9.8 ms of swing for the draws. Taking the full sweep instead — including
the collapsed region, where the statistic runs on a handful of surviving peaks
and should not be read closely — the rewired draws span [-15.0, +18.8] ms
against real's [-8.1, +0.1]. Real also stays closest to zero deviation at its
own optimum (§5).

This is a robustness claim, not an accuracy claim, and it is the one the data
supports. The connectome arm is not better at hitting onsets. It is less
sensitive to where you put the decision boundary.

## 7. Three nulls cannot produce a significant result

With B null draws the smallest achievable one-sided rank p-value is 1/(B+1).
At B=3 that floor is **0.25** — so every clean separation in §5, in either
direction, is consistent with chance at any conventional threshold. Reaching
p <= 0.05 needs B >= 19. At the spread run's measured 1h58m per seed that is
~37 GPU-hours for one null arm at pilot scale, before the campaign's other
arms. This is larger than the current ~31-53 GPU-hour campaign estimate
assumes, and it is a planning input, not a result.

---

## 8. What this means for Phase 4

1. **Decide the primary metric before funding the campaign.** The present
   headline picks its threshold on the split it scores, and §3 shows that
   choice is worth up to 0.128 to one arm and 0.008 to the other. The honest
   options are to select the threshold on a held-out split and report on val,
   or to fix one operating point for every arm and publish the curve beside
   it. This is the pivotal decision and it is not a measurement question.
2. **Report the curve, not just the argmax.** `evaluate` now persists
   `onset_f_curve`; `beat_alignment_error` and `groove_similarity` now take an
   optional `threshold` (defaulting to the previous hard-coded 0.3, so no
   existing number moved). Plateau width and timing swing are threshold-free
   summaries and are better campaign metrics than a selected maximum.
3. **Budget for enough null draws to say anything.** Three is not enough to
   clear p=0.25 no matter how large the effect.
4. **The hypothesis is not confirmed, and not refuted.** Real loses on onset
   F at own-optimum and ties on beat alignment and groove. Its case rests on
   operating-point robustness and timing stability, which were not the
   campaign's pre-registered endpoints.
5. **Relabel the spread run.** `pilot_b4.yaml` calls `--seeds` seed-to-seed
   spread; it is topology-draw spread (§2), and the real arm cannot have one.

## 9. An environment trap worth recording

Bare `python` on this machine is **3.14.7 with `torch 2.14.0+cpu`**; the
project environment is **`.venv/Scripts/python.exe`, 3.12.10 with
`torch 2.14.0+cu126`**. A sweep started with the wrong one runs silently on CPU
and reports `device cpu` rather than failing. Every command in this note used
`.venv`. This is the same class of silent failure `bootstrap_local.py` was
written to catch, in a different place.

---

## 10. Verified vs reasoned, for the record

**Verified by running code in this session:** every number in §1-§6 —
the pilot and spread tables were read from the three `ablation_results.json`
files written by the completed runs, and all threshold curves, plateau widths,
timing swings and z-scores were computed from `runs/threshold_response.json`,
produced this session on CUDA. The reproducibility deltas (8.7e-06 on the
recorded metric, 6.0e-04 across the 19-threshold sweep) were computed here.
The script's agreement with `evaluate`'s recorded numbers (§4) was checked
before the full run.

**Reasoned, not measured:** the p >= 0.25 floor in §7 is arithmetic, not an
experiment. The account in §4 of *why* the curves differ — that real places
less probability mass at high confidence — is inferred from the shape of the
F curves; the output distributions themselves were not histogrammed.

**Not done, and out of scope here:** no new training run, no additional null
draws, and no change to which metric is reported as the headline. §8.1 is a
recommendation awaiting a decision, not a change that has been made.

`pytest -q` in `.venv`: **141 passed** (the 4 CUDA tests that the CPU
interpreter skips are included). Changed this session: `src/train.py`
(persist `onset_f_curve`, extend the default sweep to 0.9), `src/metrics.py`
(optional `threshold` on `beat_alignment_error` and `groove_similarity`,
defaults unchanged), `configs/v1_8piece.yaml` (sweep to 0.9), and the new
`scripts/threshold_response.py`.
