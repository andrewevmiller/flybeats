# Gate P2 memo: engine phases P0, S0, P1 and P2

16 Nov 2026 · chair: Fable (this Code session, branch `claude/new-session-8rt4tg`) · for: the owner's go or no-go on the plugin shell (P3a) and the Lab UI (P3b) · status of the work: **built, tested and committed; nothing has run in FL Studio; nothing has run on a real checkpoint**

Per decision D6 this session ran P0, S0, P1 and P2 and stops here. Everything below was done by Fable: this session wrote P0 and P1 itself and ran two Fable forks (same model) for the C++ engine and for S0, then a third as red team on this memo (section 7). No Opus or Sonnet specialist was dispatched; the plan's E1 to E4 roles were filled by Fable at the owner's request of 16 Nov.

## 1. What exists now

| Phase | Deliverable | Where | State |
|---|---|---|---|
| P0 | Architecture decision records, parity tolerances, the perturbation test list, the slot-in bar, the real-time checklist | `plugin/docs/P0-decisions.md` | written; the owner ratifies at Gate P0 (it was written before any run, as the plan requires) |
| S0 | Two trivial JUCE 8 VST3 plugins (effect with sidechain and MIDI out and reported latency; instrument with MIDI out), build steps for Windows and macOS, FL Studio steps, the five-observation form | `plugin/s0/`, `plugin/reports/S0-build-report.md` | compiles on Linux (compile check only); **the owner builds and tests it in FL Studio** |
| P1 | The `.flybrain` format; the exporter (slice + checkpoint → brain file with display metadata, provenance hashes, cliff calibration, runaway limits); the numpy reference engine with every Lab perturbation; the parity harness against PyTorch; the laptop CLI; the drift test | `plugin/export/flybrain/`, tests in `plugin/export/tests/` | 33 Python tests pass on synthetic networks; the 10-minute drift at the real size passes; **the laptop run on the real slice and checkpoints is still owed** |
| P2 | The C++17 engine library (no JUCE, no SDK, no dependency): brain reader, resampler, ear, sparse core, readout, decoder, snapshot tap, control bus, runaway guard, recorder; `flybrain_render` and `flybrain_bench` tools; 8 C++ test binaries; 12 Python cross-checks | `plugin/engine/`, `plugin/reports/P2-engine-report.md` | all pass in Release and under ASan+UBSan; CPU measured |

Branch commits: `3f2727b` (P0, P1), `aa6cb5b` and `9f07d4a` (drift), `50f7f0a` (S0), `c9647ce` (engine snapshot), then this memo's commit with the final engine, the reference-engine fix in section 4 and the red-team answers.

## 2. P1 exit criterion, as met here

The plan's P1 exit: parity with PyTorch on a current checkpoint (hits identical, rates within tolerance, a 10-minute drift pass), each perturbation matching its PyTorch equivalent, the exported cliff margin equal to the repo's silence test.

What was possible in this container: **no slice, checkpoint or audio exists here** (all git-ignored on the laptop), so every number is on synthetic networks that copy the slice's schema and the real network's size, with noise input, through the real `FlyNet` class. The parity tolerances were fixed in P0 before the first run and were not moved afterwards.

| Check | Result |
|---|---|
| Reference engine vs PyTorch, 183-neuron synthetic, 2 s noise, float64 and float32 | rates within 2×10⁻⁷ of the largest rate (limit: 10⁻⁶ + 10⁻⁵ × max rate); bands 9×10⁻⁷; probabilities 2×10⁻⁶; hit sets identical (7 hits) |
| Block sizes 1, 7, 64, 480, 4,096 | bit-identical bands, rates, probabilities and hits |
| Neutral controls, repeat runs | bit-identical to the baseline |
| Each perturbation (lesion, deafen, gain, tau, sign flip, band offset, un-silence, injection) vs its PyTorch equivalent | passes the same tolerance, hits identical, and each changes the output against the baseline (rule H3) |
| Streaming decoder vs `loss.peaks`, including plateaus (ties) | identical, one frame late, last frame decided at flush |
| Pre-roll, reset, runaway guard | hits muted during pre-roll with rates untouched; reset reproduces the first run; a 50× gain never gives NaN or Inf, the guard trips and the engine recovers to the baseline bit for bit |
| Checkpoint round trip | a `best.pt`-shaped checkpoint loaded into a freshly built `FlyNet` exports an array-identical brain (every array, `g` and the eigenvalue; the files differ only in the `created` timestamp) |
| Cliff calibration | the exporter's bisection on the repo's `silence_test` finds a last-pass and first-fail gain on a synthetic network and restores `gain_scale`; the runaway limits derive from it (ADR-7) |
| History readout | a brain with four 50 ms blocks exports and matches PyTorch |
| 10-minute drift, 3,326 neurons and 109,967 connections, reference f64 vs PyTorch f32 | rate error 6.4×10⁻⁶ against a limit of 2.2×10⁻⁴, flat across the ten minutes (5.5 to 6.4×10⁻⁶ per minute); probabilities 2.7×10⁻⁶; 18 vs 19 hits, the one difference on a saturated plateau inside the ambiguity rule, 0 unexplained |

**What the hit comparison is worth.** The red team found that the first synthetic readout saturated two drums at probability 1.0, so most "identical hits" were plateau ties the ambiguity rule could excuse, and a deliberate tie bug in the decoder passed. Fixed in this memo's commit: the synthetic readout no longer saturates (40 of its 43 hits are genuine), `compare` reports the genuine count and the parity tests require at least 10, and both decoders now have exact tie tests on constructed plateaus (Python already had one; the C++ decoder's is new). The large drift network's readout still saturates one drum; its hit arm remains weak and the rate and probability arms carry it.

**Two things the drift run taught.** First, the first large synthetic export blew up to infinity within a minute in PyTorch and the reference engine alike: perturbing the type-pair dials and time constants of a 3,326-neuron network pushes it past its cliff. The drift network now keeps the network at its critical gain and moves only the readout. The real network sits near its cliff too (fails the silence test at 1.45×), so the runaway guard is not decorative. Second, the large synthetic readout saturates (probabilities flat at 1.0 for a drum), which makes hit comparison weak there; the hit contract is carried by the small network and by the C++ tests.

## 3. P2 exit criterion, as met here

The plan's P2 exit: parity with the numpy engine; identical output for block sizes 1, 64, 480, 4,096; 44.1, 48 and 96 kHz within tolerance; stable on silence, noise and clipped input; neutral controls equal baseline; perturbed runs match the reference; a gain sweep through the cliff never gives NaN or Inf and recovers; no audio-thread allocation; measured CPU cost. The engine fork's full table is in `plugin/reports/P2-engine-report.md`. First-hand by the chair: a from-scratch Release build, the test-data regeneration, ctest (8/8, in which the parity, block-size, host-rate, stability, perturbation, gain-sweep, allocation, bus and recorder checks run) and the whole Python suite (45/45, the 12 C++ cross-checks and the C++ drift comparison included), rerun again after the red-team fixes. Taken from the fork's report, not rerun by the chair: the ASan+UBSan build, the bus stress counts, the resampler's alias figure and the CPU table.

| Check | Result |
|---|---|
| Core given the reference's ear features, three brains (small, history readout, large) | **rates bit-exact** against the numpy reference in float and in double; hits identical |
| Full chain at 16 kHz, double | rates within 7×10⁻¹⁶ (the FFT's rounding differs from numpy's), probabilities 6×10⁻¹⁶, hits 96/96 |
| Full chain, float | rates 4×10⁻⁷, probabilities 3×10⁻⁷, hits identical except plateau ties, 0 unexplained |
| Block sizes 1, 64, 480, 4,096 | bit-identical events and frames |
| 44.1, 48, 96 kHz (numpy-resampled reference) | rates 4×10⁻¹⁶, hits identical (58/58, 56/56, 60/60); reported latency 537, 576 and 1,056 host samples (12.2, 12.0 and 11.0 ms: resampler delay plus two hops) |
| Resampler | bit-identical across block sizes; max difference 0.0 against its numpy port; 1 kHz gain within 0.0005 dB; a 9 kHz tone's alias at −92 dB |
| Stability, 60 s each of silence, white noise, ±1 square, float and double | finite, no runaway; peak rates 0.95, 5.1 and 12.7 |
| Neutral controls, every perturbation, mid-stream control change, injection with a start frame | neutral equals the baseline bit for bit; each perturbation equals the reference's run to 10⁻¹⁵ with identical hits and changes the output; identical before the swap frame |
| Gain sweep 1 to 50× with guard limits | never NaN or Inf, peak clamped, 57 guard resets at 50×, then a neutral run equals the baseline both in a fresh engine and after `reset()` |
| The same at 50× on a brain with **no** calibration (the exporter's default) | before the red team: 120 of 394 frames carried NaN and 12 Inf. Now the guard runs on hard default limits (ceiling 10⁴, mean limit 10³, 20 frames) when the brain carries none: every snapshot and event finite in float and double, 16 resets, `guardCalibrated()` false so the Cliff Meter can say "unmeasured" |
| Hits beyond the caller's event array | counted in `droppedEventCount()`, never silently lost (test: a one-slot array over 4,096-sample blocks) |
| Denormal mode | FTZ/DAZ is set for the block and the host thread's control word restored on exit |
| 10-minute drift at the real size | C++ double vs reference 1.4×10⁻¹⁴ and C++ float vs reference f32 7.6×10⁻⁶, both flat over the ten minutes, 0 unexplained hit differences, 0 runaways |
| Allocation hook around `process`, `flush`, `reset`, `stepFeatures` with control swaps and runaway resets inside | 0 allocations, 0 frees (float and double, 16, 44.1, 48 kHz) |
| Snapshot tap, control bus | a writer published 456,090 slots in 0.5 s while a slow reader took 790, every read consistent and increasing; the ring drops and counts when full, order kept |
| Recorder, 60 s | 12,000 frames written equal to the snapshots, 0 dropped, control change logged at the right frame, brain hash in the sidecar; a stalled consumer never blocks the writer |
| Sanitizers | ASan+UBSan build: 8/8, nothing reported |

**CPU cost, measured** (`plugin/reports/bench_large.txt`; one vCPU of this container's Xeon at 2.1 GHz, 48 kHz host, 512-sample blocks, resampler included):

| Precision | Silence | Noise |
|---|---|---|
| float | 187 µs per frame, 3.8 % of one core | 202 µs, 4.0 % |
| double | 204 µs, 4.1 % | 207 µs, 4.2 % |

Inside the plan's 2 to 8 % estimate, which was arithmetic; this is a measurement, on a cloud core rather than the laptop's.

## 4. Bit-exactness, honestly

ADR-3 asked for bit-identical engines in the same precision. Achieved for the rates given the same ear features. Not achieved end to end, and not fixable without changing the reference: the in-house FFT rounds differently from numpy's pocketfft (10⁻¹⁴ in double); the reference's readout is a BLAS product whose summation order is OpenBLAS's, and `np.exp` and `std::exp` differ in the last bit (10⁻¹⁶). The tolerance contract holds with ten orders of headroom in double and three in float, and hits are identical except where the reference itself decides by under 10⁻⁵, which is listed every time. The chair records this as the contract's state rather than relaxing ADR-3's wording.

Two reference-engine quirks the fork found after a runaway trigger (pending ear samples discarded, so the result depended on the block size; the frame just pushed never decided) were fixed in the reference in this memo's commit so that it matches the engine: the frame is published with no hits and the pending samples are kept. Post-runaway behaviour is outside the parity contract either way.

## 5. What the owner does next (the P1/P2 items that need the laptop)

1. **Build and run S0 in FL Studio** on Windows and on the Mac (`plugin/s0/README.md`), fill in `plugin/s0/OBSERVATIONS.md`. The shell's bus layout and MIDI path (ADR-6) are provisional until then.
2. **Export a real brain**: from `plugin\export`, `.venv\Scripts\python.exe -I -m flybrain.cli --set baby --checkpoint runs\<run>\best.pt --out brains\<name>.flybrain --name "<name>" [--control 1] [--history]`. It rebuilds the network from the slice, loads the checkpoint strictly, calibrates the cliff, writes provenance and runs the parity harness on 20 training clips (the plan's 50 of 16 s: `--clips 50`). Then `plugin\engine\build\flybrain_render` on the same clips, compared through `test_cpp_engine.py`'s functions. This is the P1 exit on real weights; the untrained real network, the baby-trained real network and a control are the three to run.
3. **Decide Gate P2**: go or no-go on P3a (shell) and P3b (Lab UI), with the S0 findings.
4. Ratify or amend `P0-decisions.md` (Gate P0), in particular the slot-in bar's clause 2.

## 6. Risks and open items carried forward

- **Portability.** Built with g++ 13 on Linux only. The code avoids GNU extensions and guards the x86 and arm64 denormal controls, but MSVC and Apple clang have not compiled it; `M_PI` may need `_USE_MATH_DEFINES` on MSVC. Q1's first job in P3.
- **Real network near its cliff.** The drift margin was measured on a network at its critical gain; the real one is closer to runaway. Float state may suffice (7.6×10⁻⁶ flat over ten minutes) or the double mode may be needed; the laptop run decides, and the engine has both.
- **Hit timing.** The engine reports a hit at the host sample where it became known (end of the following frame); `latencyHostSamples()` is resampler delay plus two hops. Whether the shell back-dates hits by one frame where the host allows is a P3 question, as is whether FL Studio compensates the reported latency (S0 observation 4).
- **Recorder polling** (1 ms sleep on the worker) and the UI-side control-log mutex are fine for the Lab, not for a hard real-time contract; noted, not changed.
- **Snapshot after reset.** The tap can still hold the last pre-reset frame, which carries its own older index; the UI must read the index.
- **Format wish for v2.** A `presumed_sign` per silenced neuron, so Wake the Modulators needs no choice in the UI. v1 is unchanged.
- **Test data generation.** `plugin/engine/tests/data/make_data.py` (needs torch) must run before ctest; without it the data tests print SKIP and pass, which CI must not mistake for coverage.
- **JUCE splash screen.** S0 sets it off; L1 decides whether the owner's JUCE tier requires it on.
- **Rules carried into P3 from the red team.** The Lab's gain control does not exceed 1× on a brain whose `guardCalibrated()` is false, and the Cliff Meter shows "unmeasured" with the ±0.02× resolution of a measured margin. The plugin treats every brain as observation-only, whatever its `capability` flag, until a signed approval scheme replaces the `gate_b` record. A control swap replaces the injection queue, so Tickle gets its own path or re-sends with the swap. Feature #32 shows the lateness range (one to two hops plus the resampler), not one number. The output limiter is a P3a requirement; the runaway guard is a last resort, not ear protection.
- **The plan's run brief** (`claude/vst-code-session-brief.md`) and the feature file are not in the repository; this session worked from the plan the owner attached. If the brief differs in anything that matters, say so and it gets folded in.

## 7. Red team

A second Fable fork attacked this memo, the reports and the code: 15 findings, none blocking Gate P2 as written, five to fix before P3. All five are fixed in this memo's commit (default guard limits so an uncalibrated brain never emits NaN; a non-saturating synthetic readout, a genuine-hit floor and a C++ decoder tie test; the denormal control word restored; the S0 README's observation 4 corrected; the CLI's parity default raised to the plan's 50 clips). The findings and the chair's answer to each are in `plugin/docs/gate-P2-red-team.md`.
