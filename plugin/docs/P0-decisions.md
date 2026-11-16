# P0: architecture decision records, parity tolerances, perturbation tests, slot-in bar

16 Nov 2026 · chair: Fable (this Code session) · ratified by: nobody yet (Gate P0 is the owner's) · plan: the VST plugin plan of 15/16 Nov

Settled owner decisions carried in unchanged: D2 closed source with a JUCE-free engine, D3 VST3 for FL Studio on Windows and macOS, D4 live monitoring default with print mode kept, D6 P0/S0/P1/P2 now, D7 Fly Lab first, D8 neuroscientist producers, D9 Lab switch, D10 Quantize button in Standard mode. The two assumptions the plan asked to confirm (print mode kept; Quantize in Standard mode) are taken as yes until the owner says otherwise.

## ADRs

**ADR-1 Engine library, no framework.** `plugin/engine` is a C++17 static library (`flybrain_engine`) with no JUCE, no VST3 SDK and no third-party dependency: brain file reader, resampler, ear, sparse core, readout, decoder, snapshot tap, control bus, runaway guard, recorder tap. The plugin shell (P3) wraps it. Command-line tools and tests link it directly. Reason: D2 (the engine stays ours whatever the shell's licence) and testability without a host.

**ADR-2 The brain file is the only interface between training and the plugin.** Format `.flybrain` version 1 (`plugin/export/flybrain/brainfile.py`, mirrored in C++): magic, version, JSON header, 64-byte-aligned little-endian arrays, SHA-256 trailer. Weights are the realised sparse weights in CSR by receiving neuron (about 0.43 MB signed plus 0.43 MB unsigned for the Lab's un-silencing, plus 0.43 MB of column indices at the real size), never a dense matrix. The hash is the brain's identity. The exporter rebuilds `FlyNet` from the slice and the checkpoint (because `g` and `gain_scale` are not in the `state_dict`, confirmed by reading `network.py` and by the exporter test) and writes `g`, `gain_scale` and the eigenvalue into the header. The schema is versioned; a reader refuses any other version. Readers verify shapes, ranges and the hash before use.

**ADR-3 Precision policy.** The engine computes in float32 (the model's precision) and offers a float64 state mode. The numpy reference engine runs in either. The core's sparse product adds each row's terms in index order, in both numpy (scipy CSR) and C++, so the two engines agree bit for bit in the same precision; PyTorch's dense product adds in another order, so against PyTorch the contract is the tolerance below. Flush-to-zero and denormals-are-zero are set on the audio thread.

**ADR-4 Streaming semantics.** The ear is causal (window ends at the frame); the engine emits one frame per 80 samples at 16 kHz; a hit is decided one frame late (the peak rule needs the next frame); at the end of a stream (print mode) the last frame is decided with "next = −∞", exactly as `loss.peaks` does. Reset zeroes everything (state, ear buffer, resampler, history ring, decoder) and starts the pre-roll (default 2 s, hits muted, network running). Output is identical whatever the block size.

**ADR-5 Threading.** All inference on the audio thread. UI reads a lock-free triple-buffered snapshot (bands, rates, motor rates, probabilities, hits, flags) written once per frame. Controls arrive through a double-buffered control block swapped at a frame boundary. The recorder is a single-producer single-consumer ring drained by a worker thread. No allocation, lock, I/O or exception on the audio thread after `prepare()`; an allocation-hook test enforces it.

**ADR-6 Shape of the plugin (per the plan).** Main build: VST3 effect, stereo main in and out plus a sidechain in, listening-source selector, built-in sampler to audio out, optional MIDI out. Secondary: instrument with MIDI out. S0 decides nothing here; it records how FL Studio behaves. Mono conversion is the **mean** of the channels (the training mix averages each stem's channels, `sampler.py`), not their sum.

**ADR-7 Runaway guard limits per brain.** From the exporter's cliff calibration at gain 1.0: mean-rate limit = `bound_factor` (50, the repo's own "bounded" rule) × the music-level mean rate, held for 20 frames; per-neuron clamp at 1000 × that limit. Stored in the brain file's `runaway` block. A brain exported without calibration has no limits and the Cliff Meter shows "unmeasured".

## Parity tolerances (fixed before the first run)

Constants in `plugin/export/flybrain/parity.py`. Evidence for the numbers: on the first synthetic export the reference engine differed from PyTorch by 2×10⁻⁷ of the largest rate and 1.4×10⁻⁶ in hit probability, so the limits give 30 to 50× headroom and still catch any real error.

| Quantity | Tolerance | Over |
|---|---|---|
| Rates | max abs error ≤ 1×10⁻⁶ + 1×10⁻⁵ × max rate | every neuron, every frame |
| Band features | max abs error ≤ 1×10⁻⁵ | every band, every frame |
| Hit probability | max abs error ≤ 1×10⁻⁵ | every drum, every frame |
| Decoded hits | the sets are identical, except hits PyTorch decides by under 1×10⁻⁵ (probability within that of the threshold or of a neighbouring frame); those are listed in the report, never silently excused | whole clip |
| Drift | the same limits over a 10-minute clip, reported per minute | the drift run |
| Engines against each other | C++ and numpy in the same precision: bit-identical rates, probabilities and hits (ADR-3); if a platform's floating point breaks this, the tolerance above applies and the memo says so | every test |
| Block size and sample rate | block sizes 1, 64, 480, 4,096: bit-identical. 44.1, 48, 96 kHz: hits identical except ambiguous ones, rates within the rate tolerance after the resampler's own error is measured and reported | P2 |

Clips for parity: 50 clips of 16 s is the plan's number for the laptop run with real checkpoints (P1 on the laptop); here, synthetic networks and noise, which exercise the arithmetic but not the music. Both are reported as what they are.

## Perturbation test list (control bus, rule H3)

Each control has a PyTorch equivalent, formed by editing the FlyNet or its input, not by re-implementing the control. Each test asserts (a) parity with that equivalent and (b) that the output changed against the baseline.

| Control | Engine field | PyTorch equivalent |
|---|---|---|
| Lesion Brush | `lesion[n] = 0` | `model.alive[n] = 0` |
| Deafen an Ear | `ear_mute[e] = 0` | ear input column zeroed |
| Gain to the Cliff | `gain_scale` | `model.gain_scale` |
| Tempo of Thought | `tau_scale` | `log_tau += log(scale)` before the clamp |
| Tickle / inject | current, frames | `+ drive` in the frame loop (exact: bias and drive add in the same term) |
| Sign Flip | `sign_flip[n] = −1` | `neurons.sign[n]` negated and the matrix rebuilt, `g` kept |
| Wake the Modulators | `unsilence[n] = ±1` | `neurons.sign[n] = ±1`, `silenced = False`, rebuilt, `g` kept |
| Per-band Ear EQ | `band_offset` | `model.band_offset += offset` |
| Hit threshold | `threshold` | `peaks(prob, threshold)` |
| Neutral | all at default | the baseline, bit for bit |

## Slot-in bar (Gate B), drafted from the v0.15 rubric

A brain may be marked `drum_capable` only when all of these hold; the chair checks them and the owner approves.

1. The v0.15 decision memo names this checkpoint and cites its scorecard on the test split, scored once, under `prereg-v5.2` or a later locked tag.
2. For the Accompanist product (P-A): the main win rule holds (real beats all five controls on beat alignment by more than the seed spread) **and** the brain's rescaled beat alignment is above the failure floor of 20 on a majority of test songs. The second clause is the chair's addition: winning against controls that are also at chance is not "it works".
3. For the Transcriber (P-B) or Reservoir (P-C) products, the memo's own pass criterion replaces clause 2 and is quoted in the brain's Scoreboard text.
4. It passes exporter parity (P1 contract above, 50 clips of 16 s, 10-minute drift) and the plugin's Lab checks (the cliff margin agrees with the repo's silence test; every control passes its effect test).
5. The Scoreboard text states the chance level, the floor and ceiling, the control results and the verdict in the memo's words; the chair and the red team sign it off; L1 checks every string.
6. The plugin team never opens the sealed confirmation sets.

Until then every brain is `observation_only`, and the drum mode for it is available only in Lab mode under the "chance-level" banner.

## Real-time safety checklist (P2 exit)

No `new`/`malloc`, no mutex, no file or console I/O, no exceptions, no `std::function` allocation and no unbounded loops in the audio path after `prepare()`. Enforced by an allocation-hook test around `process()` and by sanitizer builds. Snapshot and control exchange are wait-free; the recorder never blocks the writer (a full ring drops frames and counts them).

## What starts before v0.15 decides

Everything in P1 and P2 (this session), S0's test build, and the Tier 1 Lab features on untrained and baby-trained checkpoints. What waits: the product choice, the Scoreboard wording, Tier 3, threshold and drum-balance tuning.

## Not verified here

- No real checkpoint, slice or audio is in this cloud container (the data directories are git-ignored), so every parity number in this session is on synthetic networks of the slice's schema and size, with noise input. The laptop run of the same harness on the real slice and the baby-trained checkpoints is a P1 exit item for the owner.
- The tolerance numbers rest on one small synthetic network's measurements plus the large drift run; the real network sits nearer its stability cliff, so the drift limit may need revisiting once the laptop run reports.
