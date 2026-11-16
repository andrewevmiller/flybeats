# P2 engine report: the C++ headless engine

16 Nov 2026 · fork E2+E3 (Fable) · code: `plugin/engine/` · Python side it talks to: `plugin/export/flybrain/{resample,recording}.py`, `plugin/export/tests/test_cpp_engine.py`, `plugin/engine/tests/data/make_data.py`

Everything below was built and run in this cloud container (4 vCPU Intel Xeon at 2.10 GHz, g++ 13.3, cmake 3.28, ninja, Release `-O3`-equivalent `CMAKE_BUILD_TYPE=Release`, and a second build with `-fsanitize=address,undefined`). No real checkpoint or slice exists here: every network is synthetic (`flybrain.synthetic.make_slice`, the slice's schema), at the small size (183 neurons) and at the real slice's size (3,326 neurons, 109,967 connections, from the parent's drift export). Audio is noise.

## What was built

`flybrain_engine`, a C++17 static library (368 KB, 2,724 lines including tools and tests) with no third-party dependency, compiled with `-ffp-contract=off -fno-fast-math` (`/fp:precise` on MSVC) so rounding is the IEEE per-operation result.

| Component | File | What it does |
|---|---|---|
| `BrainFile` | `brainfile.{h,cpp}`, `util.{h,cpp}` | Reads `.flybrain` v1: magic, version, JSON header (own parser), 64-byte-aligned little-endian arrays, SHA-256 trailer (own implementation, FIPS known answers checked). Refuses bad magic, wrong version, hash mismatch, truncated file, arrays outside the file, every range and shape rule of `verify_brain`. Exposes the full header text for the UI. |
| `Resampler` | `dsp.{h,cpp}` | Host rate to 16 kHz, rational L/M polyphase windowed sinc (Kaiser β 8.96, 96 taps per side per output, stopband from 8 kHz). Streaming, bit-identical whatever the block size, production delay 96 input samples, time alignment exact. 16 kHz is a pass-through. |
| `RealFFT`, `Ear` | `dsp.{h,cpp}` | In-house radix-2 complex FFT (512), periodic Hann, band sums from the file's band matrix, `log1p`. The engine applies `(x − mean)/std`, then gain and offset, then the control-bus band offset, in its own precision. |
| `Core<T>`, `Readout<T>`, `PeakDecoder<T>`, `RunawayGuard<T>` | `network.h` | CSR step with each row's terms added in index order; the update in exactly reference.py's order; readout with the history ring; the streaming peak rule (one frame late, flush at end of stream); clamp and mean-rate runaway rule with soft reset. `T` is float or double. |
| `Controls`, `CompiledControls<T>` | `controls.h`, `network.h` | The control block (lesion, ear mute, gain, tau scale, sign flips, un-silence, band offset, up to 64 injections with a start frame, threshold), compiled off the audio thread into a weight/mask/alpha set. |
| `TripleBuffer`, `SpscRing`, `Snapshot` | `bus.h` | Wait-free exchange: the control bus (UI writes, audio acquires at a frame boundary) and the snapshot tap (audio writes once per decided frame, UI reads the newest) are the same triple buffer; the recorder tap is an SPSC ring that drops and counts when full. |
| `Recorder` | `recorder.{h,cpp}` | Worker thread drains the ring to `FLYREC01` binary plus a JSON sidecar (brain hash, host, sample rate, block size, precision, control log with frame indices, frames written and dropped). |
| `Engine` | `engine.{h,cpp}` | `prepare` (allocates everything), `reset`, `process(inputs, channels, samples, events, max)` (mean of channels, input trim in dB, resample, ear, frame steps, hit events with the host-sample offset at which each hit became known), `flush` (print-mode end of stream), `stepFeatures` (ear features in, for the bit-exact tests), `setControls`, `readSnapshot`, `recorder()`, `latencyHostSamples()`. FTZ/DAZ set on the processing thread (x86 via MXCSR, arm64 via FPCR), switchable off for the exactness tests. |
| Tools | `tools/render.cpp`, `tools/bench.cpp` | `flybrain_render` (brain + raw f32 or WAV in; hits as JSON; `--dump` per-frame binary; `--record`; `--controls c.json`; `--features` bypass; `--double`; `--no-ftz`; `--preroll`; `--trim`). `flybrain_bench` (µs per frame and % of one core, float/double × silence/noise). |

## Tests and results

C++ (`ctest`, 8 binaries, **8/8 pass** in both the Release build and the ASan+UBSan build; the sanitizer build reported nothing). The data they read is produced by `tests/data/make_data.py` from the Python side (git-ignored; the tests print SKIP when it is absent, so run the script first).

| Test | Result |
|---|---|
| `test_brainfile` | small brain loads with the right sizes; corrupt byte → "hash mismatch"; bad magic, wrong version, truncation, index out of range, alpha out of range, missing file all refused with the named reason; SHA-256 known answers; JSON cases. 20 checks. |
| `test_resampler` | block sizes 1/7/64/480/4096 bit-identical at 44.1, 48, 88.2, 96, 176.4, 192 kHz; **agreement with the numpy port: max diff 0.0** (bit-identical, 44.1/48/96 kHz, 32k samples each); 1 kHz tone gain −0.0005/−0.0003/−0.0003 dB; 9 kHz alias at 7 kHz **−92.1/−92.7/−93.8 dB** (bar: −80); impulse at 48 kHz: output peak at index 333 (time-aligned), produced when input 1000+96 arrived. 46 checks. |
| `test_ear` | normalised band features vs the reference engine's over 400 frames: max abs diff 1.4×10⁻¹⁴ (values up to ~4), bar 10⁻⁹; reset reproduces the first frames; FFT vs direct DFT 10⁻¹² level. |
| `test_core` | engine fed the reference's own ear features: **rates bit-exact in double (vs `run_clip` f64) and in float (vs f32)**, also on the history-readout brain (4 blocks of 10 frames). Probabilities differ by 1.1×10⁻¹⁶ (double) and 1.2×10⁻⁷ (float): not bit-exact (see "Bit-exactness"). Hits identical (96/96, 74/74, 77/77). |
| `test_engine` | full chain at 16 kHz double: rate err 6.7×10⁻¹⁶ (limit 2.8×10⁻⁵), prob err 5.6×10⁻¹⁶, hits 96/96; with FTZ off the same; float: rate err 3.6×10⁻⁷, prob 2.5×10⁻⁷, hits identical except near-tie frames that are all explained (74 vs 96 counted hits: the float path resolves plateau ties differently; 0 unexplained). Block sizes 1/64/480/4096 bit-identical; whole-block run gives the same events; hit offset = (frame+2)·80−1; latency 160 samples at 16 kHz. Host rates 44.1/48/96 kHz against the reference on numpy-resampled audio: rate err 4.4×10⁻¹⁶, hits identical (58/58, 56/56, 60/60); latency 537 (12.18 ms), 576 (12.00 ms), 1056 (11.00 ms) host samples. Stereo (x,x) equals mono x; (x,−x) is silence. 60 s of silence, noise and a ±1 square, float and double: all finite, peak rates 0.95 / 5.1 / 12.7, no runaway. Neutral controls equal the baseline bit for bit. Each perturbation (lesion, deafen, gain, tau, sign flip, band offset, unsilence, inject) equals the reference engine's run to 10⁻¹⁵ with identical hits and changes the output vs the baseline (rule H3). Injection with a start frame: identical before it. Mid-stream control change: identical before the swap frame. Gain sweep 1/2/5/10/20/50× with guard limits (ceiling 10³, mean limit 1.0, 5 frames): never NaN/Inf, peak clamped at 1000, 57 resets at 50×; afterwards a neutral run equals the baseline both in a fresh engine and in the same engine after `reset()`. Pre-roll 100 frames mutes hits and leaves the rates untouched. Reset restores the initial state; input trim changes the output. 886 checks. |
| `test_bus` | writer published 456,090 slots in 0.5 s while a reader sleeping 500 µs took 790: every read consistent (checksum) and increasing; the ring drops and counts when full, order kept. |
| `test_recorder` | 60 s recorded: 12,000 frames written, equal to the snapshots frame by frame (rates and hits), 0 dropped, the control change logged at frame 6,001 with the brain hash in the sidecar; a 4-frame ring under a stalled consumer writes 17 and counts 182 dropped out of 199, without blocking. |
| `test_alloc` | global `operator new/delete` counted around `process`, `flush`, `reset`, `stepFeatures` with a control swap and runaway resets inside, float and double, 16k/44.1k/48k: **0 allocations, 0 frees**. |

Python (`pytest plugin/export/tests/test_cpp_engine.py`, **12/12 pass**, 152 s): core bit-exact given the reference ear (double and float); full chain vs reference and vs **PyTorch** (`parity.compare` passes); block sizes; host rates with the latency formula; the numpy resampler ports agree with each other and with the C++ test data; recorder round trip vs the reference with 0 drops and 1 logged control; perturbed run vs `run_perturbed`; a history-readout brain vs reference and PyTorch; and the **10-minute drift at the real size** below.

### 10-minute drift, 3,326 neurons / 109,967 connections (`plugin/reports/cpp_drift_large_{double,float}.json`)

Compared minute by minute (each engine's rates for 10 minutes are 3.2 GB, so the comparison streams).

| Engine pair | Rate error per minute (max abs) | Limit | Prob error | Hits | Unexplained hit diffs |
|---|---|---|---|---|---|
| C++ double vs reference f64 | 1.1×10⁻¹⁴ … 1.4×10⁻¹⁴, flat across the 10 minutes | 2.2×10⁻⁴ | 5.6×10⁻¹⁵ | 19 vs 19 | 0 |
| C++ float vs reference f32 | 5.7×10⁻⁶ … 7.6×10⁻⁶, flat | 2.2×10⁻⁴ | 4.5×10⁻⁶ | 18 vs 18 | 0 |
| Reference f64 vs PyTorch f32 (parent's run) | 6.4×10⁻⁶ overall | 2.2×10⁻⁴ | 2.7×10⁻⁶ | 18 vs 19, the 1 difference ambiguous | 0 |

No growth over time in either precision: the error is rounding in the ear's FFT, not drift. Note the network is one kept at its critical gain with only the readout moved (`make_model(lively="readout")`); the first large export, with theta and log_tau perturbed, blew up to infinity in both PyTorch and the reference engine within a minute, which the parent replaced. The real network sits near its cliff too; the laptop run decides whether float state suffices.

### Bit-exactness status (ADR-3)

- **Rates: bit-exact** against the numpy reference in the same precision, given the same ear features (`test_core`, three brains, float and double). The CSR product in index order and the elementwise update in the reference's order reproduce scipy's CSR product and numpy's elementwise arithmetic exactly, with FMA contraction off.
- **Full chain: not bit-exact, by 10⁻¹⁴ (double) / 10⁻⁷ (float) in the rates.** The ear's FFT differs from pocketfft's (numpy's) in rounding; so does the band sum order. Nothing to fix: the P0 tolerance holds with 10 orders of headroom in double.
- **Probabilities: not bit-exact** even from identical rates: the reference's `(eff * w_hit) @ onehot` is a BLAS gemv whose summation order is OpenBLAS's (I checked: in float32 it happened to equal a sequential sum on this machine, in float64 it did not), and `np.exp` and `std::exp` differ in the last bit. Errors 10⁻¹⁶ (double) / 10⁻⁷ (float). Hits are identical in every test except where the reference itself decides by less than 10⁻⁵, and those are listed, never excused silently.
- **Resampler: bit-exact** with the numpy port (max diff 0.0 in both ports' comparison), which was not required.
- Denormals: FTZ/DAZ is on in the plugin; the reference does not flush. On the test clips it made no difference (the no-FTZ run equals the FTZ run); over long silence a float state would reach denormals after ~17 s where the two could differ by ~10⁻³⁸, far inside the tolerance.

### CPU cost, large brain (`plugin/reports/bench_large.txt`)

3,326 neurons, 109,967 connections, 232 readout neurons, 48 kHz host, 512-sample blocks, 20 s each, one core of this container's Xeon at 2.1 GHz:

| Precision | Input | µs per frame | % of one core at 200 frames/s |
|---|---|---|---|
| float | silence | 187.4 | 3.75 |
| float | noise | 201.9 | 4.04 |
| double | silence | 204.4 | 4.09 |
| double | noise | 207.4 | 4.15 |

Inside the plan's 2–8 % estimate. The resampler (193 taps per output sample) is included. This is a cloud vCPU; a laptop core will differ.

## Not verified

- Only synthetic networks and noise; no real checkpoint, slice or music (none in this container). The laptop P1/P2 run on the real slice is still owed.
- Portability: built with g++ 13 on Linux only. MSVC and Apple clang were kept in mind (no GNU extensions, no VLA, `<xmmintrin.h>` guarded, FPCR inline asm guarded) but not compiled. `M_PI` needs `_USE_MATH_DEFINES` on MSVC (not added; a one-line fix if it bites).
- Real-time safety was checked by the allocation hook and by reading the code for locks and I/O on the audio path (`Recorder::push` is lock-free; `logControl` takes a mutex but is UI-side). No priority-inversion or jitter measurement under a real host.
- The recorder's worker polls with a 1 ms sleep; the control log mutex is UI-side only.
- `latencyHostSamples()` = resampler delay + two hops at the host rate; it reports when a hit is known, not a perceptual alignment, and the host matrix (P3) decides whether hosts compensate it.
- The hit-event `sampleOffset` is the host sample at which the hit was decided (end of the following frame); whether the shell should back-date hits by one frame when the host allows is a P3 question.

## Things the parent should know (not changed here)

1. **Build directories were committed** in `c9647ce` (88 files under `plugin/engine/build` and `build-asan`). I added `plugin/engine/.gitignore` (`build/`, `build-*/`); please `git rm -r --cached plugin/engine/build plugin/engine/build-asan` before the next commit.
2. `tests/data/` is git-ignored except `make_data.py`; CI must run `python -I plugin/engine/tests/data/make_data.py` (needs torch) before `ctest`, or the data tests print SKIP and pass.
3. Reference-engine quirks I did not copy (they only matter after a runaway trigger, which is outside the parity contract): on a trigger `reference.py` calls `ear.reset()`, which discards the samples pending toward the next hop (block-size dependent), and `decoder.reset()`, which leaves the frame just pushed undecided forever (it stays in `pending` and `run_clip` then stacks one frame fewer). The C++ engine keeps the pending samples and publishes that frame with no hits. Suggest changing the reference to match (two lines).
4. `parity.PERTURBATIONS` is defined in `test_reference.py`, not in `parity.py`; `make_data.py` carries its own copy of the list.
5. Format: nothing needs to change in `.flybrain` v1. One wish for v2: an explicit `presumed_sign` per silenced neuron, so Wake the Modulators does not need the UI to choose a sign.
6. The `Snapshot` is published one frame late (when the frame's hits are known) so the UI always sees hits with their frame; after `reset()` the tap can still hold the last pre-reset frame, which carries its own (older) index.
