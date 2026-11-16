# P3 voices report: the synthesized kit, Motor Sonification, Fly Sidechain and the limiter

16 Nov 2026 · fork E7 (Fable) · code: `plugin/voices/` (static library `flybrain_voices`, JUCE-free, no third-party dependency, 1 source file, 1 header) · brief: `plugin/docs/P3-brief.md` section 7 · **nobody has listened to any of this**: every number below is a measurement in this container; the WAVs are rendered for the owner's ears.

## What was built

Namespace `flybrain::voices`, header `plugin/voices/include/flybrain/voices.h` (the exact interface the brief asked for; the shell links against it):

| Class | Methods | Behaviour |
|---|---|---|
| `Kit` | `prepare(sampleRate, maxBlock)`, `reset()`, `setVoicing(int)`, `setDrum(drum, gainDb, mute, pan)`, `setVelocityCurve(c)`, `trigger(drum, velocity01, sampleOffset)`, `process(L, R, n)` (adds), `anyActive()` | Eight synthesized voices, two slots per drum (a retrigger starts the other slot and fades the old one over 1 ms), `reset()` fades every sounding voice over 5 ms, open hat choked over 5 ms by the closed and the pedal hat, velocity curve v' = v^(2^−c), velocity also drives brightness (low-pass cutoff), equal-power pan, per-drum gain and mute, up to 64 queued triggers per block, offsets clamped to the block. No samples anywhere. |
| `Sonification` | `prepare`, `reset`, `setLevelDb`, `setEnabled` (20 ms fade), `setGroupRates(z, n)`, `process` (adds), `noteHz(group)` | Eight sines on C2 G2 C3 G3 C4 E4 G4 C5 (65.41, 98.00, 130.81, 196.00, 261.63, 329.63, 392.00, 523.25 Hz); amplitude = tanh(max(z, 0)/3) smoothed with a 20 ms time constant; the eight at full amplitude sum to `level`; NaN or negative z counts as 0. |
| `Sidechain` | `prepare`, `reset`, `setAmount`, `setMode(Duck/Gate)`, `setGroupRate(z)`, `process` (multiplies), `currentGain()` | z' = tanh(max(z, 0)/2); duck gain 1 − amount·z', gate gain amount·z' (amount 0 = untouched); 10 ms attack (gain falling), 80 ms release (rising); clamped to [0, 1]. |
| `Limiter` | `prepare`, `reset`, `process`, `ceiling()`, `currentGain()` | Gain envelope follows min(1, ceiling/peak) with 1 ms attack and 50 ms release, then a hard clip at −0.3 dBFS (0.96605) makes the guarantee; NaN or Inf input samples become 0. No look-ahead, as the brief asked, so the clip is what holds the ceiling on a sudden peak. |

Voice design (Studio / Tight / Room decay seconds): kick sine sweep 150→45 Hz over 60 ms plus a 4 ms noise click (0.35 / 0.20 / 0.50); pedal hat dark short noise (0.04 / 0.025 / 0.06); snare 180 + 330 Hz tone (0.12 / 0.08 / 0.18) plus high-passed noise (0.18 / 0.12 / 0.26); closed hat bright short noise (0.06 / 0.04 / 0.09); open hat long noise (0.45 / 0.30 / 0.70); tom sweep 180→90 Hz over 100 ms (0.30 / 0.18 / 0.45); crash noise through 3.5 kHz and 6 kHz resonances (1.2 / 0.8 / 1.8); ride noise plus 2.1, 3.3 and 4.9 kHz partials and a 4.2 kHz resonance (0.9 / 0.6 / 1.3).

## Tests and results

`voices_tests` (88 checks) and `voices_alloc_test`: **2/2 pass** in Release and in the ASan+UBSan build (nothing reported). Zero compiler warnings with `-Wall -Wextra`. 48 kHz throughout.

| Drum (Studio, velocity 1) | Peak | −60 dB after | Bound in the test | Largest sample-to-sample jump within one hit |
|---|---|---|---|---|
| kick | 0.679 | 264 ms | 600 | 0.117 |
| hihat_pedal | 0.258 | 40 ms | 120 | 0.360 |
| snare | 0.437 | 177 ms | 400 | 0.704 |
| hihat_closed | 0.317 | 60 ms | 150 | 0.475 |
| hihat_open | 0.291 | 438 ms | 800 | 0.444 |
| tom | 0.685 | 255 ms | 500 | 0.040 |
| crash | 0.366 | 1,092 ms | 1,800 | 0.286 |
| ride | 0.269 | 842 ms | 1,400 | 0.184 |

- Every drum is silent (< 10⁻⁴) 2.5 s after the hit; all three voicings decay (crash silent by 2.8 s in each).
- **Bounded**: 2,000 random triggers over 4 s across all drums and velocities, 4,096-sample blocks: peak 0.98 (before any limiter), all finite. Louder velocity gives a louder hit; the velocity curve at −1 is softer and at +1 harder at mid velocity.
- **Click-free retrigger and reset**: with two retriggers (30 and 45 ms) and a `reset()` at 70 ms, the largest jump outside the 2 ms attack windows is within 1.5× the voice's own largest jump within a single hit, for all eight drums. The brief's "no jump above 0.05" cannot hold for noise voices (their jumps are the sound; see the table), so the test compares against each voice's own natural jump instead. **Reset never cuts**: the first millisecond after `reset()` still carries energy (> 10 % of the RMS just before), and the output is exactly 0 six milliseconds later with no voice active.
- **Choke**: the open hat is silent 8 ms after a closed-hat or pedal-hat trigger, and not choked by a kick.
- **Sonification**: exactly 0 for z = 0; |z| = 10 on all groups at 0 dB gives peak 0.654 (bounded, below 1); a NaN rate is treated as 0; the envelope reaches 75 % of its settled value at 20 ms after a step (one-pole, 20 ms time constant); the scale frequencies are as documented.
- **Sidechain**: duck gain stays in [0, 1]; attack 63 % at 9.9 ms, release 63 % at 79.5 ms; gate closes (< 0.01) at z = 0 and opens (> 0.95) at z = 10; amount 0 leaves the signal untouched bit for bit.
- **Limiter**: a +20 dB sine (amplitude 10) comes out at peak 0.96605 = the ceiling, never above; a −6 dB sine passes bit for bit; NaN and Inf inputs come out finite.
- **Allocation hook**: 200 blocks exercising trigger, setVoicing, setDrum, setVelocityCurve, reset, and all four `process` calls, with sonification and sidechain parameter changes: **0 allocations, 0 frees**.
- **CPU** (one vCPU of this container, 2.1 GHz Xeon, 512-sample blocks): kit with all eight drums retriggered every 20 blocks (up to 16 voices sounding) 96 µs per block = 0.90 % of one core; sonification 62 µs = 0.58 %. Per voice roughly 6 µs per block.

## Listening material

`voices_render <dir>` writes `kit_studio.wav`, `kit_tight.wav`, `kit_room.wav` (every drum at velocities 0.3, 0.6, 1.0, 0.6 s apart, through the limiter; 15.4 s, peaks 0.76–0.83, no DC) and `sonification.wav` (8 s, the eight voices driven by sine-modulated z in [0, 4] at −6 dB, peak 0.26). Rendered here to the scratchpad; the owner runs the tool on the laptop to listen (`plugin/voices/build/voices_render <dir>`).

## Not verified

- Nobody has listened: the voices are a first synthesized design that meets the measurable brief (decay, bounds, clicks, choke); whether they sound like a usable kit is the owner's call, and the voicings are three parameter sets, not three tuned kits.
- Built with g++ 13 on Linux only; `M_PI` in the tests and the tool needs `_USE_MATH_DEFINES` on MSVC (the library itself defines its own constant).
- The limiter's guarantee comes from the hard clip; the 1 ms attack envelope reduces but does not remove clipping on a sudden full-scale transient (by design, no look-ahead per the brief). If the shell reports audible clipping on crash hits at +6 dB per-drum gain, a 1 ms look-ahead would be the change.
- The retrigger test's criterion (1.5× the voice's own jump) is the chair's to ratify; it replaces an unmeetable one.
- The sonification's "soft curve" (tanh(z/3)) and the sidechain's (tanh(z/2)) are choices, not derived from any data; z is the standardised motor-group rate the shell computes from the snapshot (mean over the group's readout neurons of z), which this library does not compute.

## For the shell fork

Link `flybrain_voices` (CMake target in `plugin/voices/CMakeLists.txt`, `add_subdirectory` it with `FLYBRAIN_VOICES_BUILD_TESTS=OFF` or on). Call order per block: zero or fill the output buffers with the dry signal, `kit.trigger(...)` for each hit at its sample offset, `kit.process(L, R, n)` (adds), `son.setGroupRates` once per 5 ms frame then `son.process` (adds), `sc.setGroupRate` per frame then `sc.process` (multiplies the wet mix), `lim.process` last. All four classes must be `prepare`d at `prepareToPlay` and `reset` on the plugin's resets.
