# P3 brief: the plugin shell (P3a) and the Fly Lab UI (P3b)

16 Nov 2026 · chair: Fable · owner's go at Gate P2 (16 Nov: "Start P3a and P3b now, same approach") · builds on `P0-decisions.md`, `gate-P2-memo.md` and the plan's sections 3, 5, 6 and 7. Fable forks build against this brief; the chair integrates, a Fable red team attacks the P3 memo; L1's string audit is part of that review.

## 1. Layout

```
plugin/shell/                       the JUCE 8 plugin (closed-source shell; the engine stays JUCE-free)
  CMakeLists.txt                    targets FlyLab (VST3 effect) and FlyLabMidi (VST3 instrument with MIDI out)
  Source/Processor.{h,cpp}          AudioProcessor: buses, parameters, state, engine, voices, quantize, MIDI out
  Source/Params.h                   every parameter ID and range (section 3), one place
  Source/State.{h,cpp}              the saved state (section 4): ValueTree <-> XML, brain hash, Lab block
  Source/BrainSlot.{h,cpp}          brain loading off the audio thread, hash check, frame-boundary swap
  Source/Quantize.{h,cpp}           the output stage that snaps hits to the host grid (section 6)
  Source/Listening.{h,cpp}          main / sidechain selector, frame-boundary switch with reset
  Source/Strings.h                  every user-visible string (section 8), nowhere else
  Source/ui/                        P3b: one file per panel (section 9), UiModel.h is the only bridge to the processor
  Source/EmbeddedBrain.{h,cpp}      the default brain, bytes in the binary (section 5)
  tests/                            headless tests built as console targets: offline == realtime, state round trip, quantize, listening, lab switch, strings
plugin/voices/                      E7: JUCE-free C++ DSP library flybrain_voices: the 8-piece synthesized kit, the Motor Sonification voices, Fly Sidechain, the output limiter; tests
```

The shell links `flybrain_engine` (plugin/engine) and `flybrain_voices`. Nothing in `plugin/engine` or `plugin/voices` includes JUCE.

## 2. Buses, latency, MIDI, transport (P3a)

- **FlyLab (effect)**: stereo main in, stereo main out, stereo auxiliary input named "Sidechain", MIDI out. **FlyLabMidi (instrument)**: optional stereo in, stereo out, MIDI out; same processor class, a compile-time flag.
- **Mono for the ear** = mean of the chosen input's channels (ADR-6), then input trim, then the engine.
- **Latency reported** = `Engine::latencyHostSamples()` plus the Quantize stage's extra latency in print mode (section 6). Reported at `prepareToPlay` and whenever the mode or Quantize settings change, through `setLatencySamples`; the UI shows the true lateness range (feature #32): resampler delay plus one to two hops, plus the Quantize delay when on.
- **Reset** (zero state, pre-roll starts) on: `prepareToPlay`, sample-rate change, transport start from stop, a play-head jump backwards or forwards by more than one block, loop wrap, brain swap, listening-source switch, Lab switch off (controls to neutral within one frame: no reset, just neutral), and the user's Reset button. Pre-roll default 2 s, parameter 0–4 s. During pre-roll hits are muted; the Wake-up View shows the pre-roll.
- **Live and print modes**: live is the default. Print mode = offline render (`isNonRealtime()`) or the user's "Print" switch; the only differences are Quantize's nearest-grid snapping and its reported latency. The print render of a clip must equal the live render of the same clip bit for bit with Quantize off (test: render the same 16 s through `processBlock` at 64, 480 and 4,096 samples and offline; identical events and audio).
- **MIDI out**: every decided hit, in both targets, as note-on at the kit's first GM note for the drum (`gm_notes` in the brain file: kick 36, snare 38, closed hat 42, pedal hat 44, open hat 46, tom 47, crash 49, ride 51) with velocity = round(1 + 126 × velocity curve(v)), note-off 10 ms later, on channel 10. A parameter "MIDI out" (on/off, default on).
- **Hit timing**: a hit becomes known at `sampleOffset` within the block; the shell triggers the voice and the MIDI note at that sample. No back-dating in v1 (noted for the host matrix).
- **Output**: dry/wet between the main input pass-through (dry) and the kit (wet); in the effect the sidechain input is never passed through. The output limiter (plugin/voices) sits last, always on, ceiling −0.3 dBFS, so no Lab setting can exceed it.

## 3. Parameters (`Params.h`; automatable unless marked)

| ID | Name | Range, default | Notes |
|---|---|---|---|
| `trim` | Input trim | −24..+24 dB, 0 | before the ear; the Ear Panel's comfort zone reads it |
| `listen` | Listening source | Main / Sidechain, Main | not automatable in v1; switch at a frame boundary with a reset |
| `threshold` | Hit threshold | 0.05..0.95, brain's (0.5) | Standard-mode safe control; passes to the control bus |
| `preroll` | Pre-roll | 0..4 s, 2 s | |
| `mode` | Mode | Live / Print, Live | non-automatable |
| `kit` | Kit | Studio / Tight / Room, Studio | three synthesized voicings |
| `drum<k>_gain`, `drum<k>_mute`, `drum<k>_pan` | per drum k = 0..7 | −inf..+6 dB (0); off; −1..1 (per-drum default spread) | |
| `velcurve` | Velocity curve | −1..1, 0 | 0 linear, negative softer, positive harder: v' = v^(2^(−c)) |
| `drywet` | Dry / wet | 0..100 %, 100 % | |
| `midiout` | MIDI out | on/off, on | |
| `quant_on`, `quant_grid`, `quant_strength` | Quantize (DAW-corrected) | off; 1/8, 1/16, 1/32 (1/16); 0..100 % (100) | section 6 |
| `sidechain_amount`, `sidechain_group`, `sidechain_mode` | Fly Sidechain | 0..100 % (0); motor group 0..7 (kick); duck / gate | feature #21: a motor group's rate ducks or gates the plugin's output; off at 0 % |
| `sonif_level`, `sonif_on` | Motor Sonification | −inf..0 dB (−12); off | feature #18 |
| `viz_view`, `viz_fps` | Visualizer | Off / Hop Rings / Motor Strip / Activity Raster / Glow / Fly Scope (Hop Rings); 30 / 60 (30) | non-automatable |
| `lab` | Lab mode | off | **non-automatable**, saved in state |
| `lab_gain` | Gain to the cliff | 0.25..2.0×, 1.0 | Lab only; capped at 1.0 when `guardCalibrated()` is false |
| `lab_tau` | Tempo of thought | 0.25..4×, 1.0 | Lab only |
| `lab_deafen` | Deafen an ear | none / left / right, none | Lab only |
| `lab_bench`, `lab_bench_kind`, `lab_bench_level` | Stimulus Bench | off; click / sweep / AM tone / song pulse; −40..0 dB | Lab only; replaces the input |
| (no parameter) | Lesion Brush | per cell type, UI only, saved in the Lab block | Lab only; automation not offered in v1 |

Automation aimed at a Lab parameter while Lab is off is ignored and the UI shows "ignored (Lab off)" next to it for 2 s (rule from the plan).

## 4. State

A ValueTree `FlyLab` with: `version` (1), every parameter, `brain` {`hash`, `name`, `source`: embedded / file, `path`}, `lab` {every Lab control's value including the lesion mask as a list of silenced cell types, kept when Lab is off}, `ui` {view, window size}. Save and recall round-trips bit for bit. On recall with a missing brain file: report "Brain file not found: <name>, <hash>. The embedded brain is NOT substituted; audio is muted until a brain is chosen." and stay muted (plan: never silently replaced). The hash check on load refuses a file whose hash differs from the recorded one with the same message plus "its contents differ".

## 5. Brain slot

- The default brain is embedded: a `.flybrain` compiled into the binary from `plugin/shell/brains/default.flybrain`. Until the owner exports a real one on the laptop, the build embeds the small synthetic brain made by `plugin/engine/tests/data/make_data.py` and its name says so: "synthetic test network (not the fly)". The CMake option `FLYLAB_DEFAULT_BRAIN=<path>` swaps it.
- Loading another file: off the audio thread (a background thread reads, verifies via `BrainFile::load`, builds the `Engine` and `prepare`s it); the swap happens at a frame boundary by exchanging an engine pointer through the control bus pattern (a `TripleBuffer<Engine*>`-style handoff or a lock-free pointer swap at the block start), followed by a reset; the old engine is deleted on the background thread.
- The brain's `capability` is read but **every brain is treated as observation-only** (P2 rule): the drum mode (kit and MIDI out) is available for every brain only in Lab mode under the banner `Strings::chanceLevelBanner`, except that the embedded synthetic brain also gets it in Standard mode with the banner "synthetic test network" so the plugin is usable before a real export. (The chair's call; the red team may argue.)

## 6. Quantize (DAW-corrected)

From the plan, section 5. Reads tempo, time signature and the musical position from `AudioPlayHead::PositionInfo` every block; grid = 1/8, 1/16 or 1/32 note. Live mode: a hit at host sample s with grid points g_k: target = the first g_k ≥ s; emitted at s + strength × (target − s), never earlier than s; the UI shows the current maximum delay in ms (one grid step at the tempo). Print mode: nearest grid point either side; the stage holds every hit for half a grid step (reported as extra latency, recomputed on tempo change) so earlier moves are possible; emitted at s + strength × (nearest − s). Off: hits pass through untouched, bit for bit. No tempo information from the host: the button greys out with "no tempo from host". Raw hits always go to the Recorder and the Motor Strip; the Motor Strip marks each quantized hit's move. Counts preserved; no hit moves more than one grid step (live) or half (print). Tests: all of the plan's, with a scripted play head (tempo changes, loops, seeks).

## 7. Sampler, sonification, sidechain, limiter (`plugin/voices`, E7)

- **Kit**: eight synthesized voices, no samples (licence-clean by construction): kick (sine sweep 150→45 Hz with a click), snare (tone + filtered noise), closed hat (short filtered noise), pedal hat (shorter, darker), open hat (long noise, choked by closed and pedal), tom (sine sweep 180→90), crash (long noise with band resonances), ride (noise + metallic partials). Three voicings (Studio / Tight / Room) vary decay and filter settings. Velocity scales level and brightness. Per-drum gain, mute and pan, then a stereo sum. Retrigger is click-free (a 1 ms fade on retrigger). `reset()` fades out in 5 ms, never cuts.
- **Motor Sonification** (#18): eight voices, one per motor group, tuned to a documented scale (kick C2, pedal hat G2, snare C3, closed hat G3, open hat C4, tom E4, crash G4, ride C5), amplitude = the standardised motor-group rate z mapped through a soft curve, with 20 ms smoothing; silent input → silent output; bounded level.
- **Fly Sidechain** (#21): a motor group's smoothed rate drives a gain on the plugin's wet output: duck (gain = 1 − amount × z') or gate (gain = amount × z'), with 10 ms attack and 80 ms release. The UI labels it "Fly Sidechain (the fly drives the output)" to keep it apart from the sidechain input.
- **Limiter**: look-ahead-free brick-wall with 1 ms attack and 50 ms release at −0.3 dBFS, on the summed output; a test proves no sample above the ceiling for a +20 dB input.
- Tests (console target `voices_tests`): each voice plays and decays to silence within its documented time; output bounded for any velocity sequence; retrigger and reset have no sample jump above 0.05 between consecutive samples at 48 kHz except at an attack; the sonification is silent for z = 0 and bounded for |z| = 10; the sidechain gain stays in [0, 1]; the limiter ceiling holds; every function is allocation-free after `prepare` (reuse the engine's allocation-hook test idea).

## 8. Strings (`Strings.h`; rules H1–H8; L1 audits them in the P3 review)

- `labBadge` = "Lab mode: perturbed. This is not the network as trained."
- `chanceLevelBanner` = "Chance-level network. No network has learned timing (v0.1). Drum output is for exploration, not a claim."
- `syntheticBanner` = "Synthetic test network (not the fly). For checking the plugin only."
- `quantizeLabel` = "Quantize (DAW-corrected)"
- `quantizeBadge` = "DAW-corrected: the host grid is doing the timing, not the fly. The Scoreboard and every claim about this network describe the un-quantized output."
- `scoreboardTitle` = "Honest Scoreboard" ; `scoreboardNoVerdict` = "No v0.15 verdict. This network has no demonstrated timing skill. Chance, floor and ceiling are shown so that nothing here can be read as a result." ; `scoreboardPerturbed` = "Results describe the Standard (unperturbed) network." ; `scoreboardQuantized` = "Greyed out: Quantize is on; these results describe the un-quantized output."
- `cliffUnmeasured` = "Cliff: unmeasured for this brain (no calibration in the file). Gain is capped at 1×." ; `cliffMeasured(x)` = "Cliff at about {x}× (±0.02×), measured with the repo's silence test."
- `brainMissing(name, hash)` as in section 4; `brainDiffers`.
- `lateByDesign(ms_lo, ms_hi)` = "Late by design: hits are known {lo}–{hi} ms after the sound (resampler + 1–2 frames)." plus " + up to {q} ms from Quantize" when on.
- `wakeUp(s)` = "Waking up: {s} s of pre-roll, hits muted while the network settles."
- `ignoredLabOff` = "ignored (Lab off)".
- `mappingNote` = "Drum-to-limb mapping is our design choice, not biology."
- `credits` = "MaleCNS v1.0 (Janelia FlyEM, CC BY 4.0). Slakh2100 (Manilow et al., CC BY 4.0). VST is a trademark of Steinberg Media Technologies GmbH. Built with JUCE." (L1 checks the VST trademark wording before release.)
- `sidechainLabel` = "Fly Sidechain (the fly drives the output)"; `listenLabel` = "Listening to: Main input / Sidechain input".
- No string anywhere may contain: happy, aroused, focused, mood, IQ, accuracy, score (other than the Scoreboard's facets), or any adjective about how well the network plays.

## 9. Lab UI (P3b), one panel per file, all reading `UiModel`

`UiModel` (in `Source/ui/UiModel.h`) is the only bridge: `readSnapshot(Snapshot&)`, `brain()` (the `BrainFile` for metadata: type names, hop, sign, side, drum, body IDs, band range, cliff, scoreboard JSON), `params()` (the APVTS), `setLabControls(const LabControls&)` (lesions, deafen, gain, tau, bench; compiled by the processor into an engine `Controls` off the audio thread), `recorder` start/stop, `loadBrain(path)`, `reset()`. Panels never touch the processor or the engine. A 30 or 60 fps timer drives repaint; a panel paints from the newest snapshot only.

| Panel | Shows | Test |
|---|---|---|
| Motor Strip (#2) | 8 lanes: motor-group mean standardised rate, hit probability, threshold line, decoded hits, quantized moves | lane values equal the snapshot's |
| Ear Panel + Level Comfort Zone (#3, #33) | 16 bands with each band's training range (band_range) and a dB trim readout always visible | bands equal the snapshot's; range from the brain |
| Hop Rings (#1, #4, #7) | 3,302 dots in rings by hop, lit by rate (documented colour map, no smoothing); modes: sign colours, left/right mirror | pixel colour from the rate map is checked on 20 random neurons against the snapshot |
| Gain to the Cliff + Cliff Meter (#11, #8) | the Lab gain knob with a red zone from the brain's cliff; a gauge of distance to the cliff; "unmeasured" when absent | cap at 1× when uncalibrated |
| Lesion Brush + Deafen an Ear (#9, #10) | cell-type list with silenced toggles; left/right ear toggles | each toggle changes the output on a reference clip (H3), checked headlessly through the processor |
| Honest Scoreboard (#31) | the brain's scoreboard JSON or `scoreboardNoVerdict`; greyed while Quantize is on; `scoreboardPerturbed` in Lab | strings present exactly |
| Experiment Recorder (#28) | start/stop, file path, frames written/dropped, "perturbed" flag in the sidecar | sidecar carries mode and control log |
| Late by Design, Wake-up View, Credits (#32, #37, #40) | the lateness range; the pre-roll countdown; MaleCNS body ID on hover and the credits text | strings present |
| Fly Sidechain (#21), Motor Sonification (#18) | their controls, labelled per section 8 | parameters wired |
| Stimulus Bench (#29) | Lab only: click, sweep, AM tone, song-like pulse replace the input | the bench signal reaches the ear (bands change) |
| Lab switch | toggle, badge, "perturbed" whenever any Lab control is non-neutral; Standard mode forces neutral | Standard-mode output with Quantize off equals the parity baseline whatever the stored Lab block |
| Visualizer (#41, #42) | view selector with master Off, Activity Raster (time × neurons grouped by hop and type, colour by rate), Glow (trails and bloom over Hop Rings, "stylised" tag always visible), Fly Scope (motor-group traces), pop-out window, fps cap | scientific views equal the snapshot; Glow goes dark at baseline rates; cost measured at 30 and 60 fps |
| Brain slot | current brain name, hash, capability, load button, missing/differs messages | load, swap, reload identical; corrupt refused; missing reported |
| Quantize (#new) | button labelled `quantizeLabel`, badge `quantizeBadge`, grid, strength, the delay in ms | strings present whenever on |
| Table stakes | kit, per-drum gain/mute/pan, velocity curve, dry/wet | parameters wired |

**Isolation test**: with the editor open at 60 fps under Xvfb and the processor driven with 64-sample blocks at 48 kHz for 60 s on a thread with a deadline, no block may exceed its budget (1.33 ms) more than 0 times after the first second; measured and reported.

## 10. Exit criteria (from the plan's P3a and P3b rows, made testable here)

P3a: VST3 validator passes on Linux (the SDK's `validator` built from JUCE's bundled SDK) and pluginval strictness 5 if obtainable; reset on loop and seek (scripted play head); state round trip; automation of every automatable parameter changes the output; offline render equals real-time; MIDI out documented per host (only Linux hosts here: none; the host matrix remains the owner's); the listening-source tests from the plan; Quantize's tests.
P3b: every panel's test above; the Lab switch test; the isolation test; every string from section 8 present and nothing outside it containing the banned words; the Recorder sidecar test; the brain slot tests.

What cannot be verified in this container and stays with the owner: FL Studio behaviour on Windows and macOS, the host matrix, signing, listening tests with real music, and every claim that depends on a real brain.
