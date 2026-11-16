# S0 build report (FL Studio test build)

16 Nov 2026 · fork of the Code session, role E4's S0 item · **nothing was run in FL Studio: that is the owner's step** (`plugin/s0/OBSERVATIONS.md` is the form to fill in)

## What was built

`plugin/s0/` is a JUCE 8.0.8 CMake project (JUCE fetched from GitHub by CMake; a checkout at `plugin/s0/_deps/JUCE` is used when present and is git-ignored) with two VST3 targets, manufacturer "flybeats", codes `Flyb`/`Fs0e` and `Flyb`/`Fs0i`, bundle IDs `com.flybeats.flys0effect` and `com.flybeats.flys0instrument`, `JUCE_VST3_CAN_REPLACE_VST2=0`, no code signing in CMake, macOS native arm64 (`-DFLYS0_UNIVERSAL=ON` for universal).

| Target | Buses | Behaviour | Answers |
|---|---|---|---|
| Fly S0 Effect (`Source/EffectProcessor.cpp`) | stereo main in, stereo main out, stereo auxiliary "Sidechain" in, MIDI out | *Listen* (Main / Sidechain) feeds a 20 ms RMS onset detector with a threshold parameter; each onset plays a synthesized 5 ms click and sends MIDI note 36; *Reported latency* (0/50/100/200 ms) is reported to the host and the main input is passed through delayed by that amount; the editor shows "main input" and "sidechain input" meters, whether the sidechain bus is enabled, the onset count and the latency in samples | observations 1, 3, 4 |
| Fly S0 Instrument (`Source/InstrumentProcessor.cpp`) | stereo out, optional stereo in, MIDI out | clicks and sends MIDI note 36 once per beat from the play head's tempo and position; the editor shows whether the host enabled the input bus and its level, and the transport state | observations 2, 3 |

`plugin/s0/README.md` has the Windows (VS 2022, x64), macOS (Xcode/Ninja, arm64) and Linux build steps, where the `.vst3` bundles land and where to copy them, and the FL Studio steps for each observation (mixer-insert routing, sidechain assignment in the wrapper's Processing tab, the wrapper's MIDI output port, plugin delay compensation, loading the instrument as a generator). The observation form is `plugin/s0/OBSERVATIONS.md`.

## What was verified here

- **Linux compile check passes** for both targets (GCC 13.3, Ninja, Release, LTO): `Fly S0 Effect.vst3` and `Fly S0 Instrument.vst3` link, with JUCE's helper writing each bundle's `moduleinfo.json` (VST3 SDK 3.7.12 as bundled in JUCE 8.0.8). Zero warnings from the S0 sources after two fixes (the hidden double-precision `processBlock` overload, a float equality on the beat counter). JUCE's Linux packages were installed with apt for this; they are not needed on the laptop.
- Not run: the VST3 validator and pluginval (neither is installed, and building the SDK's validator was outside "trivially available"); no host loaded the bundles. Windows and macOS builds were not attempted (no toolchain here). CMake's first configure took about 80 s with the local JUCE checkout.

## Things to know before the owner tests

- The first configure fetches JUCE (about 100 MB) unless a checkout is placed at `plugin/s0/_deps/JUCE`.
- `JUCE_DISPLAY_SPLASH_SCREEN` is set to 0 in `CMakeLists.txt`; if the owner builds under JUCE's Starter terms that require the splash screen, set it to 1 (a one-line change; the plan's licence check, L1, decides).
- The effect's clicks are **not** delayed by the reported latency while its pass-through is, on purpose: with compensation on, the pass-through sits on the grid and the clicks show the true lateness.
- The instrument uses small `std::vector`s on the audio thread for note-offs; fine for a test build, not a pattern for the real engine (P2 polices allocation there).
- Out of my scope but noticed: nothing in the plan says which FL Studio sample rate and buffer size to test at; the form has fields for both.
