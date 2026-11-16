# S0: the FL Studio test build

Two throwaway VST3 plugins that answer the five S0 observations in the plan (section 8) before the real plugin shell is built. They are not the flybeats engine; they click and send a MIDI note so you can see and hear what FL Studio does with an effect's inputs, an instrument's inputs, plugin MIDI output and reported latency. Record what you see in `OBSERVATIONS.md`.

| Target | What it is | Buses |
|---|---|---|
| **Fly S0 Effect** (`FlyS0Effect`) | VST3 effect for a mixer insert | stereo main in, stereo main out, stereo **sidechain** in, MIDI out |
| **Fly S0 Instrument** (`FlyS0Instrument`) | VST3 instrument (a generator channel) | stereo out, an optional stereo in, MIDI out |

**Effect behaviour.** Parameter *Listen* picks which input (Main or Sidechain) feeds a crude onset detector (20 ms RMS crossing the *Onset threshold* upward, 100 ms refractory). On each onset it plays a 5 ms click to its audio output and sends MIDI note 36 (C1, velocity from level) on its MIDI output. *Reported latency* (0, 50, 100, 200 ms) is reported to the host, and the main input is passed through delayed by exactly that amount. Its window shows two meters, **main input** and **sidechain input**, whether the sidechain bus is enabled, the onset count and the latency in samples.

**Instrument behaviour.** Once per beat of the host transport it clicks and sends MIDI note 36. Its window shows the level on its input bus (silent if the host connects nothing), whether the host enabled that bus, and the transport state.

## Building

JUCE 8.0.8 is fetched by CMake from GitHub on the first configure (about 100 MB; or put a checkout at `plugin/s0/_deps/JUCE` and it is used as is). CMake 3.22 or newer is needed. No code signing happens in the build.

### Windows (x64, Visual Studio 2022)

```bat
cd plugin\s0
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release --target FlyS0Effect_VST3 FlyS0Instrument_VST3
```

The bundles land at `build\FlyS0Effect_artefacts\Release\VST3\Fly S0 Effect.vst3` and `build\FlyS0Instrument_artefacts\Release\VST3\Fly S0 Instrument.vst3`. Copy each whole `.vst3` folder to `C:\Program Files\Common Files\VST3\`. In FL Studio: Options > Manage plugins > *Find more plugins* (or *Verify plugins*), then the two appear under *Installed*; tick *Favorite* so they show in the plugin menus.

### macOS (Apple Silicon, native arm64)

```bash
cd plugin/s0
cmake -S . -B build -G Xcode            # or: -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release --target FlyS0Effect_VST3 FlyS0Instrument_VST3
```

The build is arm64 only, so FL Studio loads it natively (observation 5). For a universal binary add `-DFLYS0_UNIVERSAL=ON`. Bundles land at `build/FlyS0Effect_artefacts/Release/VST3/Fly S0 Effect.vst3` (and the instrument likewise). Copy each to `~/Library/Audio/Plug-Ins/VST3/`. An unsigned local build normally loads on your own Mac; if macOS or FL Studio refuses it, ad-hoc sign it (`codesign --force --deep -s - "Fly S0 Effect.vst3"`). Proper signing and notarisation are your steps, outside this build (plan, section 3). Then run FL Studio's plugin manager scan as on Windows.

### Linux (compile check only)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target FlyS0Effect_VST3 FlyS0Instrument_VST3
```

Needs JUCE's Linux packages (`libasound2-dev libfreetype6-dev libx11-dev libxcomposite-dev libxcursor-dev libxext-dev libxinerama-dev libxrandr-dev libxrender-dev libfontconfig1-dev libgl1-mesa-dev`). FL Studio does not run on Linux, so this only proves the code compiles.

## The FL Studio steps

Use a project with at least two instrument tracks that make clearly separate sounds (say a bass on one, a lead on another), each on its own mixer track. Do each step on Windows and on macOS.

**Observation 1: an effect on a mixer insert hears other tracks.**
1. Add *Fly S0 Effect* to a slot of an otherwise empty mixer insert, say Insert 10.
2. *Through routing/sends:* in the mixer, select the bass's mixer track and route it to Insert 10 (right-click Insert 10's send switch at the bottom of the mixer, or use "Route to this track only"). Play. The effect's **main input** meter should move with the bass, and you should hear clicks on the bass's onsets. Set *Listen* to *Main input*.
3. *Through the sidechain:* route the lead's mixer track to Insert 10 as a **sidechain** (right-click the send switch > *Sidechain to this track*). Open the effect's wrapper settings (the gear icon > *Processing* tab) and under *Sidechain* / inputs assign that sidechain to the plugin's second input ("Sidechain"). Play. The **sidechain input** meter should move with the lead while the main meter still follows the bass. Set *Listen* to *Sidechain input*: clicks now follow the lead.
4. Record yes/no for each path.

**Observation 2: an instrument channel hears none.**
1. Add *Fly S0 Instrument* as a generator (Channel rack > + > the plugin). Assign it to a mixer track and route the bass track to that mixer track, and also try the sidechain assignment in the wrapper's Processing tab if the wrapper offers inputs for a generator.
2. Play. The plugin's window shows whether the host enabled its input bus and the level on it. Record whether any audio ever reaches it.

**Observation 3: does FL route MIDI out from the effect, from the instrument, or from neither?**
1. For each plugin: open the wrapper settings (gear icon > *Settings* tab) and set the **MIDI output port** to a number, say 1.
2. Add a *MIDI Out* channel or any synth (FL Keys) in the channel rack and set its **MIDI input port** to the same number (synth: wrapper Settings > *Input port*).
3. Play. If the synth sounds on each onset (effect) or each beat (instrument), FL routes that plugin's MIDI out. Record per plugin.

**Observation 4: delay compensation follows the reported latency.**
1. With the effect on Insert 10 and the bass routed to it, set *Reported latency* to 200 ms. Set *Listen* to *Main input* and the threshold so the bass triggers clicks.
2. Play a pattern with the bass on the beat, metronome on. The effect's output is the bass delayed by 200 ms plus an undelayed click at each onset decision, so the click always leaves the plugin about 200 ms *before* the delayed bass. **Judge the clicks against the metronome, not against the bass.** With **plugin delay compensation on** (mixer menu > *Plugin delay compensation* > automatic) the host moves the plugin's whole output 200 ms earlier: the delayed bass lines up with the metronome and the other track, and the clicks land about 200 ms *early* against the metronome. With PDC off, the clicks are on time against the metronome and the bass is 200 ms late. If the clicks sit 200 ms early, compensation follows the reported latency; if they sit on the beat and the bass is late, it does not.
3. Set *Reported latency* **before** pressing play, then stop and start the transport: some hosts read a plugin's latency only when they (re)activate it, so also note whether FL picks up a change made while playing or needs the plugin reloaded. Record the setting, and whether the effect's *reported latency: N samples* line matches 200 ms at your sample rate (9,600 at 48 kHz). Repeat at 50 ms if you want a finer check. Also note what FL shows as the plugin's latency in the wrapper (the *Processing* tab lists it).

**Observation 5: the macOS build loads natively on Apple Silicon.**
1. On the Mac, after copying the bundles, scan plugins. Open the plugin manager's details for *Fly S0 Effect*: it should not say it is bridged. In Activity Monitor, FL Studio should show *Kind: Apple* and no bridge process for the plugin.
2. Record the macOS and FL Studio versions.

Write everything into `OBSERVATIONS.md` and bring it to Gate P2.
