# Plan: the flybeats VST plugin

Status: **plan only, parked.** No code, no settings change. Written 14 Oct 2026.

**This plan breaks a v0.1 rule, on purpose and on paper only.** `RETROSPECTIVE-v0.0.md` says: *no UI, DAW, soundbank
or release plans until the win rule has been evaluated on the test split, once.* Writing the plan down is cheap.
**Building any of it waits for that evaluation** (step 0 below). Nothing here changes `config/locked.yaml`, the
model, training or scoring.

## What the plugin is

A plugin you put on a drumless track or bus in a DAW. It listens to the audio and plays a drum part live, through
the trained fly network: the same ear, the same wiring, the same readout as training.

- **Input:** audio, at the host's sample rate, stereo or mono.
- **Output:**
  - MIDI notes on the 8 kit pieces at General MIDI pitches: kick 36, snare 38, closed hi-hat 42, pedal hi-hat 44,
    open hi-hat 46, tom 45, crash 49, ride 51. Velocity comes from the loudness output: training targets were
    velocity / 127, so it's `round(127 x vel)`, clamped to 1-127.
  - Audio: the input passed through, plus the drums from a small built-in sample kit (mix knob, drums-only option).
    This covers DAWs that can't route MIDI out of an effect.
  - Recorded hits as a MIDI clip you can drag into the DAW.

## What the model needs at run time (from the code)

| Piece | Source | What it costs per 5 ms frame |
|---|---|---|
| Resample to 16 kHz, mono | new; training audio is Slakh redux 16 kHz | small |
| Ear: causal 512-point FFT every 80 samples, Hann window, 16 bands, `log1p`, fixed mean/std, learned gain/offset | `model/ear.py`, `network.py: ear_input` | one 512-point FFT |
| Network step: `v += alpha * (-v + W r + bias + drive)`, `r = relu(v) * alive` | `network.py: run` | 108,593 multiply-adds (sparse), 3,302 neuron updates |
| Readout: standardise the readout motor neurons, hit logit and loudness per drum (and the 200 ms history blocks, if adopted) | `network.py: readout` | a few hundred multiply-adds |
| Hit = local peak of hit probability at or above 0.5 | `loss.py: peaks` | needs the next frame: +5 ms |

- **It is small.** After training, `W = fixed x g x softplus(theta[pair])` is a fixed sparse matrix: 108,593 values,
  about 1 MB with everything else. One frame is about 0.1 million multiply-adds, 200 frames a second, so about
  20 million a second. **One CPU core handles this easily; no GPU, no PyTorch, no ONNX.**
- **Latency is about 10-15 ms:** the 5 ms frame, the 5 ms peak rule, plus the resampler. The plugin reports it to the
  host, so in playback the hits line up with the audio (plugin delay compensation). Live input is late by that much.
- **The first 2 s are unreliable.** Training ignored each clip's first 2 s while the network settles. The plugin
  runs continuously and mutes output for 2 s after a reset (transport start, sample-rate change).

## The steps

### 0. Gate: what model, and is it worth it (before any building)

- **The test-split evaluation is done** and reported, win or lose.
- **Choose which network ships.** If the real network loses, a random control may play better (in memorisation,
  control 1's F1 was 0.81 against the real network's 0.58). Shipping the real wiring is the point of the project,
  but it should be a stated choice, not a default. *Your call.*
- **Listen to it offline first** (step 1's render script) on:
  - Slakh test songs with the drum stem removed (stem dropout means training sometimes heard drums; a plugin is used
    on music without drums);
  - 5-10 real, non-Slakh recordings. Slakh is rendered from MIDI, so real mixes are a shift in distribution.
- **Go / no-go:** if the drum parts aren't worth listening to on real music, stop here. None of steps 2-6 would
  change that.

### 1. Python: export and a streaming reference (in this repo)

- `scripts/export_plugin_model.py`: `best.pt` → `flybeats-<tag>.fbm`, one file with:
  - the collapsed sparse matrix (CSR), `alpha` and `bias` per neuron, `alive`, ear indices and bands;
  - band mean/std/gain/offset, readout indices, `mn_mean`/`mn_std`, `w_hit`, `u_vel`, `c_hit`, `e_vel`, the drum map,
    and the history profiles if adopted;
  - a header: format version, locked tag, git commit, slice and control seed, checksum.
- `StreamingFlyNet` (numpy, frame by frame, carried state, streaming STFT). A test that it matches `FlyNet` on whole
  clips. This catches anything in the batched code that quietly depends on seeing the whole clip.
- `scripts/render_midi.py`: audio file → MIDI file, through the streaming reference. Usable on its own; it is also
  step 0's listening tool.
- Golden files: 3 fixed clips → band frames, motor-neuron rates, hit logits, hit list. The C++ must reproduce them.

### 2. C++ core library (no plugin framework)

- `plugin/core/`: `Resampler`, `Ear`, `Net` (CSR step), `Readout`, `PeakPicker`, `ModelFile`. About 500 lines.
- **Real-time safe:** everything allocated on load; nothing allocated, locked or logged in `process()`.
- **Checks (Catch2), against the golden files:**
  - band frames and one-frame network step: within float tolerance;
  - whole 16 s clips: **the same hits within one frame**, and every rate within a stated tolerance. Raw float
    agreement over 3,200 frames may not hold: the network starts at the critical gain (1.0 x the largest eigenvalue),
    so tiny rounding differences can grow. If they do, the hit-list comparison is what decides, and that is written
    down before the check runs.
  - a benchmark: CPU per second of audio, target under 2% of one core on the development laptop.
- FFT: `pffft` or JUCE's `dsp::FFT`. Resampler: `r8brain-free` (MIT). Training's 16 kHz files were resampled offline;
  the bands stop at 4 kHz, far below the new Nyquist, so filter differences shouldn't matter. The golden check
  confirms it at 44.1 and 48 kHz.

### 3. Plugin shell

- **JUCE 8 with CMake**, VST3 and standalone first; AU (macOS) and CLAP (`clap-juce-extensions`) later.
- **Licence decision needed:** JUCE 8 is AGPLv3 or paid. The repository has no licence yet. Open-sourcing under
  AGPLv3 is the free route. *Your call.*
- Buses: stereo in; stereo out (dry + kit); MIDI out. The model runs on its own fixed 80-sample frames whatever the
  host's block size, with each note timestamped to its frame inside the block.
- Parameters (saved with the project): input trim, dry/drums mix, drums-only, per-drum mute, velocity scale, MIDI
  channel, kit choice, and **sensitivity** (hit threshold; 0.5 = as scored, marked as such).
- Input level matters: the band normalisation expects Slakh-mix levels. A meter shows where the input sits against
  the training range, with an input trim. No automatic gain until it has been checked against the render script.

### 4. Interface

- Input meter and the 16 bands; 8 drum lanes scrolling hits as they happen; mute and solo per drum.
- "Capture" records the last N bars of hits into a MIDI clip to drag into the DAW.
- About box: MaleCNS v1.0 (Janelia FlyEM, CC BY 4.0) and Slakh2100 (CC BY 4.0) credits, the model's tag and commit.
- Later, if wanted: a live view of the wing and hind-leg motor neurons driving each drum. That's the project's story,
  but only after the plugin plays well.
- Optional: load more than one model file and switch between them, e.g. the real wiring and a random control.

### 5. Testing in DAWs (Windows first)

- `pluginval` at its strictest level, in CI.
- By hand: FL Studio, Reaper, Bitwig, Ableton Live. **Ableton can't take MIDI out of an audio effect**, so there the
  built-in kit or the drag-out clip is the route; check the side-chain route on a MIDI track.
- Checks: sample rates 44.1 / 48 / 96 kHz; block sizes 32-2048 and variable; bypass; offline bounce matching
  real-time playback; project reload restoring state; two instances at once.

### 6. Release

- Windows installer (Inno Setup) and code signing. macOS needs an Apple developer account, signing and notarisation.
- Kit samples: CC0 or self-recorded only, licences listed in `SOURCES.md`.
- Release notes state what the plugin is and isn't: trained on Slakh, playing what it hears, with the test-split
  result linked, whichever way it went.

## Order and rough size

| Step | Depends on | Rough size |
|---|---|---|
| 0. Gate | test-split evaluation | a few evenings of listening |
| 1. Export, streaming reference, render script | 0 (the chosen checkpoint) | 2-3 days |
| 2. C++ core and golden checks | 1 | 3-5 days |
| 3. Plugin shell | 2 | 3-4 days |
| 4. Interface | 3 | 3-5 days |
| 5. DAW testing | 3-4 | 2-3 days, then ongoing |
| 6. Release | 5 | 2 days (Windows), more for macOS |

Step 1 is the main risk-reducer: the render script answers "is this worth a plugin?" before any C++ exists, and its
golden files make every later step checkable.

## Decisions for you

1. Ship the real network regardless of the result, or the best player?
2. Licence: AGPLv3 open source (JUCE free), or pay for JUCE?
3. Windows only at first, or macOS as well (developer account needed)?
4. MIDI out only to start, or the built-in kit from the start?
