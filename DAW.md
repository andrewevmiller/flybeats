# flybeats in a DAW

`src/realtime.py` is the only thing that reaches a DAW, and it does so in three
ways: a MIDI file, an audio file, or a live MIDI port. This page covers all
three, what the notes and velocities actually contain, and which of the model's
controls are reachable from the command line.

Everything here is FL Studio-first because that is what gets asked for, but only
the two setup sections are FL-specific — the rest is any host that reads a `.mid`
or opens a MIDI input.

> **You need a model.** Every command below wants `--bundle x.fb` or
> `--checkpoint runs/…/best.pt`. No bundle is published yet
> ([ROADMAP.md](ROADMAP.md) Phase C), so until one is, this path needs a model
> you trained and exported with `scripts/export_bundle.py`. `--config` loads an
> *untrained* model and is for `--benchmark` only; it will emit notes, and they
> will be noise.
>
> Read [Results, and what they are not](README.md#results-and-what-they-are-not)
> before forming an opinion about what comes out.

---

## Which path

| path | you get | extra installs | timing |
|---|---|---|---|
| **A — MIDI file** | a `.mid` to import | none beyond `requirements.txt` | sub-frame, baked into the file |
| **B — audio file** | a 22.05 kHz mono wav | none | baked in, tempo-independent |
| **C — live MIDI port** | notes into a running DAW | `sounddevice`, `mido`, `python-rtmidi`, plus a virtual MIDI cable | quantised to the audio block — 20 ms by default |

Start with A or B. They need no audio device, no MIDI stack and no virtual
cable, and their timing is strictly better than the live path's. Path C exists
because playing into a DAW is the point of a drummer, but it has had **no
hardware testing at all** — see [What live mode is not](#what-live-mode-is-not).

Run everything from the repo root. The modules in `src/` import each other by
bare name, so `python src/realtime.py …` works and `python -m src.realtime` does
not.

---

## Path A — render a MIDI file and import it

```bash
python src/realtime.py --bundle flybeats-8piece.fb --render song.wav --out drums.mid
```

`--out` defaults to `runs/render.mid` when no sample kit is in play. The command
prints the hit count it wrote.

What is in the file:

| | |
|---|---|
| tracks | one, `is_drum=True`, program 0, named `flybeats` |
| channel | 10, the GM percussion channel (`channel=9` zero-indexed) |
| resolution | 960 ticks/beat — a ~0.5 ms grid, below the ~5 ms the model resolves |
| note length | a fixed 50 ms, for every hit |
| velocity | 40–127 (see [Velocity](#velocity-and-why-it-starts-at-40)) |
| timing | absolute seconds from the first sample of `song.wav` |

### The tempo trap

`realtime.py` never sets `initial_tempo`, so the file carries pretty_midi's
default **120 BPM** tempo map, and the hit times are absolute seconds converted
through it. A host that reads the ticks under its own project tempo instead puts
every hit at `t × 120 / project_BPM` seconds — at 140 BPM a hit at 10.000 s lands
at 8.571 s, and the further into the take, the worse it gets.

So: **either set the project to 120 BPM before importing, or accept the file's
tempo when the DAW offers to.** If neither is possible, use path B and drop in a
wav, whose timing cannot be reinterpreted.

Related, and more fundamental: flybeats does not estimate tempo and never has.
The output is a timestamped performance, not a pattern on a grid. Bars will not
line up with the project unless the source audio happened to start on one.

### FL Studio

1. Set the project tempo to **120**.
2. `File → Import → MIDI file`, or drag `drums.mid` onto the playlist. If FL
   offers to take the tempo from the file, say yes.
3. It arrives as one channel on channel 10. Route it at whatever drum sampler
   you use — FPC, DirectWave, Slicex, a layered sampler — and check the pad
   notes against [The note map](#the-note-map), because a sampler's default
   layout is its own, not necessarily GM's.
4. Line the clip up with the source audio at **sample 0**. Hit times are measured
   from the first sample of the wav fed to `--render`, so any offset you give the
   audio track you must give the MIDI clip too.

### Other hosts

Ableton Live, Logic, Reaper, Bitwig, Studio One all import a plain type-1 MIDI
file. The only thing to watch is the same 120 BPM tempo map; in Live, dropping
the file into a MIDI track imports the notes without touching the project tempo,
so set the tempo yourself first.

---

## Path B — render audio and drop it on a track

The one that needs nothing installed beyond `requirements.txt`, and the one to
use when the tempo trap is in the way.

```bash
python src/realtime.py --bundle flybeats-8piece.fb --render song.wav \
    --sound-source samples --kit-dir kits/synth --out drums.wav
```

- Output is **mono at the model's sample rate**, 22 050 Hz. Your DAW will
  resample it; if that matters, resample it yourself with something better than
  a linear interpolator first.
- Input is resampled to 22 050 Hz by linear interpolation and peak-normalised
  before inference. Feeding 22.05 kHz audio skips the resampler entirely.
- A 2-second tail is appended so cymbals ring out instead of being cut off.
- The mix is only normalised if it clips, never quietly.
- The last partial block — under 20 ms — is dropped, so the render can end up to
  one block short of the input.
- `kits/synth/` is the shipped starter kit. It is generated from sines and
  filtered noise, it is not a good kit, and it is there so that this path works
  on a machine with no sampler. Point `--kit-dir` at a real one; see
  [Making sound](README.md#making-sound) for the folder layout, and note that
  third-party kits are auto-mapped by filename with `manifest.yaml`'s `aliases`
  as the override.

Both renders run the **same `StreamingDrummer`** the live path uses, so what a
file contains is what the live path would have played.

---

## Path C — live MIDI into a running DAW

### 1. Install the realtime extras

They are commented out of `requirements.txt` because they need system audio and
MIDI libraries:

```bash
pip install sounddevice mido python-rtmidi
```

`python-rtmidi` ships Windows wheels for **Python 3.11 and 3.12 only**; on 3.13+
pip builds from source and wants the MSVC Build Tools. This is one of the
reasons [SETUP.md](SETUP.md) recommends 3.12.

### 2. Create a virtual MIDI cable

flybeats opens a MIDI *output*; the DAW opens a MIDI *input*. Something has to
join them, and only macOS ships it.

| OS | what to use |
|---|---|
| Windows | [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) — install, add a port, name it something you will recognise |
| macOS | built in: `Audio MIDI Setup → Window → Show MIDI Studio → IAC Driver → tick "Device is online"` |
| Linux | ALSA sequencer ports are virtual already; `mido` opens one directly, or use `a2jmidid` for a JACK host |

### 3. Find the port name

There is no `--list-ports` flag. This is the one-liner:

```bash
python -c "import mido; print(mido.get_output_names())"
```

Pass one of those strings to `--midi-port`. **Always pass it.** With no
`--midi-port`, `mido.open_output()` takes the system default, which on Windows
is usually the Microsoft GS Wavetable Synth — you will hear a General MIDI drum
kit from the OS, the DAW will receive nothing, and it looks exactly like a
routing bug.

### 4. Check the machine can keep up first

```bash
python src/realtime.py --bundle flybeats-8piece.fb --benchmark
python src/realtime.py --bundle flybeats-8piece.fb --benchmark --speed 4
```

It prints mean and p95 inference per block and exits **0 within budget, 1 over**,
so it drops into a shell script. Measured on 4 CPU cores, no GPU:

| subgraph tier | nodes | edges | inference / 20 ms block | headroom |
|---|---|---|---|---|
| live (small) | 10,000 | 643k | 5.4 ms mean, 5.9 ms p95 | 3.7× realtime |
| live (default) | 30,000 | 2.94M | 14.2 ms mean, 16.8 ms p95 | 1.4× realtime |

A latency number from someone else's machine tells you nothing about yours; that
is what the flag is for. Over budget, the fixes are a smaller
`subgraph.max_nodes`, a larger `audio.step_ms`, or path A.

### 5. Play

```bash
python src/realtime.py --bundle flybeats-8piece.fb --midi-port "loopMIDI Port 1"
```

No `--render` means live. It opens a **mono 22 050 Hz input stream** on the
default recording device, prints the port it is sending to, and runs until
Ctrl-C.

Before the stream opens it re-runs the benchmark and **halves `--speed` until the
p95 fits the block**, printing what it capped to. An audio callback that overruns
does not degrade gracefully — it drops buffers and clicks, which sounds like a
broken model rather than a machine out of headroom. If even speed 1 misses, it
says so and plays anyway; expect dropouts.

### 6. FL Studio

1. `Options → MIDI Settings` (F10). Under **Input**, select the loopMIDI port,
   click **Enable**, and give it a **Port number** — say 1.
2. Add your drum sampler to a channel. In the plugin wrapper's settings, set
   **Input port** to the same number, so the notes reach that plugin regardless
   of which channel is selected.
3. Map the pads to [the note map](#the-note-map).
4. To capture it: arm the transport for recording with **Notes** enabled in the
   recording filter, then run flybeats and play the source material into your
   input device.

### Other hosts

- **Ableton Live** — `Preferences → Link/Tempo/MIDI`, turn on **Track** for the
  port, then set a MIDI track's input to it and arm the track.
- **Logic** — any enabled MIDI input reaches the selected track; select the
  drum-sampler track.
- **Reaper** — `Preferences → MIDI Devices`, enable the input, then arm a track
  with that input and monitoring on.

### What live mode is not

Read this before blaming the model for something the plumbing is doing.

- **Live is always MIDI.** `--sound-source samples` is only read on the
  `--render` branch; `run_live` builds a `MidiBank` unconditionally. There is no
  live sample playback, whatever a passing mention elsewhere suggests.
- **Emission is quantised to the block.** Hits carry a sub-frame timestamp — that
  is the whole point of `Trigger.t` — but live, every hit in a block is sent when
  that block finishes. At the default `--block-ms 20` your DAW sees 20 ms
  quantisation on top of its own latency. Lowering `--block-ms` tightens it at
  the cost of more inference calls per second, and below about 5 ms (the encoder
  hop, 110 samples at 22.05 kHz) a block contains no steps at all and nothing
  ever fires.
- **Notes have no length.** Each hit is a `note_on` immediately followed by a
  `note_off` at velocity 0. One-shot drum samplers do not care. Anything that
  needs a held note gets nothing.
- **Everything is on channel 10.** There is no per-class channel option.
- **No clock, no sync, no transport.** flybeats does not send or receive MIDI
  clock, MMC or MTC, and knows nothing about the project tempo. It is free
  running against the audio it hears.
- **The input device must accept 22 050 Hz mono.** PortAudio will often resample
  for you; some interfaces refuse outright.
- **It is untested on hardware.** The live path is written, reviewed and
  unexercised — no audio device has ever been attached to it in this project. If
  it misbehaves, that is the likeliest place to look.

---

## The note map

`src/decoder.py` holds one General MIDI map, and `DrumKit.notes` is the only
thing that reaches MIDI. Both banks use it, so file and live agree.

Note *names* are a DAW convention, not a fact: FL Studio calls MIDI note 60 C5,
while Ableton and Logic call it C3. Match the **numbers**.

| class | note | FL Studio | Ableton / Logic | 3piece | 8piece | articulated |
|---|---|---|---|---|---|---|
| `kick` | 36 | C3 | C1 | ● | ● | ● |
| `sidestick` | 37 | C#3 | C#1 | | | ● |
| `snare` | 38 | D3 | D1 | ● | ● | ● |
| `hat_closed` | 42 | F#3 | F#1 | ● | ● | ● |
| `hat_pedal` | 44 | G#3 | G#1 | | | ● |
| `tom_low` | 45 | A3 | A1 | | ● | ● |
| `hat_open` | 46 | A#3 | A#1 | | ● | ● |
| `tom_mid` | 47 | B3 | B1 | | ● | ● |
| `crash` | 49 | C#4 | C#2 | | ● | ● |
| `tom_high` | 50 | D4 | D2 | | ● | ● |
| `ride` | 51 | D#4 | D#2 | | | ● |
| `ride_bell` | 53 | F4 | F2 | | | ● |

`clap` (39) and `cowbell` (56) are in the map but in no tier.

Which tier you get is `kit.tier` in the config the model was trained with, and
it is fixed at training time — a bundle plays the kit it was trained on. Kit
size is capped by anatomy rather than by the GM chart: the tiers exist because
Phase 0 confirmed 15 wing steering motor types, and nothing above `articulated`
fits without leg motor neurons.

### Choke groups are not in the MIDI

One hi-hat cannot be open and closed at once, and `SampleBank` enforces that
with a 5 ms fade — but choking lives in the sound layer, and `MidiBank` does not
do it. The `note_off` is not a choke either; it goes out immediately after every
`note_on`.

So **set the choke in your sampler**. Most call it a mute group, cut group or
choke group: put `hat_closed` (42) and `hat_open` (46) in one, which is what
`kits/synth/manifest.yaml` declares for the sample path.

---

## Velocity, and why it starts at 40

`MidiBank.to_midi_velocity` is `clip(round(40 + 87 × v), 1, 127)` for a model
velocity `v` in 0..1 — so the DAW **never sees a velocity below 40**, by design,
so that a quiet hit still speaks.

The consequence in a sampler: the bottom third of your velocity layers is
unreachable. If a kit's soft layers matter, either remap 40–127 onto 1–127 with
a velocity curve in the sampler, or pick a kit whose layers are laid out over
the top two thirds.

Where the number comes from matters too. The decoder has a **velocity head** — a
second readout over the same motor pool, trained against the drummer's own MIDI
velocities and wearing the same hemisphere mask, so a hit's strength comes from
the hemisphere that produced the hit. A bundle exported before that head existed
(format v1) still loads and falls back to *peak height above the threshold*,
which is detection confidence wearing dynamics' clothes: a hesitant model plays
quietly and a certain one plays loudly, which is not what a drummer does.

And honestly: the head is built and trained but does not yet produce dynamics on
real audio — it gets two classes backwards. See
[What the velocity head can and cannot do](README.md#what-the-velocity-head-can-and-cannot-do).
Expect a narrow velocity spread in the DAW for now.

---

## The controls, from the command line

All of these work on every path — file render, audio render and live — unless
noted.

| flag | what it does |
|---|---|
| `--speed N` | core updates per encoder frame: 1 as trained, 4 the fly's own timescale |
| `--class-speed kick=1 …` | hold named classes back to a human timescale while the rest run fast |
| `--slider NAME VALUE` | `drive`, `tightness`, `pocket`, `gate` — repeatable |
| `--lesion POP` | mute a confirmed population mid-performance, e.g. `pIP10` |
| `--style N` | swap the groove without swapping checkpoints (needs a multi-style model) |
| `--threshold X` | peak-picking threshold; default is whatever the model's own eval sweep chose |
| `--block-ms X` | audio block size; 20 by default |
| `--midi-port NAME` | which MIDI output to open, live |
| `--kit-dir DIR` | sample kit, for `--sound-source samples` |

### Speed

Inference-time only. No retraining, and at speed 1 the output is unchanged.

| speed | effective τ | |
|---|---|---|
| 1 | 20 ms | as trained |
| 4 | 5 ms | **full fly** — the wingbeat period, and the model's own τ floor |
| 8+ | ≤2.5 ms | faster than the animal; a musical effect, not a biological one |

Cost is linear in the dial, which is why the live path caps it and the offline
renderer does not. The per-class refractory scales with speed by default, so
`--class-speed kick=1` pins the kick to a human timescale while the hats
flutter:

```bash
python src/realtime.py --bundle m.fb --render song.wav --out fly.mid \
    --speed 8 --class-speed kick=1 snare=1
```

See [SPEED_PLAN.md](SPEED_PLAN.md).

### Sliders and lesions

Each one addresses a Phase-0-confirmed population, not a knob someone invented:

| slider | population | effect |
|---|---|---|
| `drive` | pC1 (155 neurons) | fill density, intensity |
| `tightness` | inhibitory neurons (GABA / Glu / His) | tight vs. sloppy timing |
| `pocket` | membrane time constant | slower τ drags the beat |
| `gate` | pIP10 (2 neurons) | song on/off, near-binary |

```bash
python src/realtime.py --bundle m.fb --render song.wav --out busier.mid \
    --slider drive 1.5 --slider pocket 1.2

python src/realtime.py --bundle m.fb --render song.wav --out lesioned.mid \
    --lesion pIP10
```

`--lesion` takes any population in the model's role index — an unknown name
fails with the list of what is available. There is no live control surface for
any of this: sliders are set at launch and hold for the run. Automating them
from the DAW would need a MIDI *input* that flybeats does not have.

---

## Timing, precisely

Worth knowing before you conclude the model is drifting.

- The clock is built from the **true encoder hop**, not the nominal `step_ms`.
  The hop is a whole number of samples — 110 at 22.05 kHz, which is 4.9887 ms,
  not 5 — so a clock built from `step_ms` would drift 0.23 % against the audio:
  9 ms over a four-second clip, 2.3 s over an album side.
- Hits carry a timestamp rather than a step count, because with a fractional
  `--speed` there is no uniform grid to be on.
- The per-class refractory is `50 / speed` ms by default, and a `--class-speed`
  override is absolute.
- Peak picking looks one step ahead **within the block only**. At a block
  boundary the next block sees the peak as a rising edge instead, so changing
  `--block-ms` can change which peaks are picked. Keep it fixed between takes
  you intend to compare.
- Align an imported render to sample 0 of the source audio. Everything is
  measured from the first sample fed to `--render`.

---

## Troubleshooting

| symptom | cause | fix |
|---|---|---|
| a GM drum kit plays from the OS, DAW gets nothing | no `--midi-port`, so `mido` took the system default | pass `--midi-port`, from `mido.get_output_names()` |
| `mido.get_output_names()` is empty on Windows | no virtual cable installed | install loopMIDI and add a port |
| MIDI arrives but nothing sounds | sampler's pads are not on the GM notes, or the wrapper's input port is unset | check [the note map](#the-note-map); set the plugin's **Input port** in FL |
| imported MIDI drifts further out the longer it plays | project is not at 120 BPM | set 120, adopt the file's tempo, or use path B |
| hats sound like both are ringing | no choke group in the sampler — MIDI carries none | put notes 42 and 46 in one mute/cut group |
| every hit is loud, dynamics are flat | v1 bundle with no velocity head, or the head's known limits | re-export from a checkpoint trained with `velocity_head: true`; see [Velocity](#velocity-and-why-it-starts-at-40) |
| velocities never go below 40 | intentional floor in `to_midi_velocity` | remap with a velocity curve in the sampler |
| clicks and dropouts live | inference overran the block | `--benchmark` first; lower `--speed`, `subgraph.max_nodes`, or use `--render` |
| `--speed` printed a "capping to" line | `fit_speed_to_budget` found the dial too fast for this machine | expected; the offline renderer has no such ceiling |
| live mode emits nothing at all | `--block-ms` below the ~5 ms encoder hop, so blocks contain no steps | raise `--block-ms` |
| 154 hits in four seconds | the threshold was not the one the model was scored at | drop `--threshold`, or re-export the bundle with `--threshold` so it ships with the model |
| `--sound-source samples` live produces no audio | live is MIDI-only | use `--render`, or a sampler in the DAW |
| `python-rtmidi` will not install | no wheel for your Python version | use Python 3.12, or install the MSVC Build Tools |
| `ModuleNotFoundError: No module named 'build'` | `src/` modules import each other by bare name | run `python src/realtime.py` from the repo root, not `python -m src.realtime` |

---

## Known limits

Collected in one place, so none of it is a surprise mid-session.

- No MIDI in. No clock sync, no transport, no external control of sliders,
  speed, style or lesions — they are launch-time arguments.
- No VST or AU plugin. flybeats is a command-line process next to the DAW; the
  connection is a MIDI cable or a file.
- Live emission is quantised to `--block-ms`, and the live path is untested on
  real hardware.
- The kit is fixed at training time.
- No tempo estimation, so no grid, no bars, no quantise.
- The MIDI file's 120 BPM tempo map is pretty_midi's default, not a decision
  about your project.
- And the model itself is undertrained. Expect a busy, snare-heavy performance
  that follows the music's energy rather than its groove. The mechanism is
  verified end to end; the musicality is not.
