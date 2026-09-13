# flybeats: Customizable Drum Sounds — SoundBank Implementation Plan

> **Rebrand note (do this first):** project renames from **FlyDrums** → **flybeats**,
> exclusively. No dual-naming, no "formerly FlyDrums." Before starting the SoundBank
> work below, do a single clean rename pass:
> - Repo/folder name
> - `configs/*.yaml` headers and comments
> - CLI tool name + `--help` text
> - MIDI device name string in `realtime.py` (Phase 5)
> - Any docstrings/README/comments referencing the old name (grep for stragglers)

---

## Goal

Decouple *what the connectome decided to hit* from *what it sounds like*. The
decoder already emits `(drum_class, velocity, timestamp)` — this plan only
touches what happens after that tuple is produced. Zero retraining required
to swap kits.

## New module: `src/soundbank.py`

Abstract base + three interchangeable backends, all conforming to one
interface so the realtime engine never needs to know which is active:

```python
class SoundBank(ABC):
    def load(config: dict) -> None: ...
    def trigger(cls: str, velocity: float, t: float) -> None: ...
    def hot_swap(new_config: dict) -> None: ...
    def validate(kit_classes: list[str]) -> list[str]:  # returns missing classes
```

| Backend | Description |
|---|---|
| `SampleBank` | WAV layers, local playback via `sounddevice` |
| `SoundFontBank` | FluidSynth-backed `.sf2` / `.sfz` |
| `MidiBank` | thin wrapper around existing Phase 5 `mido` output |

## Config surface

```yaml
# configs/kit_sound.yaml
kit_tier: 8piece
sound_source: samples   # samples | soundfont | midi_passthrough
mapping:
  kick:  {dir: samples/808/kick,  layers: 4}
  snare: {dir: samples/808/snare, layers: 4}
  chh:   {dir: samples/808/chh,   layers: 3, choke_group: hihat}
  ohh:   {dir: samples/808/ohh,   layers: 3, choke_group: hihat}
  # ... rest of confirmed classes from Phase 0 kit map
```

## Velocity layers + round-robin

Avoid a hard `floor(v * N)` layer cutoff — it produces an audible seam where
sample identity jumps.

- Crossfade adjacent layers: `layer_idx = v * (N-1)`, blend `floor` and
  `ceil` layers by the fractional part
- Within a chosen layer, cycle variants with **no-immediate-repeat
  round-robin** (shuffle a bag, refill when empty) — pure random noticeably
  repeats over short loops

## Choke groups

Config: `{hihat: [chh, ohh]}` → live `active_voices: dict[group -> Voice]`.

On trigger:
1. If the incoming class belongs to a group with a currently sounding voice,
   fade that voice out fast (~5 ms) before starting the new one
2. Otherwise allocate a new voice

Lives in `SoundBank`, not the model — it's acoustic physics (one hi-hat can't
be open and closed at once), not something the connectome should learn.

## Playback engine constraints

Given Phase 5's ~20 ms buffer + ~10 ms inference budget, the mixer must not
do disk I/O or allocation inside the audio callback:

- Preload every sample as an in-memory numpy array at `load()` /
  `hot_swap()` time — drum one-shots are tiny, cheap even for a full
  kit × layers × variants
- Fixed-size voice pool (steal-oldest on overflow), summed each callback
  block
- `hot_swap()` preloads the new bank on a background thread, then does an
  atomic pointer swap between callback blocks — never mutate the live bank
  in place

## Kit layout & validation

```
kits/808/
  manifest.yaml   # declares choke groups, layer velocity ranges
  kick/   {01_soft.wav, 02_mid.wav, 03_hard.wav}
  snare/  {...}
  chh/    {..., choke_group: hihat}
  ohh/    {..., choke_group: hihat}
```

`validate()` diffs the kit's classes against Phase 0's confirmed
motor-neuron-derived class list and returns anything missing. Missing
classes fall back to silence with a logged warning rather than crashing —
lesion mode already needs "some classes just don't fire" to be a first-class
outcome, not an error state.

## User-uploaded kits

- Filename-keyword auto-mapping (`kick|bd`, `snare|sd`, `hat|hh`, etc.)
- Manual-mapping fallback UI when confidence is low
- Same `validate()` path applies before the kit can be activated

## Where this touches existing phases

- **Phase 5 (`realtime.py`)**: MIDI-out path gets this for free via
  `MidiBank`; add a parallel local-render path using `sounddevice` for users
  without an external synth
- **Lesion mode**: more legible with distinct sounds — muting a cell type is
  easier to hear with a crisp 808 than a mushy default kit

## Suggested file additions

```
flybeats/
  src/
    soundbank.py          # SoundBank ABC + SampleBank/SoundFontBank/MidiBank
    voice.py               # Voice, choke-group, and mixer/voice-pool logic
  kits/
    808/                    # example sample kit + manifest.yaml
    acoustic/               # example sample kit + manifest.yaml
  configs/
    kit_sound.yaml
```

## Implementation order (suggested)

1. Rebrand pass (see note at top)
2. `SoundBank` ABC + `SampleBank` (covers the common case first)
3. Voice pool + choke groups + crossfaded velocity layers
4. `validate()` against Phase 0's confirmed class list, with silent-fallback
   behavior for missing classes
5. `hot_swap()` with background preload + atomic pointer swap
6. Wire into `realtime.py` alongside existing `MidiBank`/`mido` path
7. `SoundFontBank` (FluidSynth) as a later addition — same interface, no
   changes needed upstream
8. User-upload auto-mapping + manual-mapping fallback
