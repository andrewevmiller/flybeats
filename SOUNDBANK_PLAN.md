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
5. **Done.** `hot_swap()` with preload and an atomic pointer swap — and the
   swap had to become genuinely atomic to earn the name. A kit was three
   attributes rebound one at a time, which is three stores against three loads
   in `trigger`: a callback landing between them reads the new `layers` with
   the old round-robin table and indexes `rr[(cls, idx)]` for a layer the old
   kit never had, raising `KeyError` inside the audio callback. Swapping to a
   kit with *more* layers is what turns it from a wrong sample into a crash.
   The three now live in a frozen `_Kit` and `trigger` snapshots the reference
   once, so a trigger holds either the whole old kit or the whole new one.
   Tested against a thread triggering and mixing without pause while the main
   thread swaps 40 times; the test fails against the old three-store version.
6. **Done.** Wired into `realtime.py` alongside the `MidiBank`/`mido` path.
7. `SoundFontBank` (FluidSynth) as a later addition — same interface, no
   changes needed upstream
8. **Done.** User-upload auto-mapping + manual-mapping fallback.
   `guess_class` maps arbitrary folder and file names onto classes, and a flat
   folder of wavs — how most uploads arrive — loads as a kit without a manifest.

   Two rules the keyword table lives by. Specific beats generic, because every
   generic name is a substring of a specific one in the direction that breaks
   things: "open hat" must not file as a closed hat, "ride bell" must not file
   as a ride. And the two-letter abbreviations match whole words only: "oh" is
   inside "ohio" and "ch" is inside "chorus", so substring-matching them
   mis-files half a sample library.

   Nothing recognisable is *not* guessed at. It is recorded, and
   `mapping_report()` prints what was auto-mapped, what was dropped, and the
   `aliases` stanza that overrides it — which is the manual-mapping fallback in
   the form this repo has, since `aliases` was already read and already wins.

### Note on step 7

Not done, deliberately. FluidSynth is not installed in this environment and
there is no audio device, so a `SoundFontBank` written here could not be
exercised at all — not one note. The repo already carries "live audio has never
been run" as a thing that will bite; adding a second unexercised backend makes
that worse rather than better. It wants a machine with a sound card.
