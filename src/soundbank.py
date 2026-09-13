"""What a drum hit sounds like, decoupled from the decision to hit it.

The model emits ``(class, velocity, time)``. Everything downstream of that tuple
is taste, not neuroscience, so it lives behind one interface with interchangeable
backends and nothing upstream needs to know which is active:

    MidiBank    notes out to a sampler or DAW -- the original Phase 5 path
    SampleBank  WAV layers mixed here, so the model makes sound on its own

``SampleBank`` exists because "MIDI out" is not a working instrument on a
machine with no sampler installed: it emits notes that nothing plays. With a
kit loaded, ``src/realtime.py --sound-source samples`` renders or plays audio
with no other software involved.

Velocity is a float in 0..1 at this interface, not a MIDI 1..127. The MIDI
scaling belongs at the MIDI edge; a sample backend converting 1..127 back into
a gain would be a lossy round trip through a unit it should never have seen.

    Velocity now comes from the decoder's own velocity head, trained against
    the drummer's MIDI velocities, so the layers crossfade on dynamics rather
    than on detection confidence. A model exported before that head existed has
    no such output and falls back to peak height, which is the old behaviour
    and reads as a hesitant model playing quietly.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from voice import RoundRobin, VoicePool, layer_gains


#: Filename keywords for auto-mapping a kit nobody wrote a manifest for.
#: Ordered most specific first and matched first-wins, because the generic
#: names are substrings of the specific ones in every direction that matters:
#: "open hat" has to beat "hat", "ride bell" has to beat "ride", and "floor
#: tom" has to beat "tom". Two forms per class -- ``phrases`` are matched
#: against the name with its separators removed, so "open hat", "open-hat" and
#: "OpenHat" are one pattern; ``tokens`` must match a whole word, because the
#: two-letter drum abbreviations are substrings of ordinary words ("oh" is
#: inside "ohio", "ch" is inside "chorus") and a substring match on them
#: mis-files half a sample library.
_KIT_KEYWORDS: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...] = (
    ("ride_bell",  ("ridebell", "bellride"),                     ("bell", "rb")),
    ("hat_pedal",  ("pedalhat", "hatpedal", "foothat", "hhpedal"), ("ph", "phh")),
    ("hat_open",   ("openhat", "hatopen", "openhh", "hhopen", "hihatopen"),
                                                                 ("oh", "ohh", "opnhat")),
    ("hat_closed", ("closedhat", "hatclosed", "closedhh", "hhclosed", "hihatclosed",
                    "hihat", "highhat"),                         ("ch", "chh", "hh", "hat")),
    ("sidestick",  ("sidestick", "crossstick", "rimshot", "rimclick"),
                                                                 ("rim", "xstick", "ss")),
    ("tom_low",    ("floortom", "lowtom", "tomlow", "tomfloor", "tom3"),
                                                                 ("ft", "lt")),
    ("tom_mid",    ("midtom", "tommid", "middletom", "tom2"),    ("mt",)),
    ("tom_high",   ("hightom", "tomhigh", "racktom", "tom1"),    ("ht",)),
    ("crash",      ("crash", "cymbalcrash"),                     ("cr", "cy")),
    ("ride",       ("ride", "cymbalride"),                       ("rd",)),
    ("cowbell",    ("cowbell",),                                 ("cow", "cb")),
    ("clap",       ("handclap", "clap"),                         ("cp",)),
    ("kick",       ("kick", "bassdrum", "basedrum"),             ("bd", "kik", "kd")),
    ("snare",      ("snare",),                                   ("sd", "sn", "snr")),
    ("tom_low",    ("tom",),                                     ("tom",)),
)


def guess_class(name: str) -> str | None:
    """Map an arbitrary kit folder or filename onto a drum class.

    For kits a user uploaded rather than authored for this repo: sample
    libraries name things "BD_01.wav", "Snr Hard", "closed hh". Returns None
    when nothing matches, which is not an error -- an unmapped folder is
    reported rather than guessed at, and the manifest's ``aliases`` is the
    manual override for exactly that case.

    A bare "tom" resolves to ``tom_low`` last of all, after every numbered and
    named variant has had its chance: a kit with one unqualified tom is more
    often a floor tom than anything else, and putting it on a real class beats
    dropping it.
    """
    flat = re.sub(r"[^a-z0-9]+", "", name.lower())
    tokens = set(re.split(r"[^a-z0-9]+", name.lower())) - {""}
    for cls, phrases, toks in _KIT_KEYWORDS:
        if any(p in flat for p in phrases) or (tokens & set(toks)):
            return cls
    return None


@dataclass(frozen=True)
class _Kit:
    """Everything ``trigger`` needs, held as one object so it swaps atomically.

    These three used to be three attributes on the bank, and ``hot_swap``
    rebound them one at a time. That is three separate stores against three
    separate loads in ``trigger``, and an audio callback lands between them:
    it reads the new ``layers`` and the old ``rr``, then indexes
    ``rr[(cls, idx)]`` for a layer the old kit did not have and raises
    ``KeyError`` inside the callback. Rebinding one reference cannot tear --
    a trigger holds either the whole old kit or the whole new one.
    """
    layers: dict[str, list[list[np.ndarray]]] = field(default_factory=dict)
    groups: dict[str, str] = field(default_factory=dict)
    rr: dict[tuple[str, int], RoundRobin] = field(default_factory=dict)
    name: str = ""


class SoundBank(ABC):
    """One way of turning ``(class, velocity, time)`` into something audible."""

    #: Sample backends mix into the audio stream; MIDI backends do not.
    produces_audio: bool = False

    @abstractmethod
    def load(self, config: dict) -> None:
        """Preload everything needed to sound a hit without further I/O."""

    @abstractmethod
    def trigger(self, cls: str, velocity: float, t: float = 0.0) -> None:
        """Sound one hit. ``velocity`` is 0..1; ``t`` is seconds, for backends
        that place events on a timeline rather than playing them now."""

    def mix(self, n: int) -> np.ndarray | None:
        """Render ``n`` samples, or ``None`` for a backend that makes no audio."""
        return None

    def validate(self, kit_classes: list[str]) -> list[str]:
        """Classes the model can emit that this bank cannot sound."""
        return []

    def hot_swap(self, new_config: dict) -> None:
        """Replace the loaded kit. Default: a plain reload."""
        self.load(new_config)

    def close(self) -> None:
        pass


class MidiBank(SoundBank):
    """Notes out, exactly as Phase 5 always did.

    Kept as a backend rather than a special case so the realtime loop has one
    code path, and so "does the refactor change the output?" is a testable
    question: the note and velocity mapping here is the same arithmetic that
    was inline in ``realtime.py``.
    """

    produces_audio = False

    def __init__(self, kit, port=None):
        self.kit = kit
        self.port = port
        self.events: list[tuple[float, int, int]] = []     # (t, note, velocity)
        self._note = {c: n for c, n in zip(kit.classes, kit.notes)}

    def load(self, config: dict) -> None:
        return None

    @staticmethod
    def to_midi_velocity(velocity: float) -> int:
        """0..1 -> 1..127, floored at 40 so a quiet hit still speaks."""
        return int(np.clip(round(40 + 87 * float(np.clip(velocity, 0.0, 1.0))), 1, 127))

    def trigger(self, cls: str, velocity: float, t: float = 0.0) -> None:
        note = self._note.get(cls)
        if note is None:
            return
        vel = self.to_midi_velocity(velocity)
        self.events.append((t, note, vel))
        if self.port is not None:
            import mido
            self.port.send(mido.Message("note_on", channel=9, note=note, velocity=vel))
            self.port.send(mido.Message("note_off", channel=9, note=note, velocity=0))

    def validate(self, kit_classes: list[str]) -> list[str]:
        return [c for c in kit_classes if c not in self._note]

    def close(self) -> None:
        if self.port is not None:
            self.port.close()


class SampleBank(SoundBank):
    """WAV layers, mixed here. The kit that needs no other software.

    A kit directory holds one folder per drum class, each containing WAV files
    sorted quietly-to-loudly; ``manifest.yaml`` names the choke groups and any
    aliases. Everything is decoded into memory at ``load`` time -- drum
    one-shots are small, and the alternative is disk I/O inside an audio
    callback, which is how you get dropouts.
    """

    produces_audio = True

    def __init__(self, sample_rate: int = 22_050, max_voices: int = 32, seed: int = 0):
        self.sample_rate = sample_rate
        self.pool = VoicePool(max_voices=max_voices, sample_rate=sample_rate)
        self.seed = seed
        self._kit = _Kit()
        self.unmapped: list[str] = []          # files no keyword matched
        self.automapped: dict[str, str] = {}   # source name -> class

    # Read-only views, so nothing outside can rebind half a kit.
    @property
    def layers(self) -> dict:
        return self._kit.layers

    @property
    def groups(self) -> dict:
        return self._kit.groups

    @property
    def rr(self) -> dict:
        return self._kit.rr

    @property
    def name(self) -> str:
        return self._kit.name

    # -- loading ------------------------------------------------------------
    def load(self, config: dict) -> None:
        layers, groups, rr = self._read_kit(config)
        self._kit = _Kit(layers, groups, rr,
                         str(config.get("name") or config.get("dir") or ""))
        self.pool.reset()

    def _read_kit(self, config: dict):
        import soundfile as sf
        import yaml

        root = Path(config["dir"])
        manifest_path = root / "manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text()) if manifest_path.exists() else {}

        groups: dict[str, str] = {}
        for group, members in (manifest.get("choke_groups") or {}).items():
            for m in members:
                groups[m] = group
        # a kit may name its folders in its own dialect (chh, bd); the model's
        # class names are the repo's, so aliases resolve one onto the other
        alias = {a: c for c, aliases in (manifest.get("aliases") or {}).items()
                 for a in aliases}

        layers: dict[str, list[list[np.ndarray]]] = {}
        rr: dict[tuple[str, int], RoundRobin] = {}
        self.unmapped: list[str] = []
        self.automapped: dict[str, str] = {}

        folders = sorted(p for p in root.iterdir() if p.is_dir())
        by_class = ({d.name: sorted(d.glob("*.wav")) for d in folders} if folders
                    else self._group_flat(sorted(root.glob("*.wav"))))

        for name, files in by_class.items():
            if not files:
                continue
            # A manifest alias is a manual override and always wins; then the
            # filename keywords; then the folder's own name, on the assumption
            # that a kit authored for this repo already uses our class names.
            if name in alias:
                cls = alias[name]
            else:
                guess = guess_class(name)
                cls = guess or name
                if guess and guess != name:
                    self.automapped[name] = guess
            # <layer>_<variant>.wav groups by layer; a flat folder is one layer
            by_layer: dict[str, list[Path]] = {}
            for f in files:
                by_layer.setdefault(f.stem.split("_")[0], []).append(f)
            ordered = [by_layer[k] for k in sorted(by_layer)]
            loaded = [[self._read_wav(sf, f) for f in group] for group in ordered]
            layers[cls] = loaded
            for i, variants in enumerate(loaded):
                rr[(cls, i)] = RoundRobin(len(variants), seed=self.seed + i)
        return layers, groups, rr

    def _group_flat(self, files: "list[Path]") -> "dict[str, list[Path]]":
        """A kit that is one folder of wavs, which is how most uploads arrive.

        Each file is mapped on its own name and the class becomes the group, so
        ``BD_01.wav BD_02.wav Snr.wav`` lands as two classes rather than three.
        Files nothing matches are recorded, not guessed at -- ``mapping_report``
        prints them with the manifest stanza that would fix them.
        """
        out: dict[str, list[Path]] = {}
        for f in files:
            cls = guess_class(f.stem)
            if cls is None:
                self.unmapped.append(f.name)
                continue
            self.automapped[f.name] = cls
            out.setdefault(cls, []).append(f)
        return out

    def mapping_report(self) -> str:
        """What auto-mapping did, and the manifest to write when it got it wrong.

        The plan calls for a "manual-mapping fallback UI when confidence is
        low". This is that fallback in the form this repo actually has: say
        what was guessed, say what was dropped, and print the stanza that
        overrides it -- ``aliases`` is already read and already wins.
        """
        lines = []
        if self.automapped:
            lines.append("auto-mapped by filename:")
            for src, cls in sorted(self.automapped.items()):
                lines.append(f"  {src:<24} -> {cls}")
        if self.unmapped:
            lines.append("unmapped (silent; add an alias to override):")
            for name in sorted(self.unmapped):
                lines.append(f"  {name}")
            lines.append("\nmanifest.yaml:\naliases:\n  <class>: [<folder-or-file-stem>]")
        return "\n".join(lines) or "every folder matched a class name directly"

    def _read_wav(self, sf, path: Path) -> np.ndarray:
        audio, sr = sf.read(path, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != self.sample_rate:      # resample once, here, never in the callback
            n = int(len(audio) * self.sample_rate / sr)
            audio = np.interp(np.linspace(0, len(audio) - 1, n),
                              np.arange(len(audio)), audio).astype(np.float32)
        return np.ascontiguousarray(audio, dtype=np.float32)

    # -- playing ------------------------------------------------------------
    def trigger(self, cls: str, velocity: float, t: float = 0.0) -> None:
        # One read of the kit reference, then work entirely off that snapshot.
        # A hot swap concurrent with this call rebinds self._kit; taking it
        # once means this trigger finishes against the kit it started with
        # rather than half of each.
        kit = self._kit
        stack = kit.layers.get(cls)
        if not stack:
            return                      # missing class: silence, not a crash
        group = kit.groups.get(cls)
        for idx, gain in layer_gains(velocity, len(stack)):
            variants = stack[idx]
            buf = variants[kit.rr[(cls, idx)].next()]
            self.pool.start(buf, gain=gain, group=group, cls=cls)

    def mix(self, n: int) -> np.ndarray:
        return self.pool.mix(n)

    def validate(self, kit_classes: list[str]) -> list[str]:
        """Classes with no samples. Missing is a warning, never an error.

        Lesion mode already needs "some classes simply do not fire" to be a
        normal outcome rather than a failure, so a kit that cannot sound a
        crash should play everything else and say so.
        """
        return [c for c in kit_classes if not self.layers.get(c)]

    def hot_swap(self, new_config: dict) -> None:
        """Preload the new kit, then swap it in between blocks.

        Built first, swapped after: mutating the live dictionaries would let a
        callback read a half-loaded kit. Sounding voices keep their own buffers
        and ring out naturally, because a Voice holds a reference to the array
        rather than an index into the bank.
        """
        layers, groups, rr = self._read_kit(new_config)
        # Built first, then one store. See _Kit for why this cannot be three.
        self._kit = _Kit(layers, groups, rr,
                         str(new_config.get("name") or new_config.get("dir") or ""))


def make_bank(source: str, kit, sample_rate: int, kit_dir: str | Path | None = None,
              port=None) -> SoundBank:
    """``--sound-source`` -> a bank. One place, so the CLI and tests agree."""
    if source in ("midi", "midi_passthrough"):
        return MidiBank(kit, port=port)
    if source == "samples":
        if kit_dir is None:
            raise ValueError("--sound-source samples needs --kit-dir")
        bank = SampleBank(sample_rate=sample_rate)
        bank.load({"dir": str(kit_dir)})
        return bank
    raise ValueError(f"unknown sound source {source!r}; have midi, samples")
