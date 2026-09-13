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

    One caveat worth stating plainly: the model was trained on binary onset
    targets, so its velocity is the peak height of the detection, not a learned
    dynamic. The layers crossfade correctly; what they are crossfading on is
    confidence. GMD's real MIDI velocities are in the corpus and unused -- see
    the README's open items.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from voice import RoundRobin, VoicePool, layer_gains


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
        self.layers: dict[str, list[list[np.ndarray]]] = {}   # cls -> layer -> variants
        self.groups: dict[str, str] = {}                      # cls -> choke group
        self.rr: dict[tuple[str, int], RoundRobin] = {}
        self.name = ""

    # -- loading ------------------------------------------------------------
    def load(self, config: dict) -> None:
        self.layers, self.groups, self.rr = self._read_kit(config)
        self.name = str(config.get("name") or config.get("dir") or "")
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
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            cls = alias.get(d.name, d.name)
            files = sorted(d.glob("*.wav"))
            if not files:
                continue
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
        stack = self.layers.get(cls)
        if not stack:
            return                      # missing class: silence, not a crash
        group = self.groups.get(cls)
        for idx, gain in layer_gains(velocity, len(stack)):
            variants = stack[idx]
            buf = variants[self.rr[(cls, idx)].next()]
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
        self.layers, self.groups, self.rr = layers, groups, rr
        self.name = str(new_config.get("name") or new_config.get("dir") or "")


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
