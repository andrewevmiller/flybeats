"""Voice allocation, choke groups and block mixing for sample playback.

Everything here is pure numpy over a block of samples, deliberately: the same
code has to run inside a ``sounddevice`` callback (where allocation and disk
I/O are forbidden) and inside an offline render (where there is no audio device
at all). One implementation, so what you hear live is what the file contains.

The audio callback rules this obeys:

* no allocation beyond the output block -- samples are preloaded by the bank
* no disk access
* no unbounded work: the voice pool is fixed-size and steals the oldest voice
  rather than growing

Choke groups live here rather than in the model. One hi-hat cannot be open and
closed at the same time; that is a fact about the instrument, not something the
connectome should have to learn.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: How fast a choked voice is silenced. Long enough not to click, short enough
#: to read as the hat closing rather than as a crossfade.
CHOKE_MS = 5.0


@dataclass(eq=False)
class Voice:
    """One sounding sample: where it came from, how far through it is.

    ``eq=False`` on purpose. A dataclass would generate ``__eq__`` comparing
    every field, and two of them are numpy arrays -- so ``voices.remove(v)``
    would compare buffers elementwise and raise as soon as two voices held
    samples of different lengths. Voices are identities, not values.
    """

    buf: np.ndarray                 # mono float32, preloaded
    gain: float
    pos: int = 0
    group: str | None = None
    cls: str = ""
    age: int = 0                    # allocation order, for steal-oldest
    fade_left: int = 0              # samples remaining in a choke fade
    fade_total: int = 0

    @property
    def done(self) -> bool:
        return self.pos >= len(self.buf) or (self.fade_total > 0 and self.fade_left <= 0)


@dataclass
class VoicePool:
    """Fixed-size mixer. ``start`` allocates, ``mix`` renders a block."""

    max_voices: int = 32
    sample_rate: int = 22_050
    voices: list[Voice] = field(default_factory=list)
    _age: int = 0

    def start(self, buf: np.ndarray, gain: float = 1.0, group: str | None = None,
              cls: str = "") -> Voice:
        """Begin a sample, choking its group first if one is sounding."""
        if group is not None:
            for v in self.voices:
                if v.group == group and not v.done and v.fade_total == 0:
                    self._choke(v)
        if len(self.voices) >= self.max_voices:
            # steal the oldest rather than refuse: a dropped hit is more
            # audible than an early tail cut, and this bounds the work per block
            self.voices.remove(min(self.voices, key=lambda v: v.age))
        self._age += 1
        v = Voice(buf=buf, gain=float(gain), group=group, cls=cls, age=self._age)
        self.voices.append(v)
        return v

    def _choke(self, v: Voice) -> None:
        n = max(1, int(self.sample_rate * CHOKE_MS / 1000.0))
        v.fade_total = n
        v.fade_left = n

    def mix(self, n: int) -> np.ndarray:
        """Render ``n`` samples of everything currently sounding."""
        out = np.zeros(n, dtype=np.float32)
        for v in list(self.voices):
            take = min(n, len(v.buf) - v.pos)
            if take <= 0:
                self.voices.remove(v)
                continue
            chunk = v.buf[v.pos: v.pos + take] * v.gain
            if v.fade_total > 0:
                # linear ramp across whatever part of the fade lands in this block
                a = v.fade_left / v.fade_total
                b = max(0, v.fade_left - take) / v.fade_total
                chunk = chunk * np.linspace(a, b, take, dtype=np.float32)
                v.fade_left -= take
            out[:take] += chunk
            v.pos += take
            if v.done:
                self.voices.remove(v)
        return out

    def reset(self) -> None:
        self.voices.clear()
        self._age = 0

    @property
    def active(self) -> int:
        return len(self.voices)


class RoundRobin:
    """No-immediate-repeat variant picker.

    Pure random repeats audibly over a short loop -- two identical snares in a
    row is exactly what a sampled kit is supposed to avoid. This shuffles a bag
    and refills it when empty, re-drawing if the refill would repeat the last
    variant played.
    """

    def __init__(self, n: int, seed: int = 0):
        self.n = max(1, int(n))
        self.rng = np.random.default_rng(seed)
        self._bag: list[int] = []
        self._last: int | None = None

    def next(self) -> int:
        if self.n == 1:
            return 0
        if not self._bag:
            bag = list(self.rng.permutation(self.n))
            # drawn from the tail, so the tail is what must not repeat the
            # variant that ended the previous bag
            if self._last is not None and bag[-1] == self._last and len(bag) > 1:
                bag[-1], bag[-2] = bag[-2], bag[-1]
            self._bag = bag
        self._last = int(self._bag.pop())
        return self._last


def layer_gains(velocity: float, n_layers: int) -> list[tuple[int, float]]:
    """Velocity -> ``[(layer index, gain), ...]``, crossfaded between neighbours.

    A hard ``floor(v * N)`` cutoff puts a seam in the middle of a crescendo:
    the sample identity jumps at the boundary and you hear the kit switch
    instruments. Blending the two adjacent layers by the fractional part moves
    that seam into a gradual change of timbre, which is what a real drum does.
    """
    if n_layers <= 1:
        return [(0, 1.0)]
    v = float(np.clip(velocity, 0.0, 1.0))
    x = v * (n_layers - 1)
    lo = int(np.floor(x))
    frac = x - lo
    if lo >= n_layers - 1:
        return [(n_layers - 1, 1.0)]
    if frac <= 1e-6:
        return [(lo, 1.0)]
    return [(lo, 1.0 - frac), (lo + 1, frac)]
