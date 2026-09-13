"""Phase 2 decoder: motor-neuron rates -> drum velocities.

A single linear layer. No hidden units, no temporal filtering, no smoothing --
anything the readout could do for itself is something the connectome would not
have to do, and the whole claim under test is that the connectome is doing the
work.

The drum map is built from the Phase 0 wing motor types, so kit size is capped
by real anatomy rather than being picked to fit a General MIDI chart.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

#: General MIDI percussion notes for the shipped 8-piece kit.
GM_NOTE = {
    "kick": 36, "snare": 38, "sidestick": 37, "clap": 39,
    "tom_low": 45, "tom_mid": 47, "tom_high": 50,
    "hat_closed": 42, "hat_pedal": 44, "hat_open": 46,
    "crash": 49, "ride": 51, "ride_bell": 53, "cowbell": 56,
}

#: Kit tiers from PLAN.md. Every tier must fit inside the confirmed wing motor
#: type count from Phase 0 (15 steering types, 25 in the 'wm' subclass), which
#: it does -- leg MNs are not needed below the 20-piece tier.
KIT_TIERS: dict[str, list[str]] = {
    "3piece": ["kick", "snare", "hat_closed"],
    "8piece": ["kick", "snare", "hat_closed", "hat_open",
               "tom_low", "tom_mid", "tom_high", "crash"],
    "articulated": ["kick", "snare", "sidestick", "hat_closed", "hat_open",
                    "hat_pedal", "tom_low", "tom_mid", "tom_high",
                    "crash", "ride", "ride_bell"],
}


@dataclass
class DrumKit:
    """The classes the decoder emits, and how they reach a sampler."""

    classes: list[str]

    @property
    def n(self) -> int:
        return len(self.classes)

    @property
    def notes(self) -> list[int]:
        return [GM_NOTE[c] for c in self.classes]

    @classmethod
    def from_tier(cls, tier: str, max_classes: int | None = None) -> "DrumKit":
        if tier not in KIT_TIERS:
            raise KeyError(f"unknown kit tier {tier!r}; have {sorted(KIT_TIERS)}")
        classes = KIT_TIERS[tier]
        if max_classes is not None and len(classes) > max_classes:
            raise ValueError(
                f"kit tier {tier!r} needs {len(classes)} classes but only "
                f"{max_classes} motor units are available -- Phase 0 caps kit size"
            )
        return cls(classes)


class MotorToDrums(nn.Module):
    """Motor-neuron rate -> per-class drum onsets, and how hard they are hit.

    Two linear readouts over the same motor pool, through the same hemisphere
    mask. ``forward`` answers *whether* a class fires at this step;
    :meth:`velocity` answers *how hard*. They were one output until now, and
    conflating them meant the "velocity" a drum machine received was really the
    height of a detection peak -- confidence wearing dynamics' clothes. A
    hesitant model played quietly and a certain one played loudly, which is not
    what a drummer does.

    ``bilateral`` splits the readout by hemisphere: left-hemisphere motor
    neurons drive the left-hand classes and right the right-hand ones, with the
    cross-hemisphere weights masked out. That is limb independence for free --
    the constraint is anatomical, not a regulariser we invented. The velocity
    head wears the same mask: a hit's strength has to come from the same
    hemisphere that produced the hit.

    ``velocity_head=False`` builds the detection half alone, which is what a
    checkpoint trained before this head existed contains.
    """

    def __init__(
        self,
        n_motor: int,
        kit: DrumKit,
        motor_side: np.ndarray | None = None,
        bilateral: bool = False,
        velocity_head: bool = True,
        velocity_activation: str = "sigmoid",
    ):
        super().__init__()
        self.kit = kit
        self.readout = nn.Linear(n_motor, kit.n, bias=True)
        nn.init.normal_(self.readout.weight, std=1.0 / max(n_motor, 1) ** 0.5)
        nn.init.constant_(self.readout.bias, -2.0)   # onsets are sparse; start quiet

        if velocity_activation not in ("sigmoid", "linear"):
            raise ValueError(f"velocity_activation must be sigmoid or linear, "
                             f"got {velocity_activation!r}")
        self.velocity_activation = velocity_activation
        if velocity_head:
            self.vel_readout = nn.Linear(n_motor, kit.n, bias=True)
            nn.init.normal_(self.vel_readout.weight, std=1.0 / max(n_motor, 1) ** 0.5)
            # sigmoid(0) and linear 0.5 are both "mid velocity" at init
            nn.init.constant_(self.vel_readout.bias,
                              0.0 if velocity_activation == "sigmoid" else 0.5)
        else:
            self.vel_readout = None

        mask = torch.ones(kit.n, n_motor)
        if bilateral:
            if motor_side is None:
                raise ValueError("bilateral readout needs motor_side")
            side = np.asarray([str(s) for s in motor_side])
            left = torch.from_numpy(side == "L")
            right = torch.from_numpy(side == "R")
            for c, name in enumerate(kit.classes):
                hand = _hand_of(name)
                if hand == "L":
                    mask[c] = left.float()
                elif hand == "R":
                    mask[c] = right.float()
                # feet and shared classes read from both hemispheres
                if mask[c].sum() == 0:      # no MN on that side: fall back to all
                    mask[c] = 1.0
        self.register_buffer("mask", mask)

    @property
    def has_velocity(self) -> bool:
        return self.vel_readout is not None

    def forward(self, rates: torch.Tensor) -> torch.Tensor:
        """``(batch, steps, n_motor)`` -> ``(batch, steps, n_classes)`` logits."""
        w = self.readout.weight * self.mask
        return torch.nn.functional.linear(rates, w, self.readout.bias)

    def velocity(self, rates: torch.Tensor) -> torch.Tensor | None:
        """How hard each class is struck, in 0..1. ``None`` without the head.

        ``sigmoid`` bounds the output, because every consumer downstream -- the
        sample player's velocity layers, MIDI's 1..127 -- wants 0..1 and would
        have to clamp an unbounded head anyway.

        ``linear`` exists because that argument has a cost the bounded version
        hides: the sigmoid's gradient is flattest at its extremes, and GMD's
        velocities cluster in the middle-to-upper range where a head sitting
        near the mean has least reason to move. A linear head can leave 0..1,
        and the streaming path already clips it; being visibly out of range is
        more useful than being squashed into range by a saturating unit.
        """
        if self.vel_readout is None:
            return None
        w = self.vel_readout.weight * self.mask
        out = torch.nn.functional.linear(rates, w, self.vel_readout.bias)
        return torch.sigmoid(out) if self.velocity_activation == "sigmoid" else out


def _hand_of(drum_class: str) -> str:
    """Which hand a drummer normally plays a class with.

    Kick and hi-hat pedal are feet and read from both hemispheres; hats and ride
    sit under the right hand in a standard right-handed setup, snare and toms
    under the left. Crude, but it is only the *initial* hemisphere assignment --
    the point is that left and right motor pools stay separated at all.
    """
    if drum_class in ("kick", "hat_pedal"):
        return "B"
    if drum_class in ("hat_closed", "hat_open", "ride", "ride_bell", "cowbell"):
        return "R"
    return "L"
