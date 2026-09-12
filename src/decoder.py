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
    """One linear layer: motor-neuron rate -> per-class drum velocity logit.

    ``bilateral`` splits the readout by hemisphere: left-hemisphere motor
    neurons drive the left-hand classes and right the right-hand ones, with the
    cross-hemisphere weights masked out. That is limb independence for free --
    the constraint is anatomical, not a regulariser we invented.
    """

    def __init__(
        self,
        n_motor: int,
        kit: DrumKit,
        motor_side: np.ndarray | None = None,
        bilateral: bool = False,
    ):
        super().__init__()
        self.kit = kit
        self.readout = nn.Linear(n_motor, kit.n, bias=True)
        nn.init.normal_(self.readout.weight, std=1.0 / max(n_motor, 1) ** 0.5)
        nn.init.constant_(self.readout.bias, -2.0)   # onsets are sparse; start quiet

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

    def forward(self, rates: torch.Tensor) -> torch.Tensor:
        """``(batch, steps, n_motor)`` -> ``(batch, steps, n_classes)`` logits."""
        w = self.readout.weight * self.mask
        return torch.nn.functional.linear(rates, w, self.readout.bias)


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
