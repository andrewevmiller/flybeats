"""Phase 2 encoder: audio -> Johnston's Organ input current.

Deliberately thin. The DSP front end is fixed (a real gammatone filterbank,
half-wave rectification, envelope, spectral flux) and the only trainable part
is a single linear map from 64 bands onto the JO channels. If this were a deep
net it would learn the task by itself and the connectome would be decoration
rather than the thing under test.

Frequency->zone mapping follows the fly: JO-B is the sound-particle-velocity
zone tuned to the courtship song band, so it is weighted toward 100-500 Hz.
JO-A covers higher-frequency vibration; JO-E is the gravity/wind zone and gets
the low band. The weighting is a prior on the *initial* linear map, not a hard
constraint -- gradient descent can move it.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

#: Courtship-song band. JO-B afferents are most sensitive here.
SONG_BAND_HZ = (100.0, 500.0)


def erb_space(low_hz: float, high_hz: float, n: int) -> np.ndarray:
    """Centre frequencies spaced evenly on the ERB scale (Glasberg & Moore)."""
    ear_q, min_bw = 9.26449, 24.7
    return -(ear_q * min_bw) + np.exp(
        np.arange(1, n + 1) * (-np.log(high_hz + ear_q * min_bw)
                               + np.log(low_hz + ear_q * min_bw)) / n
    ) * (high_hz + ear_q * min_bw)


def gammatone_filterbank(
    n_bands: int, sample_rate: int, low_hz: float = 40.0, high_hz: float = 6000.0,
    order: int = 4, length: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """FIR gammatone impulse responses, one per band.

    Returns ``(kernels[n_bands, length], centre_freqs[n_bands])``. An FIR bank
    is used rather than the usual IIR cascade because it runs as a single
    strided conv1d, which is what both training and the Phase 5 streaming path
    want.
    """
    cf = erb_space(low_hz, high_hz, n_bands)[::-1].copy()   # ascending
    t = np.arange(length) / sample_rate
    bandwidth = 1.019 * 24.7 * (4.37 * cf / 1000.0 + 1.0)

    kernels = np.zeros((n_bands, length), dtype=np.float32)
    for i, (f, b) in enumerate(zip(cf, bandwidth)):
        g = t ** (order - 1) * np.exp(-2 * np.pi * b * t) * np.cos(2 * np.pi * f * t)
        n = np.linalg.norm(g)
        kernels[i] = (g / n if n > 0 else g).astype(np.float32)
    return kernels, cf.astype(np.float32)


def zone_frequency_prior(cf: np.ndarray, zone_of_channel: list[str]) -> np.ndarray:
    """Initial band->channel weights, biased by each JO zone's real tuning."""
    prior = np.zeros((len(cf), len(zone_of_channel)), dtype=np.float32)
    lo, hi = SONG_BAND_HZ
    log_cf = np.log(np.maximum(cf, 1.0))
    song_centre = math.log(math.sqrt(lo * hi))

    for j, zone in enumerate(zone_of_channel):
        if zone == "JO_B":          # song band, narrow
            w = np.exp(-0.5 * ((log_cf - song_centre) / 0.55) ** 2)
        elif zone == "JO_A":        # higher-frequency vibration
            w = 1.0 / (1.0 + np.exp(-(log_cf - math.log(800.0)) / 0.4))
        elif zone == "JO_E":        # gravity / wind, low frequency
            w = 1.0 / (1.0 + np.exp((log_cf - math.log(150.0)) / 0.4))
        else:                       # unassigned zones: flat
            w = np.ones_like(log_cf)
        s = w.sum()
        prior[:, j] = w / s if s > 0 else w
    return prior


class AudioToJO(nn.Module):
    """Fixed gammatone DSP block + one trainable linear layer. Nothing deeper.

    Input  : ``(batch, samples)`` mono waveform
    Output : ``(batch, steps, n_channels)`` input current, one channel per JO
             afferent in the subgraph
    """

    def __init__(
        self,
        n_channels: int,
        zone_of_channel: list[str],
        sample_rate: int = 22_050,
        step_ms: float = 5.0,
        n_bands: int = 64,
        flux_weight: float = 1.0,
        trainable_dsp: bool = False,
    ):
        super().__init__()
        if len(zone_of_channel) != n_channels:
            raise ValueError("zone_of_channel must have one entry per JO channel")

        self.sample_rate = sample_rate
        self.step_ms = step_ms
        self.hop = max(1, int(round(sample_rate * step_ms / 1000.0)))
        self.n_bands = n_bands
        self.flux_weight = flux_weight

        kernels, cf = gammatone_filterbank(n_bands, sample_rate)
        # Fixed by default: the filterbank is meant to be a model of the ear,
        # not another set of free parameters competing with the connectome.
        self.register_buffer("cf", torch.from_numpy(cf))
        self.filters = nn.Parameter(
            torch.from_numpy(kernels).unsqueeze(1), requires_grad=trainable_dsp
        )

        # Envelope smoothing, ~15 ms one-pole, applied as a fixed conv.
        tau_samples = 0.015 * sample_rate
        env_len = int(4 * tau_samples)
        env = np.exp(-np.arange(env_len) / tau_samples).astype(np.float32)
        env /= env.sum()
        self.register_buffer("env_kernel", torch.from_numpy(env).view(1, 1, -1))

        # The one trainable layer. 2*n_bands in: envelope and spectral flux.
        self.to_jo = nn.Linear(2 * n_bands, n_channels, bias=True)
        with torch.no_grad():
            prior = torch.from_numpy(zone_frequency_prior(cf, zone_of_channel))
            self.to_jo.weight.zero_()
            self.to_jo.weight.add_(torch.cat([prior, prior * flux_weight], dim=0).T)
            self.to_jo.bias.zero_()

    @property
    def n_features(self) -> int:
        return 2 * self.n_bands

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """Fixed DSP: waveform -> ``(batch, steps, 2*n_bands)``."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        x = wav.unsqueeze(1)                                     # (B, 1, T)

        pad = self.filters.shape[-1] - 1
        sub = torch.nn.functional.conv1d(torch.nn.functional.pad(x, (pad, 0)), self.filters)
        sub = torch.relu(sub)                                    # half-wave rectify

        epad = self.env_kernel.shape[-1] - 1
        b, c, t = sub.shape
        env = torch.nn.functional.conv1d(
            torch.nn.functional.pad(sub.reshape(b * c, 1, t), (epad, 0)), self.env_kernel
        ).reshape(b, c, t)

        env = env[:, :, :: self.hop]
        env = torch.log1p(env * 1e3)                             # compressive, as the ear is

        # Spectral flux: positive part of the frame-to-frame difference. This is
        # the onset function the drum targets are actually aligned to.
        flux = torch.relu(env[:, :, 1:] - env[:, :, :-1])
        flux = torch.nn.functional.pad(flux, (1, 0))

        return torch.cat([env, flux], dim=1).transpose(1, 2)     # (B, steps, 2*bands)

    @property
    def context_samples(self) -> int:
        """Left context the causal DSP block needs before its output is exact."""
        return int(self.filters.shape[-1] + self.env_kernel.shape[-1])

    def forward_window(self, wav: torch.Tensor, start_step: int, n_steps: int) -> torch.Tensor:
        """Encode only steps ``[start_step, start_step + n_steps)``.

        Truncated BPTT backwards through one chunk at a time, so the encoder
        cannot be run once over the whole clip -- its graph would be freed by
        the first chunk's backward pass. Re-encoding a window with enough left
        context gives bit-comparable output and keeps memory flat. It is also
        exactly what the Phase 5 streaming path does, so training and live
        inference run the same code.
        """
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        hop = self.hop
        ctx_steps = -(-self.context_samples // hop)          # ceil, in steps
        first = max(0, start_step - ctx_steps)
        lo = first * hop
        hi = min(wav.shape[-1], (start_step + n_steps) * hop + hop)
        feats = self.features(wav[:, lo:hi])
        off = start_step - first
        return self.to_jo(feats[:, off: off + n_steps])

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        return self.to_jo(self.features(wav))


def zones_for_channels(types: np.ndarray, verified: dict) -> list[str]:
    """Label each JO channel with its Phase 0 zone concept (JO_A/JO_B/JO_E/...)."""
    lookup: dict[str, str] = {}
    for concept in ("JO_A", "JO_B", "JO_E", "JO_other"):
        rec = verified["concepts"].get(concept)
        if rec and rec["confirmed"]:
            for t in rec["types"]:
                lookup[t["type"]] = concept
    return [lookup.get(str(t), "JO_other") for t in types]
