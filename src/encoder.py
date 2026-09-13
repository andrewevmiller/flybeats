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
constraint -- gradient descent can move it, subject to the sign constraint below.

Two properties of the drive matter more than the prior does, and the first
training runs got both wrong. ``log1p`` band energy is a large positive
constant plus a small fluctuation, so an unstandardised feature vector is
mostly DC: the linear map's useful output is a few percent of its magnitude.
Trained against a regulariser that penalised activity, the cheapest response
was to cancel the DC with negative weights -- which also cancels the signal,
and the encoder silenced its own sensory input (96% of ``to_jo`` weights went
negative; see the README). So:

  * features are **standardised per channel** by a fixed affine calibrated once
    on training audio. Fixed, not per-clip: a per-clip statistic would be
    non-causal and would break the streaming path's agreement with the training
    path, which ``tests/test_encoder.py`` pins exactly.
  * ``to_jo`` is optionally constrained **non-negative**, by projection after
    each optimiser step. An onset function driving JO afferents should excite
    them; the DC offset that used to be fought with negative weights is now the
    bias's job, and the bias is free.
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
    order: int = 4, length: int = 512,
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
        filter_taps: int = 512,
        standardize: bool = True,
        nonneg: bool = True,
    ):
        super().__init__()
        if len(zone_of_channel) != n_channels:
            raise ValueError("zone_of_channel must have one entry per JO channel")

        self.sample_rate = sample_rate
        self.step_ms = step_ms
        self.hop = max(1, int(round(sample_rate * step_ms / 1000.0)))
        self.n_bands = n_bands
        self.flux_weight = flux_weight
        self.standardize = bool(standardize)
        self.nonneg = bool(nonneg)

        # 512 taps is 23 ms at 22.05 kHz. The slowest band (40 Hz) has an ERB
        # bandwidth of ~29.5 Hz, so a 5.4 ms time constant -- 23 ms is over four
        # of those and the tail is already numerically dead. 1024 taps doubled
        # the convolution cost to buy nothing.
        kernels, cf = gammatone_filterbank(n_bands, sample_rate, length=filter_taps)
        # Fixed by default: the filterbank is meant to be a model of the ear,
        # not another set of free parameters competing with the connectome.
        self.register_buffer("cf", torch.from_numpy(cf))
        self.filters = nn.Parameter(
            torch.from_numpy(kernels).unsqueeze(1), requires_grad=trainable_dsp
        )

        # Envelope smoothing, ~15 ms one-pole. Applied *after* decimation to
        # the step grid, not at the audio rate: at 22.05 kHz a 15 ms one-pole
        # is a 1,323-tap FIR run on every sample, which measured at 140 ms per
        # 20 ms audio block -- 81% of the entire encoder and the single thing
        # keeping this out of real time. On the 5 ms step grid the same filter
        # is ~12 taps at 1/110th the rate. Decimating by a box mean first is
        # also the correct anti-aliasing step, which sampling every hop-th
        # value (the earlier version) skipped.
        tau_steps = 15.0 / step_ms
        env_len = max(2, int(4 * tau_steps))
        env = np.exp(-np.arange(env_len) / tau_steps).astype(np.float32)
        env /= env.sum()
        self.register_buffer("env_kernel", torch.from_numpy(env).view(1, 1, -1))

        # Per-channel standardisation of the DSP features. Identity until
        # ``calibrate`` runs, so an uncalibrated model behaves exactly as before
        # and nothing depends on calibration having happened silently.
        self.register_buffer("feat_mean", torch.zeros(2 * n_bands))
        self.register_buffer("feat_scale", torch.ones(2 * n_bands))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))

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

    def _subband(self, x: torch.Tensor, prepadded: bool) -> torch.Tensor:
        """Rectified filterbank output.

        With ``prepadded``, ``x`` already carries the filter's ``taps - 1``
        samples of left context and the output aligns to ``x[..., taps-1:]``;
        the convolution then produces only the samples that are kept, instead
        of a full window that is mostly discarded.
        """
        if not prepadded:
            x = torch.nn.functional.pad(x, (self.filters.shape[-1] - 1, 0))
        return torch.relu(torch.nn.functional.conv1d(x, self.filters))

    def _assemble(self, sub: torch.Tensor) -> torch.Tensor:
        """Rectified sub-bands -> ``(batch, steps, 2*n_bands)`` features."""
        b, c, t = sub.shape
        n_steps = t // self.hop
        if n_steps == 0:
            return sub.new_zeros(b, 0, 2 * self.n_bands)
        # Decimate to the step grid by a box mean over each hop window -- the
        # anti-aliasing and the downsample in one cheap operation.
        env = sub[:, :, : n_steps * self.hop].reshape(b, c, n_steps, self.hop).mean(-1)

        epad = self.env_kernel.shape[-1] - 1
        env = torch.nn.functional.conv1d(
            torch.nn.functional.pad(env.reshape(b * c, 1, n_steps), (epad, 0)), self.env_kernel
        ).reshape(b, c, n_steps)
        env = torch.log1p(env * 1e3)                             # compressive, as the ear is

        # Spectral flux: positive part of the frame-to-frame difference. This is
        # the onset function the drum targets are actually aligned to.
        flux = torch.relu(env[:, :, 1:] - env[:, :, :-1])
        flux = torch.nn.functional.pad(flux, (1, 0))

        return torch.cat([env, flux], dim=1).transpose(1, 2)     # (B, steps, 2*bands)

    def _standardize(self, feats: torch.Tensor) -> torch.Tensor:
        """Fixed per-channel affine. Constant in time, so the streaming path
        and the training path still produce identical numbers block by block."""
        if not self.standardize:
            return feats
        return (feats - self.feat_mean) / self.feat_scale

    def raw_features(self, wav: torch.Tensor) -> torch.Tensor:
        """DSP features before standardisation -- what ``calibrate`` measures."""
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        return self._assemble(self._subband(wav.unsqueeze(1), prepadded=False))

    @torch.no_grad()
    def calibrate(self, wavs) -> dict:
        """Set the standardisation affine from a sample of training audio.

        One fixed mean and scale per feature channel, measured once and then
        frozen into the checkpoint: live playback must see exactly the affine
        training saw. Channels that never move (a silent band) keep scale 1 so
        the division cannot amplify nothing into noise.

        Returns the drive statistics it measured, so a caller can log what the
        encoder was calibrated against rather than trusting that it happened.
        """
        if not self.standardize:
            return {"calibrated": False}
        n = torch.zeros(())
        s1 = torch.zeros(2 * self.n_bands)
        s2 = torch.zeros(2 * self.n_bands)
        for wav in wavs:
            f = self.raw_features(wav.to(self.feat_mean.device)).float().reshape(-1, 2 * self.n_bands)
            n += f.shape[0]
            s1 += f.sum(0)
            s2 += (f * f).sum(0)
        if float(n) < 2:
            raise ValueError("calibrate needs at least two feature frames")
        mean = s1 / n
        var = (s2 / n - mean * mean).clamp(min=0.0)
        scale = var.sqrt()
        scale[scale < 1e-6] = 1.0
        self.feat_mean.copy_(mean)
        self.feat_scale.copy_(scale)
        self.calibrated.fill_(True)
        return {"calibrated": True, "frames": int(n),
                "mean_abs": float(mean.abs().mean()), "scale_mean": float(scale.mean()),
                "flat_channels": int((var.sqrt() < 1e-6).sum())}

    @torch.no_grad()
    def project_(self) -> None:
        """Projected-gradient step for the non-negativity constraint.

        Clamping after the optimiser (rather than reparameterising through a
        softplus or an exp) keeps the initial weights *exactly* the zone prior
        and leaves a weight sitting at zero with a live gradient, so a channel
        that should come back up can.
        """
        if self.nonneg:
            self.to_jo.weight.clamp_(min=0.0)

    def features(self, wav: torch.Tensor) -> torch.Tensor:
        """Fixed DSP: waveform -> ``(batch, steps, 2*n_bands)``, standardised."""
        return self._standardize(self.raw_features(wav))

    @property
    def context_samples(self) -> int:
        """Left context the causal DSP block needs before its output is exact.

        The filterbank needs its full impulse response; the envelope now works
        on the step grid, so it needs its taps expressed back in samples.
        """
        return int(self.filters.shape[-1] + self.env_kernel.shape[-1] * self.hop)

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
        hop, taps = self.hop, self.filters.shape[-1]
        # the envelope needs its taps of step-grid context, and the flux one more
        ctx_steps = self.env_kernel.shape[-1]
        first = max(0, start_step - ctx_steps)

        a, b = first * hop, min(wav.shape[-1], (start_step + n_steps) * hop)
        lo = a - (taps - 1)
        x = wav[:, max(0, lo): b].unsqueeze(1)
        if lo < 0:
            x = torch.nn.functional.pad(x, (-lo, 0))
        feats = self._standardize(self._assemble(self._subband(x, prepadded=True)))
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
