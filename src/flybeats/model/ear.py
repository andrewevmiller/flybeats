"""Phase 3, step 3: the thin ear. Mono 16 kHz audio -> 16 log-spaced bands, one frame per 5 ms.

The window for frame t ends at sample (t + 1) * hop, so no frame ever sees audio from its future.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

N_BANDS = 16
F_LO, F_HI = 40.0, 4000.0
N_FFT, HOP = 512, 80


def band_matrix(sr=16000, n_fft=N_FFT, n_bands=N_BANDS):
    """(n_fft//2 + 1, n_bands) 0/1 matrix: each FFT bin sums into the band containing its frequency.
    A band too narrow to contain a bin (the lowest, 40-53 Hz) takes its nearest bin instead."""
    freqs = np.fft.rfftfreq(n_fft, 1 / sr)
    edges = np.geomspace(F_LO, F_HI, n_bands + 1)
    m = np.zeros((len(freqs), n_bands), np.float32)
    for b in range(n_bands):
        inside = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if inside.any():
            m[inside, b] = 1
        else:
            m[np.argmin(np.abs(np.log(freqs[1:] / np.sqrt(edges[b] * edges[b + 1])))) + 1, b] = 1
    return torch.from_numpy(m)


class Bands(torch.nn.Module):
    """log(1 + band energy), normalised per band with fixed stats from training clips."""

    def __init__(self, mean=None, std=None, sr=16000):
        super().__init__()
        self.register_buffer("bands", band_matrix(sr))
        self.register_buffer("window", torch.hann_window(N_FFT))
        self.register_buffer("mean", torch.zeros(N_BANDS) if mean is None else torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("std", torch.ones(N_BANDS) if std is None else torch.as_tensor(std, dtype=torch.float32))

    def raw(self, audio):
        """audio (B, samples) -> (B, samples // HOP, 16)"""
        x = torch.nn.functional.pad(audio, (N_FFT - HOP, 0))
        spec = torch.stft(x, N_FFT, HOP, window=self.window, center=False, return_complex=True)
        return torch.log1p(spec.abs().pow(2).transpose(1, 2) @ self.bands)

    def forward(self, audio):
        return (self.raw(audio) - self.mean) / self.std


def compute_band_stats(clips, sr=16000):
    """Per-band mean and spread over a list of clip dicts (from ClipSampler), warm-up frames included."""
    bands = Bands(sr=sr)
    with torch.no_grad():
        x = torch.cat([bands.raw(torch.from_numpy(c["audio"])[None]) for c in clips], 1)[0]
    return {"mean": x.mean(0).tolist(), "std": x.std(0).clamp_min(1e-6).tolist(), "clips": len(clips)}


def load_band_stats(path):
    s = json.loads(Path(path).read_text(encoding="utf-8"))
    return s["mean"], s["std"]


def ear_bands(neurons):
    """Fixed wiring: JO-B ears take the 8 low bands and JO-A ears the 8 high bands, round-robin by
    bodyId; a mirrored ear gets its original's band. Returns (ear row indices, band per ear)."""
    t = neurons.type.fillna("")
    real = neurons[(neurons.mirror_of.isna()) & t.str.match(r"^JO-[AB]")].sort_values("bodyId")
    band_of = {}
    for prefix, first in (("JO-B", 0), ("JO-A", 8)):
        ids = real.bodyId[real.type.str.startswith(prefix)]
        band_of.update({b: first + k % 8 for k, b in enumerate(ids)})
    ears = neurons[t.str.match(r"^JO-[AB]")]
    band = [band_of[b if pd.isna(m) else m] for b, m in zip(ears.bodyId, ears.mirror_of)]
    return ears.index.to_numpy(), np.array(band)
