"""Phase 3, steps 1-4: the fly network.

Learned: per cell type a time constant and a bias; one strength dial per (sending type, receiving type)
pair; a gain and offset per ear band; and a hit and loudness weight per readout motor neuron plus a
bias per drum. Signs and synapse counts come from the slice and never change. The same class loads the
real slice or any random control: only the edge table differs.
"""
import math

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from ..connectome.slice import largest_eigenvalue, type_keys
from .ear import N_BANDS, Bands, check_ear, ear_bands

THETA_ONE = math.log(math.e - 1)          # softplus(THETA_ONE) == 1, so the untrained dials change nothing


class FlyNet(nn.Module):
    def __init__(self, neurons, edges, cfg, band_stats=None, sr=16000):
        super().__init__()
        check_ear(cfg, sr)
        mc = cfg["model"]
        self.dt_ms = cfg["training_data"]["frame_ms"]
        self.tau_min, self.tau_max = float(mc["tau_ms"]["min"]), float(mc["tau_ms"]["max"])
        self.pieces = cfg["kit"]["pieces"]
        n = len(neurons)
        self.n = n

        # Cell types (each untyped neuron is its own type) and type pairs, one dial per pair
        tk = type_keys(neurons)
        type_idx, type_names = pd.factorize(tk, sort=True)
        pre, post = edges.pre.to_numpy(np.int64), edges.post.to_numpy(np.int64)
        pairs = pd.MultiIndex.from_arrays([type_idx[pre], type_idx[post]])
        pair_idx, pair_names = pd.factorize(pairs, sort=True)
        self.type_names = list(type_names)
        self.register_buffer("type_idx", torch.from_numpy(type_idx.astype(np.int64)))
        self.register_buffer("pair_idx", torch.from_numpy(pair_idx.astype(np.int64)))

        # Fixed wiring: sender sign x synapse count x global gain
        sign = neurons.sign.to_numpy(np.float32)
        self.register_buffer("pre", torch.from_numpy(pre))
        self.register_buffer("post", torch.from_numpy(post))
        self.register_buffer("fixed", torch.from_numpy(sign[pre] * edges.synapses.to_numpy(np.float32)))
        self.eigenvalue = largest_eigenvalue(neurons, edges)
        self.g = cfg["model"]["initial_gain"] / self.eigenvalue
        self.gain_scale = 1.0                     # the stability sweep multiplies the gain by this
        self.register_buffer("alive", torch.from_numpy((~neurons.silenced.to_numpy()).astype(np.float32)))

        # Ear: fixed band per ear neuron
        ear_idx, ear_band = ear_bands(neurons)
        self.register_buffer("ear_idx", torch.from_numpy(ear_idx.astype(np.int64)))
        self.register_buffer("ear_band", torch.from_numpy(ear_band.astype(np.int64)))
        mean, std = band_stats if band_stats is not None else (None, None)
        self.bands = Bands(mean, std, sr)

        # Readout: each drum reads only its own motor neurons
        drum = neurons.drum
        mn = np.flatnonzero(drum.notna().to_numpy())
        drum_of = drum.iloc[mn].map({p: i for i, p in enumerate(self.pieces)}).to_numpy(np.int64)
        onehot = np.zeros((len(mn), len(self.pieces)), np.float32)
        onehot[np.arange(len(mn)), drum_of] = 1
        self.register_buffer("mn_idx", torch.from_numpy(mn.astype(np.int64)))
        self.register_buffer("mn_drum", torch.from_numpy(onehot))

        # Learned parameters
        n_types, n_pairs, n_mn, n_drums = len(type_names), len(pair_names), len(mn), len(self.pieces)
        self.log_tau = nn.Parameter(torch.full((n_types,), math.log(mc["tau_ms"]["init"])))
        self.bias = nn.Parameter(torch.zeros(n_types))
        self.theta = nn.Parameter(torch.full((n_pairs,), THETA_ONE))
        self.band_gain = nn.Parameter(torch.ones(N_BANDS))
        self.band_offset = nn.Parameter(torch.zeros(N_BANDS))
        # Small random readout weights: exactly zero would block every gradient into the network
        self.w_hit = nn.Parameter(mc["readout_init_std"] * torch.randn(n_mn))
        self.u_vel = nn.Parameter(mc["readout_init_std"] * torch.randn(n_mn))
        self.c_hit = nn.Parameter(torch.full((n_drums,), float(mc["hit_bias_init"])))   # starts near "no hit"
        self.e_vel = nn.Parameter(torch.zeros(n_drums))

    def weight_matrix(self):
        """Dense (post, pre) matrix, rebuilt once per batch from the edge list."""
        vals = self.fixed * (self.g * self.gain_scale) * F.softplus(self.theta[self.pair_idx])
        w = torch.zeros(self.n, self.n, device=vals.device, dtype=vals.dtype)
        return w.index_put((self.post, self.pre), vals)

    def tau_ms(self):
        return self.log_tau.exp().clamp(self.tau_min, self.tau_max)

    def init_state(self, batch, device):
        return torch.zeros(batch, self.n, device=device)

    def ear_input(self, audio):
        """(B, samples) -> (B, frames, n_ears) current into each ear neuron."""
        x = self.bands(audio)
        x = x * self.band_gain + self.band_offset
        return x[:, :, self.ear_band]

    def run(self, ear_in, v=None, record=False):
        """Step the network over ear_in (B, T, n_ears). Returns (outputs, final v).

        outputs: hit_logits (B, T, drums), vel (B, T, drums), mean_rate (B, T),
                 mn_rates (B, T, readout neurons); rates (B, T, N) too when record=True.
        """
        b, t_len, _ = ear_in.shape
        dev = ear_in.device
        if v is None:
            v = self.init_state(b, dev)
        w_t = self.weight_matrix().t()
        alpha = (self.dt_ms / self.tau_ms())[self.type_idx]
        bias = self.bias[self.type_idx]
        r = F.relu(v) * self.alive
        hit, vel, mean_rate, mn_rates, rates = [], [], [], [], []
        for t in range(t_len):
            drive = torch.zeros(b, self.n, device=dev).index_copy(1, self.ear_idx, ear_in[:, t])
            v = v + alpha * (-v + r @ w_t + bias + drive)
            r = F.relu(v) * self.alive
            rm = r[:, self.mn_idx]
            hit.append((rm * self.w_hit) @ self.mn_drum + self.c_hit)
            vel.append(torch.sigmoid((rm * self.u_vel) @ self.mn_drum + self.e_vel))
            mean_rate.append(r.mean(1))
            mn_rates.append(rm)
            if record:
                rates.append(r)
        out = {"hit_logits": torch.stack(hit, 1), "vel": torch.stack(vel, 1), "mean_rate": torch.stack(mean_rate, 1),
               "mn_rates": torch.stack(mn_rates, 1)}
        if record:
            out["rates"] = torch.stack(rates, 1)
        return out, v

    def forward(self, audio, v=None, record=False):
        return self.run(self.ear_input(audio), v, record)
