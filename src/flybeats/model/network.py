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


class _WeightGradOnce(torch.autograd.Function):
    """The connection matrix's gradient, formed once per clip instead of once per frame.

    The same matrix is used at every frame, so its gradient is a sum over frames: rates before the frame, times the
    gradient arriving at that frame's recurrent input. Left to autograd, each frame forms a dense (neurons x
    neurons) gradient and adds it to a running total: 3,200 full-matrix multiplies and adds per clip, most of a
    training step (reports/profile_step.json). Here the frames use a detached copy of the matrix, each adds one of
    these outputs (zeros) to its recurrent input, and the gradients arriving at them are gathered and multiplied
    by the saved rates, in blocks of CHUNK frames added up in float64. Same quantity, added up in a different order,
    so it may differ in the last digits: scripts/check_cuda_graph.py checks it is at least as close to a
    double-precision reference as the per-frame way."""

    @staticmethod
    def forward(ctx, w_t, rates, t_len, batch):
        ctx.rates = rates                          # run() appends each frame's rates; complete by the backward pass
        return tuple(w_t.new_zeros(batch, w_t.shape[1]) for _ in range(t_len))

    @staticmethod
    def backward(ctx, *grads):
        r = torch.stack(ctx.rates)                 # (frames, batch, neurons): the rates each frame multiplied
        ctx.rates = None                           # no longer needed; free them now
        g = torch.stack([gr if gr is not None else torch.zeros_like(r[0]) for gr in grads])
        # One multiply over all frames would add each entry's 25,600 terms (3,200 frames x 8 clips) one after
        # another in float32: on the GPU less accurate than the per-frame way (check of 30 Sep). So: blocks of
        # CHUNK frames (at most 400 terms in a row), their results added up in float64.
        total = torch.zeros(r.shape[2], g.shape[2], dtype=torch.float64, device=r.device)
        for s in range(0, r.shape[0], _WeightGradOnce.CHUNK):
            total += (r[s:s + _WeightGradOnce.CHUNK].flatten(0, 1).t() @ g[s:s + _WeightGradOnce.CHUNK].flatten(0, 1))
        return total.to(r.dtype), None, None, None

    CHUNK = 50


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
        self.weight_grad_once = True              # False: autograd's per-frame weight gradient (the reference)
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
        # Readout normalisation: fixed per-neuron mean and spread, set once by fit_readout_norm();
        # until then the identity, so the readout sees raw rates
        self.register_buffer("mn_mean", torch.zeros(len(mn)))
        self.register_buffer("mn_std", torch.ones(len(mn)))

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
        self._setup_readout_history(mc.get("readout_history"), n_drums)

    def _setup_readout_history(self, rh, n_drums):
        """Readout history (reports/history-readout-proposal.md; not locked): off unless model.readout_history is set.
        Each drum weighs its motor neurons' present frame and the means of len(blocks_ms) earlier blocks through a
        learned profile per drum, starting at (1, 0, ...), so the untrained readout equals the present-only one."""
        self.history_blocks = []
        self.p_hit = self.p_vel = None
        if not rh:
            return
        expected = {"present": True, "profile": "per_drum", "init": "present_only"}
        for k, v in expected.items():
            if rh.get(k) != v:
                raise ValueError(f"readout_history.{k} must be {v!r}, not {rh.get(k)!r}")
        if rh.get("velocity") not in ("same", "present_only"):
            raise ValueError(f"readout_history.velocity must be 'same' or 'present_only', not {rh.get('velocity')!r}")
        self.history_blocks = [int(round(ms / self.dt_ms)) for ms in rh["blocks_ms"]]
        if min(self.history_blocks) < 1:
            raise ValueError(f"readout_history.blocks_ms must each be at least one frame: {rh['blocks_ms']}")
        init = torch.zeros(n_drums, 1 + len(self.history_blocks))
        init[:, 0] = 1
        self.p_hit = nn.Parameter(init.clone())
        if rh["velocity"] == "same":
            self.p_vel = nn.Parameter(init.clone())

    def weight_matrix(self):
        """Dense (post, pre) matrix, rebuilt once per batch from the edge list."""
        vals = self.fixed * (self.g * self.gain_scale) * F.softplus(self.theta[self.pair_idx])
        w = torch.zeros(self.n, self.n, device=vals.device, dtype=vals.dtype)
        return w.index_put((self.post, self.pre), vals)

    def tau_ms(self):
        return self.log_tau.exp().clamp(self.tau_min, self.tau_max)

    def init_state(self, batch, device, dtype=torch.float32):
        return torch.zeros(batch, self.n, device=device, dtype=dtype)

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
            v = self.init_state(b, dev, ear_in.dtype)
        w_t = self.weight_matrix().t()
        once = self.weight_grad_once and torch.is_grad_enabled() and w_t.requires_grad
        if once:                                   # the weight gradient in one multiply (_WeightGradOnce)
            used = []
            zeros = _WeightGradOnce.apply(w_t, used, t_len, b)
            w_t = w_t.detach()
        alpha = (self.dt_ms / self.tau_ms())[self.type_idx]
        bias = self.bias[self.type_idx]
        r = F.relu(v) * self.alive
        mean_rate, mn_rates, rates = [], [], []
        for t in range(t_len):
            drive = torch.zeros(b, self.n, device=dev, dtype=ear_in.dtype).index_copy(1, self.ear_idx, ear_in[:, t])
            if once:
                used.append(r.detach())            # detached: kept with its graph, r would tie the graph to itself
                recurrent = r @ w_t + zeros[t]     # adding exact zeros: the same numbers as without
            else:
                recurrent = r @ w_t
            v = v + alpha * (-v + recurrent + bias + drive)
            r = F.relu(v) * self.alive
            mean_rate.append(r.mean(1))
            mn_rates.append(r[:, self.mn_idx])
            if record:
                rates.append(r)
        mn = torch.stack(mn_rates, 1)
        out = {"mean_rate": torch.stack(mean_rate, 1), "mn_rates": mn, **self.readout(mn)}
        if record:
            out["rates"] = torch.stack(rates, 1)
        return out, v

    def readout(self, mn):
        """Hit logits and loudness from the readout motor neurons' rates mn (B, T, readout neurons), for all frames at
        once: the readout never feeds back into the network, so it runs after the frame loop. Each frame reads that
        frame's scaled rates z (and, with readout history on, the means of earlier blocks of z) of its drum's own
        motor neurons only."""
        z = (mn - self.mn_mean) / self.mn_std
        eff_hit = eff_vel = z
        if self.history_blocks:
            feats = [z] + self.history_means(z)
            eff_hit = sum(f * (self.mn_drum @ self.p_hit)[:, k] for k, f in enumerate(feats))
            if self.p_vel is not None:
                eff_vel = sum(f * (self.mn_drum @ self.p_vel)[:, k] for k, f in enumerate(feats))
        return {"hit_logits": (eff_hit * self.w_hit) @ self.mn_drum + self.c_hit,
                "vel": torch.sigmoid((eff_vel * self.u_vel) @ self.mn_drum + self.e_vel)}

    def history_means(self, z):
        """For each history block, the mean of z over its frames before each frame (block j: frames edges[j] + 1 to
        edges[j + 1] back); frames before the clip count as z = 0 (each neuron's average activity). A moving average
        over the block's length (each mean adds only that many frames, so no precision is lost over a long clip),
        on z with the longest lookback of zeros put in front."""
        t_len = z.shape[1]
        edges = [int(e) for e in np.cumsum([0] + self.history_blocks)]
        back = edges[-1]
        zp = F.pad(z.transpose(1, 2), (back, 0))                          # (B, n, back + T)
        means = []
        for j, n_j in enumerate(self.history_blocks):
            pooled = F.avg_pool1d(zp, n_j, stride=1)                       # pooled[k] = mean of zp[k : k + n_j]
            first = back - edges[j + 1]                                    # frame t's block starts at t - edges[j + 1]
            means.append(pooled[:, :, first:first + t_len].transpose(1, 2))
        return means

    def forward(self, audio, v=None, record=False):
        return self.run(self.ear_input(audio), v, record)

    @torch.no_grad()
    def fit_readout_norm(self, audio_batches, warmup_frames, std_floor):
        """Set each readout motor neuron's mean and spread from music through this (untrained) network.
        Untrained rates vary by about 0.001, far too little for readout weights near 0.01 to use."""
        rm = torch.cat([self(a)[0]["mn_rates"][:, warmup_frames:].flatten(0, 1) for a in audio_batches])
        self.mn_mean.copy_(rm.mean(0))
        self.mn_std.copy_(rm.std(0).clamp_min(std_floor))
