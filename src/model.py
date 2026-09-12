"""Phase 3 model: a rate-based recurrent network whose topology is the fly.

The connectome fixes *which* neurons connect and in *which direction*; the
transmitter fixes the sign. Training may only change what each connection is
worth. Concretely, the recurrent weight is

    W[i,j] = sign[i] * softplus(log_gain[i,j])

with ``sign`` frozen and ``log_gain`` initialised at ``log(synapse_count)``, so
the network starts at the animal's own relative connection strengths.

Dynamics are a rate relaxation of LIF rather than surrogate-gradient spiking.
PLAN.md's open question 3 asks which to use; the rate model is here because it
converges reliably, and ``--spiking`` is left as the follow-on once the rate
model is solid on the 3-piece sanity task.

    tau_i dv_i/dt = -v_i + sum_j W_ij r_j + I_i + b_i
    r_i          = softplus(v_i - theta_i)

Learnable: per-edge log-gain, per-neuron log-tau, per-neuron threshold.
Frozen:    topology, edge signs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


class SparseSpMM(torch.autograd.Function):
    """``out = A @ r`` for a fixed-sparsity A, differentiable in A's values.

    torch's own CSR autograd returns a gradient sized to the *deduplicated*
    values when the index list contains repeated (row, col) pairs, which the
    Phase 4 random-rewiring ablation can produce. Rather than depend on the
    edge list staying unique, the backward is written out: it is two lines of
    exact algebra and it removes the failure mode entirely.

        d out[b, i] / d A[i, j] = r[b, j]   ->  grad_v[e] = sum_b go[b, row_e] * r[b, col_e]
        grad_r = A^T @ go

    Forward via CSR is ~190x faster on CPU than rebuilding and coalescing a COO
    tensor every timestep, which matters a lot inside a 1,600-step BPTT loop.
    """

    @staticmethod
    def forward(ctx, values, r, crow, col, crow_t, col_t, perm_t, row, n):
        out = torch.sparse.mm(
            torch.sparse_csr_tensor(crow, col, values, size=(n, n)), r.t()
        ).t()
        ctx.save_for_backward(values, r, crow_t, col_t, perm_t, row, col)
        ctx.n = n
        return out

    @staticmethod
    def backward(ctx, grad_out):
        values, r, crow_t, col_t, perm_t, row, col = ctx.saved_tensors
        grad_out = grad_out.contiguous()

        grad_r = grad_v = None
        if ctx.needs_input_grad[1]:
            at = torch.sparse_csr_tensor(crow_t, col_t, values[perm_t], size=(ctx.n, ctx.n))
            grad_r = torch.sparse.mm(at, grad_out.t()).t()
        if ctx.needs_input_grad[0]:
            grad_v = (grad_out[:, row] * r[:, col]).sum(0)
        return grad_v, grad_r, None, None, None, None, None, None, None


@dataclass
class ModelConfig:
    step_ms: float = 5.0
    tau_ms_init: float = 20.0
    tau_ms_min: float = 5.0
    tau_ms_max: float = 200.0
    threshold_init: float = 0.0
    gain_scale: float | str = "auto"  # global scale on softplus(log_gain);
                                      # "auto" normalises the spectral radius
    spectral_radius: float = 0.9      # target rho(W) when gain_scale is "auto"
    input_scale: float = 1.0
    state_clip: float = 20.0          # keeps the relaxation from blowing up
    log_gain_clamp: tuple = (-15.0, 12.0)   # keeps exp() in range
    spiking: bool = False


class ConnectomeRNN(nn.Module):
    """Sparse recurrent network with fixed connectome topology.

    The recurrence is a single ``torch.sparse`` mat-mul per step. The sparse
    values are rebuilt from ``log_gain`` each forward pass so gradients flow to
    the per-edge parameters while the indices stay constant.
    """

    def __init__(
        self,
        edge_index: np.ndarray,
        edge_sign: np.ndarray,
        weight: np.ndarray,
        n_nodes: int,
        sensory_idx: np.ndarray,
        motor_idx: np.ndarray,
        modulatory_idx: dict[str, np.ndarray] | None = None,
        cfg: ModelConfig | None = None,
    ):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.n_nodes = int(n_nodes)

        # The recurrence computes  out[post] = sum_pre W[post, pre] * r[pre],
        # so the sparse matrix has row = post, col = pre. Edges are sorted by
        # row once here and the CSR index arrays are built once; only the
        # *values* change during training, so the structure is genuinely frozen.
        ei = torch.from_numpy(np.ascontiguousarray(edge_index)).long()
        row_all, col_all = ei[1], ei[0]
        order = torch.argsort(row_all * self.n_nodes + col_all)
        row, col = row_all[order].contiguous(), col_all[order].contiguous()

        crow = torch.zeros(self.n_nodes + 1, dtype=torch.int64)
        crow[1:] = torch.bincount(row, minlength=self.n_nodes).cumsum(0)
        # transpose, for the gradient w.r.t. the rate vector
        perm_t = torch.argsort(col * self.n_nodes + row)
        crow_t = torch.zeros(self.n_nodes + 1, dtype=torch.int64)
        crow_t[1:] = torch.bincount(col, minlength=self.n_nodes).cumsum(0)

        self.register_buffer("edge_row", row)
        self.register_buffer("edge_col", col)
        self.register_buffer("crow", crow)
        self.register_buffer("crow_t", crow_t)
        self.register_buffer("col_t", row[perm_t].contiguous())
        self.register_buffer("perm_t", perm_t)
        # NB: stored as [row, col] = [post, pre], the matrix convention -- not
        # the [pre, post] convention the constructor argument uses.
        self.register_buffer("edge_index", torch.stack([row, col]))
        self.register_buffer(
            "edge_sign", torch.from_numpy(edge_sign.astype(np.float32))[order].contiguous()
        )
        self.register_buffer("sensory_idx", torch.from_numpy(sensory_idx.astype(np.int64)))
        self.register_buffer("motor_idx", torch.from_numpy(motor_idx.astype(np.int64)))

        # Per-edge gain, initialised at the animal's own synapse counts.
        w = np.maximum(weight.astype(np.float32), 1.0)
        self.log_gain = nn.Parameter(torch.from_numpy(np.log(w))[order].contiguous())

        # Per-neuron time constant and threshold.
        n = self.n_nodes
        self.log_tau = nn.Parameter(torch.full((n,), float(np.log(self.cfg.tau_ms_init))))
        self.threshold = nn.Parameter(torch.full((n,), float(self.cfg.threshold_init)))
        self.bias = nn.Parameter(torch.zeros(n))

        # Global gain. Left free, each ablation arm starts at a wildly different
        # operating point -- the real subgraph's recurrent operator has
        # rho ~ 4,000 against ~400 for a degree-matched rewiring of it, so a
        # shared constant puts one arm in saturation and the other near-silent.
        # That would make the Phase 4 comparison a test of which topology
        # happened to land in range, not of whether the topology helps. So the
        # scale is normalised per arm to a common target spectral radius. The
        # connectome's own *relative* synapse counts are untouched; only the
        # overall scale moves, and it stays trainable through log_gain.
        if self.cfg.gain_scale == "auto":
            rho = self._spectral_radius()
            scale = self.cfg.spectral_radius / rho if rho > 0 else 1.0
        else:
            scale = float(self.cfg.gain_scale)
        self.register_buffer("gain_scale", torch.tensor(float(scale)))

        # Lesion / slider gates. Multiplicative on each neuron's output rate;
        # 1.0 is intact. Not trained -- set at inference by lesion mode and the
        # biological sliders.
        self.register_buffer("gate", torch.ones(n))

        self.modulatory_idx = {k: torch.from_numpy(v.astype(np.int64))
                               for k, v in (modulatory_idx or {}).items()}

    # -- parameter views ----------------------------------------------------
    @property
    def tau_steps(self) -> torch.Tensor:
        tau_ms = self.log_tau.exp().clamp(self.cfg.tau_ms_min, self.cfg.tau_ms_max)
        return tau_ms / self.cfg.step_ms

    def edge_weight(self) -> torch.Tensor:
        """W[i,j] = sign[i] * exp(log_gain[i,j]) * scale.

        exp, not softplus: with log_gain initialised at log(synapse_count) this
        makes the starting weight *exactly* the synapse count, so the network
        begins at the animal's own relative connection strengths and the ratios
        between connections survive the global rescaling untouched. softplus
        would give log(1 + synapse_count), which quietly compresses a 2,591-
        synapse connection and a 26-synapse one into a factor of ~2.4 instead
        of ~100.
        """
        lo, hi = self.cfg.log_gain_clamp
        return self.edge_sign * self.log_gain.clamp(lo, hi).exp() * self.gain_scale

    @torch.no_grad()
    def _spectral_radius(self, iters: int = 60) -> float:
        """|lambda_max| of the unscaled recurrent operator, by power iteration."""
        lo, hi = self.cfg.log_gain_clamp
        vals = self.edge_sign * self.log_gain.clamp(lo, hi).exp()
        a = torch.sparse_csr_tensor(self.crow, self.edge_col, vals,
                                    size=(self.n_nodes, self.n_nodes))
        x = torch.randn(self.n_nodes, 1, generator=torch.Generator().manual_seed(0))
        x /= x.norm()
        lam = 0.0
        for _ in range(iters):
            y = torch.sparse.mm(a, x)
            nrm = float(y.norm())
            if nrm == 0.0:
                return 0.0
            x, lam = y / nrm, nrm
        return lam

    def recurrent(self, values: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
        return SparseSpMM.apply(
            values, r, self.crow, self.edge_col, self.crow_t, self.col_t,
            self.perm_t, self.edge_row, self.n_nodes,
        )

    # -- dynamics -----------------------------------------------------------
    def initial_state(self, batch: int, device=None, dtype=None) -> torch.Tensor:
        return torch.zeros(batch, self.n_nodes,
                           device=device or self.log_tau.device,
                           dtype=dtype or self.log_tau.dtype)

    def rate(self, v: torch.Tensor) -> torch.Tensor:
        r = torch.nn.functional.softplus(v - self.threshold)
        return r * self.gate

    def forward(
        self,
        drive: torch.Tensor,                 # (B, T, n_sensory) from the encoder
        state: torch.Tensor | None = None,
        tonic: torch.Tensor | None = None,   # (B, n_nodes) neuromodulatory bias
        return_all: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the relaxation. Returns ``(motor_rates, final_state)``.

        ``motor_rates`` is ``(B, T, n_motor)``; with ``return_all`` it is the
        full ``(B, T, n_nodes)`` tensor instead (needed by the rate regulariser
        and by lesion analysis).
        """
        b, t, _ = drive.shape
        v = self.initial_state(b, drive.device, drive.dtype) if state is None else state
        w = self.edge_weight()
        alpha = (1.0 / self.tau_steps).clamp(max=1.0)
        base = self.bias if tonic is None else self.bias + tonic

        out = []
        for k in range(t):
            r = self.rate(v)
            rec = self.recurrent(w, r)
            inp = torch.zeros_like(v)
            inp.index_add_(
                1, self.sensory_idx,
                drive[:, k, :].to(v.dtype) * self.cfg.input_scale,
            )
            v = v + alpha * (-v + rec + inp + base)
            v = v.clamp(-self.cfg.state_clip, self.cfg.state_clip)
            out.append(self.rate(v) if return_all else self.rate(v)[:, self.motor_idx])

        return torch.stack(out, dim=1), v

    # -- lesion / slider machinery -----------------------------------------
    def set_gate(self, idx: np.ndarray | torch.Tensor, value: float) -> None:
        """Scale a population's output. ``value=0`` is a lesion."""
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.gate.device)
        self.gate[idx] = value

    def reset_gates(self) -> None:
        self.gate.fill_(1.0)

    def n_params(self) -> dict[str, int]:
        return {
            "edge_gain": self.log_gain.numel(),
            "tau": self.log_tau.numel(),
            "threshold": self.threshold.numel(),
            "bias": self.bias.numel(),
            "total": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }


class GenreModulation(nn.Module):
    """Genre as tonic drive into the octopaminergic pool, not a checkpoint swap.

    PLAN.md open question 2 asks whether tonic octopaminergic drive is
    numerically stable as a plain bias. It is not, unbounded: the OA pool is
    small (25 neurons) and feeds high-gain drive populations, so an unconstrained
    embedding saturates pC1/pIP10 within a few steps. So the embedding is passed
    through a tanh and a learned-but-bounded scale, which keeps the injected
    current inside a fixed envelope while still letting genres separate. One
    model, swap the style vector at inference, and interpolation between genres
    stays meaningful because the map is smooth.
    """

    def __init__(self, n_styles: int, target_idx: np.ndarray, n_nodes: int,
                 dim: int = 8, max_current: float = 0.5):
        super().__init__()
        self.embed = nn.Embedding(n_styles, dim)
        nn.init.normal_(self.embed.weight, std=0.1)
        self.project = nn.Linear(dim, len(target_idx))
        nn.init.zeros_(self.project.bias)
        nn.init.normal_(self.project.weight, std=0.1)
        self.register_buffer("target_idx", torch.from_numpy(target_idx.astype(np.int64)))
        self.n_nodes = int(n_nodes)
        self.max_current = float(max_current)

    def forward(self, style_id: torch.Tensor) -> torch.Tensor:
        """``(B,)`` style ids -> ``(B, n_nodes)`` tonic current."""
        e = self.embed(style_id)
        cur = torch.tanh(self.project(e)) * self.max_current
        out = torch.zeros(style_id.shape[0], self.n_nodes,
                          device=cur.device, dtype=cur.dtype)
        out.index_add_(1, self.target_idx, cur)
        return out

    def interpolate(self, a: int, b: int, alpha: float, device=None) -> torch.Tensor:
        """Blend two style vectors -- the thing per-genre checkpoints cannot do."""
        device = device or self.embed.weight.device
        ea = self.embed.weight[a]
        eb = self.embed.weight[b]
        e = (1 - alpha) * ea + alpha * eb
        cur = torch.tanh(self.project(e)) * self.max_current
        out = torch.zeros(1, self.n_nodes, device=device, dtype=cur.dtype)
        out.index_add_(1, self.target_idx, cur.unsqueeze(0))
        return out


class FlyDrums(nn.Module):
    """Encoder -> ConnectomeRNN -> decoder, plus the slider/lesion surface."""

    def __init__(self, encoder, rnn: ConnectomeRNN, decoder, genre: GenreModulation | None = None):
        super().__init__()
        self.encoder = encoder
        self.rnn = rnn
        self.decoder = decoder
        self.genre = genre
        self._slider_base: dict[str, torch.Tensor] = {}

    def forward(self, wav, state=None, style_id=None, return_rates=False):
        drive = self.encoder(wav)
        tonic = self.genre(style_id) if (self.genre is not None and style_id is not None) else None
        rates, state = self.rnn(drive, state=state, tonic=tonic, return_all=return_rates)
        motor = rates[:, :, self.rnn.motor_idx] if return_rates else rates
        logits = self.decoder(motor)
        return (logits, state, rates) if return_rates else (logits, state)

    # -- biological sliders -------------------------------------------------
    def set_slider(self, name: str, value: float, roles: dict[str, np.ndarray]) -> None:
        """Each slider maps to a real, Phase-0-confirmed population.

        drive     -- pC1 gain: fill density and intensity
        tightness -- gain on inhibitory (GABA/Glu) neurons: tight vs sloppy timing
        pocket    -- membrane time-constant scale: slower tau drags the beat
        gate      -- pIP10 gain: song on/off, near-binary
        """
        if name == "drive":
            self.rnn.set_gate(roles.get("pC1", np.array([], dtype=np.int64)), value)
        elif name == "gate":
            self.rnn.set_gate(roles.get("pIP10", np.array([], dtype=np.int64)), value)
        elif name == "tightness":
            idx = roles.get("inhibitory", np.array([], dtype=np.int64))
            self.rnn.set_gate(idx, value)
        elif name == "pocket":
            if "log_tau" not in self._slider_base:
                self._slider_base["log_tau"] = self.rnn.log_tau.detach().clone()
            with torch.no_grad():
                self.rnn.log_tau.copy_(self._slider_base["log_tau"] + float(np.log(max(value, 1e-3))))
        else:
            raise KeyError(f"unknown slider {name!r}")

    def reset_sliders(self) -> None:
        self.rnn.reset_gates()
        if "log_tau" in self._slider_base:
            with torch.no_grad():
                self.rnn.log_tau.copy_(self._slider_base["log_tau"])
