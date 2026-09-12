"""Phase 4: the controls that decide whether any of this means anything.

PLAN.md's core claim is that *real* connectome topology is a useful inductive
bias for this task. That claim is only testable against controls, so all five
arms run here on identical data, identical hyperparameters and identical
encoder/decoder, with only the recurrent core changed:

  real          the actual MaleCNS subgraph
  rewired       degree-matched random rewiring of that same subgraph
  sign_shuffled real topology, neurotransmitter signs shuffled
  gru           dense GRU, parameter count matched, no connectome at all
  shortcut      encoder straight to decoder, connectome bypassed (sanity floor)

If ``real`` does not beat ``rewired`` and ``sign_shuffled``, the connectome is
not contributing and that is the result. Report it either way.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from subgraph import SubGraph

ROOT = Path(__file__).resolve().parents[1]

ARMS = ("real", "rewired", "sign_shuffled", "gru", "shortcut")


# ---------------------------------------------------------------- topology --
def degree_matched_rewire(sg: SubGraph, seed: int = 0) -> SubGraph:
    """Rewire while preserving every node's in- and out-degree exactly.

    Uses the configuration model on the degree sequence: split each edge into a
    out-stub and an in-stub, shuffle the in-stubs, reconnect. Degree sequence,
    edge count and total synaptic weight are all preserved, so the only thing
    that changes is *which* neuron talks to which. That is precisely the
    variable under test.

    Synapse counts travel with the edges (shuffled alongside) so the weight
    distribution is preserved too, not just the count.
    """
    rng = np.random.default_rng(seed)
    src, dst = sg.edge_index
    new_dst = dst.copy()
    rng.shuffle(new_dst)                      # permute in-stubs

    # Signs stay a property of the presynaptic neuron -- Dale's law must
    # survive rewiring, otherwise this arm would be testing two changes at once.
    presign = np.zeros(sg.n_nodes, dtype=np.float32)
    presign[src] = sg.edge_sign
    order = rng.permutation(len(src))

    return replace(
        sg,
        edge_index=np.stack([src, new_dst]),
        weight=sg.weight[order].copy(),
        edge_sign=presign[src],
        meta={**sg.meta, "ablation": "rewired", "seed": seed},
    )


def shuffle_signs(sg: SubGraph, seed: int = 0) -> SubGraph:
    """Keep the real wiring, permute which neurons are excitatory.

    Shuffled per *neuron*, not per edge, so Dale's law still holds -- a neuron
    is still uniformly excitatory or inhibitory, just possibly the wrong one.
    This isolates the contribution of the neurotransmitter predictions from the
    contribution of the topology.
    """
    rng = np.random.default_rng(seed)
    src, _ = sg.edge_index
    per_node = np.zeros(sg.n_nodes, dtype=np.float32)
    per_node[src] = sg.edge_sign
    rng.shuffle(per_node)
    return replace(
        sg, edge_sign=per_node[src],
        meta={**sg.meta, "ablation": "sign_shuffled", "seed": seed},
    )


# ------------------------------------------------------------- baselines ----
class GRUCore(nn.Module):
    """Dense GRU with no connectome structure, parameter-count matched.

    Exposes the same interface as ConnectomeRNN so train.py needs no special
    case: same forward signature, same ``motor_idx``, same gate machinery
    (inert here -- there is no cell type to lesion, which is itself the point).
    """

    def __init__(self, n_sensory: int, n_motor: int, target_params: int, cfg):
        super().__init__()
        self.cfg = cfg
        hidden = _gru_hidden_for(target_params, n_sensory, n_motor)
        self.hidden = hidden
        self.gru = nn.GRU(n_sensory, hidden, batch_first=True)
        self.readout = nn.Linear(hidden, n_motor)
        self.register_buffer("motor_idx", torch.arange(n_motor))
        self.register_buffer("gate", torch.ones(n_motor))
        self.n_nodes = hidden

    def initial_state(self, batch, device=None, dtype=None):
        return torch.zeros(1, batch, self.hidden,
                           device=device or self.readout.weight.device,
                           dtype=dtype or self.readout.weight.dtype)

    def forward(self, drive, state=None, tonic=None, return_all=False):
        if state is not None and state.dim() == 2:
            state = state.unsqueeze(0)
        h, state = self.gru(drive, state)
        rates = torch.nn.functional.softplus(self.readout(h))
        return rates, state

    def set_gate(self, idx, value):     # no cell types to lesion
        pass

    def reset_gates(self):
        pass

    def n_params(self):
        return {"total": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "hidden": self.hidden}


def _gru_hidden_for(target: int, n_in: int, n_out: int) -> int:
    """Largest hidden size whose parameter count does not exceed ``target``.

    GRU params = 3*(h*n_in + h*h + 2*h); readout adds h*n_out + n_out. Solved by
    bisection rather than algebra so the formula cannot drift out of sync with
    whatever nn.GRU actually allocates.
    """
    def count(h: int) -> int:
        return 3 * (h * n_in + h * h + 2 * h) + h * n_out + n_out

    lo, hi = 1, 1
    while count(hi) < target and hi < 1 << 16:
        hi *= 2
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(mid) <= target:
            lo = mid
        else:
            hi = mid - 1
    # No minimum-size floor. A floor would let the GRU quietly exceed the budget
    # on a small target, and "parameter-matched" is the entire point of this arm.
    return max(lo, 1)


class ShortcutCore(nn.Module):
    """Encoder straight to decoder. The floor: no recurrence at all.

    If an arm with no memory scores near the connectome arm, the task is not
    testing what we think it is testing.
    """

    def __init__(self, n_sensory: int, n_motor: int, cfg):
        super().__init__()
        self.cfg = cfg
        self.proj = nn.Linear(n_sensory, n_motor)
        self.register_buffer("motor_idx", torch.arange(n_motor))
        self.register_buffer("gate", torch.ones(n_motor))
        self.n_nodes = n_motor

    def initial_state(self, batch, device=None, dtype=None):
        return torch.zeros(batch, self.n_nodes,
                           device=device or self.proj.weight.device,
                           dtype=dtype or self.proj.weight.dtype)

    def forward(self, drive, state=None, tonic=None, return_all=False):
        return torch.nn.functional.softplus(self.proj(drive)), self.initial_state(
            drive.shape[0], drive.device, drive.dtype)

    def set_gate(self, idx, value):
        pass

    def reset_gates(self):
        pass

    def n_params(self):
        return {"total": sum(p.numel() for p in self.parameters() if p.requires_grad)}


# ------------------------------------------------------------- lesion mode --
def lesion(model, roles: dict, population: str, level: float = 0.0) -> None:
    """Mute (or scale) a confirmed cell population mid-performance.

    Same machinery as the ablation arms, applied at inference instead of before
    training -- which is what makes lesion mode cheap to ship.
    """
    idx = roles.get(population)
    if idx is None or len(idx) == 0:
        raise KeyError(f"no population {population!r} in this subgraph; "
                       f"have {sorted(k for k, v in roles.items() if len(v))}")
    model.rnn.set_gate(idx, level)


def lesion_sweep(model, loader, cfg, device, roles: dict, populations, evaluate_fn) -> list[dict]:
    """Score the model intact, then with each population silenced in turn."""
    model.rnn.reset_gates()
    rows = [{"population": "intact", "n_neurons": 0, **evaluate_fn(model, loader, cfg, device)}]
    baseline = rows[0]["onset_f"]
    for pop in populations:
        idx = roles.get(pop, np.array([], dtype=np.int64))
        if len(idx) == 0:
            continue
        model.rnn.reset_gates()
        lesion(model, roles, pop, 0.0)
        m = evaluate_fn(model, loader, cfg, device)
        rows.append({"population": pop, "n_neurons": int(len(idx)), **m,
                     "delta_onset_f": m["onset_f"] - baseline})
    model.rnn.reset_gates()
    return rows


# ------------------------------------------------------------------- driver --
def build_arm(arm: str, cfg: dict, sg: SubGraph, n_styles: int):
    """Build one ablation arm. Encoder and decoder are identical across arms."""
    from build import build_model

    if arm == "real":
        return build_model(cfg, sg, n_styles=n_styles)
    if arm == "rewired":
        return build_model(cfg, degree_matched_rewire(sg, cfg["train"].get("seed", 0)),
                           n_styles=n_styles)
    if arm == "sign_shuffled":
        return build_model(cfg, shuffle_signs(sg, cfg["train"].get("seed", 0)),
                           n_styles=n_styles)

    # GRU and shortcut reuse the real encoder/decoder, swapping only the core.
    model, kit = build_model(cfg, sg, n_styles=n_styles)
    n_sensory = len(sg.role("sensory"))
    n_motor = len(sg.role("motor"))
    target = model.rnn.n_params()["total"]
    model.rnn = (GRUCore(n_sensory, n_motor, target, model.rnn.cfg) if arm == "gru"
                 else ShortcutCore(n_sensory, n_motor, model.rnn.cfg))
    return model, kit


def main(argv=None) -> int:
    import train as train_mod
    from build import device_of, get_subgraph, load_config, role_index
    from decoder import DrumKit

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece.yaml")
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--lesion", action="store_true",
                    help="after training the real arm, sweep lesions over confirmed populations")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    if a.smoke:
        cfg["data"].update(synthetic=True, n_clips=8, seconds=1.0)
        cfg["train"].update(batch_size=2, epochs=1, tbptt_steps=50)
    if a.epochs is not None:
        cfg["train"]["epochs"] = a.epochs

    device = device_of(cfg)
    sg = get_subgraph(cfg)
    roles = role_index(sg)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    train_loader, val_loader, n_styles = train_mod.build_loaders(cfg, kit)
    n_styles = max(int(n_styles or 1), 1)

    out = a.out or ROOT / "runs" / f"ablation_{a.config.stem}"
    out.mkdir(parents=True, exist_ok=True)
    results, lesion_rows = [], []

    for arm in a.arms:
        # identical seed per arm: the topology is the only thing that differs
        torch.manual_seed(cfg["train"].get("seed", 0))
        np.random.seed(cfg["train"].get("seed", 0))

        model, kit = build_arm(arm, cfg, sg, n_styles)
        model = model.to(device)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=cfg["train"].get("lr", 3e-3))
        n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n=== {arm} === params {n_par:,} | core {type(model.rnn).__name__}")

        for ep in range(cfg["train"].get("epochs", 10)):
            tr = train_mod.run_epoch(model, train_loader, opt, cfg, device, train=True)
            ev = train_mod.evaluate(model, val_loader, cfg, device)
            print(f"  ep{ep:>3} loss {tr['loss']:.4f} | onset_F {ev['onset_f']:.4f} "
                  f"@thr {ev['best_threshold']:.2f} | groove {ev['groove_sim']:.3f}")

        ev = train_mod.evaluate(model, val_loader, cfg, device)
        results.append({"arm": arm, "params": n_par, **ev})
        torch.save({"model": model.state_dict(), "arm": arm, "config": cfg}, out / f"{arm}.pt")

        if a.lesion and arm == "real":
            pops = [p for p in ("pC1", "pIP10", "octopaminergic", "aPN1", "vPN1",
                                "pC2", "sensory", "inhibitory") if len(roles.get(p, []))]
            lesion_rows = lesion_sweep(model, val_loader, cfg, device, roles, pops,
                                       train_mod.evaluate)

    (out / "ablation_results.json").write_text(json.dumps(
        {"config": str(a.config), "arms": results, "lesion": lesion_rows}, indent=2))
    print("\n" + format_table(results))
    if lesion_rows:
        print("\n" + format_lesion(lesion_rows))
    print(f"\nwrote {out / 'ablation_results.json'}")
    return 0


def format_table(rows: list[dict]) -> str:
    head = (f"{'arm':<16}{'params':>12}{'onset F':>10}{'@thr':>7}{'groove':>9}"
            f"{'beat ms':>10}{'dev ms':>9}")
    lines = [head, "-" * len(head)]
    for r in rows:
        lines.append(f"{r['arm']:<16}{r['params']:>12,}{r['onset_f']:>10.4f}"
                     f"{r.get('best_threshold', float('nan')):>7.2f}"
                     f"{r['groove_sim']:>9.3f}{r['beat_align_ms']:>10.1f}{r['mean_dev_ms']:>9.1f}")
    lines.append("\nonset F is reported at each arm's best peak-picking threshold, "
                 "chosen on this split;\na single fixed threshold would score arms "
                 "at a point that suits whichever one\nhappens to sit at that output scale.")
    return "\n".join(lines)


def format_lesion(rows: list[dict]) -> str:
    head = f"{'lesion':<18}{'n':>6}{'onset F':>10}{'delta':>9}"
    lines = ["Lesion sweep (real arm)", head, "-" * len(head)]
    for r in rows:
        d = r.get("delta_onset_f")
        lines.append(f"{r['population']:<18}{r['n_neurons']:>6}{r['onset_f']:>10.4f}"
                     f"{('' if d is None else f'{d:>+9.4f}')}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
