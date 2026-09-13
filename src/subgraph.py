"""Phase 1: carve a trainable subgraph out of the 162k-neuron connectome.

Full-graph BPTT is not viable at the sequence lengths this task needs (a 4-bar
loop at 5 ms is ~1,600 steps). So we keep the part of the circuit that can
actually carry signal from the ears to the wings:

    forward k hops from the JO afferents      (what the ear can reach)
  ∩ backward j hops from the wing motor pool  (what can reach the wings)
  ∪ seeds ∪ targets ∪ required populations    (sliders and genre need these)

Read the hop counts honestly. In a brain this small-world they barely
constrain anything: 3 hops forward from the JO afferents reaches 154,853 of
162,517 neurons and 3 hops backward from the wing motor pool reaches 130,934,
for an intersection of 128,433. Everything below ``max_nodes`` is therefore
chosen by ``_trim``, not by anatomy -- **the trim is the selection**, and its
score is a one-hop-in x one-hop-out heuristic rather than a path-based measure
of ear->wing throughput. Anyone reading "k-hop subgraph" as a meaningful
anatomical restriction here would be mistaken. Replacing the heuristic with a
defensible path criterion is on the open list in the README; until then the
claim this module supports is the weaker one.

Everything is driven by ``data/verified_types.json`` -- the Phase 0 gate output
-- so no type string is hardcoded here.

Synaptic signs come from the presynaptic neuron's transmitter (Dale's law) and
are **frozen**. Training may only change the magnitude of a connection, never
its sign and never its existence.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from connectome import Graph, build_neuron_graph, load_verified_types, types_for

ROOT = Path(__file__).resolve().parents[1]

#: Phase 0 concepts that seed the forward sweep (the ear).
DEFAULT_SEED_CONCEPTS = ("JO_A", "JO_B", "JO_E")
#: Phase 0 concepts that terminate the backward sweep (the wings).
DEFAULT_TARGET_CONCEPTS = ("wing_motor_all",)
#: Populations the feature surface needs regardless of whether the sweeps find
#: them: the drive/tightness sliders and the genre embedding attach here.
DEFAULT_REQUIRED_CONCEPTS = ("pC1", "pC2", "pIP10", "octopaminergic", "aPN1", "vPN1")


@dataclass
class SubGraph:
    """A trainable slice of the connectome, with frozen signs and roles."""

    node: np.ndarray                    # (n,) int32 -- indices into the parent graph
    edge_index: np.ndarray              # (2, e) int32 -- local indices
    weight: np.ndarray                  # (e,) float32 -- synapse counts
    edge_sign: np.ndarray               # (e,) float32 -- +1/-1, frozen
    body_ids: np.ndarray
    types: np.ndarray
    side: np.ndarray
    nt: np.ndarray
    roles: dict[str, np.ndarray] = field(default_factory=dict)  # concept -> local idx
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.node)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def role(self, name: str) -> np.ndarray:
        return self.roles.get(name, np.array([], dtype=np.int32))

    def nodes_of_types(self, names) -> np.ndarray:
        return np.flatnonzero(np.isin(self.types, list(names))).astype(np.int32)

    def side_mask(self, which: str) -> np.ndarray:
        return np.asarray([s == which for s in self.side], dtype=bool)

    def summary(self) -> str:
        e_exc = int((self.edge_sign > 0).sum())
        e_inh = int((self.edge_sign < 0).sum())
        dens = self.n_edges / max(self.n_nodes ** 2, 1)
        return (
            f"SubGraph: {self.n_nodes:,} neurons, {self.n_edges:,} edges "
            f"(density {dens:.2e}, mean in-degree {self.n_edges / max(self.n_nodes,1):.1f})\n"
            f"  signs: {e_exc:,} excitatory / {e_inh:,} inhibitory\n"
            f"  roles: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(self.roles.items()) if len(v))
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, node=self.node, edge_index=self.edge_index, weight=self.weight,
            edge_sign=self.edge_sign, body_ids=self.body_ids, types=self.types,
            side=self.side, nt=self.nt, meta=json.dumps(self.meta),
            role_names=np.array(list(self.roles), dtype=object),
            **{f"role__{k}": v for k, v in self.roles.items()},
        )

    @classmethod
    def load(cls, path: Path) -> "SubGraph":
        z = np.load(path, allow_pickle=True)
        roles = {str(k): z[f"role__{k}"] for k in z["role_names"]}
        return cls(
            node=z["node"], edge_index=z["edge_index"], weight=z["weight"],
            edge_sign=z["edge_sign"], body_ids=z["body_ids"], types=z["types"],
            side=z["side"], nt=z["nt"], roles=roles, meta=json.loads(str(z["meta"])),
        )


def _csr(g: Graph, n: int, min_weight: int) -> sp.csr_matrix:
    src, dst = g.edge_index
    w = g.weight
    if min_weight > 1:
        keep = w >= min_weight
        src, dst, w = src[keep], dst[keep], w[keep]
    return sp.csr_matrix((w.astype(np.float32), (src, dst)), shape=(n, n))


def _sweep(adj: sp.csr_matrix, seeds: np.ndarray, hops: int) -> np.ndarray:
    """Boolean reachability within ``hops`` steps along ``adj``."""
    n = adj.shape[0]
    reached = np.zeros(n, dtype=bool)
    reached[seeds] = True
    frontier = reached.copy()
    for _ in range(hops):
        # adj[i, j] != 0 means i -> j, so one hop forward is frontier @ adj.
        nxt = (frontier.astype(np.float32) @ adj) > 0
        nxt &= ~reached
        if not nxt.any():
            break
        reached |= nxt
        frontier = nxt
    return reached


def extract(
    g: Graph,
    verified: dict,
    seed_concepts=DEFAULT_SEED_CONCEPTS,
    target_concepts=DEFAULT_TARGET_CONCEPTS,
    required_concepts=DEFAULT_REQUIRED_CONCEPTS,
    forward_hops: int = 3,
    backward_hops: int = 3,
    min_weight: int = 3,
    max_nodes: int | None = 30_000,
    full_graph: bool = False,
) -> SubGraph:
    n = g.n_nodes
    seed_types = types_for(verified, *seed_concepts)
    target_types = types_for(verified, *target_concepts)
    seeds = g.nodes_of_types(seed_types)
    targets = g.nodes_of_types(target_types)
    if len(seeds) == 0 or len(targets) == 0:
        raise SystemExit("seed or target population is empty -- re-run the Phase 0 gate")

    required: dict[str, np.ndarray] = {}
    for c in required_concepts:
        try:
            required[c] = g.nodes_of_types(types_for(verified, c))
        except KeyError:
            required[c] = np.array([], dtype=np.int32)

    if full_graph:
        keep = np.ones(n, dtype=bool)
        print(f"  --full-graph: keeping all {n:,} neurons (render mode)")
    else:
        adj = _csr(g, n, min_weight)
        print(f"  sweeping from {len(seeds)} JO afferents / {len(targets)} wing MNs "
              f"(min_weight={min_weight})")
        fwd = _sweep(adj, seeds, forward_hops)
        bwd = _sweep(adj.T.tocsr(), targets, backward_hops)
        keep = fwd & bwd
        print(f"  forward {forward_hops}-hop: {fwd.sum():,} | backward {backward_hops}-hop: "
              f"{bwd.sum():,} | intersection: {keep.sum():,}")

        keep[seeds] = True
        keep[targets] = True
        for idx in required.values():
            keep[idx] = True

        if max_nodes is not None and keep.sum() > max_nodes:
            keep = _trim(adj, keep, seeds, targets, required, max_nodes)
            print(f"  trimmed to {keep.sum():,} (cap {max_nodes:,})")

    node = np.flatnonzero(keep).astype(np.int32)
    remap = np.full(n, -1, dtype=np.int32)
    remap[node] = np.arange(len(node), dtype=np.int32)

    src, dst = g.edge_index
    w = g.weight
    emask = keep[src] & keep[dst]
    if min_weight > 1 and not full_graph:
        emask &= w >= min_weight
    src, dst, w = remap[src[emask]], remap[dst[emask]], w[emask].astype(np.float32)

    # Dale's law: the sign of a synapse is a property of the *presynaptic*
    # neuron's transmitter. Frozen -- only magnitudes are learned.
    presign = g.sign[node][src].astype(np.float32)
    presign[presign == 0] = 1.0   # modulatory neurons act positively in the fast graph

    sg = SubGraph(
        node=node,
        edge_index=np.stack([src, dst]),
        weight=w,
        edge_sign=presign,
        body_ids=g.body_ids[node],
        types=g.types[node],
        side=g.side[node],
        nt=g.nt[node],
        roles={
            "sensory": remap[seeds][remap[seeds] >= 0],
            "motor": remap[targets][remap[targets] >= 0],
            **{c: remap[idx][remap[idx] >= 0] for c, idx in required.items()},
        },
        meta=dict(
            dataset=verified["dataset"],
            seed_concepts=list(seed_concepts),
            target_concepts=list(target_concepts),
            forward_hops=forward_hops, backward_hops=backward_hops,
            min_weight=min_weight, max_nodes=max_nodes, full_graph=full_graph,
            parent_nodes=int(n),
        ),
    )
    return sg


def _trim(adj, keep, seeds, targets, required, max_nodes) -> np.ndarray:
    """Shrink to ``max_nodes`` by dropping the weakest pathway members.

    Score = (input from the kept set) * (output to the kept set), geometric
    mean. A neuron that both hears and acts scores high; a dead-end scores zero.

    Two hops of context, so it is a *local* proxy for ear->wing throughput and
    not a path measure: a neuron wired into a busy cluster with no route to the
    motor pool still scores well. Since the k-hop sweeps keep ~128k of 162k
    neurons, this heuristic -- not the anatomy -- is what actually picks the
    10k-30k neurons that get trained. That is a weaker claim than "the subgraph
    is the ear-to-wing pathway" and it is the one the code supports.
    """
    kf = keep.astype(np.float32)
    inflow = adj.T @ kf
    outflow = adj @ kf
    score = np.sqrt(np.maximum(inflow, 0) * np.maximum(outflow, 0))

    protected = np.zeros_like(keep)
    protected[seeds] = True
    protected[targets] = True
    for idx in required.values():
        protected[idx] = True

    candidates = np.flatnonzero(keep & ~protected)
    budget = max_nodes - int(protected.sum())
    if budget <= 0:
        return protected
    best = candidates[np.argsort(-score[candidates])[:budget]]
    out = protected.copy()
    out[best] = True
    return out


def hops_from(sg, src: np.ndarray, max_hops: int = 8) -> np.ndarray:
    """BFS hop distance from ``src`` along the subgraph's directed edges.

    ``-1`` for anything the sweep never reaches. This is the same forward sweep
    Phase 1 uses to select the subgraph, run again here on the trimmed graph so
    the depth labels match the network that actually ran.
    """
    n = sg.n_nodes
    pre, post = sg.edge_index[0], sg.edge_index[1]
    a = sp.csr_matrix((np.ones(len(pre), dtype=bool), (pre, post)), shape=(n, n))
    dist = np.full(n, -1, dtype=np.int32)
    seen = np.zeros(n, dtype=bool)
    frontier = np.zeros(n, dtype=bool)
    frontier[src] = True
    dist[src] = 0
    seen |= frontier
    for h in range(1, max_hops + 1):
        nxt = (a.T @ frontier.astype(np.int8)) > 0
        nxt &= ~seen
        if not nxt.any():
            break
        dist[nxt] = h
        seen |= nxt
        frontier = nxt
    return dist


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--forward-hops", type=int, default=3)
    ap.add_argument("--backward-hops", type=int, default=3)
    ap.add_argument("--min-weight", type=int, default=3,
                    help="drop connections weaker than this many synapses")
    ap.add_argument("--max-nodes", type=int, default=30_000)
    ap.add_argument("--full-graph", action="store_true",
                    help="render mode: keep all 162k neurons, offline single-batch use only")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "cache" / "subgraph.npz")
    a = ap.parse_args(argv)

    g = build_neuron_graph()
    print(g.summary())
    sg = extract(
        g, load_verified_types(),
        forward_hops=a.forward_hops, backward_hops=a.backward_hops,
        min_weight=a.min_weight,
        max_nodes=None if a.full_graph else a.max_nodes,
        full_graph=a.full_graph,
    )
    print(sg.summary())
    sg.save(a.out)
    print(f"saved -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
