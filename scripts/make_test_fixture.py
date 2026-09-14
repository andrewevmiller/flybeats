"""Derive a small, *real* subgraph fixture from a cached one.

The tests that matter most for shipping -- bundle round-trips, gradient
checkpointing, the velocity head riding along with one forward pass -- all need
a SubGraph to build a model from. Building one needs the 1.1 GB connectome
download, so on a fresh checkout (and in CI) they skipped, and the export path
that v0.1 depends on was covered by nothing at all.

A synthetic fixture would not do: Dale's law, the transmitter table and the
confirmed populations are the properties these graphs are supposed to carry.
So the fixture is an *induced* subgraph of a real cached one -- every neuron,
edge, sign and type in it came out of MaleCNS v1.0 -- reduced to the point
where a test can build a model in under a second.

    python scripts/make_test_fixture.py \
        --source tests/fixtures/subgraph_10k.npz \
        --out tests/fixtures/subgraph_2k.npz --max-nodes 2000

What it keeps: every node carrying a role (sensory, motor, pC1, pC2, pIP10,
octopaminergic, aPN1, vPN1), because a model cannot be built without them, and
then the most strongly connected of the rest up to ``--max-nodes``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from subgraph import SubGraph  # noqa: E402


def induce(sg: SubGraph, max_nodes: int) -> SubGraph:
    """Induce a ``max_nodes``-node subgraph, keeping every role node."""
    required = np.unique(np.concatenate(
        [v for v in sg.roles.values() if len(v)] or [np.array([], dtype=np.int32)]
    )).astype(np.int64)
    if len(required) > max_nodes:
        raise SystemExit(
            f"{len(required)} role nodes exceed --max-nodes {max_nodes}; "
            f"a model cannot be built without them"
        )

    # Rank the rest by total synaptic traffic, so the fixture keeps the part of
    # the graph that actually carries signal rather than an arbitrary slice.
    strength = np.zeros(sg.n_nodes, dtype=np.float64)
    np.add.at(strength, sg.edge_index[0].astype(np.int64), sg.weight)
    np.add.at(strength, sg.edge_index[1].astype(np.int64), sg.weight)
    strength[required] = np.inf
    keep = np.sort(np.argsort(-strength, kind="stable")[:max_nodes]).astype(np.int64)

    remap = np.full(sg.n_nodes, -1, dtype=np.int64)
    remap[keep] = np.arange(len(keep))
    src, dst = sg.edge_index[0].astype(np.int64), sg.edge_index[1].astype(np.int64)
    live = (remap[src] >= 0) & (remap[dst] >= 0)

    meta = dict(sg.meta)
    meta.update(max_nodes=int(len(keep)), parent_nodes=int(sg.n_nodes),
                derived_from="induced subgraph of a real cached SubGraph, "
                             "for tests only -- not an extraction from the connectome")
    return SubGraph(
        node=sg.node[keep],
        edge_index=np.stack([remap[src[live]], remap[dst[live]]]).astype(np.int32),
        weight=sg.weight[live], edge_sign=sg.edge_sign[live],
        body_ids=sg.body_ids[keep], types=sg.types[keep],
        side=sg.side[keep], nt=sg.nt[keep],
        roles={k: remap[v.astype(np.int64)].astype(np.int32)
               for k, v in sg.roles.items()},
        meta=meta,
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path,
                    default=ROOT / "tests" / "fixtures" / "subgraph_10k.npz")
    ap.add_argument("--out", type=Path,
                    default=ROOT / "tests" / "fixtures" / "subgraph_2k.npz")
    ap.add_argument("--max-nodes", type=int, default=2000)
    a = ap.parse_args(argv)

    sg = SubGraph.load(a.source)
    small = induce(sg, a.max_nodes)
    small.save(a.out)
    print(small.summary())
    print(f"  -> {a.out} ({a.out.stat().st_size / 1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
