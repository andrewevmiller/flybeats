"""The ablations are the evidence, so their invariants get tested.

A rewiring that quietly changed the degree sequence, or a sign shuffle that
also perturbed topology, would make the comparison meaningless while still
producing a plausible-looking number.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ablations import _gru_hidden_for, degree_matched_rewire, shuffle_signs  # noqa: E402
from subgraph import SubGraph  # noqa: E402


def _toy(n=200, e=2000, seed=0) -> SubGraph:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, n, e).astype(np.int32)
    dst = rng.integers(0, n, e).astype(np.int32)
    per_node = rng.choice([-1.0, 1.0], n).astype(np.float32)
    return SubGraph(
        node=np.arange(n, dtype=np.int32),
        edge_index=np.stack([src, dst]),
        weight=rng.integers(1, 100, e).astype(np.float32),
        edge_sign=per_node[src],
        body_ids=np.arange(n, dtype=np.int64),
        types=np.array([f"T{i % 7}" for i in range(n)], dtype=object),
        side=np.array(["L" if i % 2 else "R" for i in range(n)], dtype=object),
        nt=np.array(["acetylcholine"] * n, dtype=object),
        roles={"sensory": np.arange(5, dtype=np.int32), "motor": np.arange(5, 10, dtype=np.int32)},
    )


def test_rewire_preserves_degrees_edges_and_weight():
    sg = _toy()
    rw = degree_matched_rewire(sg, seed=1)

    assert rw.n_edges == sg.n_edges
    assert rw.n_nodes == sg.n_nodes
    # out-degree is preserved exactly (the out-stubs are untouched)
    assert np.array_equal(np.bincount(rw.edge_index[0], minlength=sg.n_nodes),
                          np.bincount(sg.edge_index[0], minlength=sg.n_nodes))
    # in-degree is preserved exactly too: permuting the destination array
    # changes which edge lands where but not how often each node appears
    assert np.array_equal(np.bincount(rw.edge_index[1], minlength=sg.n_nodes),
                          np.bincount(sg.edge_index[1], minlength=sg.n_nodes))
    assert rw.weight.sum() == sg.weight.sum()
    assert np.array_equal(np.sort(rw.weight), np.sort(sg.weight))
    # it must actually have rewired something
    assert not np.array_equal(rw.edge_index[1], sg.edge_index[1])


def test_rewire_keeps_dales_law():
    """Every neuron must stay uniformly excitatory or inhibitory after rewiring."""
    sg = _toy()
    rw = degree_matched_rewire(sg, seed=2)
    for node in np.unique(rw.edge_index[0]):
        signs = rw.edge_sign[rw.edge_index[0] == node]
        assert len(np.unique(signs)) == 1, f"node {node} emits mixed signs"


def test_sign_shuffle_keeps_topology_identical():
    """Only the signs may move -- otherwise the arm tests two things at once."""
    sg = _toy()
    ss = shuffle_signs(sg, seed=3)
    assert np.array_equal(ss.edge_index, sg.edge_index)
    assert np.array_equal(ss.weight, sg.weight)
    assert sorted(ss.edge_sign.tolist()) != [] and not np.array_equal(ss.edge_sign, sg.edge_sign)
    # A zero sign is a deleted edge that edge_index cannot show, so the two
    # assertions above passed straight through a shuffle that was dropping
    # edges. This is the one that catches it.
    assert (ss.edge_sign != 0).all(), "sign shuffle silently deleted edges"
    # Per-neuron signs are reassigned, not invented: the excitatory/inhibitory
    # balance over neurons must survive. Edge-level counts will not, because
    # neurons differ in out-degree.
    emit = np.unique(sg.edge_index[0])
    before = np.zeros(sg.n_nodes, dtype=np.float32); before[sg.edge_index[0]] = sg.edge_sign
    after = np.zeros(sg.n_nodes, dtype=np.float32); after[ss.edge_index[0]] = ss.edge_sign
    assert sorted(before[emit].tolist()) == sorted(after[emit].tolist())
    # and Dale's law still holds
    for node in np.unique(ss.edge_index[0]):
        assert len(np.unique(ss.edge_sign[ss.edge_index[0] == node])) == 1


def test_sign_shuffle_preserves_excitatory_fraction():
    sg = _toy()
    ss = shuffle_signs(sg, seed=4)
    src = sg.edge_index[0]
    before = np.zeros(sg.n_nodes, dtype=np.float32); before[src] = sg.edge_sign
    after = np.zeros(sg.n_nodes, dtype=np.float32); after[src] = ss.edge_sign
    assert np.array_equal(np.sort(before), np.sort(after))


def test_gru_hidden_matches_parameter_budget():
    for target in (10_000, 250_000, 672_569, 5_000_000):
        h = _gru_hidden_for(target, n_in=405, n_out=66)
        got = 3 * (h * 405 + h * h + 2 * h) + h * 66 + 66
        nxt = 3 * ((h + 1) * 405 + (h + 1) ** 2 + 2 * (h + 1)) + (h + 1) * 66 + 66
        assert got <= target, f"GRU exceeds budget at target={target}"
        assert nxt > target, f"GRU is not the largest fit at target={target}"
