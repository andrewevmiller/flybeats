"""Invariants on the extracted subgraph.

These run against the real cached subgraph when one exists, because the
properties that matter (Dale's law holding across a real transmitter table, the
required populations surviving the trim) are exactly the ones a synthetic
fixture would not exercise.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from connectome import NT_SIGN  # noqa: E402
from subgraph import SubGraph, hops_from  # noqa: E402

from conftest import real_graph  # noqa: E402

@pytest.fixture(scope="module")
def sg():
    """The real cached subgraph.

    A locally built cache if there is one -- that is the graph being trained on
    -- and otherwise the committed 10k extraction, which is a real one too.
    Either way these invariants are checked against real anatomy; a synthetic
    fixture would not exercise Dale's law across a real transmitter table.
    """
    cache = real_graph()
    if cache is None:
        pytest.skip("no cached subgraph and no fixture in this checkout")
    return SubGraph.load(cache)


def test_edge_signs_obey_dales_law(sg):
    """Every neuron is uniformly excitatory or inhibitory across its outputs."""
    src = sg.edge_index[0]
    order = np.argsort(src)
    s, sign = src[order], sg.edge_sign[order]
    bounds = np.flatnonzero(np.diff(s)) + 1
    for a, b in zip(np.r_[0, bounds], np.r_[bounds, len(s)]):
        assert len(np.unique(sign[a:b])) == 1, f"neuron {s[a]} emits mixed signs"


def test_edge_signs_match_the_presynaptic_transmitter(sg):
    """The sign must be derived from the transmitter, not assigned freely."""
    src = sg.edge_index[0]
    sample = np.random.default_rng(0).choice(len(src), size=min(5000, len(src)), replace=False)
    for e in sample:
        nt = str(sg.nt[src[e]])
        expected = NT_SIGN.get(nt, 0)
        if expected == 0:            # modulatory or unknown -> positive in the fast graph
            expected = 1
        assert sg.edge_sign[e] == expected, f"{nt} mapped to sign {sg.edge_sign[e]}"


def test_signs_are_only_plus_or_minus_one(sg):
    assert set(np.unique(sg.edge_sign).tolist()) <= {-1.0, 1.0}


def test_indices_are_in_range(sg):
    assert sg.edge_index.min() >= 0
    assert sg.edge_index.max() < sg.n_nodes
    for name, arr in (("body_ids", sg.body_ids), ("types", sg.types),
                      ("side", sg.side), ("nt", sg.nt)):
        assert len(arr) == sg.n_nodes, f"{name} is not one entry per node"


def test_required_populations_survive_the_trim(sg):
    """The sliders and the genre embedding attach to these; losing one to a node
    budget would silently disable a shipped feature."""
    for pop in ("sensory", "motor", "pC1", "pIP10", "octopaminergic"):
        idx = sg.role(pop)
        assert len(idx) > 0, f"population {pop!r} was trimmed away"
        assert idx.max() < sg.n_nodes


def test_motor_pool_is_bilateral(sg):
    """The bilateral readout needs motor neurons on both sides."""
    sides = {str(s) for s in sg.side[sg.role("motor")]}
    assert "L" in sides and "R" in sides, sides


def test_weights_are_positive_synapse_counts(sg):
    assert sg.weight.min() >= 1
    assert np.isfinite(sg.weight).all()


def test_no_duplicate_body_ids(sg):
    assert len(np.unique(sg.body_ids)) == sg.n_nodes


def test_roundtrip_preserves_everything(sg, tmp_path):
    p = tmp_path / "rt.npz"
    sg.save(p)
    back = SubGraph.load(p)
    assert np.array_equal(back.edge_index, sg.edge_index)
    assert np.array_equal(back.edge_sign, sg.edge_sign)
    assert np.array_equal(back.weight, sg.weight)
    assert np.array_equal(back.body_ids, sg.body_ids)
    assert sorted(back.roles) == sorted(sg.roles)
    for k in sg.roles:
        assert np.array_equal(back.roles[k], sg.roles[k])


def test_hops_from_counts_real_path_lengths():
    """Depth from the JO afferents is what tells "the recurrence attenuates
    gradually" apart from "the first synapse loses everything"."""
    from types import SimpleNamespace

    # 0 -> 1 -> 2 -> 3, plus 4 hanging off nothing
    edges = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    g = SimpleNamespace(n_nodes=5, edge_index=edges)
    d = hops_from(g, np.array([0]))
    assert list(d) == [0, 1, 2, 3, -1]

    # direction matters: the sweep follows pre -> post only
    assert list(hops_from(g, np.array([3]))) == [-1, -1, -1, 0, -1]

    # max_hops truncates rather than mislabelling
    assert list(hops_from(g, np.array([0]), max_hops=2)) == [0, 1, 2, -1, -1]
