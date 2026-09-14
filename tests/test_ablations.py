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


# --- the control arms have to be callable the way the model calls them ------
#
# GRUCore and ShortcutCore replace model.rnn wholesale, so FlyBeats.forward
# calls them with ConnectomeRNN's signature. When the speed work added
# `substeps` to that signature, neither control arm got it, and both raised
# TypeError on their first forward pass -- which meant `python src/ablations.py`
# trained real, rewired and sign_shuffled and then died on the gru arm. Nothing
# here tested the replacement cores against the real one's calling convention,
# so nothing noticed.

import inspect  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

from ablations import GRUCore, ShortcutCore  # noqa: E402
from model import ConnectomeRNN, ModelConfig  # noqa: E402


def _core(kind, n_in=4, n_out=3, hidden=16):
    cfg = ModelConfig()
    return (GRUCore(n_in, n_out, hidden, cfg) if kind == "gru"
            else ShortcutCore(n_in, n_out, cfg))


@pytest.mark.parametrize("kind", ["gru", "shortcut"])
def test_a_control_core_accepts_every_argument_the_real_core_does(kind):
    """Derived from ConnectomeRNN rather than spelled out, so the next
    parameter added to the real core fails here instead of mid-run."""
    real = set(inspect.signature(ConnectomeRNN.forward).parameters)
    mine = set(inspect.signature(_core(kind).forward).parameters)
    missing = real - mine - {"self"}
    assert not missing, f"{kind} core cannot be called with: {sorted(missing)}"


@pytest.mark.parametrize("kind", ["gru", "shortcut"])
def test_a_control_core_returns_one_step_per_substep(kind):
    """The real core returns sum(schedule) steps, not one per frame. A control
    arm that ignored substeps would return the wrong length and score against
    a misaligned target."""
    core = _core(kind)
    drive = torch.randn(2, 5, 4)

    plain, _ = core(drive)
    assert plain.shape[:2] == (2, 5)

    trebled, _ = core(drive, substeps=3)
    assert trebled.shape[:2] == (2, 15)

    scheduled, _ = core(drive, substeps=[1, 2, 1, 3, 1])
    assert scheduled.shape[:2] == (2, 8)


@pytest.mark.parametrize("kind", ["gru", "shortcut"])
def test_a_control_core_refuses_a_nonsense_schedule(kind):
    core = _core(kind)
    drive = torch.randn(1, 4, 4)
    with pytest.raises(ValueError):
        core(drive, substeps=0)
    with pytest.raises(ValueError):
        core(drive, substeps=[1, 2])          # four frames, two entries


# --- the lesion sweep ------------------------------------------------------
#
# lesion_sweep silences one population at a time and reports each one's delta
# against the intact model. The failure that would not look like a failure is
# lesions accumulating: every row after the first would carry the previous
# ones, deltas would grow monotonically down the table, and it would read as a
# dose-response curve. Nothing else in the file would object.

from ablations import lesion, lesion_sweep  # noqa: E402


class _Core:
    """Just enough core to be gated."""

    def __init__(self, n):
        self.gate = np.ones(n)

    def set_gate(self, idx, value):
        self.gate[np.asarray(idx)] = value

    def reset_gates(self):
        self.gate[:] = 1.0


class _Model:
    def __init__(self, n):
        self.rnn = _Core(n)


def _score_by_how_much_is_silenced(model, loader, cfg, device):
    """onset F falls by 0.01 for each neuron currently muted, so the score says
    exactly which neurons were off when it was measured."""
    silenced = int((model.rnn.gate == 0).sum())
    return {"onset_f": 0.5 - 0.01 * silenced, "groove_sim": 0.0,
            "beat_align_ms": 0.0, "mean_dev_ms": 0.0}


ROLES = {"two": np.array([0, 1]), "one": np.array([5]),
         "none": np.array([], dtype=np.int64)}


def _sweep():
    return lesion_sweep(_Model(10), None, None, None, ROLES,
                        ["two", "one", "none"], _score_by_how_much_is_silenced)


def test_each_lesion_is_measured_against_the_intact_model_alone():
    rows = {r["population"]: r for r in _sweep()}
    assert rows["intact"]["onset_f"] == pytest.approx(0.50)
    # 'one' must score with one neuron muted, not with three
    assert rows["one"]["onset_f"] == pytest.approx(0.49), "lesions accumulated"
    assert rows["two"]["onset_f"] == pytest.approx(0.48)


def test_deltas_are_against_intact():
    rows = {r["population"]: r for r in _sweep()}
    assert rows["two"]["delta_onset_f"] == pytest.approx(-0.02)
    assert rows["one"]["delta_onset_f"] == pytest.approx(-0.01)
    assert "delta_onset_f" not in rows["intact"]


def test_a_population_the_subgraph_does_not_have_is_skipped_not_reported_as_null():
    """Reporting it with a zero delta would say the population was silenced and
    nothing happened, which is the opposite of what happened."""
    assert "none" not in {r["population"] for r in _sweep()}


def test_the_model_is_left_intact_afterwards():
    model = _Model(10)
    lesion_sweep(model, None, None, None, ROLES, ["two", "one"],
                 _score_by_how_much_is_silenced)
    assert (model.rnn.gate == 1.0).all(), "the sweep left neurons muted"


def test_lesioning_a_population_that_is_not_there_says_which_ones_are():
    with pytest.raises(KeyError) as e:
        lesion(_Model(10), ROLES, "pC9")
    assert "two" in str(e.value) and "none" not in str(e.value)
