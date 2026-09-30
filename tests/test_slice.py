"""Phase 1 gate tests. Run after build_slice.py and build_controls.py."""
import hashlib

import numpy as np
import pandas as pd
import pytest

from flybeats.connectome.controls import check_control, control_classes, reciprocal_mask, rewire, rewire_classes
from flybeats.connectome.slice import build_slice
from flybeats.connectome.store import control_path, load_slice, save_table, slice_paths


@pytest.fixture(scope="module")
def real(paths):
    return load_slice(paths["work_dir"])


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rebuild_is_byte_identical(cfg, paths, tmp_path):
    neurons, edges, _ = build_slice(cfg, paths)
    save_table(neurons, tmp_path / "n.parquet")
    save_table(edges, tmp_path / "e.parquet")
    saved_n, saved_e = slice_paths(paths["work_dir"])
    assert sha(tmp_path / "n.parquet") == sha(saved_n)
    assert sha(tmp_path / "e.parquet") == sha(saved_e)


def test_edges_stay_inside_the_slice(real):
    neurons, edges = real
    n = len(neurons)
    assert edges.pre.between(0, n - 1).all() and edges.post.between(0, n - 1).all()
    assert (neurons["index"].to_numpy() == np.arange(n)).all()


def test_mirrored_ears(cfg, real):
    neurons, _ = real
    ears = neurons[neurons.type.fillna("").str.match(cfg["connectome"]["ears"]["types"])]
    mirrored = ears[ears.mirror_of.notna()]
    original = ears[ears.mirror_of.isna()]
    assert (mirrored.bodyId < 0).all() and (mirrored.bodyId == -mirrored.mirror_of).all()
    assert (original.bodyId > 0).all() and (original.side == "L").all()
    assert set(mirrored.mirror_of) <= set(original.bodyId)     # no real JO-A/B other than the left ears
    assert (neurons.mirror_of.notna() == (neurons.bodyId < 0)).all()


def test_every_drum_has_its_own_motor_neurons(cfg, real):
    neurons, _ = real
    drums = cfg["kit"]["pieces"]
    counts = neurons.drum.value_counts()
    assert all(counts.get(d, 0) >= 1 for d in drums)
    assert set(neurons.drum.dropna()) == set(drums)             # one column, so no neuron serves two drums
    assert (neurons.role[neurons.drum.notna()] == "motor").all()


def test_signs_follow_roles(real):
    neurons, _ = real
    expected = neurons.role.map({"excitatory": 1, "inhibitory": -1, "motor": -1, "modulatory": 0, "unknown": 0})
    assert (neurons.sign == expected).all()
    assert (neurons.silenced == (neurons.sign == 0)).all()


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_control_is_a_fair_opponent(cfg, paths, real, seed):
    neurons, edges = real
    ctrl = pd.read_parquet(control_path(paths["work_dir"], seed))
    ears = cfg["connectome"]["ears"]
    classes = (control_classes(neurons, edges, ears)
               if cfg["connectome"]["random_controls"]["method"] == "degree_class_reciprocity_preserving_swaps" else None)
    checks = check_control(neurons, edges, ctrl, ears, classes)   # raises on degree, synapse, class or pair mismatch
    assert checks["surviving_real_connections_share"] < 0.1


def test_rewire_on_a_small_graph():
    rng = np.random.default_rng(0)
    pairs = {(int(a), int(b)) for a, b in rng.integers(0, 40, (300, 2)) if a != b}
    edges = pd.DataFrame(sorted(pairs), columns=["pre", "post"])
    edges["synapses"] = rng.integers(5, 50, len(edges)).astype(float)
    out, accepted = rewire(edges, 40, 10, seed=1)
    assert accepted > 0
    assert np.array_equal(np.bincount(edges.pre, minlength=40), np.bincount(out.pre, minlength=40))
    assert np.array_equal(np.bincount(edges.post, minlength=40), np.bincount(out.post, minlength=40))
    assert np.allclose(np.bincount(edges.pre, edges.synapses, 40), np.bincount(out.pre, out.synapses, 40))
    assert not (out.pre == out.post).any() and not out.duplicated(["pre", "post"]).any()


def test_rewire_classes_on_a_small_graph():
    """Class blocks, reciprocal pairs and degrees survive; connections still move; blocks with no real connections
    (here: class 0 to class 3, as ears to motor neurons) stay empty."""
    rng = np.random.default_rng(0)
    n = 60
    classes = np.repeat([0, 1, 2, 3], 15)
    pairs = set()
    for a, b in rng.integers(0, n, (500, 2)).tolist():
        if a != b and not (classes[a] == 0 and classes[b] == 3) and not (classes[b] == 0 and classes[a] == 3):
            pairs.add((a, b))
            if rng.random() < 0.15:
                pairs.add((b, a))
    edges = pd.DataFrame(sorted(pairs), columns=["pre", "post"])
    edges["synapses"] = rng.integers(5, 50, len(edges)).astype(float)
    out, accepted = rewire_classes(edges, classes, 10, seed=1)
    assert accepted > 0
    k = 4
    for a, b in (("pre", "post"), ("post", "pre")):
        assert np.array_equal(np.bincount(edges[a] * k + classes[edges[b]], minlength=n * k),
                              np.bincount(out[a] * k + classes[out[b]], minlength=n * k))
    assert np.allclose(np.bincount(edges.pre, edges.synapses, n), np.bincount(out.pre, out.synapses, n))
    assert reciprocal_mask(out, n).sum() == reciprocal_mask(edges, n).sum() > 0
    assert not ((classes[out.pre] == 0) & (classes[out.post] == 3)).any()
    assert not (out.pre == out.post).any() and not out.duplicated(["pre", "post"]).any()
    moved = set(zip(out.pre, out.post)) - set(zip(edges.pre, edges.post))
    assert len(moved) > len(edges) / 2
