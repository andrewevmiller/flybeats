"""The ablations are the evidence, so their invariants get tested.

A rewiring that quietly changed the degree sequence, or a sign shuffle that
also perturbed topology, would make the comparison meaningless while still
producing a plausible-looking number.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ablations import (ARMS, _gru_hidden_for, arm_config,  # noqa: E402
                       degree_matched_rewire, shuffle_signs)
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


# ----------------------------------------------------- per-arm operating point --
def test_shipped_config_carries_the_measured_per_arm_radii():
    """The Phase 4 arms do not share a radius, and these are the measured ones.

    From sections 3 and 4 of
    results/local/2026-09-15-probe-reproducibility.md: each arm's best cell
    with no wing motor neuron pinned at state_clip for more than half the
    settled window. Pinned here so a casual edit to the config has to argue
    with a test rather than quietly move the campaign's operating point.
    """
    from build import load_config

    cfg = load_config(ROOT / "configs" / "v1_8piece.yaml")
    by_arm = cfg["model"]["spectral_radius_by_arm"]
    assert by_arm["real"] == 5.0
    assert by_arm["sign_shuffled"] == 2.0
    assert by_arm["rewired"] == 0.5
    # A key that is not an arm name is silently ignored by build_arm, which is
    # the failure mode this catches: the arm would run at the fallback scalar
    # and nothing would say so.
    assert set(by_arm) <= set(ARMS), f"not arm names: {set(by_arm) - set(ARMS)}"

    # train.py builds a single model through build_model and knows nothing
    # about arms, so it trains the real arm off the scalar. If the scalar and
    # real's entry disagree, `train.py` and `ablations.py --arms real` train
    # two different models and the headline run is not the real arm.
    assert cfg["model"]["spectral_radius"] == by_arm["real"]


def test_arm_config_overrides_per_arm_and_falls_back_otherwise():
    cfg = {"model": {"gain_scale": "auto", "spectral_radius": 3.0,
                     "spectral_radius_by_arm": {"real": 7.0, "rewired": 0.25}},
           "train": {"seed": 0}}

    assert arm_config("real", cfg)["model"]["spectral_radius"] == 7.0
    assert arm_config("rewired", cfg)["model"]["spectral_radius"] == 0.25
    # gru and shortcut throw the recurrent core away, so rho never reaches
    # them; anything unlisted keeps the scalar.
    for arm in ("gru", "shortcut", "sign_shuffled"):
        assert arm_config(arm, cfg)["model"]["spectral_radius"] == 3.0
    # one cfg is shared across every arm of a run, so overriding for one arm
    # must not leak into the next
    assert cfg["model"]["spectral_radius"] == 3.0
    assert cfg["model"]["spectral_radius_by_arm"] == {"real": 7.0, "rewired": 0.25}

    # no mapping at all is the old behaviour, unchanged
    plain = {"model": {"spectral_radius": 3.0}}
    assert arm_config("real", plain) is plain


def test_build_arm_realises_each_arms_own_spectral_radius():
    """The override has to reach the built operator, not just the config dict.

    gain_scale="auto" rescales the core to hit the target, so the assertion is
    on the radius the model will actually run at: rho of the unscaled operator
    times the gain_scale buffer that multiplies it.
    """
    import torch

    from ablations import build_arm
    from build import get_subgraph, load_config
    from conftest import use_small_graph

    cfg = use_small_graph(load_config(ROOT / "configs" / "sanity_3piece.yaml"))
    sg = get_subgraph(cfg)
    # Sentinels, deliberately not the shipped values: a mapping that was being
    # ignored would still give the right answer if the test asked for the
    # number the fallback scalar already holds.
    cfg["model"]["gain_scale"] = "auto"
    cfg["model"]["spectral_radius"] = 3.0
    cfg["model"]["spectral_radius_by_arm"] = {
        "real": 7.0, "rewired": 0.25, "sign_shuffled": 1.5}

    for arm, want in (("real", 7.0), ("rewired", 0.25), ("sign_shuffled", 1.5)):
        torch.manual_seed(0)
        model, _ = build_arm(arm, cfg, sg, n_styles=1, seed=0)
        assert model.rnn.cfg.spectral_radius == want, f"{arm} got the wrong target"
        realised = model.rnn._spectral_radius() * float(model.rnn.gain_scale)
        assert abs(realised - want) < 0.01 * want, f"{arm} realised rho {realised}"

    # And an arm with no entry is built at the fallback, not at whatever the
    # previous arm left behind.
    torch.manual_seed(0)
    model, _ = build_arm("gru", cfg, sg, n_styles=1, seed=0)
    assert model.rnn.cfg.spectral_radius == 3.0


# --- every arm must actually run (TEST_PLAN T1.1) --------------------------

def _arm_fixture():
    """A small real graph and a config that cannot wander off this machine.

    ``device: cpu`` is pinned rather than inherited: ``v1_8piece.yaml`` sets
    ``device: auto``, which resolves to CUDA wherever a card exists, and a test
    that silently changes device between machines is not testing one thing.
    """
    from build import get_subgraph, load_config

    from conftest import use_small_graph

    cfg = use_small_graph(load_config(ROOT / "configs" / "sanity_3piece.yaml"))
    cfg["train"]["device"] = "cpu"
    sg = get_subgraph(cfg)
    wav = torch.randn(2, 11025, generator=torch.Generator().manual_seed(2)) * 0.1
    return cfg, sg, wav


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_builds_forward_passes_and_backprops(arm):
    """The failure with a precedent: an arm that cannot run at all.

    The CI workflow's own header records it -- "two arms that could not do a
    forward pass, because a signature changed in model.py and the replacement
    cores in ablations.py did not. A test existed for neither." CI was the
    response, but CI ran a suite that still never called an arm, so the same
    drift recurred immediately: ``substeps`` was added to ``ConnectomeRNN`` and
    to ``FlyBeats.forward``, and ``GRUCore``/``ShortcutCore`` went on not
    accepting it. Both raised ``TypeError`` on every call through
    ``FlyBeats.forward`` -- playback, transcription, bundle export -- while
    training stayed green, because ``run_epoch`` calls ``model.rnn`` directly.

    Phase D is a GPU day. A broken arm fails after the money is spent.
    """
    from ablations import build_arm

    cfg, sg, wav = _arm_fixture()
    torch.manual_seed(0)
    model, kit = build_arm(arm, cfg, sg, n_styles=3, seed=0)
    model.train()

    logits, state = model(wav)
    assert logits.shape[0] == wav.shape[0]
    assert logits.shape[2] == len(kit.classes), f"{arm} predicts the wrong kit"
    assert logits.shape[1] == wav.shape[-1] // model.encoder.hop
    assert torch.isfinite(logits).all(), f"{arm} produced non-finite logits"

    # Backward too: an arm that forward-passes and then has no gradient path to
    # its own parameters trains for a day and learns nothing.
    logits.square().mean().backward()
    trained = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    assert trained, f"{arm} has no trainable parameters"
    got = [(n, p) for n, p in trained if p.grad is not None]
    assert got, f"{arm} reached no parameter with a gradient"
    for n, p in got:
        assert torch.isfinite(p.grad).all(), f"{arm}.{n} has a non-finite gradient"


@pytest.mark.parametrize("arm", ARMS)
def test_every_arm_honours_the_speed_dial(arm):
    """``substeps`` has to mean the same thing to every core.

    ``StreamingDrummer`` reads timestamps off the returned row count, so a core
    that accepted the argument and ignored it would not fail -- it would place
    every hit at the wrong moment, which is worse.
    """
    from ablations import build_arm

    cfg, sg, wav = _arm_fixture()
    torch.manual_seed(0)
    model, _ = build_arm(arm, cfg, sg, n_styles=3, seed=0)
    model.eval()

    with torch.no_grad():
        base, _ = model(wav)
        fast, _ = model(wav, substeps=2)
        # A per-frame schedule is the fractional dial: 1,2,1,2,... averages 1.5.
        frames = base.shape[1]
        sched = [1 + (i % 2) for i in range(frames)]
        frac, _ = model(wav, substeps=sched)

    assert fast.shape[1] == 2 * base.shape[1], f"{arm} ignored substeps=2"
    assert frac.shape[1] == sum(sched), f"{arm} ignored the per-frame schedule"
    assert torch.isfinite(fast).all() and torch.isfinite(frac).all()

    with pytest.raises(ValueError):
        model(wav, substeps=0)
    with pytest.raises(ValueError):
        model(wav, substeps=[1] * (frames + 3))


# --- seed discipline (TEST_PLAN T1.4) --------------------------------------

def _weights(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def _build(arm, cfg, sg, seed):
    """One arm, from a fixed initialisation RNG. Only ``seed`` varies.

    ``ablations.main`` does exactly this -- ``torch.manual_seed(base_seed)``
    before every rep, with ``seed = base_seed + rep`` going to ``build_arm`` --
    so the repeats vary the topology draw and nothing else.
    """
    from ablations import build_arm

    torch.manual_seed(0)
    np.random.seed(0)
    model, _ = build_arm(arm, cfg, sg, n_styles=3, seed=seed)
    return _weights(model)


@pytest.mark.parametrize("arm", ARMS)
def test_one_seed_rebuilds_the_same_arm(arm):
    cfg, sg, _ = _arm_fixture()
    assert _same(_build(arm, cfg, sg, 0), _build(arm, cfg, sg, 0)), \
        f"{arm} is not reproducible under its own seed"


@pytest.mark.parametrize("arm", ARMS)
def test_only_a_stochastic_arm_moves_with_the_seed(arm):
    """`STOCHASTIC_ARMS` exists; nothing checked that the distinction holds.

    It decides how many repeats each arm gets (`n_reps = seeds if arm in
    STOCHASTIC_ARMS else 1`), so an arm wrongly outside it would be run once
    and quoted without an error bar, and one wrongly inside it would spend
    four extra GPU runs re-measuring optimiser noise.
    """
    from ablations import STOCHASTIC_ARMS

    cfg, sg, _ = _arm_fixture()
    moved = not _same(_build(arm, cfg, sg, 0), _build(arm, cfg, sg, 1))
    assert moved == (arm in STOCHASTIC_ARMS), (
        f"{arm} {'moved with' if moved else 'ignored'} the seed, "
        f"but is {'in' if arm in STOCHASTIC_ARMS else 'not in'} STOCHASTIC_ARMS")


def test_a_stochastic_arms_seed_cannot_reach_the_real_arm():
    """Phase D's claim is one observed topology against N draws from a null.

    Every arm is built from one shared `SubGraph`, so a rewiring that wrote
    through it would change the real arm's graph too -- and the comparison
    would be against a control that had already contaminated its own baseline.
    """
    from ablations import degree_matched_rewire, shuffle_signs

    cfg, sg, _ = _arm_fixture()
    before = _build("real", cfg, sg, 0)
    edges, signs = sg.edge_index.copy(), sg.edge_sign.copy()

    for seed in range(3):
        _build("rewired", cfg, sg, seed)
        _build("sign_shuffled", cfg, sg, seed)
        degree_matched_rewire(sg, seed)
        shuffle_signs(sg, seed)

    assert np.array_equal(sg.edge_index, edges), "the shared subgraph was rewired in place"
    assert np.array_equal(sg.edge_sign, signs), "the shared subgraph's signs were shuffled in place"
    assert _same(before, _build("real", cfg, sg, 0)), \
        "the real arm changed after the null arms were drawn"
