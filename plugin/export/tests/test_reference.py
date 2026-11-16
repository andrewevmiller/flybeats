"""P1 exit criteria on a synthetic network: parity with PyTorch, block-size independence, every perturbation
against its PyTorch equivalent, the streaming decoder against loss.peaks, and the runaway guard."""
import numpy as np
import pytest
import torch

from flybrain.parity import (PERTURBATIONS, TIE_EPS, ambiguous_hits, compare, make_model, run_perturbed, torch_hits, torch_outputs,
                             torch_perturbed)
from flybrain.reference import Controls, PeakDecoder, ReferenceEngine, hit_list, run_clip

SR = 16000


def audio(seconds=2.0, seed=5, level=0.1):
    return (np.random.default_rng(seed).standard_normal(int(seconds * SR)) * level).astype(np.float32)


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_parity_with_pytorch(brain_small, dtype):
    b, model, _, _ = brain_small
    a = audio()
    tor = torch_outputs(model, a)
    ref = run_clip(b, a, dtype=dtype)
    rep = compare(ref, tor)
    assert rep["passes"], rep
    assert rep["genuine_hits"] >= 10, rep              # hits the ambiguity rule cannot excuse carry the contract


def test_block_size_independence(brain_small):
    b = brain_small[0]
    a = audio(1.0)
    base = run_clip(b, a)
    for block in (1, 7, 64, 480, 4096):
        r = run_clip(b, a, block=block)
        for k in ("bands", "rates", "prob", "hits"):
            assert np.array_equal(r[k], base[k]), (block, k)


def test_neutral_controls_equal_baseline(brain_small):
    b = brain_small[0]
    a = audio(1.0)
    base = run_clip(b, a)
    c = Controls(b)
    assert c.is_neutral(b)
    r = run_clip(b, a, controls=c)
    for k in ("rates", "prob", "hits"):
        assert np.array_equal(r[k], base[k]), k


def test_repeat_runs_are_identical(brain_small):
    b = brain_small[0]
    a = audio(1.0)
    r1, r2 = run_clip(b, a), run_clip(b, a)
    assert all(np.array_equal(r1[k], r2[k]) for k in ("rates", "prob", "hits"))


@pytest.mark.parametrize("kind,kw", PERTURBATIONS)
def test_perturbation_matches_pytorch(brain_small, cfg, kind, kw):
    b, model, neurons, edges = brain_small
    a = audio(1.0)
    m, ear_mask, extra = torch_perturbed(model, neurons, edges, cfg, kind, **kw)
    tor = torch_outputs(m, a, ear_mask=ear_mask, extra_drive=extra)
    ref = run_perturbed(b, a, kind, **kw)
    rep = compare(ref, tor)
    assert rep["passes"], (kind, rep)
    base = torch_outputs(model, a)
    assert not np.array_equal(base["rates"], tor["rates"]), "the perturbation changed nothing (rule H3)"


def test_unsilence_matches_pytorch(brain_small, cfg):
    b, model, neurons, edges = brain_small
    silenced = np.flatnonzero(neurons.silenced.to_numpy())[:2].tolist()
    a = audio(1.0)
    m, _, _ = torch_perturbed(model, neurons, edges, cfg, "unsilence", targets=silenced, sign=1)
    tor = torch_outputs(m, a)
    ref = run_perturbed(b, a, "unsilence", targets=silenced, sign=1)
    rep = compare(ref, tor)
    assert rep["passes"], rep
    assert ref["rates"][:, silenced].max() > 0        # they are awake now


def test_injection_matches_pytorch(brain_small, cfg):
    b, model, neurons, edges = brain_small
    a = audio(1.0)
    t_len = len(a) // 80
    target = int(np.flatnonzero(neurons.type.isna().to_numpy())[0])   # an untyped neuron: its own type
    kw = dict(neuron=target, current=2.0, start=50, frames=30, t_len=t_len)
    m, _, extra = torch_perturbed(model, neurons, edges, cfg, "inject", **kw)
    tor = torch_outputs(m, a, extra_drive=extra)
    ref = run_perturbed(b, a, "inject", **kw)
    rep = compare(ref, tor)
    assert rep["passes"], rep
    base = torch_outputs(model, a)
    assert np.array_equal(base["rates"][:50], tor["rates"][:50])
    assert not np.array_equal(base["rates"][50:], tor["rates"][50:])


def test_streaming_decoder_equals_peaks_rule():
    from flybeats.model.loss import peaks
    rng = np.random.default_rng(3)
    prob = rng.random((300, 8)).astype(np.float32)
    prob[10:20] = 0.9                                    # plateaus: ties
    dec = PeakDecoder(8)
    got = []
    for t in range(300):
        d = dec.push(prob[t], prob[t], t, 0.5)
        if d is not None:
            got.append(d)
    got.append(dec.flush(0.5))
    hits = np.zeros((300, 8), bool)
    for f, h, _ in got:
        hits[f] = h
    want = sorted({(int(f), d) for d in range(8) for f in peaks(prob[:, d], 0.5)})
    assert hit_list(hits) == want
    assert got[-1][0] == 299                             # the last frame is decided by flush


def test_decode_is_one_frame_late(brain_small):
    b = brain_small[0]
    eng = ReferenceEngine(b)
    a = audio(0.5)
    decided = eng.push(a[:80 * 10])                      # 10 frames in
    assert [f.index for f in decided] == list(range(9))  # frame 9 waits for frame 10
    decided += eng.flush()
    assert decided[-1].index == 9


def test_preroll_mutes_hits_after_reset(brain_small):
    b = brain_small[0]
    a = audio(1.0)
    base = run_clip(b, a)
    gated = run_clip(b, a, preroll_frames=100)
    assert not gated["hits"][:100].any()
    assert np.array_equal(gated["hits"][100:], base["hits"][100:])
    assert np.array_equal(gated["rates"], base["rates"])  # the network itself is untouched


def test_reset_restores_the_initial_state(brain_small):
    b = brain_small[0]
    a = audio(1.0)
    eng = ReferenceEngine(b)
    first = eng.push(a) + eng.flush()
    eng.reset()
    second = eng.push(a) + eng.flush()
    assert all(np.array_equal(x.rates, y.rates) for x, y in zip(first, second))
    assert second[0].reset_flag and not second[1].reset_flag


def test_runaway_guard_never_gives_nan_and_recovers(brain_small):
    b = brain_small[0]
    b.meta["runaway"] = {"rate_ceiling": 1e3, "mean_rate_limit": 1.0, "frames": 5}
    a = audio(2.0, level=0.3)
    eng = ReferenceEngine(b)
    c = Controls(b)
    c.gain_scale = 50.0                                   # far past any cliff
    eng.set_controls(c)
    frames = eng.push(a) + eng.flush()
    rates = np.stack([f.rates for f in frames])
    assert np.isfinite(rates).all() and rates.max() <= 1e3
    assert eng.guard.triggered >= 1
    assert any(f.reset_flag for f in frames[1:])          # it reset at least once after the start
    c.gain_scale = 1.0
    eng.set_controls(c)
    eng.reset()
    calm = eng.push(a) + eng.flush()
    base = run_clip(b, a)
    assert np.array_equal(np.stack([f.rates for f in calm]), base["rates"])


def test_ambiguous_hits_are_reported_not_hidden():
    prob = np.full((5, 1), 0.5 + TIE_EPS / 2, np.float32)
    amb = ambiguous_hits(prob, 0.5)
    assert amb == {(f, 0) for f in range(5)}
