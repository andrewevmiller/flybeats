"""The style dial's two changes: where the genre tonic lands, and gapped audio.

``genre.targets`` picks the populations the tonic is injected into. Its default
has to build exactly the octopaminergic index every earlier config and
checkpoint used, or old runs would load into a different model. And
``train.audio_gaps`` silences spans of the training audio: it must be off by
default, reach roughly its fraction, and repeat exactly for a given seed and
epoch so a resumed run sees the same gaps.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as T  # noqa: E402
from build import (build_model, genre_target_index, get_subgraph,  # noqa: E402
                   load_config, role_index)
from bundle import build_bundle, load_bundle  # noqa: E402

from conftest import use_small_graph  # noqa: E402


def _cfg(**genre):
    cfg = use_small_graph(load_config(ROOT / "configs" / "sanity_3piece.yaml"))
    cfg["genre"].update(genre)
    return cfg


# --- genre.targets ---------------------------------------------------------

def test_default_targets_are_the_octopaminergic_pool_exactly():
    cfg = _cfg()
    assert "targets" not in cfg["genre"]
    sg = get_subgraph(cfg)
    got = genre_target_index(cfg, role_index(sg))
    assert np.array_equal(got, sg.role("octopaminergic").astype(np.int64))
    torch.manual_seed(0)
    model, _ = build_model(cfg, sg, n_styles=3)
    assert np.array_equal(model.genre.target_idx.numpy(), got)


def test_an_explicit_octopaminergic_target_builds_the_same_model():
    """Same seed, default vs. explicit: identical parameters and output."""
    sg = get_subgraph(_cfg())
    torch.manual_seed(0)
    a, _ = build_model(_cfg(), sg, n_styles=3)
    torch.manual_seed(0)
    b, _ = build_model(_cfg(targets=["octopaminergic"]), sg, n_styles=3)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k


def test_targets_are_the_union_of_the_named_roles():
    cfg = _cfg(targets=["pC1", "octopaminergic", "pC1"])
    roles = role_index(get_subgraph(cfg))
    got = genre_target_index(cfg, roles)
    want = set(roles["pC1"].tolist()) | set(roles["octopaminergic"].tolist())
    assert set(got.tolist()) == want
    assert len(got) == len(want), "a node listed twice would get double the drive"


def test_tonic_lands_only_on_the_target_indices():
    cfg = _cfg(targets=["pC1"], max_current=5.0)
    sg = get_subgraph(cfg)
    torch.manual_seed(0)
    model, _ = build_model(cfg, sg, n_styles=3)
    pc1 = sg.role("pC1").astype(np.int64)
    assert len(pc1) and model.genre.project.out_features == len(pc1)
    tonic = model.genre(torch.arange(3))
    off = np.setdiff1d(np.arange(sg.n_nodes), pc1)
    assert torch.count_nonzero(tonic[:, off]) == 0
    assert torch.count_nonzero(tonic[:, pc1]) > 0
    assert float(tonic.abs().max()) <= 5.0


def test_an_unknown_target_role_is_refused():
    cfg = _cfg(targets=["no_such_population"])
    sg = get_subgraph(cfg)
    try:
        build_model(cfg, sg, n_styles=3)
    except SystemExit as e:
        assert "no_such_population" in str(e)
    else:
        raise AssertionError("an unknown role must not build silently")


def test_a_pc1_checkpoint_and_bundle_round_trip(tmp_path):
    cfg = _cfg(targets=["pC1"], max_current=5.0)
    sg = get_subgraph(cfg)
    torch.manual_seed(0)
    model, _ = build_model(cfg, sg, n_styles=3)
    with torch.no_grad():
        model.genre.project.weight.add_(torch.randn_like(model.genre.project.weight))
    model.eval()
    ck = {"model": model.state_dict(), "config": copy.deepcopy(cfg),
          "kit": ["kick", "snare", "hat_closed"], "n_styles": 3, "best_threshold": 0.5}
    ckpt = tmp_path / "best.pt"
    torch.save(ck, ckpt)

    # checkpoint: rebuild from its own config, strict load
    loaded = torch.load(ckpt, map_location="cpu", weights_only=False)
    again, _ = build_model(loaded["config"], sg, n_styles=3)
    again.load_state_dict(loaded["model"])
    again.eval()

    # bundle
    path = tmp_path / "m.fb"
    torch.save(build_bundle(ck, sg, model), path)
    bundled, _, _, _ = load_bundle(path)

    wav = torch.randn(1, 22050, generator=torch.Generator().manual_seed(1)) * 0.1
    sid = torch.tensor([2])
    with torch.no_grad():
        a, _ = model(wav, style_id=sid)
        b, _ = again(wav, style_id=sid)
        c, _ = bundled(wav, style_id=sid)
    assert torch.equal(a, b)
    assert torch.equal(a, c)


# --- train.audio_gaps ------------------------------------------------------

GAPS = {"fraction": 0.5, "min_ms": 250, "max_ms": 1000}


def test_gap_mask_reaches_about_its_fraction():
    sr, n = 22050, 22050 * 4
    for seed in range(5):
        m = T.audio_gap_mask(n, sr, GAPS, torch.Generator().manual_seed(seed))
        f = float(m.float().mean())
        assert 0.5 <= f < 0.5 + 1000 / 4000 + 1e-6, f


def test_gap_mask_spans_respect_min_length():
    sr, n = 22050, 22050 * 4
    m = T.audio_gap_mask(n, sr, GAPS, torch.Generator().manual_seed(3)).int()
    edges = torch.diff(torch.cat([torch.zeros(1, dtype=torch.int32), m,
                                  torch.zeros(1, dtype=torch.int32)]))
    starts, ends = torch.nonzero(edges == 1).flatten(), torch.nonzero(edges == -1).flatten()
    assert len(starts) and bool(((ends - starts) >= int(0.25 * sr)).all())


def test_gaps_are_deterministic_by_seed_and_epoch():
    cfg = {"train": {"seed": 0, "audio_gaps": GAPS}, "audio": {"sample_rate": 22050}}
    wav = torch.ones(3, 44100)
    a = T.apply_audio_gaps(wav, cfg, T.audio_gap_generator(cfg, 4))
    b = T.apply_audio_gaps(wav, cfg, T.audio_gap_generator(cfg, 4))
    c = T.apply_audio_gaps(wav, cfg, T.audio_gap_generator(cfg, 5))
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert not torch.equal(a[0], a[1]), "each clip gets its own gaps"
    assert set(torch.unique(a).tolist()) <= {0.0, 1.0}


def test_gaps_are_off_by_default_and_never_touch_validation(monkeypatch):
    """No audio_gaps key: run_epoch never calls it. With the key, only train
    epochs call it."""
    calls = []
    monkeypatch.setattr(T, "apply_audio_gaps",
                        lambda wav, cfg, gen: calls.append(1) or wav)
    from decoder import DrumKit

    cfg = _cfg()
    cfg["data"].update(synthetic=True, n_clips=4, seconds=0.5)
    cfg["train"].update(batch_size=2, tbptt_steps=25, workers=0)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    batch, _, n_styles = T.build_loaders(cfg, kit)
    sg = get_subgraph(cfg)
    torch.manual_seed(0)
    model, _ = build_model(cfg, sg, n_styles=max(n_styles, 1))
    opt = torch.optim.SGD(model.parameters(), lr=0.0)

    T.run_epoch(model, batch, opt, cfg, torch.device("cpu"), train=True, epoch=0)
    assert calls == []

    cfg["train"]["audio_gaps"] = GAPS
    T.run_epoch(model, batch, None, cfg, torch.device("cpu"), train=False, epoch=0)
    assert calls == []
    T.run_epoch(model, batch, opt, cfg, torch.device("cpu"), train=True, epoch=0)
    assert len(calls) >= 1
