"""v0.15 confirmation sets (plan section 10, T4): the seal rule, the relabel, the sampler's exclusion and the guard.
The synthetic tests need no data. The last test checks the sealed files on this machine when they exist."""
import json
from pathlib import Path

import pandas as pd
import pytest

from flybeats.config import ROOT
from flybeats.data import confirm
from flybeats.data.sampler import ClipSampler


def make_manifest(n_clusters=60, per_cluster=2, split="train"):
    rows = []
    for c in range(n_clusters):
        for k in range(per_cluster if c % 3 else 1):     # some clusters of size 1, some of size 2
            rows.append({"song": f"S{c:03d}_{k}", "split": split, "drum_hash": f"h{c:03d}",
                         "duration": 60.0, "sample_rate": 16000, "input_stems": ["bass"], "path": "x"})
    return pd.DataFrame(rows)


def test_select_is_deterministic_and_disjoint():
    m = make_manifest()
    a1, b1 = confirm.select(m, [], min_songs=10)
    a2, b2 = confirm.select(m, [], min_songs=10)
    assert (a1, b1) == (a2, b2)
    assert a1 and b1 and not set(a1) & set(b1)


def test_sets_are_cluster_closed():
    m = make_manifest()
    a, b = confirm.select(m, [], min_songs=10)
    for songs in (a, b):
        hashes = set(m.set_index("song").loc[songs].drum_hash)
        members = set(m.song[m.drum_hash.isin(hashes)])
        assert members == set(songs), "a drum cluster was split between sets or across the set boundary"


def test_ineligible_song_removes_its_whole_cluster():
    m = make_manifest()
    poisoned = m.song[m.drum_hash == "h001"].iloc[0]          # one member of cluster h001 is excluded
    a, b = confirm.select(m, [poisoned], min_songs=10)
    assert poisoned not in a + b
    assert not set(m.song[m.drum_hash == "h001"]) & set(a + b)


def test_sets_fill_a_then_b_to_at_least_the_minimum():
    m = make_manifest()
    a, b = confirm.select(m, [], min_songs=20)
    assert len(a) >= 20 and len(b) >= 20
    assert len(a) < 20 + 3                                      # stops at the first cluster that reaches the minimum


def test_only_train_songs_are_sealed():
    m = make_manifest()
    m.loc[m.index[:10], "split"] = "validation"
    a, b = confirm.select(m, [], min_songs=5)
    assert not set(m.song[m.split != "train"]) & set(a + b)


def test_draw_clips_is_fixed_and_inside_the_song():
    durations = {"S1": 40.0, "S2": 16.5}
    d1 = confirm.draw_clips(["S2", "S1"], durations, 16, seed=7, per_song=3)
    d2 = confirm.draw_clips(["S1", "S2"], durations, 16, seed=7, per_song=3)
    assert d1 == d2 and len(d1) == 6
    for c in d1:
        assert 0 <= c["start"] <= durations[c["song"]] - 16


def test_apply_confirmation_relabels_and_is_idempotent():
    m = make_manifest()
    a, b = confirm.select(m, [], min_songs=10)
    once = confirm.apply_confirmation(m, a, b)
    twice = confirm.apply_confirmation(once, a, b)
    assert (once.split == "confirm_a").sum() == len(a)
    assert (once.split == "confirm_b").sum() == len(b)
    assert once.equals(twice)
    assert (m.split == "train").all(), "the input manifest is not changed"


def test_apply_confirmation_refuses_a_song_in_validation():
    m = make_manifest()
    m.loc[m.index[0], "split"] = "validation"
    with pytest.raises(ValueError):
        confirm.apply_confirmation(m, [m.song[0]], [])


def test_sampler_never_sees_a_confirmation_song():
    m = make_manifest()
    a, b = confirm.select(m, [], min_songs=10)
    sealed = confirm.apply_confirmation(m, a, b)
    cfg = {"training_data": {"clip_seconds": 16, "frame_ms": 5, "warmup_ignored_seconds": 2,
                             "stem_dropout": {"p": 0.2, "keep_at_least": 1}, "sampling_seed": 0},
           "kit": {"pieces": ["kick"]}}
    s = ClipSampler(Path("."), sealed, pd.DataFrame(columns=["song", "time", "piece", "velocity"]),
                    pd.DataFrame(columns=["song", "time", "downbeat"]), cfg, "train", seed=0)
    assert not set(s.songs.index) & set(a + b)


def test_data_gate_flags_a_leak_into_a_confirmation_set():
    m = make_manifest()
    a, b = confirm.select(m, [], min_songs=10)
    sealed = confirm.apply_confirmation(m, a, b)
    # a training song that shares a drum part with a sealed song
    leaked = sealed.iloc[[0]].copy()
    leaked["song"] = "LEAK"
    leaked["split"] = "train"
    leaked["drum_hash"] = sealed.drum_hash[sealed.split == "confirm_a"].iloc[0]
    checks = confirm.confirmation_gate_checks(pd.concat([sealed, leaked], ignore_index=True))
    assert any(not ok for ok, name, _ in checks if name.startswith("no drum part shared between training"))


def test_guard_opens_each_set_once_and_b_only_after_a(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    with pytest.raises(PermissionError):
        confirm.open_confirmation("confirm_b", "r0", "c", "t1", ledger)     # B before A
    confirm.open_confirmation("confirm_a", "r1", "c", "t1", ledger)
    with pytest.raises(PermissionError):
        confirm.open_confirmation("confirm_a", "r2", "c", "t1", ledger)     # A twice
    with pytest.raises(PermissionError):
        confirm.open_confirmation("confirm_b", "r3", "c", "t1", ledger)     # B under the tag A used
    with pytest.raises(PermissionError):
        confirm.open_confirmation("confirm_b", "r3", "c", "t2", ledger, known_tags=["t1"])   # tag missing in git
    entry = confirm.open_confirmation("confirm_b", "r4", "c", "t2", ledger, known_tags=["t1", "t2"])
    assert entry["set"] == "confirm_b"
    assert [e["set"] for e in confirm.read_ledger(ledger)] == ["confirm_a", "confirm_b"]


def test_sealed_files_on_this_machine_are_disjoint_and_clear_of_baby_and_check():
    a_path, b_path = ROOT / "config" / "confirmation_A.txt", ROOT / "config" / "confirmation_B.txt"
    if not (a_path.exists() and b_path.exists()):
        pytest.skip("the seal has not been run")
    a, b = confirm.read_list(a_path), confirm.read_list(b_path)
    assert len(a) >= 150 and len(b) >= 150
    assert not set(a) & set(b)
    check = json.loads((ROOT.parent / "data" / "work" / "data" / "full" / "train_check_clips.json").read_text())
    assert not {c["song"] for c in check} & set(a + b)
