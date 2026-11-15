"""The memorisation check's fixed clips: flybeats.data.sampler.fixed_clips must give the clips the check always drew
inline (spec 1: the 8 clips are the same by construction). Uses the real BabySlakh set in work_dir."""
import numpy as np
import pytest

from flybeats.data.manifest import load_manifest
from flybeats.data.sampler import ClipSampler, fixed_clips
from flybeats.data.slakh import unpacked_root


@pytest.fixture(scope="module")
def train_sampler(cfg, paths):
    try:
        manifest, events, beats = load_manifest(paths["work_dir"], "baby")
        root = unpacked_root(paths["work_dir"], "baby")
    except FileNotFoundError:
        pytest.skip("the baby set is not on this machine")
    return ClipSampler(root, manifest, events, beats, cfg, "train")


def test_fixed_clips_match_the_inline_draw(cfg, train_sampler):
    mc = cfg["memorisation"]
    # the lines memorise.py used before the helper existed, unchanged
    rng = np.random.default_rng(mc["clip_seed"])
    songs = sorted(train_sampler.songs.index)[:mc["songs"]]
    old = [train_sampler.clip(song, float(rng.uniform(0, train_sampler.songs.duration[song] - train_sampler.clip_s)))
           for song in songs * mc["clips_per_song"]]

    new_songs, new = fixed_clips(train_sampler, mc["songs"], mc["clips_per_song"], mc["clip_seed"])

    assert new_songs == songs
    assert len(new) == len(old) == len(songs) * mc["clips_per_song"]
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a["audio"], b["audio"])
        np.testing.assert_array_equal(a["hits"], b["hits"])
        np.testing.assert_array_equal(a["mask"], b["mask"])
