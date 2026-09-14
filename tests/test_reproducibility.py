"""Two runs of one config must see the same data.

Both guarantees here were absent, and their absence was invisible. The corpus
loader cropped every clip at a random offset drawn from numpy's *global* RNG --
on the validation split too, so onset F was scored on a different slice of the
audio every time it was read, and a checkpoint re-evaluated twice disagreed
with itself. The training loader shuffled from the global RNG as well, so
ablation arms sharing one loader saw different batches in a different order,
and whatever gap appeared between them was part topology and part luck.

Neither shows up in a loss curve, which is why they are pinned here.
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dataset import GrooveDataset, build_dataset  # noqa: E402
from decoder import DrumKit  # noqa: E402


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    """A two-clip corpus in GMD's layout, long enough for windows to differ.

    The audio is a ramp, so the window's own values say where it was cut from:
    a test can tell "same offset" from "same-looking offset".
    """
    sf = pytest.importorskip("soundfile")
    pretty_midi = pytest.importorskip("pretty_midi")

    root = tmp_path_factory.mktemp("groove")
    rows = []
    for i, split in enumerate(("train", "validation")):
        name = f"clip{i}"
        audio = np.linspace(0.0, 1.0, 4 * 22_050, dtype=np.float32)
        sf.write(root / f"{name}.wav", audio, 22_050)

        pm = pretty_midi.PrettyMIDI()
        drums = pretty_midi.Instrument(program=0, is_drum=True)
        for k in range(16):
            drums.notes.append(pretty_midi.Note(
                velocity=60 + k, pitch=36, start=0.25 * k, end=0.25 * k + 0.05))
        pm.instruments.append(drums)
        pm.write(str(root / f"{name}.mid"))

        rows.append({"audio_filename": f"{name}.wav", "midi_filename": f"{name}.mid",
                     "style": "funk/groove1", "split": split, "bpm": "120"})

    with (root / "info.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return root


def _ds(corpus, split, **kw):
    return GrooveDataset(corpus, DrumKit.from_tier("3piece").classes, split=split,
                         seconds=1.0, **kw)


def test_validation_windows_are_fixed(corpus):
    """Read the same clip twice, get the same audio -- and the same targets.

    Without this, val onset F moves between two readings of one checkpoint, and
    a difference between two ablation arms includes the difference between two
    random crops of the corpus.
    """
    va = _ds(corpus, "validation", random_windows=False)
    first_audio, first_onsets = va[0][0], va[0][1]
    for _ in range(3):
        torch.manual_seed(np.random.randint(1 << 30))   # hostile: move the RNG
        audio, onsets = va[0][0], va[0][1]
        assert torch.equal(audio, first_audio), "the validation window moved"
        assert torch.equal(onsets, first_onsets), "the validation target moved"

    # ... and a second instance of the same dataset agrees, so the window does
    # not depend on how many times the process happened to draw before it.
    again = _ds(corpus, "validation", random_windows=False)
    assert torch.equal(again[0][0], first_audio)


def test_training_windows_vary_for_augmentation_but_replay_under_one_seed(corpus):
    """Training keeps its random crop; what it loses is irreproducibility."""
    tr = _ds(corpus, "train", random_windows=True)

    torch.manual_seed(0)
    run_a = [tr[0][0].clone() for _ in range(6)]
    torch.manual_seed(0)
    run_b = [tr[0][0].clone() for _ in range(6)]
    assert all(torch.equal(a, b) for a, b in zip(run_a, run_b)), \
        "the same seed gave a different training window"

    assert any(not torch.equal(run_a[0], w) for w in run_a[1:]), \
        "the training window stopped moving; augmentation is gone"

    torch.manual_seed(1)
    run_c = [tr[0][0].clone() for _ in range(6)]
    assert any(not torch.equal(a, c) for a, c in zip(run_a, run_c)), \
        "a different seed gave the same windows"


def test_only_the_training_split_gets_random_windows(corpus):
    """The split decides, not the caller remembering to pass a flag."""
    cfg = {"root": str(corpus), "seconds": 1.0, "synthetic": False}
    assert build_dataset(cfg, DrumKit.from_tier("3piece").classes, "train").random_windows
    assert not build_dataset(cfg, DrumKit.from_tier("3piece").classes,
                             "validation").random_windows


def test_reseeding_a_loader_replays_the_same_batch_order():
    """What every ablation arm relies on: arm two must see arm one's batches.

    The loader is built once and reused, and its generator keeps advancing, so
    without an explicit reset the second arm starts mid-stream.
    """
    import train as T

    kit = DrumKit.from_tier("3piece")
    cfg = {
        "audio": {"step_ms": 5.0, "sample_rate": 22_050},
        "data": {"synthetic": True, "n_clips": 12, "seconds": 0.5},
        "train": {"batch_size": 2, "workers": 0, "seed": 0},
    }
    loader, _val, _n = T.build_loaders(cfg, kit)

    def order():
        return [int(b[3].sum()) + int(b[4].sum() * 1000) for b in loader]

    T.reseed_loader(loader, 0)
    first = order()
    second = order()                      # no reset: the stream has moved on
    T.reseed_loader(loader, 0)
    third = order()

    assert first == third, "reseeding did not replay the batch order"
    assert len(first) > 1 and first != second, \
        "the loader is not shuffling at all, so this guarantee is vacuous"
