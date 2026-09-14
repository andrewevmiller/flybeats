"""Training that survives the box going away underneath it.

This environment restarted three times in one session, twice 37 minutes apart,
against runs that need about 70. Without resume the overnight batch was not
slow, it was impossible: every run would be started and lost.

The two things worth testing are the two that fail quietly. A checkpoint that
resumes into the wrong experiment produces a plausible number from a mixture of
two runs, and a checkpoint truncated by the very restart it exists to survive
takes the run down on the way back up.
"""
import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as T  # noqa: E402


# --- the pieces ------------------------------------------------------------

def test_a_checkpoint_is_tied_to_the_config_that_made_it():
    base = {"train": {"epochs": 12, "seed": 0}, "kit": {"tier": "8piece"}}
    same = {"kit": {"tier": "8piece"}, "train": {"seed": 0, "epochs": 12}}
    assert T._fingerprint(base) == T._fingerprint(same), "key order changed it"

    for change in ({"train": {"epochs": 12, "seed": 1}, "kit": {"tier": "8piece"}},
                   {"train": {"epochs": 6, "seed": 0}, "kit": {"tier": "8piece"}}):
        assert T._fingerprint(base) != T._fingerprint(change)


def test_an_interrupted_write_leaves_the_previous_checkpoint_intact(tmp_path, monkeypatch):
    """The point of the rename. A restart during torch.save must not be able to
    destroy the checkpoint that was already there."""
    path = tmp_path / "last.pt"
    T._atomic_save({"epoch": 3}, path)

    def die(*a, **k):
        raise OSError("no space left on device")

    monkeypatch.setattr(T.torch, "save", die)
    with pytest.raises(OSError):
        T._atomic_save({"epoch": 4}, path)

    assert torch.load(path, weights_only=False)["epoch"] == 3, "the old one was lost"
    assert not (tmp_path / "last.pt.tmp").exists(), "left a partial file behind"


def test_history_is_written_the_same_way(tmp_path):
    path = tmp_path / "history.json"
    T._atomic_write_text('[{"epoch": 0}]', path)
    assert json.loads(path.read_text()) == [{"epoch": 0}]
    assert not (tmp_path / "history.json.tmp").exists()


# --- end to end ------------------------------------------------------------

CACHE = ROOT / "data" / "cache" / "subgraph_test2k.npz"


@pytest.fixture
def tiny_config(tmp_path):
    """A standalone config small enough to train in seconds."""
    if not CACHE.exists():
        pytest.skip("no cached subgraph in this checkout")
    from build import load_config

    cfg = load_config(ROOT / "configs" / "sanity_3piece.yaml")
    cfg.pop("_base_", None)
    cfg["name"] = "resume_test"
    cfg["subgraph"].update(cache=str(CACHE), max_nodes=2000)
    # As small as still trains: these tests are about control flow, not
    # learning, and every second here is a second off the batch on the box.
    cfg["data"].update(synthetic=True, n_clips=4, seconds=0.5)
    cfg["train"].update(epochs=3, batch_size=2, tbptt_steps=25)
    path = tmp_path / "tiny.yaml"
    yaml.safe_dump(cfg, path.open("w"))
    return path


def test_a_run_killed_mid_flight_comes_back_where_it_stopped(
        tiny_config, tmp_path, monkeypatch, capsys):
    out = tmp_path / "run"
    args = ["--config", str(tiny_config), "--out", str(out), "--epochs", "3"]

    # Die during epoch 2, the way a container restart does: two epochs are
    # checkpointed, the third never completes.
    real, calls = T.run_epoch, {"n": 0}

    def die_on_the_third(*a, **k):
        calls["n"] += 1
        if calls["n"] > 2:
            raise RuntimeError("container went away")
        return real(*a, **k)

    monkeypatch.setattr(T, "run_epoch", die_on_the_third)
    with pytest.raises(RuntimeError):
        T.main(args)

    ck = torch.load(out / "last.pt", weights_only=False)
    assert ck["epoch"] == 1, "the last completed epoch was not checkpointed"
    assert len(ck["history"]) == 2

    monkeypatch.setattr(T, "run_epoch", real)
    assert T.main(args) == 0

    printed = capsys.readouterr().out
    assert "RESUMED" in printed and "at epoch 2/3" in printed
    assert len(json.loads((out / "history.json").read_text())) == 3, \
        "the resumed run did not continue the history, it restarted it"
    assert not (out / "last.pt").exists(), "insurance kept after a clean finish"


def test_fresh_ignores_a_checkpoint_that_is_there(tiny_config, tmp_path, capsys):
    out = tmp_path / "run"
    base = ["--config", str(tiny_config), "--out", str(out)]
    assert T.main(base + ["--epochs", "1"]) == 0
    # a clean finish removes last.pt, so put one back the way a kill would
    T._atomic_save({"epoch": 0, "best": 0.0, "history": [{}],
                    "fingerprint": "deadbeef"}, out / "last.pt")

    capsys.readouterr()
    assert T.main(base + ["--epochs", "1", "--fresh"]) == 0
    assert "RESUMED" not in capsys.readouterr().out


def test_a_checkpoint_from_another_run_is_refused(tiny_config, tmp_path, capsys):
    """The quiet one: resuming across configs mixes two experiments into one
    number that looks like neither."""
    out = tmp_path / "run"
    out.mkdir()
    T._atomic_save({"epoch": 0, "best": 0.9, "history": [{}],
                    "fingerprint": "not-this-config"}, out / "last.pt")

    assert T.main(["--config", str(tiny_config), "--out", str(out), "--epochs", "1"]) == 0
    printed = capsys.readouterr().out
    assert "different config" in printed and "RESUMED" not in printed
