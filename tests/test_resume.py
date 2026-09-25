"""A run stopped and resumed must be the run that was never stopped.

Resume exists so a long run can be split around a shutdown or a Windows Update
restart -- the 23 September queue lost A'1 at epoch 2 of 12 and retrained it
from scratch. A resume that restores the weights but not AdamW's moments, or
not the loader's shuffle stream, still trains and still logs plausible
numbers; it is just a different run, and nothing downstream could tell. So
this compares bit for bit.
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
from build import load_config  # noqa: E402

from conftest import SMALL_GRAPH, use_small_graph  # noqa: E402

needs_graph = pytest.mark.skipif(not SMALL_GRAPH.exists(),
                                 reason="no subgraph fixture in this checkout")


def _config(tmp_path: Path, epochs: int = 3) -> Path:
    cfg = use_small_graph(load_config(ROOT / "configs" / "sanity_3piece.yaml"))
    cfg["data"].update(synthetic=True, n_clips=4, seconds=0.5)
    # workers=1 on purpose: worker seeds come off the loader's generator,
    # which is exactly the state a naive resume forgets.
    cfg["train"].update(batch_size=2, tbptt_steps=25, workers=1, epochs=epochs,
                        device="cpu", refit_velocity_head=False)
    path = tmp_path / "tiny.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


def _run(cfg: Path, out: Path, *extra: str) -> int:
    return T.main(["--config", str(cfg), "--out", str(out), *extra])


def _strip(history):
    """Everything but wall-clock time, as text: some metrics are NaN on the
    toy data, and NaN != NaN would fail two identical histories."""
    return json.dumps([{k: v for k, v in r.items() if k != "seconds"} for r in history],
                      sort_keys=True)


@needs_graph
def test_stop_then_resume_matches_an_unbroken_run(tmp_path):
    cfg = _config(tmp_path)
    whole, split = tmp_path / "whole", tmp_path / "split"

    assert _run(cfg, whole) == 0

    split.mkdir()
    (split / "STOP").touch()
    assert _run(cfg, split, "--resume") == 0          # no last.pt yet: starts fresh
    first = torch.load(split / "last.pt", weights_only=False)
    assert first["epochs_done"] == 1, "STOP must end the run after the current epoch"
    assert not (split / "STOP").exists(), "a consumed STOP must not stop the resume too"

    assert _run(cfg, split, "--resume") == 0

    a = torch.load(whole / "last.pt", weights_only=False)
    b = torch.load(split / "last.pt", weights_only=False)
    assert b["epochs_done"] == a["epochs_done"] == 3
    assert _strip(a["history"]) == _strip(b["history"])
    for k in a["model"]:
        assert torch.equal(a["model"][k], b["model"][k]), f"weights differ at {k}"
    for (i, sa), (_, sb) in zip(a["opt"]["state"].items(), b["opt"]["state"].items()):
        for k in sa:
            assert torch.equal(torch.as_tensor(sa[k]), torch.as_tensor(sb[k])), \
                f"optimiser state {k} differs for param {i}"
    ba = torch.load(whole / "best.pt", weights_only=False)["model"]
    bb = torch.load(split / "best.pt", weights_only=False)["model"]
    assert all(torch.equal(ba[k], bb[k]) for k in ba)


@needs_graph
def test_resume_refuses_another_configs_checkpoint(tmp_path):
    cfg = _config(tmp_path, epochs=1)
    out = tmp_path / "run"
    assert _run(cfg, out) == 0
    other = yaml.safe_load(cfg.read_text())
    other["train"]["lr"] = other["train"].get("lr", 3e-3) * 2
    cfg2 = tmp_path / "other.yaml"
    cfg2.write_text(yaml.safe_dump(other))
    with pytest.raises(SystemExit, match="different config"):
        _run(cfg2, out, "--resume")


@needs_graph
def test_a_finished_run_can_be_extended(tmp_path):
    """train.epochs is the one setting a resume may change."""
    out = tmp_path / "run"
    assert _run(_config(tmp_path, epochs=1), out) == 0
    assert _run(_config(tmp_path, epochs=2), out, "--resume") == 0
    assert torch.load(out / "last.pt", weights_only=False)["epochs_done"] == 2
