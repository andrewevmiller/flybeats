import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "plugin" / "export"))


@pytest.fixture(scope="session")
def cfg():
    from flybeats.config import load_locked
    return load_locked()


@pytest.fixture(scope="session")
def slice_small():
    from flybrain.synthetic import make_slice
    return make_slice(seed=1)


@pytest.fixture(scope="session")
def brain_small(cfg, slice_small, tmp_path_factory):
    """(brain, model, neurons, edges) for a small synthetic network exported through the real exporter."""
    from flybrain.brainfile import read_brain
    from flybrain.exporter import export
    from flybrain.parity import make_model
    neurons, edges = slice_small
    model = make_model(cfg, neurons, edges, seed=0)
    path = tmp_path_factory.mktemp("brain") / "small.flybrain"
    export(model, neurons, edges, cfg, path, name="synthetic-small")
    return read_brain(path), model, neurons, edges
