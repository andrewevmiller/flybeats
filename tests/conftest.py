import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flybeats.config import load_locked, load_paths  # noqa: E402


@pytest.fixture(scope="session")
def cfg():
    return load_locked()


@pytest.fixture(scope="session")
def paths():
    return load_paths()


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs the GPU (run with -m gpu)")
    config.addinivalue_line("markers", "slow: takes minutes")


def pytest_collection_modifyitems(config, items):
    """GPU tests run only when asked for with -m gpu."""
    if "gpu" in (config.getoption("-m") or ""):
        return
    skip = pytest.mark.skip(reason="needs the GPU; run with -m gpu")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)
