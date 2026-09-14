"""Where the tests find a subgraph to build models from.

Every test that needs a model needs a ``SubGraph``, and building one from
scratch needs the 1.1 GB connectome download. On a fresh checkout those tests
skipped -- including all six bundle round-trip tests, which cover the one
artefact a user ever receives. A skipped test is not a passing test, and the
export path was effectively untested everywhere except a machine that had
already downloaded the connectome.

So two fixtures are committed, both real: `subgraph_10k.npz` is a cached
extraction from MaleCNS v1.0, and `subgraph_2k.npz` is an induced subgraph of
it (``scripts/make_test_fixture.py``), small enough to build a model from in
well under a second. Real neurons, real edges, real frozen signs -- a
synthetic stand-in would not exercise Dale's law or the confirmed populations,
which are exactly the properties worth pinning.

A locally built cache still wins where one exists: it is the graph the user is
actually training on.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
sys.path.insert(0, str(ROOT / "src"))

#: 2,000 neurons induced from the 10k extraction: for anything that just needs
#: a model to exist.
SMALL_GRAPH = FIXTURES / "subgraph_2k.npz"
SMALL_GRAPH_NODES = 2000

#: The largest real extraction available here, preferring a locally built cache.
REAL_GRAPH_CANDIDATES = (
    ROOT / "data" / "cache" / "subgraph.npz",
    ROOT / "data" / "cache" / "subgraph_10k.npz",
    FIXTURES / "subgraph_10k.npz",
)


def real_graph() -> Path | None:
    """A real cached subgraph, or None if this checkout somehow has neither."""
    return next((p for p in REAL_GRAPH_CANDIDATES if p.exists()), None)


def use_small_graph(cfg: dict) -> dict:
    """Point a config at the small fixture, in place.

    The node count has to match what the fixture was built with: ``get_subgraph``
    compares the request against the cached metadata and rebuilds on a mismatch,
    which is the behaviour that stops a run silently training on the previous
    run's graph -- and which would send a test off to download the connectome.
    """
    cfg["subgraph"]["max_nodes"] = SMALL_GRAPH_NODES
    cfg["subgraph"]["min_weight"] = 5
    cfg["subgraph"]["cache"] = str(SMALL_GRAPH)
    return cfg
