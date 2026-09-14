# Test fixtures

Two subgraphs, both real. Every neuron, edge, frozen sign and type in them came
out of MaleCNS v1.0 — nothing here is synthetic, because the properties these
tests pin (Dale's law across a real transmitter table, the confirmed
populations surviving the trim, a bundle rebuilding the same edge order) are
exactly the ones a made-up graph would not exercise.

| file | what it is |
|---|---|
| `subgraph_10k.npz` | A cached Phase 1 extraction: 10,000 neurons, 642,569 edges, `min_weight` 5, 3 hops each way from the JO afferents. The live tier the CPU runs trained on. |
| `subgraph_2k.npz` | 2,000 neurons, 62,383 edges, **induced** from the above by `scripts/make_test_fixture.py` — every role node plus the most strongly connected of the rest. Not an extraction from the connectome; a slice of one. |

They are committed for one reason: building a subgraph otherwise needs the
1.1 GB connectome download, so every test that needs a model to exist skipped
on a fresh checkout. That included all six bundle round-trip tests — the ones
covering the only artefact a user ever receives — and the gradient
checkpointing test, which had never run here at all until they were added.

Regenerate the small one with:

```bash
python scripts/make_test_fixture.py \
    --source tests/fixtures/subgraph_10k.npz \
    --out tests/fixtures/subgraph_2k.npz --max-nodes 2000
```

`tests/conftest.py` prefers a locally built cache under `data/cache/` where one
exists — that is the graph you are actually training on — and falls back to
these.
