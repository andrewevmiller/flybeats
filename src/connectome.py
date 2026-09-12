"""Load MaleCNS v1.0 into a neuron-level graph, and cache it.

The raw ``connectome-weights`` table is segment-to-segment over the *whole*
segmentation: 151.8M rows spanning ~87M post-synaptic fragments, most of which
are unannotated debris. Everything downstream wants the neuron-level graph --
edges where both endpoints are annotated, traced, typed bodies. That reduction
is expensive (and easy to OOM naively), so it is done once here and cached as a
compact ``.npz``.

Node ids are dense int32 indices into ``body_ids``; ``weight`` is the raw
synapse count, which Phase 3 uses to initialise per-edge log-gain.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.feather as pf

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = ROOT / "data" / "cache"

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

#: Neurotransmitter -> synaptic sign. Frozen at build time; never trainable.
#: ACh is the excitatory workhorse; GABA and glutamate are inhibitory in the
#: fly (GluCl-alpha is a chloride channel, so glutamate is *not* excitatory
#: here the way it is in vertebrates). Histamine is inhibitory (photoreceptors).
#: Aminergic transmitters are modulatory -- they get sign 0 in the fast graph
#: and are reached instead through the neuromodulatory bias path in Phase 3.
NT_SIGN: dict[str, int] = {
    "acetylcholine": +1,
    "glutamate": -1,
    "gaba": -1,
    "histamine": -1,
    "dopamine": 0,
    "octopamine": 0,
    "serotonin": 0,
    "unclear": 0,
}

#: Sign used when a neuron's transmitter is unknown. Excitatory is the majority
#: class (104k of 156k resolved bodies), so this is the least-damaging guess --
#: but it is recorded in ``sign_known`` so ablations can measure its impact.
DEFAULT_SIGN = +1


@dataclass
class Graph:
    """A neuron-level connectome graph with frozen synaptic signs."""

    body_ids: np.ndarray          # (N,) int64 -- MaleCNS bodyId per node
    types: np.ndarray             # (N,) object -- cell-type string per node
    edge_index: np.ndarray        # (2, E) int32 -- [pre, post] dense indices
    weight: np.ndarray            # (E,) int32 -- raw synapse count
    sign: np.ndarray              # (N,) int8 -- presynaptic sign, per Dale's law
    sign_known: np.ndarray        # (N,) bool -- False where DEFAULT_SIGN was used
    nt: np.ndarray                # (N,) object -- resolved transmitter per node
    side: np.ndarray              # (N,) object -- somaSide ('L'/'R'/'M'/'')
    superclass: np.ndarray        # (N,) object

    @property
    def n_nodes(self) -> int:
        return len(self.body_ids)

    @property
    def n_edges(self) -> int:
        return self.edge_index.shape[1]

    def index_of(self, body_ids) -> np.ndarray:
        """Map bodyIds to dense node indices, dropping any not in the graph."""
        order = np.argsort(self.body_ids)
        srt = self.body_ids[order]
        q = np.asarray(list(body_ids), dtype=np.int64)
        pos = np.searchsorted(srt, q)
        pos = np.clip(pos, 0, len(srt) - 1)
        ok = srt[pos] == q
        return order[pos[ok]].astype(np.int32)

    def nodes_of_types(self, type_names) -> np.ndarray:
        wanted = set(type_names)
        return np.flatnonzero(np.isin(self.types, list(wanted))).astype(np.int32)

    def summary(self) -> str:
        exc = int((self.sign > 0).sum())
        inh = int((self.sign < 0).sum())
        mod = int((self.sign == 0).sum())
        return (
            f"Graph: {self.n_nodes:,} neurons, {self.n_edges:,} edges, "
            f"{int(self.weight.sum()):,} synapses | "
            f"signs +{exc:,} / -{inh:,} / mod {mod:,} | "
            f"{int((~self.sign_known).sum()):,} inferred"
        )


def _resolve_neurotransmitters(body_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-body transmitter, preferring ground truth > body consensus > cell-type."""
    nt = pd.read_feather(
        RAW / NEUROTRANSMITTERS,
        columns=["body", "consensus_nt", "celltype_predicted_nt", "ground_truth"],
    )
    nt = nt[nt["body"].isin(body_ids)]

    def clean(s: pd.Series) -> pd.Series:
        return s.where(s.notna() & (s != "unclear") & (s != ""))

    resolved = (
        clean(nt["ground_truth"])
        .fillna(clean(nt["consensus_nt"]))
        .fillna(clean(nt["celltype_predicted_nt"]))
    )
    table = pd.Series(resolved.to_numpy(), index=nt["body"].to_numpy())
    table = table[~table.index.duplicated(keep="first")]

    per_node = table.reindex(body_ids)
    known = per_node.notna().to_numpy()
    return per_node.fillna("unclear").to_numpy(dtype=object), known


def build_neuron_graph(
    raw: Path = RAW,
    cache: Path = CACHE,
    min_weight: int = 1,
    traced_only: bool = True,
    typed_only: bool = True,
    rebuild: bool = False,
) -> Graph:
    """Build (or load) the neuron-level graph.

    Parameters mirror the filters that meaningfully change the graph, and are
    all encoded in the cache filename so variants do not collide.
    """
    cache.mkdir(parents=True, exist_ok=True)
    tag = f"w{min_weight}_{'traced' if traced_only else 'all'}_{'typed' if typed_only else 'untyped'}"
    path = cache / f"neuron_graph_{tag}.npz"

    if path.exists() and not rebuild:
        z = np.load(path, allow_pickle=True)
        return Graph(
            body_ids=z["body_ids"], types=z["types"], edge_index=z["edge_index"],
            weight=z["weight"], sign=z["sign"], sign_known=z["sign_known"],
            nt=z["nt"], side=z["side"], superclass=z["superclass"],
        )

    ann = pd.read_feather(
        raw / ANNOTATIONS,
        columns=["bodyId", "type", "status", "somaSide", "superclass", "class", "subclass"],
    )
    if traced_only:
        ann = ann[ann["status"] == "Traced"]
    if typed_only:
        ann = ann[ann["type"].notna()]
    ann = ann.drop_duplicates("bodyId").sort_values("bodyId").reset_index(drop=True)

    body_ids = ann["bodyId"].to_numpy(dtype=np.int64)
    n = len(body_ids)
    print(f"  nodes: {n:,}")

    # Stream the edge table in batches; searchsorted keeps peak memory flat
    # instead of the multi-GB temporaries np.isin would build over 151M rows.
    src_chunks, dst_chunks, w_chunks = [], [], []
    reader = pf.read_table(raw / WEIGHTS, memory_map=True)
    total_in = 0
    for batch in reader.to_batches(max_chunksize=8_000_000):
        pre = batch.column("body_pre").to_numpy()
        post = batch.column("body_post").to_numpy()
        wt = batch.column("weight").to_numpy()
        total_in += len(pre)

        if min_weight > 1:
            keep = wt >= min_weight
            pre, post, wt = pre[keep], post[keep], wt[keep]

        ip = np.searchsorted(body_ids, pre)
        np.clip(ip, 0, n - 1, out=ip)
        hit = body_ids[ip] == pre
        if not hit.any():
            continue
        ip, post, wt = ip[hit], post[hit], wt[hit]

        iq = np.searchsorted(body_ids, post)
        np.clip(iq, 0, n - 1, out=iq)
        hit = body_ids[iq] == post
        if not hit.any():
            continue

        src_chunks.append(ip[hit].astype(np.int32))
        dst_chunks.append(iq[hit].astype(np.int32))
        w_chunks.append(wt[hit].astype(np.int32))

    del reader
    src = np.concatenate(src_chunks); dst = np.concatenate(dst_chunks)
    weight = np.concatenate(w_chunks)
    del src_chunks, dst_chunks, w_chunks
    print(f"  edges: {len(src):,} kept of {total_in:,} raw rows")

    nt, known = _resolve_neurotransmitters(body_ids)
    sign = np.array([NT_SIGN.get(x, 0) for x in nt], dtype=np.int8)
    # Neurons with no transmitter call still need a sign to be usable.
    sign[~known] = DEFAULT_SIGN

    g = Graph(
        body_ids=body_ids,
        types=ann["type"].to_numpy(dtype=object),
        edge_index=np.stack([src, dst]),
        weight=weight,
        sign=sign,
        sign_known=known,
        nt=nt,
        side=ann["somaSide"].fillna("").to_numpy(dtype=object),
        superclass=ann["superclass"].fillna("").to_numpy(dtype=object),
    )
    np.savez_compressed(
        path, body_ids=g.body_ids, types=g.types, edge_index=g.edge_index,
        weight=g.weight, sign=g.sign, sign_known=g.sign_known, nt=g.nt,
        side=g.side, superclass=g.superclass,
    )
    print(f"  cached -> {path}")
    return g


def load_verified_types(path: Path | None = None) -> dict:
    """Read the Phase 0 gate output. No module may hardcode a type string."""
    path = path or (ROOT / "data" / "verified_types.json")
    if not path.exists():
        raise SystemExit(
            f"missing {path}\nRun the Phase 0 gate first: python scripts/verify_types.py"
        )
    return json.loads(path.read_text())


def types_for(verified: dict, *concepts: str, require: bool = True) -> list[str]:
    """Collect the confirmed type strings for one or more Phase 0 concepts."""
    out: list[str] = []
    for c in concepts:
        rec = verified["concepts"].get(c)
        if rec is None or not rec["confirmed"]:
            if require:
                raise KeyError(f"concept {c!r} is not confirmed in verified_types.json")
            continue
        out.extend(t["type"] for t in rec["types"])
    return sorted(set(out))


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Build and cache the neuron-level graph")
    ap.add_argument("--min-weight", type=int, default=1)
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    print(build_neuron_graph(min_weight=a.min_weight, rebuild=a.rebuild).summary())
