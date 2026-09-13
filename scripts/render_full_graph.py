"""Render tier: run trained weights over the complete 162k-neuron graph.

PLAN.md keeps this as an offline, single-batch pass for demo renders and to
check whether the larger graph changes anything. The live subgraph is a subset
of the full graph, so the trained per-edge gains transfer directly: edges
present in both keep their learned value, and edges only in the full graph fall
back to their connectome initialisation (log synapse count). That fallback is
the point of the comparison -- it asks what the rest of the brain contributes
when it is switched on without ever having been trained.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import build_model, get_subgraph, load_config, role_index  # noqa: E402
from connectome import build_neuron_graph, load_verified_types  # noqa: E402
from subgraph import extract  # noqa: E402


def transfer_gains(live_model, full_model, live_sg, full_sg) -> dict:
    """Copy learned per-edge gains from the live subgraph onto the full graph."""
    # Match on (pre bodyId, post bodyId), not on local indices: the two graphs
    # number their nodes differently. ConnectomeRNN also re-sorts edges internally, so read the gains back in the
    # model's own order via its stored [post, pre] index pairs.
    def model_pairs(sg, model):
        row, col = model.rnn.edge_index.numpy()          # post, pre (local)
        return np.stack([sg.body_ids[col], sg.body_ids[row]])

    lp = model_pairs(live_sg, live_model)
    fp = model_pairs(full_sg, full_model)

    lmap = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(lp[0], lp[1]))}
    gains = live_model.rnn.log_gain.detach().numpy()
    target = full_model.rnn.log_gain.detach().numpy().copy()

    hits = 0
    for i, (a, b) in enumerate(zip(fp[0], fp[1])):
        j = lmap.get((int(a), int(b)))
        if j is not None:
            target[i] = gains[j]
            hits += 1
    with torch.no_grad():
        full_model.rnn.log_gain.copy_(torch.from_numpy(target))

    # per-neuron parameters transfer by bodyId
    lidx = {int(b): i for i, b in enumerate(live_sg.body_ids)}
    for name in ("log_tau", "threshold", "bias"):
        src = getattr(live_model.rnn, name).detach().numpy()
        dst = getattr(full_model.rnn, name).detach().numpy().copy()
        for i, b in enumerate(full_sg.body_ids):
            j = lidx.get(int(b))
            if j is not None:
                dst[i] = src[j]
        with torch.no_grad():
            getattr(full_model.rnn, name).copy_(torch.from_numpy(dst))

    return {"edges_transferred": hits, "edges_full": int(fp.shape[1]),
            "fraction": hits / max(fp.shape[1], 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "render_full.yaml")
    ap.add_argument("--audio", type=Path, help="wav to render; omit to just report transfer stats")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "render_full.mid")
    ap.add_argument("--max-nodes", type=int, default=None,
                    help="cap the render graph (memory guard); omit for all 162k")
    a = ap.parse_args(argv)

    device = torch.device("cpu")
    ck = torch.load(a.checkpoint, map_location=device, weights_only=False)
    live_cfg = ck["config"]
    live_sg = get_subgraph(live_cfg)
    live_model, kit = build_model(live_cfg, live_sg,
                                  n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    live_model.load_state_dict(ck["model"])

    full_cfg = load_config(a.config)
    full_cfg["kit"] = live_cfg["kit"]
    g = build_neuron_graph()
    print(g.summary())
    full_sg = extract(
        g, load_verified_types(),
        seed_concepts=tuple(live_cfg["subgraph"].get("seed_concepts", ("JO_A", "JO_B", "JO_E"))),
        target_concepts=tuple(live_cfg["subgraph"].get("target_concepts", ("wing_motor_all",))),
        min_weight=live_cfg["subgraph"].get("min_weight", 3),
        max_nodes=a.max_nodes,
        full_graph=a.max_nodes is None,
    )
    print(full_sg.summary())

    full_model, _ = build_model(full_cfg, full_sg,
                                n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    stats = transfer_gains(live_model, full_model, live_sg, full_sg)
    print(f"transferred {stats['edges_transferred']:,} of {stats['edges_full']:,} edges "
          f"({stats['fraction']:.1%}); the rest keep their connectome initialisation")

    if a.audio:
        from realtime import render_file
        n = render_file(full_model.eval(), kit, full_cfg, a.audio, a.out)
        print(f"wrote {n} notes -> {a.out}")
        stats["notes"] = n

    (a.out.parent / "render_full_stats.json").write_text(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
