"""A self-contained model file: the laptop path.

Loading a checkpoint used to mean rebuilding the subgraph, which meant the
1.1 GB connectome tables, a multi-GB graph build, and the corpus whose config
path the run happened to name. None of that is needed to *run* the model: the
trained checkpoint already carries the whole topology as buffers -- edge_index,
edge_sign, the CSR index arrays, and the sensory and motor index sets. It is
called for only because ``build_model`` takes a ``SubGraph`` to construct from.

So a bundle is the checkpoint plus the two small things that are genuinely not
in it -- the role index the sliders and lesion mode address, and the motor-side
labels -- with the derivable buffers dropped and the indices stored as int32.
That is ~9 MB against 37 MB, and it needs no ``data/`` directory at all.

    python scripts/export_bundle.py --checkpoint runs/rho10_long/best.pt
    python src/realtime.py --bundle flybeats-8piece.fb --render song.wav

The CSR arrays and the edge ordering are rebuilt by ``ConnectomeRNN.__init__``
from the edge list, exactly as they were built the first time, so a bundle is
not a second implementation of the model -- it is the same constructor fed from
a different source. ``scripts/export_bundle.py`` proves that by comparing the
two models' outputs on the same audio before it writes anything.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

#: Bumped when the on-disk layout changes in a way an older loader would
#: misread. A bundle that says a version this code does not know is refused
#: rather than loaded into a wrong-shaped model.
#:
#: v2 added the decoder's velocity head. v1 bundles stay readable, because a
#: bundle is the only durable form a trained model has here -- refusing one
#: would strand a model that cost hours to train, and the shape difference is
#: recoverable from the state dict itself.
FORMAT_VERSION = 2

#: Every version this build can load. Older entries must stay loadable; the
#: loader adapts the model to what the bundle actually contains.
READABLE_VERSIONS = (1, 2)

#: Rebuilt by ``ConnectomeRNN.__init__`` from the edge list, so storing them
#: would be storing the same information three more times.
DERIVED_BUFFERS = ("rnn.edge_row", "rnn.edge_col", "rnn.crow", "rnn.crow_t",
                   "rnn.col_t", "rnn.perm_t", "rnn.edge_index", "rnn.edge_sign")


def _pack(sd: dict) -> dict:
    """Drop the derivable buffers; the rest is stored as-is."""
    return {k: v for k, v in sd.items() if k not in DERIVED_BUFFERS}


def build_bundle(ck: dict, sg, model) -> dict:
    """Assemble the bundle payload from a checkpoint and the subgraph it used."""
    from build import role_index

    rnn = model.rnn
    # stored [row, col] = [post, pre]; the constructor wants [pre, post]
    row = rnn.edge_index[0].cpu().numpy()
    col = rnn.edge_index[1].cpu().numpy()
    roles = {k: np.asarray(v, dtype=np.int32) for k, v in role_index(sg).items()}

    return {
        "format": FORMAT_VERSION,
        "config": ck["config"],
        "kit": ck["kit"],
        "n_styles": int(ck.get("n_styles", 1) or 1),
        # what the eval sweep chose for this model, so playback does not re-guess
        "best_threshold": ck.get("best_threshold"),
        "n_nodes": int(rnn.n_nodes),
        # int32 halves the edge list; 642k edges cannot overflow it and the
        # loader casts back before anything touches torch.sparse.
        "edge_index": np.stack([col, row]).astype(np.int32),
        "edge_sign": rnn.edge_sign.cpu().numpy().astype(np.int8),
        "sensory_idx": rnn.sensory_idx.cpu().numpy().astype(np.int32),
        "motor_idx": rnn.motor_idx.cpu().numpy().astype(np.int32),
        "motor_side": np.asarray([str(s) for s in sg.side[sg.role("motor")]]),
        "roles": roles,
        "state": _pack(model.state_dict()),
    }


def load_bundle(path: str | Path, device=None):
    """``(model, kit, config, roles)`` from a bundle file. Touches no dataset.

    The edge weights passed to the constructor are placeholders: every trained
    value arrives with the state dict a few lines later. What the constructor is
    being used for here is the *structure* -- the edge sort order and the CSR
    arrays -- which has to be derived the same way it was at training time or
    the loaded ``log_gain`` would line up against the wrong edges.
    """
    from decoder import DrumKit, MotorToDrums
    from encoder import AudioToJO
    from model import ConnectomeRNN, FlyBeats, GenreModulation, ModelConfig

    device = device or torch.device("cpu")
    b = torch.load(Path(path), map_location="cpu", weights_only=False)
    if int(b.get("format", -1)) not in READABLE_VERSIONS:
        raise ValueError(
            f"{path} is bundle format {b.get('format')!r}, this build reads "
            f"{', '.join(str(v) for v in READABLE_VERSIONS)}. Re-export it with "
            f"scripts/export_bundle.py."
        )

    cfg = b["config"]
    kit = DrumKit(b["kit"])
    sensory = b["sensory_idx"].astype(np.int64)
    motor = b["motor_idx"].astype(np.int64)
    edge_index = b["edge_index"].astype(np.int64)

    mcfg = ModelConfig(
        step_ms=cfg["audio"]["step_ms"],
        tau_ms_init=cfg["model"].get("tau_ms_init", 20.0),
        # the trained gain_scale arrives as a buffer in the state dict; asking
        # for "auto" here would run a power iteration to compute a value that is
        # about to be overwritten
        gain_scale=1.0,
        input_scale=cfg["model"].get("input_scale", 1.0),
        state_clip=cfg["model"].get("state_clip", 20.0),
    )
    rnn = ConnectomeRNN(
        edge_index=edge_index, edge_sign=b["edge_sign"].astype(np.float32),
        weight=np.ones(edge_index.shape[1], dtype=np.float32),
        n_nodes=int(b["n_nodes"]), sensory_idx=sensory, motor_idx=motor, cfg=mcfg,
    )
    enc = AudioToJO(
        n_channels=len(sensory), zone_of_channel=["JO_other"] * len(sensory),
        sample_rate=cfg["audio"]["sample_rate"], step_ms=cfg["audio"]["step_ms"],
        n_bands=cfg["audio"].get("n_bands", 64),
        standardize=cfg["audio"].get("standardize_features", True),
        nonneg=cfg["audio"].get("nonneg_to_jo", True),
    )
    # What the bundle contains decides the shape, not what this build would
    # build by default: a v1 bundle has no velocity head, and constructing one
    # anyway would fail the strict-ish load below on a missing key.
    has_vel = any(k.startswith("decoder.vel_readout") for k in b["state"])
    dec = MotorToDrums(n_motor=len(motor), kit=kit, motor_side=b["motor_side"],
                       bilateral=cfg["kit"].get("bilateral", False),
                       velocity_head=has_vel,
                       velocity_activation=cfg["kit"].get("velocity_activation",
                                                          "sigmoid"))
    genre = None
    n_styles = int(b.get("n_styles", 1) or 1)
    if any(k.startswith("genre.") for k in b["state"]):
        oa = np.asarray(b["roles"]["octopaminergic"], dtype=np.int64)
        genre = GenreModulation(n_styles=n_styles, target_idx=oa, n_nodes=int(b["n_nodes"]),
                                dim=cfg["genre"].get("dim", 8),
                                max_current=cfg["genre"].get("max_current", 0.5))

    model = FlyBeats(enc, rnn, dec, genre)
    missing, unexpected = model.load_state_dict(b["state"], strict=False)
    missing = [k for k in missing if k not in DERIVED_BUFFERS]
    if missing or unexpected:
        raise ValueError(f"bundle does not match this model: missing={missing}, "
                         f"unexpected={unexpected}")

    roles = {k: np.asarray(v, dtype=np.int64) for k, v in b["roles"].items()}
    if b.get("best_threshold") is not None:
        cfg.setdefault("eval", {})["threshold"] = float(b["best_threshold"])
    return model.to(device).eval(), kit, cfg, roles
