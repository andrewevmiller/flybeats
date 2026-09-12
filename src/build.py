"""Assemble a FlyDrums model from a config. Shared by train, ablations, realtime.

Keeping this in one place is what lets the Phase 4 ablations be honest: every
arm is built through this function with the same encoder, decoder, optimiser
and data, and only the recurrent core differs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import yaml

from connectome import build_neuron_graph, load_verified_types, types_for
from decoder import DrumKit, MotorToDrums
from encoder import AudioToJO, zones_for_channels
from model import ConnectomeRNN, FlyDrums, GenreModulation, ModelConfig
from subgraph import SubGraph, extract

ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    base = cfg.pop("_base_", None)
    if base:
        parent = load_config(Path(path).parent / base)
        cfg = deep_merge(parent, cfg)
    return cfg


def deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get_subgraph(cfg: dict, rebuild: bool = False) -> SubGraph:
    sc = cfg.get("subgraph", {})
    cache = Path(sc.get("cache", ROOT / "data" / "cache" / "subgraph.npz"))
    if cache.exists() and not rebuild:
        return SubGraph.load(cache)
    g = build_neuron_graph()
    sg = extract(
        g, load_verified_types(),
        seed_concepts=tuple(sc.get("seed_concepts", ("JO_A", "JO_B", "JO_E"))),
        target_concepts=tuple(sc.get("target_concepts", ("wing_motor_all",))),
        forward_hops=sc.get("forward_hops", 3),
        backward_hops=sc.get("backward_hops", 3),
        min_weight=sc.get("min_weight", 3),
        max_nodes=sc.get("max_nodes", 30_000),
        full_graph=sc.get("full_graph", False),
    )
    sg.save(cache)
    return sg


def inhibitory_nodes(sg: SubGraph) -> np.ndarray:
    """Nodes whose transmitter is inhibitory -- the 'tightness' slider target."""
    inh = {"gaba", "glutamate", "histamine"}
    return np.flatnonzero(np.isin(sg.nt, list(inh))).astype(np.int64)


def build_model(cfg: dict, sg: SubGraph, n_styles: int = 1, verified: dict | None = None):
    verified = verified or load_verified_types()

    sensory = sg.role("sensory").astype(np.int64)
    motor = sg.role("motor").astype(np.int64)
    if len(sensory) == 0 or len(motor) == 0:
        raise SystemExit("subgraph has no sensory or motor population")

    kit = DrumKit.from_tier(cfg["kit"]["tier"], max_classes=len(motor))
    zones = zones_for_channels(sg.types[sensory], verified)

    enc = AudioToJO(
        n_channels=len(sensory), zone_of_channel=zones,
        sample_rate=cfg["audio"]["sample_rate"], step_ms=cfg["audio"]["step_ms"],
        n_bands=cfg["audio"].get("n_bands", 64),
        trainable_dsp=cfg["audio"].get("trainable_dsp", False),
    )
    mcfg = ModelConfig(
        step_ms=cfg["audio"]["step_ms"],
        tau_ms_init=cfg["model"].get("tau_ms_init", 20.0),
        gain_scale=cfg["model"].get("gain_scale", "auto"),
        spectral_radius=cfg["model"].get("spectral_radius", 0.9),
        input_scale=cfg["model"].get("input_scale", 1.0),
        state_clip=cfg["model"].get("state_clip", 20.0),
    )
    rnn = ConnectomeRNN(
        edge_index=sg.edge_index, edge_sign=sg.edge_sign, weight=sg.weight,
        n_nodes=sg.n_nodes, sensory_idx=sensory, motor_idx=motor, cfg=mcfg,
    )
    dec = MotorToDrums(
        n_motor=len(motor), kit=kit,
        motor_side=sg.side[motor], bilateral=cfg["kit"].get("bilateral", False),
    )

    genre = None
    oa = sg.role("octopaminergic").astype(np.int64)
    if cfg.get("genre", {}).get("enabled", True) and len(oa) and n_styles > 1:
        genre = GenreModulation(
            n_styles=n_styles, target_idx=oa, n_nodes=sg.n_nodes,
            dim=cfg["genre"].get("dim", 8),
            max_current=cfg["genre"].get("max_current", 0.5),
        )
    return FlyDrums(enc, rnn, dec, genre), kit


def role_index(sg: SubGraph) -> dict[str, np.ndarray]:
    """Populations the sliders and lesion mode address, by name."""
    roles = {k: v.astype(np.int64) for k, v in sg.roles.items()}
    roles["inhibitory"] = inhibitory_nodes(sg)
    return roles


def device_of(cfg: dict) -> torch.device:
    want = cfg.get("train", {}).get("device", "auto")
    if want == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(want)
