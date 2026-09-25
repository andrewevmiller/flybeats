"""Assemble a flybeats model from a config. Shared by train, ablations, realtime.

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
from model import ConnectomeRNN, FlyBeats, GenreModulation, ModelConfig
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


#: Subgraph settings that change the graph itself. A cached file is only reused
#: when every one of these matches what the config asks for.
SUBGRAPH_KEYS = ("seed_concepts", "target_concepts", "forward_hops",
                 "backward_hops", "min_weight", "max_nodes", "full_graph")


def subgraph_params(cfg: dict) -> dict:
    sc = cfg.get("subgraph", {})
    return {
        "seed_concepts": list(sc.get("seed_concepts", ["JO_A", "JO_B", "JO_E"])),
        "target_concepts": list(sc.get("target_concepts", ["wing_motor_all"])),
        "forward_hops": sc.get("forward_hops", 3),
        "backward_hops": sc.get("backward_hops", 3),
        "min_weight": sc.get("min_weight", 3),
        "max_nodes": sc.get("max_nodes", 30_000),
        "full_graph": bool(sc.get("full_graph", False)),
    }


def get_subgraph(cfg: dict, rebuild: bool = False) -> SubGraph:
    """Load the cached subgraph, or rebuild it if the config asks for a different one.

    Configs inherit a cache path from their ``_base_``, so several of them
    address the same file while asking for different graphs -- sanity_3piece
    wants 10k nodes at min_weight 5, v1_8piece wants 30k at 3. Reusing whatever
    happens to be on disk means a run silently trains on the previous run's
    graph and nothing says so. The saved metadata is checked against the request
    and the graph is rebuilt on any mismatch.
    """
    sc = cfg.get("subgraph", {})
    cache = Path(sc.get("cache", ROOT / "data" / "cache" / "subgraph.npz"))
    if not cache.is_absolute():
        cache = ROOT / cache
    want = subgraph_params(cfg)

    if cache.exists() and not rebuild:
        sg = SubGraph.load(cache)
        have = {k: sg.meta.get(k) for k in SUBGRAPH_KEYS}
        differs = {k: (have[k], want[k]) for k in SUBGRAPH_KEYS
                   if _norm(have[k]) != _norm(want[k])}
        if not differs:
            return sg
        print(f"  [subgraph] {cache.name} was built with "
              + ", ".join(f"{k}={h!r} (want {w!r})" for k, (h, w) in differs.items())
              + " -- rebuilding")

    g = build_neuron_graph()
    sg = extract(g, load_verified_types(), **{
        **want,
        "seed_concepts": tuple(want["seed_concepts"]),
        "target_concepts": tuple(want["target_concepts"]),
    })
    cache.parent.mkdir(parents=True, exist_ok=True)
    sg.save(cache)
    return sg


def _norm(v):
    """Compare lists and tuples by value; everything else as-is."""
    return list(v) if isinstance(v, (list, tuple)) else v


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
        standardize=cfg["audio"].get("standardize_features", True),
        nonneg=cfg["audio"].get("nonneg_to_jo", True),
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
        velocity_head=cfg["kit"].get("velocity_head", True),
        velocity_activation=cfg["kit"].get("velocity_activation", "sigmoid"),
        standardize_motor=cfg["kit"].get("standardize_motor", False),
    )

    genre = None
    oa = genre_target_index(cfg, role_index(sg))
    if cfg.get("genre", {}).get("enabled", True) and len(oa) and n_styles > 1:
        genre = GenreModulation(
            n_styles=n_styles, target_idx=oa, n_nodes=sg.n_nodes,
            dim=cfg["genre"].get("dim", 8),
            max_current=cfg["genre"].get("max_current", 0.5),
        )
    return FlyBeats(enc, rnn, dec, genre), kit


def genre_target_index(cfg: dict, roles: dict) -> np.ndarray:
    """Nodes the genre tonic lands on: the union of the ``genre.targets`` roles.

    Defaults to ``["octopaminergic"]``, which is what every config and
    checkpoint before this key used. A single role keeps its own node order, so
    that default builds exactly the index it always did; several roles are
    joined in listed order with repeats dropped.
    """
    names = (cfg.get("genre") or {}).get("targets", ["octopaminergic"])
    if isinstance(names, str):
        names = [names]
    missing = [n for n in names if n not in roles]
    if missing:
        raise SystemExit(f"genre.targets names unknown roles {missing}; "
                         f"known: {sorted(roles)}")
    parts = [np.asarray(roles[n], dtype=np.int64) for n in names]
    if len(parts) == 1:
        return parts[0]
    return np.asarray(list(dict.fromkeys(np.concatenate(parts).tolist())), dtype=np.int64)


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
