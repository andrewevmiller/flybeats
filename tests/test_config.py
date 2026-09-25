"""What a config actually resolves to, after `_base_` inheritance.

TEST_PLAN T1.2. Four lines of `deep_merge` implement inheritance, every config
in the repo depends on them, and nothing tested either the merge or its
results. The cost of that showed up in a real run rather than in CI: the
configs named `_cpu` inherit `device: auto` from `v1_8piece.yaml`, which
`build.device_of` resolves to CUDA wherever a card is visible -- so the whole
Phase A' velocity queue would have moved to GPU the moment a CUDA wheel landed,
with nothing in the filename or the config saying so, and A'1's CPU numbers
would have been compared against arms measured somewhere else.

The defence is not a smarter merge. It is a table that states what each shipped
config resolves to, so an inherited value that nobody chose has to be written
down here before it can reach a run.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from build import deep_merge, load_config  # noqa: E402

CONFIGS = ROOT / "configs"


# --- the merge itself ------------------------------------------------------

def test_a_nested_override_replaces_only_its_own_key():
    """The whole point of inheritance here: change one knob, keep the rest."""
    base = {"train": {"device": "auto", "epochs": 20, "lr": 0.003}, "name": "base"}
    got = deep_merge(base, {"train": {"epochs": 5}, "name": "child"})
    assert got["train"] == {"device": "auto", "epochs": 5, "lr": 0.003}
    assert got["name"] == "child"


def test_the_parent_is_not_mutated():
    """One loaded parent is shared by every config that inherits it, so a merge
    that wrote through would make load order decide what a run trained on."""
    base = {"train": {"epochs": 20}, "subgraph": {"max_nodes": 30000}}
    deep_merge(base, {"train": {"epochs": 5}})
    assert base["train"]["epochs"] == 20
    deep_merge(base, {"subgraph": {"max_nodes": 2000}})
    assert base["subgraph"]["max_nodes"] == 30000


def test_a_scalar_overriding_a_dict_replaces_it_whole():
    """Deliberate, and worth pinning: a child that writes a scalar where the
    parent had a mapping means "forget the mapping", not "merge into it"."""
    got = deep_merge({"model": {"radius_by_arm": {"real": 5.0}}},
                     {"model": {"radius_by_arm": 1.0}})
    assert got["model"]["radius_by_arm"] == 1.0


def test_a_dict_overriding_a_scalar_replaces_it_whole():
    got = deep_merge({"model": {"radius": 1.0}}, {"model": {"radius": {"real": 5.0}}})
    assert got["model"]["radius"] == {"real": 5.0}


def test_inheritance_is_transitive():
    """velocity_w5_cpu -> velocity_probe_cpu -> v1_8piece_cpu -> v1_8piece is
    four deep, so a grandparent's value has to survive two merges."""
    cfg = load_config(CONFIGS / "velocity_w5_cpu.yaml")
    assert cfg["audio"]["step_ms"] == 5.0          # from v1_8piece, 3 levels up
    assert cfg["data"]["max_files"] == 256         # from velocity_probe_cpu
    assert cfg["train"]["velocity_weight"] == 5.0  # this config's own reason to exist
    assert cfg["name"] == "velocity_w5_cpu"        # and name is never inherited


# --- the resolution table --------------------------------------------------

#: What each shipped config resolves to, after inheritance. Adding a config
#: means adding a row: an unlisted config fails the completeness test below.
#: `None` means "the key is absent", which for max_nodes means the full graph.
RESOLVED = {
    #                          device  bf16    max_nodes  min_weight  max_files
    "v1_8piece.yaml":         ("auto", "auto", 30000,     3,          None),
    "v1_8piece_cpu.yaml":     ("cpu",  "auto", 10000,     5,          64),
    "pilot_b4.yaml":          ("auto", "auto", 30000,     3,          None),
    "render_full.yaml":       ("auto", "auto", None,      3,          None),
    "sanity_3piece.yaml":     ("auto", "auto", 10000,     5,          None),
    "sanity_3piece_run.yaml": ("auto", "auto", 10000,     5,          None),
    "velocity_probe_cpu.yaml": ("cpu", "auto", 10000,     5,          256),
    "velocity_probe_s1_cpu.yaml": ("cpu", "auto", 10000,  5,          256),
    "velocity_w5_cpu.yaml":   ("cpu",  "auto", 10000,     5,          256),
    "velocity_lin_cpu.yaml":  ("cpu",  "auto", 10000,     5,          256),
    "velocity_lin_s1_cpu.yaml": ("cpu", "auto", 10000,    5,          256),
    "velocity_peak_cpu.yaml": ("cpu",  "auto", 10000,     5,          256),
    "velocity_std_cpu.yaml":  ("cpu",  "auto", 10000,     5,          256),
    "velocity_std_s1_cpu.yaml": ("cpu", "auto", 10000,    5,          256),
    "style_pc1_cpu.yaml":     ("cpu",  "auto", 10000,     5,          256),
    "style_pc1_gaps_cpu.yaml": ("cpu", "auto", 10000,     5,          256),
}


def test_the_table_covers_every_shipped_config():
    """A new config with no row here is a config nobody has resolved."""
    shipped = {p.name for p in CONFIGS.glob("*.yaml")}
    assert shipped == set(RESOLVED), (
        f"missing rows: {sorted(shipped - set(RESOLVED))}; "
        f"stale rows: {sorted(set(RESOLVED) - shipped)}"
    )


@pytest.mark.parametrize("name", sorted(RESOLVED))
def test_config_resolves_to_what_the_table_says(name):
    device, bf16, max_nodes, min_weight, max_files = RESOLVED[name]
    cfg = load_config(CONFIGS / name)
    assert cfg["train"].get("device") == device
    assert cfg["train"].get("bf16") == bf16
    assert cfg["subgraph"].get("max_nodes") == max_nodes
    assert cfg["subgraph"].get("min_weight") == min_weight
    assert cfg["data"].get("max_files") == max_files


@pytest.mark.parametrize("name", sorted(n for n in RESOLVED if n.endswith("_cpu.yaml")))
def test_a_config_named_cpu_runs_on_cpu(name):
    """Stated separately from the table because it is the invariant, not a
    value: the name is a promise about where the run happens, and `device_of`
    is what reads it. The table above could be edited to `auto` row by row;
    this cannot be satisfied except by actually resolving to CPU."""
    import torch

    from build import device_of

    assert device_of(load_config(CONFIGS / name)) == torch.device("cpu")


def test_each_velocity_arm_carries_exactly_its_own_change():
    """Phase A' is only interpretable if the arms differ by one thing each.

    A'1 raises the weight, A'2 replaces the sigmoid, A'3 scores at the peak --
    and each must inherit the *baseline* value of the other two, or the run
    that lands measures two changes and attributes them to one.
    """
    arms = {n: load_config(CONFIGS / f"velocity_{n}_cpu.yaml")
            for n in ("probe", "w5", "lin", "peak")}

    for name, cfg in arms.items():
        assert cfg["kit"]["velocity_head"] is True, f"{name} has no velocity head"
        assert cfg["data"]["max_files"] == 256, f"{name} trains on a different corpus"
        assert cfg["train"]["epochs"] == 12, f"{name} trains for a different time"
        assert cfg["train"].get("seed", 0) == 0, f"{name} starts from a different seed"

    weight = "velocity_weight"
    assert arms["probe"]["train"][weight] == 1.0
    assert arms["w5"]["train"][weight] == 5.0
    assert arms["lin"]["train"][weight] == 1.0, "A'2 must not also raise the weight"
    assert arms["peak"]["train"][weight] == 1.0, "A'3 must not also raise the weight"

    assert arms["lin"]["kit"]["velocity_activation"] == "linear"
    for other in ("probe", "w5", "peak"):
        assert "velocity_activation" not in arms[other]["kit"], (
            f"{other} changes the head's activation as well")

    assert arms["peak"]["train"]["velocity_peak_only"] == 0.95
    for other in ("probe", "w5", "lin"):
        assert "velocity_peak_only" not in arms[other]["train"], (
            f"{other} changes where velocity is scored as well")


def test_each_style_arm_carries_exactly_its_own_change():
    """The style-dial arms: the pC1 arm moves the genre tonic and nothing else;
    the gaps arm adds audio gaps to that and nothing else."""
    base = load_config(CONFIGS / "velocity_probe_cpu.yaml")
    pc1 = load_config(CONFIGS / "style_pc1_cpu.yaml")
    gaps = load_config(CONFIGS / "style_pc1_gaps_cpu.yaml")

    assert "targets" not in base["genre"] and base["genre"]["max_current"] == 0.5
    assert pc1["genre"]["targets"] == ["pC1"] and pc1["genre"]["max_current"] == 5.0
    assert gaps["genre"] == pc1["genre"]
    assert gaps["train"]["audio_gaps"] == {"fraction": 0.5, "min_ms": 250, "max_ms": 1000}
    assert "audio_gaps" not in pc1["train"] and "audio_gaps" not in base["train"]

    def strip(c):
        c = {k: dict(v) if isinstance(v, dict) else v for k, v in c.items()}
        c.pop("name")
        c["genre"] = {k: v for k, v in c["genre"].items() if k not in ("targets", "max_current")}
        c["train"].pop("audio_gaps", None)
        return c
    assert strip(pc1) == strip(base)
    assert strip(gaps) == strip(base)
