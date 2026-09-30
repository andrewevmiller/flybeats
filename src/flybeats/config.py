"""Load the locked settings (config/locked.yaml) and the local data paths (config/paths.local.yaml)."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_locked(path=None):
    path = Path(path or ROOT / "config" / "locked.yaml")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_paths(path=None):
    """malecns_dir, slakh_dir, work_dir as Paths. work_dir is created if missing."""
    path = Path(path or ROOT / "config" / "paths.local.yaml")
    with open(path, encoding="utf-8") as f:
        paths = {k: Path(v) for k, v in yaml.safe_load(f).items()}
    paths["work_dir"].mkdir(parents=True, exist_ok=True)
    return paths


def reports_dir():
    d = ROOT / "reports"
    d.mkdir(exist_ok=True)
    return d
