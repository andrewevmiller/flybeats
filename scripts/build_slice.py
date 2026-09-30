"""Phase 1, steps 1-6: build the connectome slice from locked.yaml and write reports/slice_report.json.

    .venv\\Scripts\\python.exe scripts\\build_slice.py
Needs about 8 GB of RAM while the 152-million-row weights file is filtered.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.connectome.slice import BUILD_STAGES, build_slice  # noqa: E402
from tqdm import tqdm  # noqa: E402
from flybeats.connectome.store import save_table, slice_paths  # noqa: E402

t0 = time.time()
cfg, paths = load_locked(), load_paths()
bar = tqdm(total=len(BUILD_STAGES), unit="stage")
started = [False]


def progress(stage):
    if started[0]:
        bar.update(1)
    started[0] = True
    bar.set_description(stage)


neurons, edges, report = build_slice(cfg, paths, progress)
bar.update(1)
bar.close()
neurons_path, edges_path = slice_paths(paths["work_dir"])
save_table(neurons, neurons_path)
save_table(edges, edges_path)
out = reports_dir() / "slice_report.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
print(f"\nwrote {neurons_path}\n      {edges_path}\n      {out}   ({time.time() - t0:.0f} s)")
