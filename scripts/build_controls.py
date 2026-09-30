"""Phase 1, random-wiring controls: one rewired connection table per seed, plus reports/controls_report.json.

    .venv\\Scripts\\python.exe scripts\\build_controls.py
Run after build_slice.py. About 1.1 million proposed swaps per seed.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.connectome.controls import check_control, drum_reach, rewire  # noqa: E402
from flybeats.connectome.store import control_path, load_slice, save_table  # noqa: E402
from tqdm import tqdm  # noqa: E402

cfg, paths = load_locked(), load_paths()
rc = cfg["connectome"]["random_controls"]
ears = cfg["connectome"]["ears"]
neurons, real = load_slice(paths["work_dir"])

report = {"real": {"drum_reach": drum_reach(neurons, real, ears)}, "controls": {}}
bar = tqdm(total=len(rc["seeds"]) * rc["swaps_per_edge"] * len(real), unit="swap", unit_scale=True, desc="swaps")
for seed in rc["seeds"]:
    t0 = time.time()
    bar.set_postfix_str(f"seed {seed}")
    ctrl, accepted = rewire(real, len(neurons), rc["swaps_per_edge"], seed, progress=bar.update)
    checks = check_control(neurons, real, ctrl, ears)
    checks["accepted_swaps"] = accepted
    save_table(ctrl, control_path(paths["work_dir"], seed))
    report["controls"][str(seed)] = checks
    bar.write(f"seed {seed}: {accepted} swaps accepted, {checks['surviving_real_connections_share']:.1%} of real "
          f"connections survive, unreachable drums: {checks['unreachable_drums'] or 'none'}  ({time.time() - t0:.0f} s)")

bar.close()
out = reports_dir() / "controls_report.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
