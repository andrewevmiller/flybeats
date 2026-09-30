"""Phase 1, random-wiring controls: one rewired connection table per seed, plus reports/controls_report.json.

    .venv\\Scripts\\python.exe scripts\\build_controls.py
Run after build_slice.py. About 1.1 million proposed swaps per seed. The method is locked.yaml's
connectome.random_controls.method: degree_class_reciprocity_preserving_swaps (prereg-v4: swaps within neuron classes,
reciprocal pairs kept) or degree_preserving_swaps (prereg-v3 and before: swaps across the whole slice).
Control tables that already exist are moved, not overwritten, to slice/controls/replaced-<date and time>/.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.connectome.controls import (CLASSES, check_control, control_classes, drum_reach,  # noqa: E402
                                          reciprocal_mask, rewire, rewire_classes)
from flybeats.connectome.slice import largest_eigenvalue  # noqa: E402
from flybeats.connectome.store import control_path, load_slice, save_table  # noqa: E402
from tqdm import tqdm  # noqa: E402

cfg, paths = load_locked(), load_paths()
rc = cfg["connectome"]["random_controls"]
ears = cfg["connectome"]["ears"]
neurons, real = load_slice(paths["work_dir"])
classes = None
if rc["method"] == "degree_class_reciprocity_preserving_swaps":
    if rc.get("classes") != CLASSES:
        sys.exit(f"locked.yaml classes {rc.get('classes')} are not the ones this code builds: {CLASSES}")
    classes = control_classes(neurons, real, ears)
elif rc["method"] != "degree_preserving_swaps":
    sys.exit(f"unknown random_controls.method {rc['method']!r}")

old = [control_path(paths["work_dir"], s) for s in rc["seeds"] if control_path(paths["work_dir"], s).exists()]
if old:
    keep = old[0].parent / f"replaced-{time.strftime('%Y%m%d-%H%M%S')}"
    keep.mkdir()
    for p in old:
        p.rename(keep / p.name)
    print(f"moved {len(old)} existing control tables to {keep}")

report = {"version": cfg["version"], "method": rc["method"],
          "real": {"drum_reach": drum_reach(neurons, real, ears), "largest_eigenvalue": largest_eigenvalue(neurons, real),
                   "reciprocal_share": float(reciprocal_mask(real, len(neurons)).mean())},
          "controls": {}}
if classes is not None:
    report["classes"] = {name: int((classes == i).sum()) for i, name in enumerate(CLASSES)}
bar = tqdm(total=len(rc["seeds"]) * rc["swaps_per_edge"] * len(real), unit="swap", unit_scale=True, desc="swaps")
for seed in rc["seeds"]:
    t0 = time.time()
    bar.set_postfix_str(f"seed {seed}")
    if classes is None:
        ctrl, accepted = rewire(real, len(neurons), rc["swaps_per_edge"], seed, progress=bar.update)
    else:
        ctrl, accepted = rewire_classes(real, classes, rc["swaps_per_edge"], seed, progress=bar.update)
    checks = check_control(neurons, real, ctrl, ears, classes)
    checks["accepted_swaps"] = accepted
    checks["largest_eigenvalue"] = largest_eigenvalue(neurons, ctrl)
    save_table(ctrl, control_path(paths["work_dir"], seed))
    report["controls"][str(seed)] = checks
    bar.write(f"seed {seed}: {accepted} swaps accepted, {checks['surviving_real_connections_share']:.1%} of real "
              f"connections survive, largest eigenvalue {checks['largest_eigenvalue']:.0f} "
              f"(real {report['real']['largest_eigenvalue']:.0f}), unreachable drums: "
              f"{checks['unreachable_drums'] or 'none'}  ({time.time() - t0:.0f} s)")

bar.close()
out = reports_dir() / "controls_report.json"
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
