"""Phase 1 gate: compare reports/slice_report.json with the numbers from the Sep 25 run, and check
that five control tables exist. Prints one OK/DIFF line per check; exit code = number of DIFFs.

    .venv\\Scripts\\python.exe scripts\\check_slice_gate.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.connectome.store import control_path  # noqa: E402

r = json.loads((reports_dir() / "slice_report.json").read_text(encoding="utf-8"))
wing, hind = r["wing_motor_neurons"], r["hind_leg_motor_neurons"]
checks = [
    ("Annotated neurons", r["annotated_neurons"], 166_700),
    ("Connections >= 5 synapses", r["connections_ge_min_synapses"], 6_242_118),
    ("Usable ears (total, left, right)", (r["usable_ears"], r["usable_left"], r["usable_right"]), (93, 75, 18)),
    ("Ears after mirroring", r["ears_after_mirroring"], 150),
    ("Slice neurons", r["slice_neurons"], 3_302),
    ("Slice connections", r["slice_connections"], 108_593),
    ("Left / right / midline", (r["sides"]["L"], r["sides"]["R"], r["sides"]["M"]), (1_595, 1_645, 62)),
    ("Roles exc/inh/motor/mod/unknown", tuple(r["roles"].values()), (1_863, 1_228, 189, 8, 14)),
    ("Motor wing / hind leg / middle leg",
     (r["motor_groups"].get("wing"), r["motor_groups"].get("hind_leg"), r["motor_groups"].get("middle_leg")), (67, 120, 2)),
    ("Wing MNs (count, types, same types L/R)",
     (wing["count"], wing["types"], wing["types_left"] == wing["types_right"] == wing["types"]), (67, 25, True)),
    ("Hind-leg MNs (count, of, types)", (hind["count"], hind["of_annotated"], hind["types"]), (120, 130, 26)),
    ("Song neurons", r["song_neurons"],
     {"AMMC-A1": "5/6", "aPN1": "16/17", "vPN1": "8/11", "dPR1": "2/2", "pC2l": "4/37"}),
    # The model counts each of the 22 untyped neurons as its own type (CHANGELOG, Sep 29); the Sep 25
    # gate numbers lumped them. Both are checked, so either rule drifting shows up.
    ("Cell types / type pairs (model)",
     (r["types_untyped_separate"]["cell_types"], r["types_untyped_separate"]["type_pairs"]), (1_224, 41_186)),
    ("Cell types / type pairs (untyped lumped)",
     (r["types_untyped_lumped"]["cell_types"], r["types_untyped_lumped"]["type_pairs"]), (1_203, 41_029)),
    ("Largest eigenvalue", r["largest_eigenvalue"], 730.6),
]

cfg, paths = load_locked(), load_paths()
seeds = cfg["connectome"]["random_controls"]["seeds"]
have = [s for s in seeds if control_path(paths["work_dir"], s).exists()]
checks.append(("Control tables", len(have), 5))

diffs = 0
for name, got, want in checks:
    ok = got == want
    diffs += not ok
    print(f"{'OK  ' if ok else 'DIFF'}  {name:<36} got {got}" + ("" if ok else f", expected {want}"))
print("\nGate numbers match." if not diffs else f"\n{diffs} difference(s): find the rule that differs before moving on.")
print("Also run: .venv\\Scripts\\python.exe -m pytest tests\\test_slice.py")
sys.exit(diffs)
