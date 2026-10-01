"""Power of the locked main win rule, simulated: the real network wins if its score beats all 5 controls' by more
than the seed spread. Stand-in metric: hit F1 averaged over kick, snare and closed hi-hat (the scorecard's beat
alignment does not exist yet).

Inputs from reports/feasibility.json (validation, 270 songs):
  headroom   ceiling minus floor = 0.087: the most any network could gain over a no-listening baseline
  noise      the gap's 95% interval (0.072-0.103) gives the song-sampling noise of a difference between two
             predictors on the same songs; per network sd = sd(difference) / sqrt(2), scaled to the 151 test songs
Unknown, so swept: the true spread between control seeds (sigma_c), and the real network's true advantage (delta).
"seed spread" is not defined further in locked.yaml: both readings are shown (sd of the 5 controls; their range).
Simulated, seeded (numpy seed 0); CPU, under a minute. Prints the tables; to keep them:
    .venv\\Scripts\\python.exe scripts\\power_estimate.py > reports\\power_estimate.txt
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import reports_dir  # noqa: E402

feas = json.loads((reports_dir() / "feasibility.json").read_text(encoding="utf-8"))
lo, hi = feas["three_drum"]["interval_95"]
headroom = feas["three_drum"]["gap"]
sd_diff_270 = (hi - lo) / (2 * 1.96)
sigma_m = sd_diff_270 / np.sqrt(2) * np.sqrt(270 / 151)
print(f"headroom {headroom:.3f}; song-sampling noise per network on 151 test songs: sd {sigma_m:.4f}")

rng = np.random.default_rng(0)
N = 100_000
deltas = np.array([0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.065, 0.087])
sigmas = [0.0, 0.005, 0.01, 0.02, 0.04]
for spread_name, spread in (("sd of the 5 controls", lambda c: c.std(1, ddof=1)),
                            ("range of the 5 controls", lambda c: c.max(1) - c.min(1))):
    print(f"\nseed spread read as the {spread_name}: chance the real network wins")
    print(f"{'true advantage':>15} {'(share of headroom)':>20}" + "".join(f"  sigma_c={s:<6}" for s in sigmas))
    for d in deltas:
        row = []
        for s in sigmas:
            ctrl = rng.normal(0, s, (N, 5)) + rng.normal(0, sigma_m, (N, 5))
            real = d + rng.normal(0, s, N) + rng.normal(0, sigma_m, N)
            row.append(np.mean(real > ctrl.max(1) + spread(ctrl)))
        print(f"{d:>15.3f} {d / headroom:>19.0%} " + "".join(f"  {p:>13.1%}" for p in row))

print("\nsmallest true advantage with an 80% chance of winning (sd reading):")
for s in sigmas:
    ok = None
    for d in np.linspace(0, 0.2, 401):
        ctrl = rng.normal(0, s, (20_000, 5)) + rng.normal(0, sigma_m, (20_000, 5))
        real = d + rng.normal(0, s, 20_000) + rng.normal(0, sigma_m, 20_000)
        if np.mean(real > ctrl.max(1) + ctrl.std(1, ddof=1)) >= 0.8:
            ok = d
            break
    print(f"  sigma_c {s:<6}: {ok:.3f} ({ok / headroom:.0%} of the headroom)" if ok is not None else f"  sigma_c {s}: > 0.2")
