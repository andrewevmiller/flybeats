"""Check 1 (diagnostic, CPU only): all six networks on the validation split, from reports/win_validation_full.json.
For each network: beat alignment (scored), hit count, balance b, and timing without b (raw beat alignment divided by b),
plus whether score rank follows hit-count rank across networks. Random timing scores 0.5 on the timing measure.
Writes reports/timing_check_all.json and prints a table. No network runs.
    .venv\\Scripts\\python.exe scripts\\timing_check_all.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
nets = json.loads((REP / "win_validation_full.json").read_text(encoding="utf-8"))["networks"]

table, songs_zero = {}, {}
for name, info in nets.items():
    songs = info["songs"]
    score, n, b, timing, zero = [], [], [], [], 0
    for s, v in songs.items():
        if v["m"] == 0 or v["n"] == 0:
            zero += 1  # b is 0, so timing without b is undefined: skipped below
        if v["scores"]["beat_alignment"] is not None:
            score.append(v["scores"]["beat_alignment"])
        n.append(v["n"])
        b.append(v["b"])
        if v["b"] > 0:
            timing.append(v["raw"]["beat_alignment"] / v["b"])
    songs_zero[name] = zero
    table[name] = {
        "beat_alignment_mean": float(np.mean(score)),
        "hits_mean": float(np.mean(n)),
        "balance_mean": float(np.mean(b)),
        "timing_without_b_mean": float(np.mean(timing)),
        "timing_without_b_sd": float(np.std(timing, ddof=1)),
        "songs_with_zero_hits_or_original": zero,
    }

names = list(table)
score_m = [table[k]["beat_alignment_mean"] for k in names]
hits_m = [table[k]["hits_mean"] for k in names]
bal_m = [table[k]["balance_mean"] for k in names]
timing_m = [table[k]["timing_without_b_mean"] for k in names]
rho_score_hits = spearmanr(score_m, hits_m).correlation
rho_score_bal = spearmanr(score_m, bal_m).correlation
rho_score_timing = spearmanr(score_m, timing_m).correlation

out = {"networks": table, "rank_correlation_across_networks": {
    "beat_alignment_vs_hits": float(rho_score_hits),
    "beat_alignment_vs_balance": float(rho_score_bal),
    "beat_alignment_vs_timing_without_b": float(rho_score_timing)},
    "note": "six networks only: rank correlations are indicative, not significance tests"}
(REP / "timing_check_all.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

print(f"{'network':<12}{'beat_al':>9}{'hits':>8}{'balance':>9}{'timing_noB':>12}{'sd':>7}{'zero':>6}")
for k in names:
    t = table[k]
    print(f"{k:<12}{t['beat_alignment_mean']:>9.3f}{t['hits_mean']:>8.1f}{t['balance_mean']:>9.3f}"
          f"{t['timing_without_b_mean']:>12.4f}{t['timing_without_b_sd']:>7.3f}{t['songs_with_zero_hits_or_original']:>6d}")
print(f"rank corr (6 networks): score vs hits {rho_score_hits:.2f}, vs balance {rho_score_bal:.2f}, "
      f"vs timing-without-b {rho_score_timing:.2f}")
print(f"wrote {REP / 'timing_check_all.json'}")
