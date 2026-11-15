"""Paired per-song comparison of control 1 against the real network on the validation split (diagnostic only).
Reads reports/win_validation_full.json (written by scripts/score_win.py --split validation --set full); no network runs.
Writes reports/per_song_control1_vs_real.json and prints a summary.
    .venv\\Scripts\\python.exe scripts\\per_song_compare.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REP = ROOT / "reports"
data = json.loads((REP / "win_validation_full.json").read_text(encoding="utf-8"))["networks"]
A, B = "control_1", "real"
sa, sb = data[A]["songs"], data[B]["songs"]
rng = np.random.default_rng(0)

rows = []
for song in sorted(set(sa) & set(sb)):
    x, y = sa[song], sb[song]
    ca, cb = x["scores"]["beat_alignment"], y["scores"]["beat_alignment"]
    if ca is None or cb is None:
        continue  # left out of the facet by the scorecard rule
    rows.append({"song": song, "c1": ca, "real": cb, "diff": ca - cb,
                 "n_c1": x["n"], "m": x["m"], "b_c1": x["b"],
                 "n_real": y["n"], "b_real": y["b"],
                 "align_raw_c1": x["raw"]["beat_alignment"], "align_raw_real": y["raw"]["beat_alignment"]})

d = np.array([r["diff"] for r in rows])
n = len(d)
boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(4000)])
lo, hi = np.percentile(boot, [2.5, 97.5])
order = np.argsort(-d)
top10 = d[order[:10]].sum() / d.sum() if d.sum() else float("nan")

# hit-count balance (the factor b that scales beat alignment)
b1 = np.array([r["b_c1"] for r in rows])
br = np.array([r["b_real"] for r in rows])
# alignment without the balance factor, where the network has hits
ok = (b1 > 0) & (br > 0)
a1 = np.array([r["align_raw_c1"] / r["b_c1"] for r in rows if r["b_c1"] > 0])
ar = np.array([r["align_raw_real"] / r["b_real"] for r in rows if r["b_real"] > 0])

summary = {
    "songs_compared": int(n),
    "control1_mean": float(np.mean([r["c1"] for r in rows])),
    "real_mean": float(np.mean([r["real"] for r in rows])),
    "mean_diff": float(d.mean()),
    "sd_diff": float(d.std(ddof=1)),
    "bootstrap_95_mean_diff": [float(lo), float(hi)],
    "songs_control1_higher": int((d > 0).sum()),
    "songs_real_higher": int((d < 0).sum()),
    "median_diff": float(np.median(d)),
    "top10_songs_share_of_total_diff": float(top10),
    "mean_balance_control1": float(b1.mean()),
    "mean_balance_real": float(br.mean()),
    "alignment_without_balance_control1_mean": float(a1.mean()) if len(a1) else None,
    "alignment_without_balance_real_mean": float(ar.mean()) if len(ar) else None,
    "mean_hits_control1": float(np.mean([r["n_c1"] for r in rows])),
    "mean_hits_real": float(np.mean([r["n_real"] for r in rows])),
    "mean_original_hits": float(np.mean([r["m"] for r in rows])),
}
out = {"summary": summary, "songs": rows}
(REP / "per_song_control1_vs_real.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
for k, v in summary.items():
    print(f"{k}: {v}")
print(f"wrote {REP / 'per_song_control1_vs_real.json'}")
