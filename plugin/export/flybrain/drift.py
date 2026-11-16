"""P1/P2 drift test: a synthetic network of the real slice's size, a long clip, reference engine against PyTorch.

    python -I plugin/export/flybrain/drift.py --minutes 10 --out plugin/reports/drift_synthetic_large.json

Reports the parity contract (parity.compare) on the whole clip and the rate error per minute, so growth shows.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "plugin" / "export"))

from flybeats.config import load_locked  # noqa: E402
from flybrain.brainfile import read_brain  # noqa: E402
from flybrain.exporter import export  # noqa: E402
from flybrain.parity import compare, make_model, torch_outputs  # noqa: E402
from flybrain.reference import run_clip  # noqa: E402
from flybrain.synthetic import make_slice  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=10)
ap.add_argument("--large", action="store_true", default=True)
ap.add_argument("--dtype", choices=["f32", "f64"], default="f64")
ap.add_argument("--out", default=str(ROOT / "plugin" / "reports" / "drift_synthetic_large.json"))
ap.add_argument("--brain", default=None, help="write the exported brain here too (for the C++ engine)")
ap.add_argument("--audio", default=None, help="write the clip as float32 .npy here too")
args = ap.parse_args()

cfg = load_locked()
t0 = time.time()
neurons, edges = make_slice(seed=2, large=args.large)
model = make_model(cfg, neurons, edges, seed=2)
path = Path(args.brain or (ROOT / "plugin" / "reports" / "drift_synthetic_large.flybrain"))
digest, meta = export(model, neurons, edges, cfg, path, name="synthetic-large")
b = read_brain(path)
print(f"network: {b.n} neurons, {b.meta['n_edges']} connections, exported in {time.time() - t0:.1f}s", flush=True)
rng = np.random.default_rng(11)
n = int(args.minutes * 60 * 16000)
# music-like: coloured noise with slow amplitude modulation, so the input is never stationary
audio = rng.standard_normal(n).astype(np.float32)
audio = np.convolve(audio, np.ones(6, np.float32) / 6, mode="same")
env = 0.5 + 0.5 * np.sin(2 * np.pi * np.arange(n) / (16000 * 2.3)) ** 2
audio = (audio * env * 0.1 / audio.std()).astype(np.float32)
if args.audio:
    np.save(args.audio, audio)
t1 = time.time()
tor = torch_outputs(model, audio)
print(f"pytorch: {time.time() - t1:.1f}s", flush=True)
t2 = time.time()
# stream the reference engine and compare frame by frame, keeping nothing but the running errors
from flybrain.parity import ambiguous_hits, PROB_ABS, RATE_ABS, RATE_REL, BAND_ABS  # noqa: E402
from flybrain.reference import ReferenceEngine  # noqa: E402
eng = ReferenceEngine(b, np.float64 if args.dtype == "f64" else np.float32)
per_minute = []
rate_err = band_err = prob_err = 0.0
hits_engine = set()
block = 16000
done = []
def take(frames):
    global rate_err, band_err, prob_err
    for f in frames:
        i = f.index
        m = i // 12000
        while len(per_minute) <= m:
            per_minute.append({"minute": len(per_minute), "rate_err": 0.0, "prob_err": 0.0, "hits_torch": 0})
        e_r = float(np.abs(f.rates - tor["rates"][i]).max())
        e_p = float(np.abs(f.prob - tor["prob"][i]).max())
        e_b = float(np.abs(f.bands - tor["bands"][i]).max())
        rate_err, prob_err, band_err = max(rate_err, e_r), max(prob_err, e_p), max(band_err, e_b)
        pm = per_minute[m]
        pm["rate_err"], pm["prob_err"] = max(pm["rate_err"], e_r), max(pm["prob_err"], e_p)
        for d in np.flatnonzero(f.hits):
            hits_engine.add((int(i), int(d)))
for s in range(0, len(audio), block):
    take(eng.push(audio[s:s + block]))
take(eng.flush())
print(f"reference engine ({args.dtype}): {time.time() - t2:.1f}s", flush=True)
tor_hits = set(tor["hits"])
for f, d in tor_hits:
    per_minute[f // 12000]["hits_torch"] += 1
amb = ambiguous_hits(tor["prob"], 0.5)
diff = hits_engine ^ tor_hits
r_lim = RATE_ABS + RATE_REL * float(np.abs(tor["rates"]).max())
rep = {"frames": int(tor["rates"].shape[0]), "rate_err": rate_err, "rate_limit": r_lim, "band_err": band_err,
       "prob_err": prob_err, "hits_torch": len(tor_hits), "hits_engine": len(hits_engine), "hit_differences": len(diff),
       "ambiguous_in_torch": len(amb), "unexplained_hit_differences": sorted(diff - amb)}
rep["passes"] = rate_err <= r_lim and band_err <= BAND_ABS and prob_err <= PROB_ABS and not (diff - amb)
rep["per_minute"] = per_minute
rep["network"] = {"neurons": b.n, "edges": b.meta["n_edges"], "brain_sha256": digest}
rep["minutes"] = args.minutes
rep["dtype"] = args.dtype
rep["seconds"] = {"pytorch": round(t2 - t1, 1), "reference": round(time.time() - t2, 1)}
Path(args.out).write_text(json.dumps(rep, indent=1) + "\n")
print(json.dumps({k: v for k, v in rep.items() if k != "per_minute"}, indent=1))
print("PASS" if rep["passes"] else "FAIL")
