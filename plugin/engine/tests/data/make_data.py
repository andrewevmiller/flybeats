"""Makes the engine tests' data from the Python side (brain files and reference-engine dumps).

    python -I plugin/engine/tests/data/make_data.py
Writes into this directory (git-ignored except this script). The C++ tests skip what is missing.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "plugin" / "export"))

from flybeats.config import load_locked  # noqa: E402
from flybrain.brainfile import read_brain, write_brain  # noqa: E402
from flybrain.exporter import export  # noqa: E402
from flybrain.parity import PERTURBATIONS, make_model, run_perturbed  # noqa: E402
from flybrain.reference import run_clip  # noqa: E402
from flybrain.resample import resample_fast  # noqa: E402
from flybrain.synthetic import make_slice  # noqa: E402

cfg = load_locked()
neurons, edges = make_slice(seed=1)
model = make_model(cfg, neurons, edges, seed=0)
export(model, neurons, edges, cfg, HERE / "small.flybrain", name="synthetic-small")
b = read_brain(HERE / "small.flybrain")

# a brain with the history readout on (the same weights plus profiles that use the past)
import copy  # noqa: E402
import torch  # noqa: E402
hcfg = copy.deepcopy(cfg)
hcfg["model"]["readout_history"] = {"present": True, "blocks_ms": [50, 50, 50, 50], "profile": "per_drum",
                                    "init": "present_only", "velocity": "same"}
hmodel = make_model(hcfg, neurons, edges, seed=0)
hmodel.load_state_dict(model.state_dict(), strict=False)
with torch.no_grad():
    hmodel.p_hit[:, 1:] = torch.tensor([0.3, -0.2, 0.1, -0.4])
    hmodel.p_vel[:, 1:] = 0.25
export(hmodel, neurons, edges, hcfg, HERE / "history.flybrain", name="synthetic-history")
hb = read_brain(HERE / "history.flybrain")

# refused files: bad index and bad alpha (write_brain does not range-check, the readers do)
a = dict(b.arrays); a["w_indices"] = a["w_indices"].copy(); a["w_indices"][0] = b.n + 5
write_brain(HERE / "bad_index.flybrain", b.meta, a)
a = dict(b.arrays); a["alpha"] = a["alpha"].copy(); a["alpha"][3] = 0.0
write_brain(HERE / "bad_alpha.flybrain", b.meta, a)

rng = np.random.default_rng(5)


def dump(name, r):
    (HERE / f"{name}_ear.f64").write_bytes(np.ascontiguousarray(r["ear"], np.float64).tobytes())
    (HERE / f"{name}_bands.f64").write_bytes(np.ascontiguousarray(r["bands"], np.float64).tobytes())
    (HERE / f"{name}_rates.f64").write_bytes(np.ascontiguousarray(r["rates"], np.float64).tobytes())
    (HERE / f"{name}_prob.f64").write_bytes(np.ascontiguousarray(r["prob"], np.float64).tobytes())
    (HERE / f"{name}_vel.f64").write_bytes(np.ascontiguousarray(r["vel"], np.float64).tobytes())
    (HERE / f"{name}_hits.u8").write_bytes(np.ascontiguousarray(r["hits"], np.uint8).tobytes())


audio = (rng.standard_normal(16000 * 2) * 0.1).astype(np.float32)
(HERE / "audio16k.f32").write_bytes(audio.tobytes())
r64 = run_clip(b, audio, dtype=np.float64)
dump("ref64", r64)
r32 = run_clip(b, audio, dtype=np.float32)
(HERE / "ref32_ear.f32").write_bytes(np.ascontiguousarray(r32["ear"], np.float32).tobytes())
(HERE / "ref32_rates.f32").write_bytes(np.ascontiguousarray(r32["rates"], np.float32).tobytes())
(HERE / "ref32_prob.f32").write_bytes(np.ascontiguousarray(r32["prob"], np.float32).tobytes())
(HERE / "ref32_hits.u8").write_bytes(np.ascontiguousarray(r32["hits"], np.uint8).tobytes())
dump("hist64", run_clip(hb, audio, dtype=np.float64))

# host-rate audio: noise at the host rate, resampled by the numpy port to 16 kHz for the reference
for rate in (44100, 48000, 96000):
    x = (rng.standard_normal(rate * 2) * 0.1).astype(np.float32)
    x = np.convolve(x, np.ones(3, np.float32) / 3, mode="same").astype(np.float32)
    (HERE / f"host{rate}.f32").write_bytes(x.tobytes())
    y = resample_fast(x, rate)
    (HERE / f"host{rate}_16k.f32").write_bytes(y.tobytes())
    dump(f"host{rate}", run_clip(b, y, dtype=np.float64))

# perturbations: the controls as the render tool reads them, and the reference output
kinds = list(PERTURBATIONS) + [("unsilence", dict(targets=np.flatnonzero(neurons.silenced.to_numpy())[:2].tolist(), sign=1)),
                               ("inject", dict(neuron=int(np.flatnonzero(neurons.type.isna().to_numpy())[0]), current=2.0, start=50, frames=30, t_len=400))]
for kind, kw in kinds:
    if kind == "lesion": c = {"lesion": kw["targets"]}
    elif kind == "deafen": c = {"ear_mute": kw["ears"]}
    elif kind == "gain": c = {"gain_scale": kw["scale"]}
    elif kind == "tau": c = {"tau_scale": kw["scale"]}
    elif kind == "sign_flip": c = {"sign_flip": kw["targets"]}
    elif kind == "band_offset": c = {"band_offset": [float(v) for v in kw["offset"]]}
    elif kind == "unsilence": c = {"unsilence": [[int(i), kw["sign"]] for i in kw["targets"]]}
    elif kind == "inject": c = {"inject": [[kw["neuron"], kw["current"], kw["start"], kw["frames"]]]}
    (HERE / f"pert_{kind}.json").write_text(json.dumps(c))
    r = run_perturbed(b, audio, kind, **kw)
    (HERE / f"pert_{kind}_rates.f64").write_bytes(np.ascontiguousarray(r["rates"], np.float64).tobytes())
    (HERE / f"pert_{kind}_hits.u8").write_bytes(np.ascontiguousarray(r["hits"], np.uint8).tobytes())
json.dump({"kinds": [k for k, _ in kinds], "frames": int(len(audio) // 80), "n": b.n, "n_mn": int(len(b["mn_idx"])),
           "n_drums": int(b.meta["n_drums"]), "history_n_mn": int(len(hb["mn_idx"]))}, open(HERE / "meta.json", "w"))
print("wrote test data to", HERE)
