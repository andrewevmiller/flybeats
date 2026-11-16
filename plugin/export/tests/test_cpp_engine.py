"""The C++ engine (plugin/engine, built into plugin/engine/build) against the numpy reference engine and PyTorch.
Skipped when the tools are not built. Builds nothing."""
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from flybrain.brainfile import read_brain
from flybrain.exporter import export
from flybrain.parity import PROB_ABS, RATE_ABS, RATE_REL, ambiguous_hits, compare, make_model, torch_outputs
from flybrain.recording import load_dump, load_recording
from flybrain.reference import hit_list, run_clip
from flybrain.resample import resample, resample_fast
from flybrain.synthetic import make_slice

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "plugin" / "engine" / "build"
RENDER = BUILD / "flybrain_render"
REPORTS = ROOT / "plugin" / "reports"

pytestmark = pytest.mark.skipif(not RENDER.exists(), reason="plugin/engine not built")


def render(brain_path, out_dir, audio=None, rate=16000, features=None, features32=None, block=512, double=True, controls=None,
           dump=True, record=False, preroll=0, extra=()):
    args = [str(RENDER), str(brain_path), "--block", str(block), "--preroll", str(preroll)]
    if double:
        args.append("--double")
    if audio is not None:
        p = out_dir / "audio.f32"
        p.write_bytes(np.asarray(audio, np.float32).tobytes())
        args += ["--audio", str(p), "--rate", str(rate)]
    if features is not None:
        p = out_dir / "ear.f64"
        p.write_bytes(np.ascontiguousarray(features, np.float64).tobytes())
        args += ["--features", str(p)]
    if features32 is not None:
        p = out_dir / "ear.f32"
        p.write_bytes(np.ascontiguousarray(features32, np.float32).tobytes())
        args += ["--features32", str(p)]
    if controls is not None:
        p = out_dir / "controls.json"
        p.write_text(json.dumps(controls))
        args += ["--controls", str(p)]
    if dump:
        args += ["--dump", str(out_dir / "frames.bin")]
    if record:
        args += ["--record", str(out_dir / "rec.bin")]
    args += list(extra)
    res = subprocess.run(args, capture_output=True, text=True, check=True)
    out = json.loads(res.stdout)
    out["dump"] = load_dump(out_dir / "frames.bin") if dump else None
    out["recording"] = load_recording(out_dir / "rec.bin") if record else None
    return out


def _check_vs_reference(dump, ref, threshold, label, require_rates_exact=False):
    n = min(len(dump["frame"]), len(ref["frame"]))
    assert n == len(ref["frame"]) == len(dump["frame"]), (label, len(dump["frame"]), len(ref["frame"]))
    rates_exact = np.array_equal(dump["rates"], ref["rates"])
    r_err = float(np.abs(dump["rates"] - ref["rates"]).max())
    p_err = float(np.abs(dump["prob"] - ref["prob"]).max())
    limit = RATE_ABS + RATE_REL * float(np.abs(ref["rates"]).max())
    amb = ambiguous_hits(ref["prob"], threshold)
    diff = set(hit_list(dump["hits"])) ^ set(hit_list(ref["hits"]))
    unexplained = diff - amb
    rep = dict(label=label, frames=n, rates_exact=rates_exact, rate_err=r_err, rate_limit=limit, prob_err=p_err,
               prob_exact=np.array_equal(dump["prob"], ref["prob"]), hits=len(hit_list(dump["hits"])),
               hits_ref=len(hit_list(ref["hits"])), hit_diffs=len(diff), unexplained=sorted(unexplained))
    assert r_err <= limit and p_err <= PROB_ABS and not unexplained, rep
    if require_rates_exact:
        assert rates_exact, rep
    return rep


def test_core_is_bit_exact_given_the_reference_ear(brain_small, tmp_path):
    b = brain_small[0]
    a = (np.random.default_rng(21).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    ref64 = run_clip(b, a, dtype=np.float64)
    out = render(b.path, tmp_path, features=ref64["ear"], double=True)
    rep = _check_vs_reference(out["dump"], ref64, b.meta["hit_threshold"], "core double", require_rates_exact=True)
    ref32 = run_clip(b, a, dtype=np.float32)
    out = render(b.path, tmp_path, features32=ref32["ear"], double=False)
    rep32 = _check_vs_reference(out["dump"], ref32, b.meta["hit_threshold"], "core float", require_rates_exact=True)
    print(rep, rep32)


def test_full_chain_matches_reference_and_pytorch(brain_small, tmp_path):
    b, model = brain_small[0], brain_small[1]
    a = (np.random.default_rng(22).standard_normal(16000 * 2) * 0.1).astype(np.float32)
    ref = run_clip(b, a, dtype=np.float64)
    out = render(b.path, tmp_path, audio=a, double=True)
    _check_vs_reference(out["dump"], ref, b.meta["hit_threshold"], "full chain double")
    tor = torch_outputs(model, a)
    rep = compare({k: out["dump"][k] for k in ("bands", "rates", "prob", "hits", "frame")}, tor)
    assert rep["passes"], rep
    # the events in the JSON equal the dump's hits, with velocities from the dump
    ev = {(f, d) for f, d, v, off in out["hits"]}
    assert ev == set(hit_list(out["dump"]["hits"]))
    for f, d, v, off in out["hits"]:
        assert abs(v - out["dump"]["vel"][f, d]) < 1e-6


def test_block_sizes_are_bit_identical(brain_small, tmp_path):
    b = brain_small[0]
    a = (np.random.default_rng(23).standard_normal(16000) * 0.1).astype(np.float32)
    base = render(b.path, tmp_path, audio=a, block=1)["dump"]
    for block in (64, 480, 4096):
        d = render(b.path, tmp_path, audio=a, block=block)["dump"]
        assert np.array_equal(d["rates"], base["rates"]) and np.array_equal(d["hits"], base["hits"]), block


@pytest.mark.parametrize("rate", [44100, 48000, 96000])
def test_host_rates(brain_small, tmp_path, rate):
    b = brain_small[0]
    x = (np.random.default_rng(rate).standard_normal(rate * 2) * 0.1).astype(np.float32)
    y = resample_fast(x, rate)
    ref = run_clip(b, y, dtype=np.float64)
    out = render(b.path, tmp_path, audio=x, rate=rate, double=True)
    rep = _check_vs_reference(out["dump"], ref, b.meta["hit_threshold"], f"host {rate}")
    assert out["latency_host_samples"] == 96 + round(2 * 80 * rate / 16000)
    print(rep)


def test_resampler_python_port_agrees_with_cpp():
    """The streaming loop port equals the vectorised port, and both the C++ (through the engine test data)."""
    x = (np.random.default_rng(1).standard_normal(44100 // 4) * 0.1).astype(np.float32)
    slow, fast = resample(x, 44100), resample_fast(x, 44100)
    assert len(slow) == len(fast) and np.abs(slow - fast).max() < 1e-7
    data = ROOT / "plugin" / "engine" / "tests" / "data"
    if (data / "host48000.f32").exists():
        want = np.frombuffer((data / "host48000_16k.f32").read_bytes(), np.float32)
        got = resample_fast(np.frombuffer((data / "host48000.f32").read_bytes(), np.float32), 48000)
        assert np.array_equal(got, want)        # the test data is this port's own output


def test_recorder_round_trip(brain_small, tmp_path):
    b = brain_small[0]
    a = (np.random.default_rng(24).standard_normal(16000 * 3) * 0.1).astype(np.float32)
    ref = run_clip(b, a, dtype=np.float64)
    out = render(b.path, tmp_path, audio=a, double=True, dump=False, record=True, controls={"gain_scale": 1.0})
    rec = out["recording"]
    assert rec["sidecar"]["frames_dropped"] == 0 and rec["sidecar"]["brain_sha256"] == b.sha256
    assert rec["sidecar"]["frames_written"] == len(ref["frame"]) == len(rec["frame"])
    assert len(rec["sidecar"]["control_log"]) == 1
    _check_vs_reference(rec, ref, b.meta["hit_threshold"], "recording")
    assert out["recorder_dropped"] == 0


def test_perturbed_run_matches_reference(brain_small, tmp_path):
    from flybrain.parity import run_perturbed
    b = brain_small[0]
    a = (np.random.default_rng(25).standard_normal(16000) * 0.1).astype(np.float32)
    ref = run_perturbed(b, a, "gain", scale=1.3)
    out = render(b.path, tmp_path, audio=a, double=True, controls={"gain_scale": 1.3})
    ref["frame"] = np.arange(len(ref["rates"]))
    _check_vs_reference(out["dump"], ref, b.meta["hit_threshold"], "gain 1.3")
    assert not out["dump"]["neutral"].any()
    base = render(b.path, tmp_path, audio=a, double=True)["dump"]
    assert base["neutral"].all() and not np.array_equal(base["rates"], out["dump"]["rates"])


def test_history_readout_brain(cfg, slice_small, tmp_path):
    import copy
    import torch
    neurons, edges = slice_small
    hcfg = copy.deepcopy(cfg)
    hcfg["model"]["readout_history"] = {"present": True, "blocks_ms": [50, 50, 50, 50], "profile": "per_drum",
                                        "init": "present_only", "velocity": "same"}
    model = make_model(hcfg, neurons, edges, seed=0)
    with torch.no_grad():
        model.p_hit[:, 1:] = torch.tensor([0.3, -0.2, 0.1, -0.4])
        model.p_vel[:, 1:] = 0.25
    path = tmp_path / "history.flybrain"
    export(model, neurons, edges, hcfg, path, name="history")
    b = read_brain(path)
    a = (np.random.default_rng(26).standard_normal(16000) * 0.1).astype(np.float32)
    ref = run_clip(b, a, dtype=np.float64)
    out = render(b.path, tmp_path, audio=a, double=True)
    _check_vs_reference(out["dump"], ref, b.meta["hit_threshold"], "history readout")
    tor = torch_outputs(model, a)
    rep = compare({k: out["dump"][k] for k in ("bands", "rates", "prob", "hits", "frame")}, tor)
    assert rep["passes"], rep


def _large_brain():
    j = REPORTS / "drift_synthetic_large.json"
    bp, ap = REPORTS / "drift_synthetic_large.flybrain", REPORTS / "drift_audio.npy"
    if not (j.exists() and bp.exists() and ap.exists()):
        pytest.skip("large drift brain not present (plugin/export/flybrain/drift.py makes it)")
    rep = json.loads(j.read_text())
    b = read_brain(bp)
    if rep["network"]["brain_sha256"] != b.sha256:
        pytest.skip("drift report and brain file disagree: the drift run is in progress")
    return b, np.load(ap), rep


@pytest.mark.parametrize("double", [True, False])
def test_large_brain_ten_minute_drift(tmp_path, double):
    """The real slice's size, 10 minutes: C++ against the reference engine, compared one minute at a time (the
    whole run is 3 GB of rates per engine, too much to hold twice), beside the reference's own PyTorch report."""
    from flybrain.recording import open_records
    from flybrain.reference import ReferenceEngine
    b, audio, drift = _large_brain()
    out = render(b.path, tmp_path, audio=audio, double=double, dump=False,
                 extra=["--dump", str(tmp_path / "frames.bin")])
    rec = open_records(tmp_path / "frames.bin")
    eng = ReferenceEngine(b, np.float64 if double else np.float32)
    thr = b.meta["hit_threshold"]
    per_minute, hits_cpp, hits_ref, unexplained, prob_err, rates_exact = [], 0, 0, [], 0.0, True
    limit = None
    n_frames = len(audio) // 80
    assert len(rec) == n_frames, (len(rec), n_frames)
    per_minute = [0.0] * 10
    cursor = 0
    for m in range(10):
        s, e = m * 12000, min((m + 1) * 12000, n_frames)
        done = eng.push(audio[s * 80:e * 80])
        if e == n_frames:
            done += eng.flush()
        done.sort(key=lambda f: f.index)
        assert [f.index for f in done] == list(range(cursor, cursor + len(done)))   # decided one frame late
        ref_rates = np.stack([f.rates for f in done]).astype(np.float64)
        ref_prob = np.stack([f.prob for f in done]).astype(np.float64)
        ref_hits = np.stack([f.hits for f in done])
        chunk = rec[cursor:cursor + len(done)]
        err = np.abs(chunk["rates"] - ref_rates).max(1)
        for f, x in zip(range(cursor, cursor + len(done)), err):
            per_minute[f // 12000] = max(per_minute[f // 12000], float(x))
        rates_exact = rates_exact and np.array_equal(chunk["rates"], ref_rates)
        prob_err = max(prob_err, float(np.abs(chunk["prob"] - ref_prob).max()))
        limit = max(limit or 0, RATE_ABS + RATE_REL * float(np.abs(ref_rates).max()))
        amb = ambiguous_hits(ref_prob, thr)
        got, want = set(hit_list(chunk["hits"].astype(bool))), set(hit_list(ref_hits))
        hits_cpp += len(got); hits_ref += len(want)
        unexplained += sorted((got ^ want) - amb)
        cursor += len(done)
        del ref_rates, chunk, done
    assert cursor == n_frames
    rep = dict(label=f"large {'double' if double else 'float'} 10 min", frames=n_frames, rates_exact=rates_exact,
               per_minute_rate_err=per_minute, rate_limit=limit, prob_err=prob_err, hits_cpp=hits_cpp, hits_ref=hits_ref,
               unexplained=[(int(f), int(d)) for f, d in unexplained], runaways=out["runaways"],
               pytorch_vs_reference={k: drift[k] for k in ("rate_err", "prob_err", "hits_torch", "hits_engine", "passes")})
    del rec
    (tmp_path / "frames.bin").unlink()
    (REPORTS / f"cpp_drift_large_{'double' if double else 'float'}.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(rep)
    assert max(per_minute) <= limit and prob_err <= PROB_ABS and not unexplained, rep
