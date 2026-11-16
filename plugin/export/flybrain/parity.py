"""The parity harness: the reference engine (and, through its recordings, the C++ engine) against PyTorch.

Contract (plugin/docs/P0-decisions.md, "Parity tolerances"):
  rates     max |r_engine - r_torch| <= RATE_ABS + RATE_REL * max |r_torch|   over every neuron and frame
  bands     max |b_engine - b_torch| <= BAND_ABS
  prob      max |p_engine - p_torch| <= PROB_ABS
  hits      the decoded hit sets are identical, except hits that PyTorch itself decides by less than TIE_EPS
            (probability within TIE_EPS of the threshold or of a neighbouring frame), which are reported, not failed
  drift     the same over a 10-minute clip, so rounding differences do not grow
Each perturbation has a PyTorch equivalent here (`torch_perturbed`), formed by changing the FlyNet or its
input exactly as the control block says, so "the perturbation matches PyTorch" is a test, not a reading.
"""
import copy
import math

import numpy as np
import torch

from .reference import Controls, hit_list, run_clip

RATE_ABS, RATE_REL = 1e-6, 1e-5
BAND_ABS = 1e-5
PROB_ABS = 1e-5
TIE_EPS = 1e-5


def make_model(cfg, neurons, edges, seed=0, band_stats=None, lively=True):
    """A FlyNet on a slice, with parameters moved off their initial values so the readout produces hits (an
    untrained network at initial_gain barely fires). Deterministic in `seed`."""
    from flybeats.model.network import FlyNet
    rng = np.random.default_rng(seed)
    if band_stats is None:
        band_stats = (rng.normal(3, 1, 16).tolist(), rng.uniform(0.5, 2, 16).tolist())
    torch.manual_seed(seed)
    model = FlyNet(neurons, edges, cfg, band_stats=band_stats)
    if lively:
        with torch.no_grad():
            g = torch.Generator().manual_seed(seed)
            # "readout": only the readout moves, so the network itself stays at its critical gain (stable);
            # True: every parameter moves, which can push a large network past its cliff (the first drift run did)
            names = {"w_hit", "u_vel", "c_hit", "e_vel", "p_hit", "p_vel"} if lively == "readout" else None
            for name, p in model.named_parameters():
                if names is None or name in names:
                    p.add_(0.3 * torch.randn(p.shape, generator=g))
            model.c_hit.fill_(-2.0)
            model.mn_std.fill_(5e-2)      # 1e-2 saturated two drums at p = 1 (red team, item 2)
    return model.eval()


@torch.no_grad()
def torch_outputs(model, audio16k, ear_mask=None, extra_drive=None, frame_bias=None):
    """FlyNet on a clip: bands (T,16), rates (T,N), prob (T,D), vel (T,D), hits list.
    ear_mask (n_ears,): multiplies the ear input (Deafen an Ear). extra_drive (T, N): current added to each
    neuron each frame (Tickle / current injection), applied by stepping one frame at a time with the bias
    changed, which is exact because bias enters the update as `+ bias + drive`."""
    model = model.cpu().eval()
    a = torch.as_tensor(np.asarray(audio16k, np.float32))[None]
    bands = model.bands(a)
    ear_in = (bands * model.band_gain + model.band_offset)[:, :, model.ear_band]
    if ear_mask is not None:
        ear_in = ear_in * torch.as_tensor(np.asarray(ear_mask, np.float32))
    if extra_drive is None:
        out, _ = model.run(ear_in, record=True)
    else:
        out = _run_with_drive(model, ear_in, torch.as_tensor(np.asarray(extra_drive, np.float32)))
    prob = torch.sigmoid(out["hit_logits"][0]).numpy()
    return {"bands": bands[0].numpy(), "rates": out["rates"][0].numpy(), "mn": out["mn_rates"][0].numpy(),
            "prob": prob, "vel": out["vel"][0].numpy(), "hits": torch_hits(prob, model_threshold(model))}


def model_threshold(model):
    return getattr(model, "hit_threshold", 0.5)


def _run_with_drive(model, ear_in, drive):
    """FlyNet.run with an extra per-frame, per-neuron current: the frame loop of network.py, verbatim, plus
    `+ drive[t]`. Only for the injection test; the readout is the model's own."""
    from torch.nn import functional as F
    b, t_len, _ = ear_in.shape
    v = model.init_state(b, ear_in.device, ear_in.dtype)
    w_t = model.weight_matrix().t()
    alpha = (model.dt_ms / model.tau_ms())[model.type_idx]
    bias = model.bias[model.type_idx]
    r = F.relu(v) * model.alive
    rates, mn = [], []
    for t in range(t_len):
        d = torch.zeros(b, model.n).index_copy(1, model.ear_idx, ear_in[:, t]) + drive[t]
        v = v + alpha * (-v + r @ w_t + bias + d)
        r = F.relu(v) * model.alive
        rates.append(r)
        mn.append(r[:, model.mn_idx])
    mn = torch.stack(mn, 1)
    return {"rates": torch.stack(rates, 1), "mn_rates": mn, **model.readout(mn)}


def torch_hits(prob, threshold):
    from flybeats.model.loss import peaks
    return sorted({(int(f), d) for d in range(prob.shape[1]) for f in peaks(prob[:, d], threshold)})


def ambiguous_hits(prob, threshold, eps=TIE_EPS):
    """Hits PyTorch decides by a margin under eps: a tolerance-level difference could flip them."""
    out = set()
    t_len, nd = prob.shape
    for d in range(nd):
        p = prob[:, d]
        for f in range(t_len):
            near_thr = abs(p[f] - threshold) < eps
            near_left = f > 0 and abs(p[f] - p[f - 1]) < eps
            near_right = f + 1 < t_len and abs(p[f] - p[f + 1]) < eps
            if (near_thr or near_left or near_right) and p[f] >= threshold - eps:
                out.add((f, d))
    return out


def compare(ref, tor, threshold=0.5):
    """Reference-engine output (run_clip) against torch_outputs. Returns a report dict with 'passes'."""
    n = min(len(ref["frame"]), tor["rates"].shape[0])
    r_err = float(np.abs(ref["rates"][:n] - tor["rates"][:n]).max())
    r_lim = RATE_ABS + RATE_REL * float(np.abs(tor["rates"]).max())
    b_err = float(np.abs(ref["bands"][:n] - tor["bands"][:n]).max())
    p_err = float(np.abs(ref["prob"][:n] - tor["prob"][:n]).max())
    ref_hits, tor_hits = set(hit_list(ref["hits"][:n])), set(h for h in tor["hits"] if h[0] < n)
    amb = ambiguous_hits(tor["prob"][:n], threshold)
    diff = (ref_hits ^ tor_hits)
    unexplained = diff - amb
    rep = {"frames": n, "rate_err": r_err, "rate_limit": r_lim, "band_err": b_err, "prob_err": p_err,
           "hits_torch": len(tor_hits), "hits_engine": len(ref_hits), "hit_differences": len(diff),
           "genuine_hits": len(tor_hits - amb),            # hits the ambiguity rule cannot excuse
           "ambiguous_in_torch": len(amb), "unexplained_hit_differences": sorted(unexplained)}
    rep["passes"] = (r_err <= r_lim and b_err <= BAND_ABS and p_err <= PROB_ABS and not unexplained)
    return rep


# ----------------------------------------------------------------------------------------------- perturbations

def torch_perturbed(model, neurons, edges, cfg, kind, **kw):
    """(model', ear_mask, extra_drive) giving the PyTorch equivalent of one control-block perturbation.
    kind: lesion (targets=[...]), deafen (ears=[...] indices into ear_idx), gain (scale), tau (scale),
          inject (neuron, current, start, frames, t_len), sign_flip (targets=[...]), unsilence (targets=[...],
          sign=+1), band_offset (offset (16,))."""
    from flybeats.model.network import FlyNet
    m = copy.deepcopy(model)
    ear_mask = extra = None
    with torch.no_grad():
        if kind == "lesion":
            m.alive[torch.as_tensor(list(kw["targets"]))] = 0.0
        elif kind == "deafen":
            ear_mask = np.ones(len(m.ear_idx), np.float32)
            ear_mask[list(kw["ears"])] = 0.0
        elif kind == "gain":
            m.gain_scale = float(kw["scale"])
        elif kind == "tau":
            m.log_tau.add_(math.log(float(kw["scale"])))
        elif kind == "inject":
            extra = np.zeros((kw["t_len"], m.n), np.float32)
            extra[kw["start"]:kw["start"] + kw["frames"], kw["neuron"]] = kw["current"]
        elif kind in ("sign_flip", "unsilence"):
            nn = neurons.copy()
            idx = list(kw["targets"])
            if kind == "sign_flip":
                nn.loc[idx, "sign"] = -nn.loc[idx, "sign"]
            else:
                nn.loc[idx, "sign"] = int(kw.get("sign", 1))
                nn.loc[idx, "silenced"] = False
            torch.manual_seed(0)
            m2 = FlyNet(nn, edges, cfg, band_stats=(model.bands.mean.tolist(), model.bands.std.tolist()))
            m2.load_state_dict(model.state_dict(), strict=False)   # buffers `fixed`/`alive` come from nn
            m2.fixed.copy_(torch.from_numpy(nn.sign.to_numpy(np.float32)[edges.pre.to_numpy()]
                                            * edges.synapses.to_numpy(np.float32)))
            m2.alive.copy_(torch.from_numpy((~nn.silenced.to_numpy()).astype(np.float32)))
            m2.g = model.g                            # keep the trained gain, not a new eigenvalue's
            m2.gain_scale = model.gain_scale
            m = m2.eval()
        elif kind == "band_offset":
            m.band_offset.add_(torch.as_tensor(np.asarray(kw["offset"], np.float32)))
        else:
            raise ValueError(kind)
    return m, ear_mask, extra


# The perturbation test list (P0): one entry per Lab control with a PyTorch equivalent. Tests and the C++ test-data
# generator import this list, so there is one copy.
PERTURBATIONS = [
    ("lesion", dict(targets=[40, 41, 42, 77])),
    ("deafen", dict(ears=[0, 2, 4, 6])),
    ("gain", dict(scale=1.3)),
    ("tau", dict(scale=2.0)),
    ("sign_flip", dict(targets=[50, 60])),
    ("band_offset", dict(offset=np.linspace(-0.5, 0.5, 16))),
]


def controls_for(brain, kind, **kw):
    c = Controls(brain)
    if kind == "lesion":
        c.lesion[list(kw["targets"])] = 0.0
    elif kind == "deafen":
        c.ear_mute[list(kw["ears"])] = 0.0
    elif kind == "gain":
        c.gain_scale = float(kw["scale"])
    elif kind == "tau":
        c.tau_scale = float(kw["scale"])
    elif kind == "inject":
        c.inject_at = (kw["neuron"], kw["current"], kw["start"], kw["frames"])
    elif kind == "sign_flip":
        c.sign_flip[list(kw["targets"])] = -1.0
    elif kind == "unsilence":
        c.unsilence[list(kw["targets"])] = int(kw.get("sign", 1))
    elif kind == "band_offset":
        c.band_offset[:] = np.asarray(kw["offset"], np.float32)
    return c


def run_perturbed(brain, audio, kind, dtype=np.float64, **kw):
    """The reference engine under one perturbation, mirroring torch_perturbed's parameters."""
    from .reference import ReferenceEngine
    eng = ReferenceEngine(brain, dtype)
    c = controls_for(brain, kind, **kw)
    if kind == "inject":
        # push frame by frame up to the start, then queue the injection
        hop = int(brain.meta["hop"])
        start = kw["start"]
        frames = eng.push(audio[:start * hop])
        c.inject = [(kw["neuron"], kw["current"], kw["frames"])]
        eng.set_controls(c)
        frames += eng.push(audio[start * hop:])
    else:
        eng.set_controls(c)
        frames = eng.push(audio)
    frames += eng.flush()
    frames.sort(key=lambda f: f.index)
    return {k: np.stack([getattr(f, k) for f in frames]) for k in ("bands", "rates", "prob", "vel", "hits")} | \
        {"frame": np.array([f.index for f in frames])}
