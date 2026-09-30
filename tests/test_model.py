"""Phase 3 gate tests. CPU, short clips, a few seconds each. Run after build_slice.py and build_controls.py.
The memorisation check is scripts/memorise.py (GPU); `pytest -m gpu` runs it through test_memorisation."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from flybeats.connectome.store import load_slice
from flybeats.model.ear import HOP
from flybeats.model.loss import drum_weights, make_loss
from flybeats.model.network import FlyNet

SR = 16000


def make(cfg, paths, control=None):
    neurons, edges = load_slice(paths["work_dir"], control)
    torch.manual_seed(0)
    return FlyNet(neurons, edges, cfg), neurons, edges


@pytest.fixture(scope="module")
def net(cfg, paths):
    return make(cfg, paths)


def one_step(model, cfg, seconds=0.5):
    """One optimiser step on random audio with a few random hits."""
    frames = int(seconds * 200)
    audio = torch.randn(2, int(seconds * SR)) * 0.1
    hits = (torch.rand(2, frames, len(model.pieces)) < 0.05).float()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    out, _ = model(audio)
    loss, _ = make_loss(cfg, torch.ones(len(model.pieces)))(out, hits, hits * 0.7, torch.ones(2, frames))
    opt.zero_grad()
    loss.backward()
    opt.step()
    return float(loss)


def edge_signs(model):
    w = model.weight_matrix().detach()
    return torch.sign(w[model.post, model.pre]), torch.sign(model.fixed)


def test_signs_match_sender_before_and_after_training(cfg, paths):
    model, _, _ = make(cfg, paths)
    got, want = edge_signs(model)
    assert torch.equal(got, want)
    one_step(model, cfg)
    got, want = edge_signs(model)
    assert torch.equal(got[want != 0], want[want != 0])


def test_silenced_neurons_output_zero(net):
    model, neurons, _ = net
    with torch.no_grad():
        out, _ = model(torch.randn(1, SR // 2), record=True)
    silenced = torch.from_numpy(neurons.silenced.to_numpy().copy())
    assert silenced.sum() == 22
    assert (out["rates"][:, :, silenced] == 0).all()


def test_gain_times_eigenvalue_is_initial_gain(cfg, net):
    model, _, _ = net
    w = model.weight_matrix().detach().double().numpy()
    lam = abs(spla.eigs(sp.csr_matrix(w), k=1, which="LM", return_eigenvectors=False, v0=np.ones(model.n))[0])
    assert lam == pytest.approx(cfg["model"]["initial_gain"], rel=1e-4)


def test_random_control_loads_and_trains(cfg, paths):
    model, _, _ = make(cfg, paths, control=1)
    before = model.theta.detach().clone()
    loss = one_step(model, cfg)
    assert np.isfinite(loss)
    assert not torch.equal(before, model.theta.detach())


def test_causality(net):
    model, _, _ = net
    torch.manual_seed(1)
    a = torch.randn(1, SR) * 0.1
    b = a.clone()
    cut = 9_000                                   # change the audio from sample 9,000 on
    b[:, cut:] = torch.randn(1, SR - cut)
    with torch.no_grad():
        oa, _ = model(a, record=True)
        ob, _ = model(b, record=True)
    safe = cut // HOP                              # frames whose window ends at or before the cut
    for k in ("hit_logits", "vel", "rates"):
        assert torch.equal(oa[k][:, :safe], ob[k][:, :safe]), k
    assert not torch.equal(oa["rates"][:, safe:], ob["rates"][:, safe:])


def test_readout_norm_standardises_motor_neurons(cfg, paths):
    model, _, _ = make(cfg, paths)
    torch.manual_seed(3)
    batches = [torch.randn(2, SR) * 0.3 for _ in range(2)]
    floor = cfg["model"]["readout_norm"]["std_floor"]
    model.fit_readout_norm(batches, 20, floor)
    with torch.no_grad():
        rm = torch.cat([model(a)[0]["mn_rates"][:, 20:].flatten(0, 1) for a in batches])
    z = (rm - model.mn_mean) / model.mn_std
    varied = rm.std(0) > floor
    assert varied.sum() > 10
    assert torch.allclose(z.mean(0), torch.zeros_like(z.mean(0)), atol=1e-3)
    assert torch.allclose(z.std(0)[varied], torch.ones(int(varied.sum())), atol=1e-3)
    assert (z.std(0)[~varied] <= 1 + 1e-3).all()      # quiet neurons are not blown up


def test_ear_must_match_locked_settings(cfg, paths):
    import copy
    bad = copy.deepcopy(cfg)
    bad["ear"]["n_fft"] = 1024
    with pytest.raises(ValueError, match="n_fft"):
        make(bad, paths)


def test_drum_weights_capped(cfg):
    import pandas as pd
    m = pd.DataFrame({"split": ["train"], "duration": [100.0], "hits_a": [10], "hits_b": [0]})
    w = drum_weights(m, ["a", "b"], 0.005, cfg["loss"]["max_drum_weight"]).numpy()
    assert w[0] == pytest.approx(min((20000 - 10) / 10, 50)) and w[1] == 50


@pytest.mark.gpu
@pytest.mark.slow
def test_memorisation():
    script = Path(__file__).resolve().parents[1] / "scripts" / "memorise.py"
    assert subprocess.run([sys.executable, str(script)]).returncode == 0


def test_schedule_runs_from_locked_rate_to_zero(cfg):
    """The one schedule train.py and memorise.py share: locked rate at step 1, falling, 0 after the last step."""
    from flybeats.model.training import make_optimiser, make_schedule
    model = torch.nn.Linear(2, 2)
    opt = make_optimiser(model, cfg)
    sched = make_schedule(opt, 10, cfg)
    rates = []
    for _ in range(10):
        rates.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert rates[0] == cfg["training"]["learning_rate"]
    assert all(a > b for a, b in zip(rates, rates[1:]))
    assert opt.param_groups[0]["lr"] == pytest.approx(0, abs=1e-12)


def test_train_step_reports_gradient_size_before_clipping(cfg, paths):
    from flybeats.model.training import train_step
    model = make(cfg, paths)[0]
    frames = 100
    batch = {"audio": torch.randn(2, int(0.5 * SR)) * 0.1,
             "hits": (torch.rand(2, frames, len(model.pieces)) < 0.05).float(), "mask": torch.ones(2, frames)}
    batch["vel"] = batch["hits"] * 0.7
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    clip = 1e-3                                           # small, so this step is surely clipped
    stats = train_step(model, opt, batch, make_loss(cfg, torch.ones(len(model.pieces))), clip)
    after = torch.sqrt(sum(p.grad.pow(2).sum() for p in model.parameters() if p.grad is not None))
    assert stats["grad_norm"] > clip and np.isfinite(stats["grad_norm"])
    assert float(after) == pytest.approx(clip, rel=1e-3)   # the gradients themselves were clipped to the limit


def test_connection_gradient_once_per_clip_matches_per_frame(cfg, paths):
    """network._WeightGradOnce: the forward pass is unchanged and every gradient is at least as close to a float64
    per-frame reference as the float32 per-frame one (the rule check_cuda_graph.py applies at full length)."""
    import copy
    base = make(cfg, paths)[0]
    with torch.no_grad():
        for p in base.parameters():
            p.add_(0.05 * torch.randn_like(p))
    frames = 100
    audio = torch.randn(2, frames * HOP) * 0.1
    hits = (torch.rand(2, frames, len(base.pieces)) < 0.05).float()

    def grads(dtype, once):
        m = copy.deepcopy(base).to(dtype)
        m.weight_grad_once = once
        out, _ = m(audio.to(dtype))
        loss, _ = make_loss(cfg, torch.ones(len(m.pieces), dtype=dtype))(
            out, hits.to(dtype), (hits * 0.7).to(dtype), torch.ones(2, frames, dtype=dtype))
        loss.backward()
        return {k: p.grad.double() for k, p in m.named_parameters()}, out["hit_logits"].detach()

    ref, _ = grads(torch.float64, False)
    old, h_old = grads(torch.float32, False)
    new, h_new = grads(torch.float32, True)
    assert torch.equal(h_old, h_new)
    for k in ref:
        e_old, e_new = [float((g[k] - ref[k]).norm() / ref[k].norm().clamp_min(1e-30)) for g in (old, new)]
        assert e_new <= 1.01 * e_old + 1e-12, k


def test_connection_gradient_once_per_clip_frees_its_graph(cfg, paths):
    """_WeightGradOnce keeps each frame's rates for the backward pass. Kept as they are, they would tie the whole
    step's autograd graph to itself, so no step's graph could ever be freed (a leak of about a gigabyte per step on
    the GPU, which also stopped the CUDA graph from recording). After a step, nothing of it may stay alive."""
    import gc
    import weakref
    model = make(cfg, paths)[0]
    out, _ = model(torch.randn(2, 50 * HOP) * 0.1)
    seen, todo, node = set(), [out["hit_logits"].grad_fn], None
    while todo and node is None:                      # find the _WeightGradOnce node in the graph
        f = todo.pop()
        if f is None or f in seen:
            continue
        seen.add(f)
        if "WeightGradOnce" in type(f).__name__:
            node = f
        todo += [g for g, _ in f.next_functions]
    probe = weakref.ref(node.rates[10])               # one frame's stored rates
    del node, seen, todo, f
    out["hit_logits"].sum().backward()
    del out
    gc.collect()
    assert probe() is None
