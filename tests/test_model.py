"""Correctness tests for the pieces where a silent bug would be invisible."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from model import ConnectomeRNN, GenreModulation, ModelConfig, SparseSpMM  # noqa: E402


def _dense_reference(edge_index, values, n, r):
    """out[b, post] = sum_pre W[post, pre] * r[b, pre], built densely.

    ``ConnectomeRNN.edge_index`` is stored as [row, col] = [post, pre] -- the
    matrix convention, not the input (pre, post) convention -- so row is index
    0 here. accumulate=True makes duplicate edges sum, matching sparse mm.
    """
    w = torch.zeros(n, n, dtype=values.dtype)
    w.index_put_((edge_index[0], edge_index[1]), values, accumulate=True)
    return r @ w.t()


def test_spmm_matches_dense_and_gradients_are_exact():
    torch.manual_seed(0)
    n, e, b = 40, 300, 3
    ei = np.stack([np.random.randint(0, n, e), np.random.randint(0, n, e)]).astype(np.int32)

    rnn = ConnectomeRNN(
        edge_index=ei,
        edge_sign=np.random.choice([-1.0, 1.0], e).astype(np.float32),
        weight=np.random.randint(1, 50, e).astype(np.float32),
        n_nodes=n,
        sensory_idx=np.arange(5), motor_idx=np.arange(5, 10),
        cfg=ModelConfig(),
    )
    r = torch.randn(b, n, dtype=torch.float64)
    vals = rnn.edge_weight().double().detach().requires_grad_(True)

    got = SparseSpMM.apply(vals.float(), r.float(), rnn.crow, rnn.edge_col,
                           rnn.crow_t, rnn.col_t, rnn.perm_t, rnn.edge_row, n)
    want = _dense_reference(rnn.edge_index, vals.float(), n, r.float())
    assert torch.allclose(got, want, atol=1e-4), (got - want).abs().max()

    # gradcheck in float64 against the dense reference
    r64 = r.clone().requires_grad_(True)
    ref = _dense_reference(rnn.edge_index, vals, n, r64)
    ref.sum().backward()
    g_ref_v, g_ref_r = vals.grad.clone(), r64.grad.clone()

    vals.grad = None
    r64b = r.clone().requires_grad_(True)
    out = SparseSpMM.apply(vals, r64b, rnn.crow, rnn.edge_col, rnn.crow_t,
                           rnn.col_t, rnn.perm_t, rnn.edge_row, n)
    out.sum().backward()
    assert torch.allclose(vals.grad, g_ref_v, atol=1e-8), (vals.grad - g_ref_v).abs().max()
    assert torch.allclose(r64b.grad, g_ref_r, atol=1e-8), (r64b.grad - g_ref_r).abs().max()


def test_spmm_handles_duplicate_edges():
    """The random-rewiring ablation can emit repeated (pre, post) pairs."""
    torch.manual_seed(1)
    n, b = 12, 2
    ei = np.array([[0, 0, 0, 3, 3], [1, 1, 2, 4, 4]], dtype=np.int32)   # duplicates
    rnn = ConnectomeRNN(
        edge_index=ei, edge_sign=np.ones(5, dtype=np.float32),
        weight=np.arange(1, 6, dtype=np.float32), n_nodes=n,
        sensory_idx=np.array([0]), motor_idx=np.array([4]), cfg=ModelConfig(),
    )
    vals = rnn.edge_weight().double().detach().requires_grad_(True)
    r = torch.randn(b, n, dtype=torch.float64, requires_grad=True)
    out = SparseSpMM.apply(vals, r, rnn.crow, rnn.edge_col, rnn.crow_t,
                           rnn.col_t, rnn.perm_t, rnn.edge_row, n)
    out.sum().backward()
    assert vals.grad.shape == vals.shape, "gradient must cover every edge, duplicates included"
    assert torch.isfinite(vals.grad).all()


def test_signs_are_frozen_under_training():
    """Gradient descent may change what a connection is worth, never its sign."""
    torch.manual_seed(0)
    n, e = 60, 400
    ei = np.stack([np.random.randint(0, n, e), np.random.randint(0, n, e)]).astype(np.int32)
    sign = np.random.choice([-1.0, 1.0], e).astype(np.float32)
    rnn = ConnectomeRNN(ei, sign, np.random.randint(1, 30, e).astype(np.float32), n,
                        np.arange(4), np.arange(4, 8), cfg=ModelConfig())
    before = torch.sign(rnn.edge_weight()).clone()

    opt = torch.optim.SGD(rnn.parameters(), lr=5.0)
    for _ in range(20):
        opt.zero_grad()
        drive = torch.randn(2, 12, 4)
        out, _ = rnn(drive)
        out.pow(2).mean().backward()
        opt.step()

    after = torch.sign(rnn.edge_weight())
    assert torch.equal(before, after), "training changed a synaptic sign"
    assert not rnn.edge_sign.requires_grad


def test_topology_is_frozen_under_training():
    """No connection may be created or destroyed."""
    n, e = 50, 200
    ei = np.stack([np.random.randint(0, n, e), np.random.randint(0, n, e)]).astype(np.int32)
    rnn = ConnectomeRNN(ei, np.ones(e, dtype=np.float32),
                        np.ones(e, dtype=np.float32) * 5, n,
                        np.arange(3), np.arange(3, 6), cfg=ModelConfig())
    idx_before = rnn.edge_index.clone()
    opt = torch.optim.SGD(rnn.parameters(), lr=1.0)
    for _ in range(5):
        opt.zero_grad()
        out, _ = rnn(torch.randn(2, 8, 3))
        out.sum().backward()
        opt.step()
    assert torch.equal(idx_before, rnn.edge_index)
    # softplus is strictly positive, so no magnitude can reach exactly zero
    assert (torch.nn.functional.softplus(rnn.log_gain) > 0).all()


def test_gain_initialised_at_synapse_counts():
    e = 100
    w = np.random.randint(1, 200, e).astype(np.float32)
    ei = np.stack([np.arange(e) % 20, np.arange(e) % 17]).astype(np.int32)
    rnn = ConnectomeRNN(ei, np.ones(e, dtype=np.float32), w, 20,
                        np.arange(2), np.arange(2, 4), cfg=ModelConfig())
    # edges get sorted internally; compare as multisets
    assert np.allclose(np.sort(rnn.log_gain.detach().numpy()), np.sort(np.log(w)), atol=1e-5)


def test_lesion_gate_silences_a_population():
    n, e = 40, 300
    ei = np.stack([np.random.randint(0, n, e), np.random.randint(0, n, e)]).astype(np.int32)
    rnn = ConnectomeRNN(ei, np.ones(e, dtype=np.float32),
                        np.random.randint(1, 20, e).astype(np.float32), n,
                        np.arange(4), np.arange(4, 8), cfg=ModelConfig())
    drive = torch.randn(1, 20, 4).abs()
    intact, _ = rnn(drive)
    rnn.set_gate(np.arange(n), 0.0)
    lesioned, _ = rnn(drive)
    assert torch.allclose(lesioned, torch.zeros_like(lesioned))
    rnn.reset_gates()
    restored, _ = rnn(drive)
    assert torch.allclose(intact, restored)


def test_genre_current_stays_bounded():
    """PLAN.md open question 2: tonic octopaminergic drive must not saturate."""
    idx = np.arange(25)
    gm = GenreModulation(n_styles=10, target_idx=idx, n_nodes=500, max_current=0.5)
    with torch.no_grad():           # push the embedding far out of distribution
        gm.embed.weight.mul_(1000.0)
        gm.project.weight.mul_(1000.0)
    cur = gm(torch.arange(10))
    assert cur.abs().max() <= 0.5 + 1e-6, f"tonic drive escaped its envelope: {cur.abs().max()}"
    assert torch.count_nonzero(cur[0]) <= len(idx), "drive leaked outside the OA pool"


def test_genre_interpolates():
    gm = GenreModulation(n_styles=4, target_idx=np.arange(5), n_nodes=50)
    a, b = gm.interpolate(0, 1, 0.0), gm.interpolate(0, 1, 1.0)
    mid = gm.interpolate(0, 1, 0.5)
    assert not torch.allclose(a, b)
    assert not torch.allclose(mid, a) and not torch.allclose(mid, b)
