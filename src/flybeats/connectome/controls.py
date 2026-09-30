"""Phase 1: random-wiring controls by degree-preserving swaps, and the checks that make each a fair opponent."""
import numpy as np
import pandas as pd

from .slice import adjacency, ear_mask, hops


def rewire(edges, n, swaps_per_edge, seed, progress=None):
    """Swap a->b, c->d into a->d, c->b, swaps_per_edge x connections times.

    A swap is rejected if it makes a self-connection or duplicates an existing connection.
    Each edge row keeps its sender and synapse count, so a->d takes a->b's count.
    progress(k), if given, is called with the number of swaps proposed in each block.
    Returns (new edges, accepted swaps).
    """
    rng = np.random.default_rng(seed)
    pre = edges.pre.to_numpy(np.int64)
    post = edges.post.to_numpy(np.int64).copy()
    m = len(pre)
    present = set((pre * n + post).tolist())
    total = swaps_per_edge * m
    accepted = 0
    block = 1 << 16
    done = 0
    while done < total:
        k = min(block, total - done)
        ii = rng.integers(0, m, k)
        jj = rng.integers(0, m, k)
        for i, j in zip(ii.tolist(), jj.tolist()):
            a, b, c, d = pre[i], post[i], pre[j], post[j]
            if i == j or a == d or c == b:
                continue
            ad, cb = a * n + d, c * n + b
            if ad in present or cb in present:
                continue
            present.discard(a * n + b); present.discard(c * n + d)
            present.add(ad); present.add(cb)
            post[i], post[j] = d, b
            accepted += 1
        done += k
        if progress:
            progress(k)
    out = pd.DataFrame({"pre": pre.astype(np.int32), "post": post.astype(np.int32),
                        "synapses": edges.synapses.to_numpy()})
    return out.sort_values(["pre", "post"]).reset_index(drop=True), accepted


def drum_reach(neurons, edges, ears_cfg):
    """Per drum: how many of its motor neurons the ears reach, and the fewest hops to any of them."""
    n = len(neurons)
    d = hops(adjacency(edges.pre.to_numpy(), edges.post.to_numpy(), n),
             np.flatnonzero(ear_mask(neurons, ears_cfg)), n)
    out = {}
    for drum in sorted(neurons.drum.dropna().unique()):
        dd = d[(neurons.drum == drum).to_numpy()]
        reached = dd[dd < 99]
        out[drum] = {"reachable": int(len(reached)), "of": int(len(dd)),
                     "min_hops": int(reached.min()) if len(reached) else None,
                     "median_hops": float(np.median(reached)) if len(reached) else None}
    return out


def check_control(neurons, real, ctrl, ears_cfg):
    """The Step 2 checks. Raises if degrees or synapse totals differ; reports overlap and reach."""
    n = len(neurons)
    for col in ("pre", "post"):
        if not np.array_equal(np.bincount(real[col], minlength=n), np.bincount(ctrl[col], minlength=n)):
            raise AssertionError(f"{col} counts per neuron differ from the real slice")
    if not np.allclose(np.bincount(real.pre, real.synapses, n), np.bincount(ctrl.pre, ctrl.synapses, n)):
        raise AssertionError("output synapses per neuron differ from the real slice")
    if (ctrl.pre.astype(np.int64) * n + ctrl.post).duplicated().any():
        raise AssertionError("control has duplicate connections")
    if (ctrl.pre == ctrl.post).sum() > (real.pre == real.post).sum():
        raise AssertionError("control has new self-connections")
    real_keys = set((real.pre.astype(np.int64) * n + real.post).tolist())
    surviving = sum(k in real_keys for k in (ctrl.pre.astype(np.int64) * n + ctrl.post).tolist())
    reach = drum_reach(neurons, ctrl, ears_cfg)
    return {"connections": len(ctrl), "total_synapses": float(ctrl.synapses.sum()),
            "surviving_real_connections_share": surviving / len(ctrl),
            "unreachable_drums": [d for d, r in reach.items() if r["reachable"] == 0],
            "drum_reach": reach}
