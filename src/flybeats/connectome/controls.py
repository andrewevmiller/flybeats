"""Phase 1: random-wiring controls by degree-preserving swaps, and the checks that make each a fair opponent.

prereg-v3 swapped across the whole slice (rewire). prereg-v4 swaps only within neuron classes and keeps reciprocal
pairs (rewire_classes): the whole-slice swaps wired ears straight to motor neurons, which the real slice never does,
and cut reciprocal connections from 11.4% to 1.6%, so the controls were easier to train for reasons unrelated to
the wiring (reports/prereg-v4-controls-proposal.md)."""
import numpy as np
import pandas as pd

from .slice import adjacency, ear_mask, hops

CLASSES = ["ear", "interneuron_1_hop", "interneuron_2_plus_hops", "motor"]


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


def control_classes(neurons, edges, ears_cfg):
    """Each neuron's class index into CLASSES: ears; interneurons 1 hop from an ear; interneurons 2 or more hops
    from an ear; motor neurons. Hops are measured on `edges`, which must be the real slice's."""
    n = len(neurons)
    ear = ear_mask(neurons, ears_cfg)
    d = hops(adjacency(edges.pre.to_numpy(), edges.post.to_numpy(), n), np.flatnonzero(ear), n)
    motor = (neurons.role == "motor").to_numpy()
    return np.where(ear, 0, np.where(motor, 3, np.where(d <= 1, 1, 2))).astype(np.int64)


def reciprocal_mask(edges, n):
    """True for each connection a->b whose reverse b->a is also a connection."""
    key = edges.pre.to_numpy(np.int64) * n + edges.post.to_numpy(np.int64)
    return np.isin(edges.post.to_numpy(np.int64) * n + edges.pre.to_numpy(np.int64), key)


def rewire_classes(edges, classes, swaps_per_edge, seed, progress=None):
    """Degree-, class- and reciprocity-preserving swaps, swaps_per_edge x connections proposals.

    One-way connections (a->b, no b->a) swap a->b, c->d into a->d, c->b, and reciprocal pairs swap {a<->b, c<->d}
    into {a<->d, c<->b}, only when a and c share a class and so do b and d. A swap is rejected if it makes a
    self-connection or a duplicate, or (one-way) makes a new reciprocal pair. So every neuron keeps its connection
    counts in and out, per class of partner, and the number of reciprocal pairs is unchanged. Each connection keeps
    its synapse count. progress(k), if given, is called with the number of swaps proposed in each block.
    Returns (new edges, accepted swaps).
    """
    rng = np.random.default_rng(seed)
    pre, post = edges.pre.to_numpy(np.int64), edges.post.to_numpy(np.int64)
    syn = dict(zip(zip(pre.tolist(), post.tolist()), edges.synapses.to_numpy().tolist()))
    present = set(syn)
    one = sorted(e for e in present if (e[1], e[0]) not in present)
    pairs = sorted({(min(a, b), max(a, b)) for a, b in present if (b, a) in present})
    one, pairs = [list(e) for e in one], [list(p) for p in pairs]
    by_one, by_pair = {}, {}
    for i, (a, b) in enumerate(one):
        by_one.setdefault((classes[a], classes[b]), []).append(i)
    for i, (a, b) in enumerate(pairs):
        by_pair.setdefault(tuple(sorted((classes[a], classes[b]))), []).append(i)
    one_key = [(classes[a], classes[b]) for a, b in one]
    pair_key = [tuple(sorted((classes[a], classes[b]))) for a, b in pairs]

    def move(old, new):
        syn[new] = syn.pop(old)
        present.discard(old)
        present.add(new)

    total, done, accepted, block = swaps_per_edge * len(edges), 0, 0, 1 << 16
    share_one = len(one) / max(len(one) + len(pairs), 1)
    while done < total:
        k = min(block, total - done)
        for u in rng.random(k).tolist():
            if u < share_one:
                i = int(rng.integers(len(one)))
                g = by_one[one_key[i]]
                j = g[int(rng.integers(len(g)))]
                (a, b), (c, d) = one[i], one[j]
                if (i == j or a == d or c == b or (a, d) in present or (c, b) in present
                        or (d, a) in present or (b, c) in present):
                    continue
                move((a, b), (a, d)); move((c, d), (c, b))
                one[i], one[j] = [a, d], [c, b]
            else:
                i = int(rng.integers(len(pairs)))
                g = by_pair[pair_key[i]]
                j = g[int(rng.integers(len(g)))]
                (a, b), (c, d) = pairs[i], pairs[j]
                if classes[a] != classes[c]:            # orient the second pair so a, c share a class
                    c, d = d, c
                if (i == j or classes[a] != classes[c] or classes[b] != classes[d] or len({a, b, c, d}) < 4
                        or any(e in present for e in ((a, d), (d, a), (c, b), (b, c)))):
                    continue
                # each sender keeps its own connection (and synapse count) and only changes target
                move((a, b), (a, d)); move((d, c), (d, a)); move((c, d), (c, b)); move((b, a), (b, c))
                pairs[i], pairs[j] = [a, d], [c, b]
            accepted += 1
        done += k
        if progress:
            progress(k)
    out = pd.DataFrame([(a, b, s) for (a, b), s in syn.items()], columns=["pre", "post", "synapses"])
    out = out.astype({"pre": np.int32, "post": np.int32, "synapses": edges.synapses.dtype})
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


def check_control(neurons, real, ctrl, ears_cfg, classes=None):
    """The Step 2 checks. Raises if degrees or synapse totals differ; reports overlap and reach. With classes
    (prereg-v4), also raises unless every neuron keeps its connection counts per class of partner in and out, the
    ear-to-motor-neuron connections match the real slice's (none), and the reciprocal pairs are as many."""
    n = len(neurons)
    if classes is not None:
        k = len(CLASSES)
        for a, b in (("pre", "post"), ("post", "pre")):
            if not np.array_equal(np.bincount(real[a].to_numpy() * k + classes[real[b].to_numpy()], minlength=n * k),
                                  np.bincount(ctrl[a].to_numpy() * k + classes[ctrl[b].to_numpy()], minlength=n * k)):
                raise AssertionError(f"per-class {'output' if a == 'pre' else 'input'} counts differ from the real slice")
        ear_motor = lambda e: int(((classes[e.pre.to_numpy()] == 0) & (classes[e.post.to_numpy()] == 3)).sum())
        if ear_motor(ctrl) != ear_motor(real):
            raise AssertionError(f"{ear_motor(ctrl)} ear-to-motor-neuron connections, real slice has {ear_motor(real)}")
        if reciprocal_mask(ctrl, n).sum() != reciprocal_mask(real, n).sum():
            raise AssertionError("reciprocal connections differ in number from the real slice")
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
            "reciprocal_share": float(reciprocal_mask(ctrl, n).mean()),
            "unreachable_drums": [d for d, r in reach.items() if r["reachable"] == 0],
            "drum_reach": reach}
