"""Phase 1: turn the three MaleCNS files into one saved slice.

build_slice() runs guide steps 1-5 and returns (neurons, edges, report):
  neurons: index, bodyId, mirror_of, type, side, role, sign, silenced, motor_group, drum
  edges:   pre, post, synapses   (pre/post are row indices into neurons)
Everything is sorted, so the same locked.yaml always gives byte-identical files.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as pf
import scipy.sparse as sp
import scipy.sparse.linalg as spla

ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
NEUROTRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"

MODULATORS = {"serotonin", "octopamine", "dopamine"}
MOTOR_GROUPS = {"wm": "wing", "hl": "hind_leg", "ml": "middle_leg", "fl": "front_leg"}
LIMBS = {"right_hind_leg": ("hind_leg", "R"), "left_hind_leg": ("hind_leg", "L"),
         "right_wing": ("wing", "R"), "left_wing": ("wing", "L")}
SONG_TYPES = ["AMMC-A1", "aPN1", "vPN1", "dPR1", "pC2l"]


# --- 1. Load neurons and connections ---------------------------------------------

def load_neurons(malecns_dir):
    """Annotation rows with a superclass, with a side and the neurotransmitter columns joined on."""
    a = pf.read_table(malecns_dir / ANNOTATIONS,
                      columns=["bodyId", "type", "superclass", "subclass", "somaSide", "rootSide", "synonyms"]).to_pandas()
    a = a[a.superclass.notna()].copy()
    a["side"] = np.where(a.somaSide.isin(["L", "R"]), a.somaSide,
                         np.where(a.rootSide.isin(["L", "R"]), a.rootSide, "M"))
    nt = pf.read_table(malecns_dir / NEUROTRANSMITTERS,
                       columns=["body", "consensus_nt", "predicted_nt", "predicted_nt_confidence"]).to_pandas()
    nt = nt.rename(columns={"body": "bodyId"})
    if nt.bodyId.duplicated().any():
        raise ValueError("neurotransmitter file has more than one row per body")
    a = a.merge(nt, on="bodyId", how="left")
    a = a.drop(columns=["somaSide", "rootSide"]).sort_values("bodyId").reset_index(drop=True)
    return a


def load_edges(malecns_dir, body_ids, min_synapses):
    """Connections with >= min_synapses between two annotated neurons (about 6.24 million)."""
    w = pf.read_table(malecns_dir / WEIGHTS)
    w = w.filter(pc.greater_equal(w["weight"], min_synapses))
    ids = pa.array(np.asarray(body_ids))
    w = w.filter(pc.and_(pc.is_in(w["body_pre"], value_set=ids), pc.is_in(w["body_post"], value_set=ids)))
    return w.to_pandas().rename(columns={"weight": "synapses"})


# --- 2. Pick and mirror the ears --------------------------------------------------

def mirror_ears(neurons, edges, ears_cfg):
    """Keep the usable left ears, drop every other ear, and add a mirrored copy of each left ear.

    Returns (neurons, edges, stats). Copies get bodyId = -original and mirror_of = original.
    """
    is_ear = neurons.type.fillna("").str.match(ears_cfg["types"])
    ear_ids = neurons.bodyId[is_ear]
    senders = set(edges.body_pre.unique())
    usable = neurons[is_ear & neurons.bodyId.isin(senders)]
    left = usable.bodyId[usable.side == "L"].to_numpy()
    stats = {"usable_ears": int(len(usable)), "usable_left": int((usable.side == "L").sum()),
             "usable_right": int((usable.side == "R").sum())}

    # Candidate targets on each side, by type. Taken before the other ears are dropped, so a
    # connection mirrored onto a dropped ear is removed with it below rather than counted as lost.
    typed = neurons[neurons.type.notna()]
    by_type_side = {k: np.sort(g.to_numpy()) for k, g in typed.groupby(["type", "side"]).bodyId}

    info = neurons.set_index("bodyId")[["type", "side"]]
    out = edges[edges.body_pre.isin(left)].join(info, on="body_post")
    out["type"] = out["type"].fillna("")   # "" = untyped, so groupby keeps them
    pre, post, syn = [], [], []
    skipped_untyped = lost_no_opposite = 0.0
    lost_types = set()
    for (ear, ttype, tside), g in out.groupby(["body_pre", "type", "side"], sort=True):
        if tside == "M":                                   # midline targets keep the same neuron
            pre += [-ear] * len(g); post += g.body_post.tolist(); syn += g.synapses.astype(float).tolist()
            continue
        total = float(g.synapses.sum())
        if ttype == "":                                    # cannot mirror "by type" without a type
            skipped_untyped += total
            continue
        opposite = {"L": "R", "R": "L"}[tside]
        cands = by_type_side.get((ttype, opposite))
        if cands is None:
            lost_no_opposite += total
            lost_types.add(ttype)
            continue
        share = total / len(cands)                         # not re-filtered, even below min_synapses
        pre += [-ear] * len(cands); post += cands.tolist(); syn += [share] * len(cands)

    copies = neurons[neurons.bodyId.isin(left)].copy()
    copies["mirror_of"] = copies.bodyId
    copies["bodyId"] = -copies.bodyId
    copies["side"] = "R"
    neurons = pd.concat([neurons.assign(mirror_of=pd.NA), copies], ignore_index=True)
    neurons["mirror_of"] = neurons.mirror_of.astype("Int64")
    neurons = neurons.sort_values("bodyId").reset_index(drop=True)

    mirrored = pd.DataFrame({"body_pre": np.array(pre, np.int64), "body_post": np.array(post, np.int64),
                             "synapses": np.array(syn, np.float64)})
    edges = pd.concat([edges.astype({"synapses": np.float64}), mirrored], ignore_index=True)

    drop = set(ear_ids) - set(left)          # every other ear goes, with all its connections
    neurons = neurons[~neurons.bodyId.isin(drop)].reset_index(drop=True)
    edges = edges[~edges.body_pre.isin(drop) & ~edges.body_post.isin(drop)]

    ear_total = float(out.synapses.sum())
    stats.update({"ears_after_mirroring": int(len(left) * 2),
                  "mirror_skipped_untyped_synapses": skipped_untyped,
                  "mirror_lost_synapses": lost_no_opposite,
                  "mirror_lost_share": lost_no_opposite / ear_total if ear_total else 0.0,
                  "mirror_lost_types": sorted(lost_types)})
    return neurons, edges, stats


# --- 3. Cut the slice by hop count --------------------------------------------------

def hops(m, sources, n, max_d=25):
    """Hop distance from any source along the rows of m (csr, m[i, j] != 0 means i -> j). 99 = unreachable."""
    d = np.full(n, 99, np.int16)
    d[sources] = 0
    frontier = np.zeros(n, bool)
    frontier[sources] = True
    for k in range(1, max_d + 1):
        nxt = (np.asarray(m[frontier].sum(0)).ravel() > 0) & (d == 99)
        if not nxt.any():
            break
        d[nxt] = k
        frontier = nxt
    return d


def adjacency(pre, post, n):
    return sp.csr_matrix((np.ones(len(pre), np.float32), (pre, post)), shape=(n, n))


def ear_mask(neurons, ears_cfg):
    return neurons.type.fillna("").str.match(ears_cfg["types"]).to_numpy()


def motor_group(neurons):
    is_motor = neurons.superclass.fillna("").str.contains("motor")
    return np.where(is_motor, neurons.subclass.map(MOTOR_GROUPS).fillna(neurons.subclass.fillna("other")), None)


def cut_slice(neurons, edges, connectome_cfg):
    n = len(neurons)
    ids = pd.Index(neurons.bodyId)
    pre, post = ids.get_indexer(edges.body_pre), ids.get_indexer(edges.body_post)
    a = adjacency(pre, post, n)
    d_ear = hops(a, np.flatnonzero(ear_mask(neurons, connectome_cfg["ears"])), n)
    groups = motor_group(neurons)
    at = a.T.tocsr()
    keep = np.zeros(n, bool)
    for g in connectome_cfg["motor_groups"]:
        d_back = hops(at, np.flatnonzero(groups == g), n)
        keep |= (d_ear.astype(np.int32) + d_back) <= connectome_cfg["hop_budget"]
    kept = neurons[keep].reset_index(drop=True)
    kept_ids = set(kept.bodyId)
    e = edges[edges.body_pre.isin(kept_ids) & edges.body_post.isin(kept_ids)]
    return kept, e


# --- 4. Assign roles ------------------------------------------------------------------

def assign_roles(neurons, signs_cfg):
    """role, sign (+1/-1, 0 when silenced) and silenced, applying the signs rules in order."""
    sign_of = {"excitatory": 1, "inhibitory": -1}
    motor = neurons.superclass.fillna("").str.contains("motor").to_numpy()
    mirrored = neurons.mirror_of.notna().to_numpy()
    nt = neurons.consensus_nt.fillna("unclear")
    modulatory = nt.isin(MODULATORS) | ((nt == "unclear") & neurons.predicted_nt.isin(MODULATORS)
                                        & (neurons.predicted_nt_confidence >= 0.5))
    role = np.select(
        [motor, mirrored,
         nt.map(lambda x: signs_cfg.get(x) == "excitatory").to_numpy(),
         nt.map(lambda x: signs_cfg.get(x) == "inhibitory").to_numpy(),
         modulatory.to_numpy()],
        ["motor", "excitatory", "excitatory", "inhibitory", "modulatory"], default="unknown")
    motor_sign = sign_of[signs_cfg[signs_cfg["motor_neurons"]]]
    sign = np.array([motor_sign if r == "motor" else sign_of.get(r, 0) for r in role], np.int8)
    out = neurons.copy()
    out["role"] = role
    out["sign"] = sign
    out["silenced"] = sign == 0
    return out


# --- 5. Attach the readout ------------------------------------------------------------

def attach_readout(neurons, readout_cfg):
    """drum per motor neuron; fails loudly on an unassigned wing type or a neuron serving two drums."""
    family_of = {t: fam for fam, types in readout_cfg["wing_families"].items() for t in types}
    groups = motor_group(neurons)
    wing_types = set(neurons.type[(groups == "wing") & neurons.type.notna()])
    missing = sorted(wing_types - set(family_of))
    if missing:
        raise ValueError(f"wing motor types with no family in locked.yaml: {missing}")

    drum = pd.Series([None] * len(neurons), index=neurons.index, dtype=object)
    for name, spec in readout_cfg.items():
        if not isinstance(spec, dict) or "limb" not in spec:
            continue
        group, side = LIMBS[spec["limb"]]
        sel = (groups == group) & (neurons.side == side) & neurons.type.notna()
        if spec["family"] != "all":
            sel &= neurons.type.map(family_of) == spec["family"]
        if drum[sel].notna().any():
            raise ValueError(f"motor neurons assigned to two drums: {name} and {set(drum[sel].dropna())}")
        drum[sel] = name
    out = neurons.copy()
    out["motor_group"] = groups
    out["drum"] = drum
    return out


# --- The whole build ------------------------------------------------------------------

def type_keys(neurons, lump_untyped=False):
    """Cell type per neuron. Each untyped neuron counts as its own type (Phase 3), unless lump_untyped,
    which puts all 22 in one type; the Phase 1 gate's 1,203 types and 41,029 pairs were counted that way."""
    untyped = "untyped" if lump_untyped else "untyped:" + neurons.bodyId.astype(str)
    return neurons.type.where(neurons.type.notna(), untyped)


def is_song_type(frame, name):
    """Song neurons go by paper names; three of the five (aPN1, vPN1, pC2l) appear only in synonyms."""
    in_synonyms = frame.synonyms.fillna("").str.contains(rf"(?:^|[:;,]\s*){name}(?:$|[;,])", regex=True)
    return (frame.type == name) | in_synonyms


def signed_matrix(neurons, edges):
    """M[post, pre] = sign(pre) * synapses; silenced senders contribute 0."""
    n = len(neurons)
    vals = neurons.sign.to_numpy()[edges.pre.to_numpy()] * edges.synapses.to_numpy()
    return sp.csr_matrix((vals, (edges.post.to_numpy(), edges.pre.to_numpy())), shape=(n, n))


def largest_eigenvalue(neurons, edges):
    val = spla.eigs(signed_matrix(neurons, edges).astype(np.float64), k=1, which="LM",
                    return_eigenvectors=False, v0=np.ones(len(neurons)))[0]
    return float(abs(val))


BUILD_STAGES = ["load neurons", "load and filter connections", "mirror ears", "cut by hops",
                "roles and readout", "report and eigenvalue"]


def build_slice(cfg, paths, progress=None):
    """progress(stage name), if given, is called as each of BUILD_STAGES begins."""
    step = progress or (lambda stage: None)
    c = cfg["connectome"]
    if cfg["data"]["malecns"]["min_confidence"] != 0.5:
        raise ValueError("only the minconf-0.5 files are supported")
    step(BUILD_STAGES[0])
    neurons = load_neurons(paths["malecns_dir"])
    n_annotated = len(neurons)
    step(BUILD_STAGES[1])
    edges = load_edges(paths["malecns_dir"], neurons.bodyId, c["min_synapses"])
    n_edges = len(edges)
    annotated = neurons                      # kept for the report's "x of y" counts

    step(BUILD_STAGES[2])
    neurons, edges, ear_stats = mirror_ears(neurons, edges, c["ears"])
    step(BUILD_STAGES[3])
    neurons, edges = cut_slice(neurons, edges, c)
    step(BUILD_STAGES[4])
    neurons = assign_roles(neurons, c["signs"])
    neurons = attach_readout(neurons, cfg["readout"])

    neurons = neurons.sort_values("bodyId").reset_index(drop=True)
    neurons.insert(0, "index", np.arange(len(neurons), dtype=np.int32))
    ids = pd.Index(neurons.bodyId)
    e = pd.DataFrame({"pre": ids.get_indexer(edges.body_pre).astype(np.int32),
                      "post": ids.get_indexer(edges.body_post).astype(np.int32),
                      "synapses": edges.synapses.to_numpy(np.float64)})
    e = e.sort_values(["pre", "post"]).reset_index(drop=True)

    step(BUILD_STAGES[5])
    report = slice_report(neurons, e, annotated, n_annotated, n_edges, ear_stats)
    cols = ["index", "bodyId", "mirror_of", "type", "side", "role", "sign", "silenced", "motor_group", "drum"]
    return neurons[cols], e, report


def slice_report(neurons, edges, annotated, n_annotated, n_edges, ear_stats):
    mg = neurons.motor_group
    motor = neurons[neurons.role == "motor"]
    wing = neurons[mg == "wing"]
    hind = neurons[mg == "hind_leg"]
    ann_groups = motor_group(annotated)
    real = neurons[neurons.mirror_of.isna()]
    def count_types(lump):
        tk = type_keys(neurons, lump).to_numpy()
        pairs = pd.DataFrame({"a": tk[edges.pre], "b": tk[edges.post]}).drop_duplicates()
        return {"cell_types": int(pd.unique(tk).size), "type_pairs": len(pairs)}
    drums = {d: {"motor_neurons": int((neurons.drum == d).sum()),
                 "types": int(neurons.type[neurons.drum == d].nunique())}
             for d in sorted(neurons.drum.dropna().unique())}
    return {
        "annotated_neurons": n_annotated,
        "connections_ge_min_synapses": n_edges,
        **ear_stats,
        "slice_neurons": len(neurons),
        "slice_connections": len(edges),
        "sides": {s: int((neurons.side == s).sum()) for s in ("L", "R", "M")},
        "roles": {r: int((neurons.role == r).sum()) for r in ("excitatory", "inhibitory", "motor", "modulatory", "unknown")},
        "motor_groups": {str(g): int((motor.motor_group == g).sum()) for g in sorted(motor.motor_group.dropna().unique())},
        "wing_motor_neurons": {"count": len(wing), "types": int(wing.type.nunique()),
                               "types_left": int(wing.type[wing.side == "L"].nunique()),
                               "types_right": int(wing.type[wing.side == "R"].nunique())},
        "hind_leg_motor_neurons": {"count": len(hind), "of_annotated": int((ann_groups == "hind_leg").sum()),
                                   "types": int(hind.type.nunique())},
        "song_neurons": {t: f"{int(is_song_type(real, t).sum())}/{int(is_song_type(annotated, t).sum())}"
                         for t in SONG_TYPES},
        "types_untyped_separate": count_types(False),
        "types_untyped_lumped": count_types(True),
        "largest_eigenvalue": round(largest_eigenvalue(neurons, edges), 1),
        "drums": drums,
    }
