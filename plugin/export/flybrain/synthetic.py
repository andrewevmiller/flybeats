"""Small networks with the slice's exact table schema, so the exporter, engines and tests run without the
connectome or any checkpoint. Not the fly: random layered wiring that only copies the real slice's shape
(ears mirrored left to right, no direct ear-to-motor connections, silenced modulators, untyped neurons, one
disjoint motor-neuron group per drum).

make_slice(...) -> (neurons, edges) as flybeats.connectome.slice.build_slice returns them.
"""
import numpy as np
import pandas as pd

PIECES = ["kick", "hihat_pedal", "snare", "hihat_closed", "hihat_open", "tom", "crash", "ride"]
DRUM_LIMB = {"kick": ("hind_leg", "R"), "hihat_pedal": ("hind_leg", "L"), "hihat_closed": ("wing", "R"),
             "ride": ("wing", "R"), "hihat_open": ("wing", "R"), "snare": ("wing", "L"), "tom": ("wing", "L"),
             "crash": ("wing", "L")}


def make_slice(n_ears=12, n_inter=120, n_motor_per_drum=4, n_modulatory=4, n_untyped=3, edges_per_neuron=12,
               n_inter_types=20, seed=0, large=False):
    """large=True gives roughly the real slice's size (3,302 neurons, about 108k connections)."""
    if large:
        n_ears, n_inter, n_motor_per_drum, n_modulatory, n_untyped = 75, 2900, 29, 22, 22
        edges_per_neuron, n_inter_types = 33, 1100
    rng = np.random.default_rng(seed)
    rows = []
    body = 1000
    # ears: real left ones (JO-A and JO-B), each mirrored to the right as bodyId -original
    ear_types = ["JO-A"] * (n_ears // 2) + ["JO-B"] * (n_ears - n_ears // 2)
    for t in ear_types:
        body += 1
        rows.append(dict(bodyId=body, mirror_of=np.nan, type=t, side="L", role="excitatory", sign=1, silenced=False,
                         motor_group=None, drum=None))
    for r in list(rows):
        rows.append(dict(r, bodyId=-r["bodyId"], mirror_of=float(r["bodyId"]), side="R"))
    # interneurons: typed (shared types), a few untyped, a few silenced modulators
    inter_types = [f"IN{k:04d}" for k in range(n_inter_types)]
    for k in range(n_inter):
        body += 1
        exc = rng.random() < 0.6
        rows.append(dict(bodyId=body, mirror_of=np.nan, type=inter_types[k % n_inter_types],
                         side=rng.choice(["L", "R", "M"], p=[0.45, 0.45, 0.1]),
                         role="excitatory" if exc else "inhibitory", sign=1 if exc else -1, silenced=False,
                         motor_group=None, drum=None))
    for k in range(n_untyped):
        body += 1
        rows.append(dict(bodyId=body, mirror_of=np.nan, type=None, side="M", role="excitatory", sign=1,
                         silenced=False, motor_group=None, drum=None))
    for k in range(n_modulatory):
        body += 1
        rows.append(dict(bodyId=body, mirror_of=np.nan, type=f"MOD{k:02d}", side="M", role="modulatory", sign=0,
                         silenced=True, motor_group=None, drum=None))
    # motor neurons: each drum its own group of its limb's side
    for d, piece in enumerate(PIECES):
        group, side = DRUM_LIMB[piece]
        for k in range(n_motor_per_drum):
            body += 1
            rows.append(dict(bodyId=body, mirror_of=np.nan, type=f"MN_{piece}_{k % 2}", side=side, role="motor",
                             sign=-1, silenced=False, motor_group=group, drum=piece))
    neurons = pd.DataFrame(rows)
    neurons = neurons.sort_values("bodyId").reset_index(drop=True)
    neurons.insert(0, "index", np.arange(len(neurons), dtype=np.int32))
    n = len(neurons)
    is_ear = neurons.type.fillna("").str.match(r"^JO-[AB]").to_numpy()
    is_motor = (neurons.role == "motor").to_numpy()
    is_inter = ~is_ear & ~is_motor
    ears, inter, motor = (np.flatnonzero(m) for m in (is_ear, is_inter, is_motor))
    # layered wiring: ears -> inter, inter <-> inter, inter -> motor, a little motor -> inter feedback
    pairs = set()

    def connect(src, dst, count):
        for _ in range(count):
            a, b = int(rng.choice(src)), int(rng.choice(dst))
            if a != b:
                pairs.add((a, b))
    e = edges_per_neuron
    connect(ears, inter, len(ears) * e)
    connect(inter, inter, len(inter) * e)
    connect(inter, motor, len(motor) * e)
    connect(motor, inter, len(motor) * 2)
    # every motor neuron and every ear needs at least one connection
    for m in motor:
        pairs.add((int(rng.choice(inter)), int(m)))
    for a in ears:
        pairs.add((int(a), int(rng.choice(inter))))
    pre, post = np.array(sorted(pairs)).T
    synapses = rng.integers(5, 60, len(pre)).astype(np.float64)
    edges = pd.DataFrame({"pre": pre.astype(np.int32), "post": post.astype(np.int32), "synapses": synapses})
    edges = edges.sort_values(["pre", "post"]).reset_index(drop=True)
    return neurons, edges


def noise_clips(n=4, seconds=4.0, sr=16000, level=0.1, seed=0):
    """Clip dicts (audio only) of coloured noise, for tests that need "music"."""
    rng = np.random.default_rng(seed)
    clips = []
    for _ in range(n):
        x = rng.standard_normal(int(seconds * sr)).astype(np.float32)
        # a little low-pass colour so the bands differ
        x = np.convolve(x, np.ones(4, np.float32) / 4, mode="same")
        clips.append({"audio": (x * level / max(x.std(), 1e-9)).astype(np.float32)})
    return clips
