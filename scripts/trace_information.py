"""Where does the real wiring lose the drum information? Memorisation found the real network's motor neurons carry
about half the hit information of control 1's, and that both networks' motor activity collapses to 3-4 signals.
Five checks, on the CPU, read-only; no network is trained:

  1 Hop by hop: how well each drum's hits can be read (best straight-line readout of the last 200 ms, average
    precision) from each stage: ears, interneurons 1 hop from the ears, interneurons 2+ hops, motor neurons.
    Real and control 1, untrained and trained (prereg-v5 final.pt), on the 8 memorisation clips.
  2 Bottlenecks in the wiring alone: per drum, the maximum flow of synapses from the ears to its motor neurons,
    and the chance that a backward walk from its motor neurons, following input synapses, reaches an ear within
    3 steps (also with signs: excitatory minus inhibitory). Real slice against all 5 controls.
  3 Silenced neurons: check 2 with the silenced neurons' outputs removed (as in the model) and kept.
  4 The collapse: the number of independent signals (participation ratio) at each stage; and the untrained
    networks at 0.5 and 0.8 x the starting gain, for the motor neurons' signals and readable information.
  5 Timing: the motor-neuron readout allowed to see 25, 50 and 100 ms past each moment (does the information
    arrive late?), and the learned time constants per stage.

Straight-line readouts are fitted and scored on the same frames: this is about what the activity holds, as in
memorisation. Each stage is first reduced to its 64 main patterns of activity.
Writes reports/trace_information.json.
    .venv\\Scripts\\python.exe scripts\\trace_information.py [--trial]
About 20-40 minutes. --trial runs a small version (2 clips, few patterns) to check the script, and writes nothing.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_flow
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flybeats.config import load_locked, load_paths, reports_dir  # noqa: E402
from flybeats.connectome.controls import CLASSES, control_classes  # noqa: E402
from flybeats.connectome.slice import ear_mask  # noqa: E402
from flybeats.connectome.store import load_slice  # noqa: E402
from flybeats.model.training import setup, to_tensors  # noqa: E402

STAGES = ["ears", "interneurons 1 hop", "interneurons 2+ hops", "motor neurons"]
BLOCK, N_BLOCKS = 10, 4                    # history: 4 blocks of 50 ms = the last 200 ms
LAGS_MS = [0, 25, 50, 100]
GAINS = [0.5, 0.8]
MAIN = ["kick", "snare", "hihat_closed"]

ap = argparse.ArgumentParser()
ap.add_argument("--trial", action="store_true", help="a small version to check the script; writes nothing")
args = ap.parse_args()
torch.set_num_threads(4)
t0 = time.time()
cfg, paths = load_locked(), load_paths()
mc, frame_ms = cfg["memorisation"], cfg["training_data"]["frame_ms"]
n_pcs = 16 if args.trial else 64
n_clips = 2 if args.trial else mc["songs"] * mc["clips_per_song"]
pieces = cfg["kit"]["pieces"]
main_idx = [pieces.index(p) for p in MAIN]
neurons, real_edges = load_slice(paths["work_dir"])
classes = control_classes(neurons, real_edges, cfg["connectome"]["ears"])     # hops measured on the real slice
assert len(CLASSES) == len(STAGES)
stage_of = classes                                                           # 0 ears .. 3 motor, as STAGES

n_dyn = 2 * 2 * (len(STAGES) + len(LAGS_MS) - 1) + 2 * len(GAINS)
bar = tqdm(total=n_dyn + 6, unit="task", desc="tracing")
say = bar.write


def average_precision(score, y):
    order = np.argsort(-score)
    y = y[order]
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / max(y.sum(), 1))


def readable(x, y):
    """Average precision of the best straight-line (logistic) readout of y from x, fitted on the same frames."""
    xt = torch.as_tensor((x - x.mean(0)) / (x.std(0) + 1e-6), dtype=torch.float32)
    t = torch.as_tensor(y, dtype=torch.float32)
    lin = torch.nn.Linear(xt.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=150, line_search_fn="strong_wolfe")
    pw = (len(t) - t.sum()) / t.sum().clamp_min(1)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(lin(xt)[:, 0], t, pos_weight=pw) \
            + 1e-3 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    with torch.no_grad():
        return average_precision(lin(xt)[:, 0].numpy(), y)


def participation(x):
    x = x[:, x.std(0) > 1e-9]
    if x.shape[1] == 0:
        return 0.0
    ev = np.clip(np.linalg.eigvalsh(np.cov(((x - x.mean(0)) / x.std(0)).T)), 0, None)
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def patterns(rates):
    """(clips, frames, neurons) -> (clips, frames, n_pcs): the stage's main patterns of activity."""
    c, f, n = rates.shape
    x = rates.reshape(-1, n).astype(np.float64)
    x = x[:, x.std(0) > 1e-9] if (x.std(0) > 1e-9).any() else x[:, :1]
    x = (x - x.mean(0)) / (x.std(0) + 1e-12)
    _, _, vt = np.linalg.svd(x[::4], full_matrices=False)                    # directions from every 4th frame
    k = min(n_pcs, vt.shape[0])
    return (x @ vt[:k].T).reshape(c, f, k)


def history(p, live, lag_frames=0):
    """Features at each scored frame: block means of the patterns over the 200 ms ending lag_frames later."""
    c, f, k = p.shape
    cs = np.concatenate([np.zeros((c, 1, k)), np.cumsum(p, 1)], 1)
    feats = []
    for j in range(N_BLOCKS):
        end = np.clip(np.arange(f) + 1 + lag_frames - j * BLOCK, 0, f)
        start = np.clip(end - BLOCK, 0, f)
        feats.append((cs[:, end] - cs[:, start]) / np.maximum(end - start, 1)[None, :, None])
    return np.concatenate(feats, 2)[live]


def run_network(model, audio):
    with torch.no_grad():
        out, _ = model(audio, record=True)
    return out["rates"].numpy()


# ---- checks 1, 4, 5: the networks' activity ----
dyn = {}
for name, control in (("real", None), ("control-1", 1)):
    torch.manual_seed(cfg["model"]["init_seed"])
    s = setup("baby", control, "cpu", cfg)
    model, train = s["model"], s["train"]
    rng = np.random.default_rng(mc["clip_seed"])
    songs = sorted(train.songs.index)[:mc["songs"]]
    clips = [train.clip(song, float(rng.uniform(0, train.songs.duration[song] - train.clip_s)))
             for song in songs * mc["clips_per_song"]][:n_clips]
    b = to_tensors(clips, "cpu")
    live = (b["mask"] > 0).numpy()
    h = b["hits"]
    y = torch.maximum(h, torch.maximum(torch.nn.functional.pad(h[:, :-1], (0, 0, 1, 0)),
                                       torch.nn.functional.pad(h[:, 1:], (0, 0, 0, 1))))[b["mask"] > 0].numpy()
    model.eval()
    gains = {}
    for g in GAINS:                                                          # check 4: untrained, lower gain
        model.gain_scale = g
        r = run_network(model, b["audio"])[..., stage_of == 3]
        x = history(patterns(r), live)
        gains[str(g)] = {"motor_signals": participation(r[live]),
                         "readable": {p: readable(x, y[:, d]) for d, p in enumerate(pieces)}}
        bar.update(1)
        say(f"{name}, untrained at {g} x gain: motor signals {gains[str(g)]['motor_signals']:.1f} "
            f"({time.time() - t0:.0f} s)")
    model.gain_scale = 1.0
    for state in ("untrained", "trained"):
        if state == "trained":
            model.load_state_dict(torch.load(reports_dir() / "memorise" / name / "final.pt", map_location="cpu")["model"])
            model.eval()
        rates = run_network(model, b["audio"])
        res = {"stages": {}, "lags": {}}
        for k, stage in enumerate(STAGES):
            r = rates[..., stage_of == k]
            p = patterns(r)
            x = history(p, live)
            res["stages"][stage] = {"neurons": int((stage_of == k).sum()),
                                    "active": int((r[live].std(0) > 1e-9).sum()),
                                    "signals": participation(r[live]),
                                    "readable": {pc: readable(x, y[:, d]) for d, pc in enumerate(pieces)}}
            bar.update(1)
            say(f"{name} {state}, {stage}: signals {res['stages'][stage]['signals']:.1f}, readable kick/snare/hi-hat "
                + "/".join(f"{res['stages'][stage]['readable'][q]:.3f}" for q in MAIN) + f" ({time.time() - t0:.0f} s)")
            if k == 3:
                motor_p = p
        res["lags"]["0"] = res["stages"]["motor neurons"]["readable"]
        for lag in LAGS_MS[1:]:                                              # check 5: arriving late?
            x = history(motor_p, live, lag // frame_ms)
            res["lags"][str(lag)] = {pc: readable(x, y[:, d]) for d, pc in enumerate(pieces)}
            bar.update(1)
        tau = model.tau_ms().detach().numpy()[model.type_idx.numpy()]
        res["tau_ms_median"] = {st: float(np.median(tau[stage_of == k])) for k, st in enumerate(STAGES)}
        if state == "untrained":
            res["gain_1.0_motor"] = {"motor_signals": res["stages"]["motor neurons"]["signals"],
                                     "readable": res["stages"]["motor neurons"]["readable"]}
        dyn[f"{name} {state}"] = res
        del rates
    dyn[f"{name} gains"] = gains
base = {p: float(y[:, d].mean()) for d, p in enumerate(pieces)}


# ---- checks 2, 3: the wiring alone ----
def wiring(edges, drop_silenced):
    n = len(neurons)
    silenced = neurons.silenced.to_numpy()
    sign = neurons.sign.to_numpy(np.float64)
    pre, post, syn = edges.pre.to_numpy(), edges.post.to_numpy(), edges.synapses.to_numpy(np.int64)
    if drop_silenced:
        keep = ~silenced[pre]
        pre, post, syn = pre[keep], post[keep], syn[keep]
    ear = ear_mask(neurons, cfg["connectome"]["ears"])
    total_in = np.bincount(edges.post.to_numpy(), weights=edges.synapses.to_numpy(np.float64), minlength=n)
    w = csr_matrix((syn / np.maximum(total_in[post], 1), (post, pre)), shape=(n, n))
    ws = csr_matrix((sign[pre] * syn / np.maximum(total_in[post], 1), (post, pre)), shape=(n, n))
    reach, signed = ear.astype(float), ear.astype(float)
    for _ in range(3):
        reach = np.where(ear, 1.0, w @ reach)
        signed = np.where(ear, 1.0, ws @ signed)
    out = {}
    src, sink = n, n + 1
    for d, p in enumerate(pieces):
        group = np.flatnonzero((neurons.drum == p).to_numpy())
        rows = np.r_[pre, np.full(ear.sum(), src), group]
        cols = np.r_[post, np.flatnonzero(ear), np.full(len(group), sink)]
        cap = np.r_[syn, np.full(ear.sum() + len(group), 10 ** 9)]
        g = csr_matrix((cap.astype(np.int64), (rows, cols)), shape=(n + 2, n + 2))
        g.sum_duplicates()
        g.data = g.data.astype(np.int32 if g.data.max() < 2 ** 31 else np.int64)
        out[p] = {"max_flow": int(maximum_flow(g, src, sink).flow_value),
                  "reach_3": float(reach[group].mean()), "signed_reach_3": float(signed[group].mean())}
    return out


struct = {}
for seed in [None, 1, 2, 3, 4, 5]:
    label = "real" if seed is None else f"control-{seed}"
    edges = real_edges if seed is None else load_slice(paths["work_dir"], seed)[1]
    struct[label] = {"silenced_removed": wiring(edges, True), "silenced_kept": wiring(edges, False)}
    bar.update(1)
    say(f"wiring {label} done ({time.time() - t0:.0f} s)")
bar.close()

# ---- report ----
print("\n1  READABLE HITS BY STAGE (average precision, last 200 ms; chance: "
      + ", ".join(f"{p} {base[p]:.3f}" for p in MAIN) + ")")
print(f"   {'stage':<22}" + "".join(f"{k:>24}" for k in ("real untrained", "control-1 untrained", "real trained",
                                                         "control-1 trained")))
for st in STAGES:
    cells = []
    for k in ("real untrained", "control-1 untrained", "real trained", "control-1 trained"):
        rd = dyn[k]["stages"][st]["readable"]
        cells.append("/".join(f"{rd[p]:.3f}" for p in MAIN))
    print(f"   {st:<22}" + "".join(f"{c:>24}" for c in cells))
print("   (each cell: kick/snare/closed hi-hat)")

print("\n4  INDEPENDENT SIGNALS BY STAGE (participation ratio / active neurons)")
for st in STAGES:
    print(f"   {st:<22}" + "".join(f"{dyn[k]['stages'][st]['signals']:>14.1f} /{dyn[k]['stages'][st]['active']:<5}"
                                   for k in ("real untrained", "control-1 untrained", "real trained", "control-1 trained")))
print("   untrained motor neurons at lower starting gain: signals; readable kick/snare/closed hi-hat")
for name in ("real", "control-1"):
    for g in GAINS + [1.0]:
        r = dyn[f"{name} untrained"]["gain_1.0_motor"] if g == 1.0 else dyn[f"{name} gains"][str(g)]
        print(f"   {name:<10} {g:>4} x  signals {r['motor_signals']:>5.1f}; "
              + "/".join(f"{r['readable'][p]:.3f}" for p in MAIN))

print("\n5  MOTOR NEURONS, ALLOWED TO SEE PAST EACH MOMENT (readable kick/snare/closed hi-hat)")
for k in ("real trained", "control-1 trained"):
    print(f"   {k:<18}" + "".join(f"  +{lag} ms " + "/".join(f"{dyn[k]['lags'][str(lag)][p]:.3f}" for p in MAIN)
                                  for lag in LAGS_MS))
print("   median learned time constant (ms) by stage:")
for k in ("real trained", "control-1 trained"):
    print(f"   {k:<18}" + "".join(f"  {st} {v:.0f}" for st, v in dyn[k]["tau_ms_median"].items()))

print("\n2-3  WIRING ALONE, per drum: max flow of synapses from the ears | chance a backward walk reaches an ear in "
      "3 steps | signed")
for mode in ("silenced_removed", "silenced_kept"):
    print(f"   silenced neurons {'removed (as in the model)' if mode == 'silenced_removed' else 'kept'}:")
    print(f"   {'drum':<13}{'real':>30}{'controls 1-5, mean (min-max)':>48}")
    for p in pieces:
        r = struct["real"][mode][p]
        cs = [struct[f"control-{i}"][mode][p] for i in range(1, 6)]
        rng_ = lambda key: (f"{np.mean([c[key] for c in cs]):.3g} ({min(c[key] for c in cs):.3g}-"
                            f"{max(c[key] for c in cs):.3g})")
        print(f"   {p:<13}{r['max_flow']:>8} | {r['reach_3']:.3f} | {r['signed_reach_3']:+.3f}    "
              f"{rng_('max_flow')} | {rng_('reach_3')} | {rng_('signed_reach_3')}")

if args.trial:
    sys.exit(f"\ntrial run: nothing written ({time.time() - t0:.0f} s)")
report = {"dynamics": dyn, "wiring": struct, "base_rate": base, "patterns_per_stage": n_pcs}
(reports_dir() / "trace_information.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
print(f"\nwrote {reports_dir() / 'trace_information.json'} ({time.time() - t0:.0f} s)")
