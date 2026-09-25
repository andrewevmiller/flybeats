"""Why do kick and low tom read backwards? Decompose the probe's head r."""
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(r"C:\Users\ricos\Documents\AI Databases\flybeats\git")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import train as T
from build import build_model, get_subgraph
from decoder import DrumKit
from probe_velocity import test_slice

ckp = Path(sys.argv[1])
np.random.seed(0); torch.manual_seed(0)
ck = torch.load(ckp, map_location="cpu", weights_only=False)
cfg = ck["config"]; cfg["train"]["workers"] = 0; kit = DrumKit(ck["kit"])
model, _ = build_model(cfg, get_subgraph(cfg), n_styles=max(ck["n_styles"], 1))
model.load_state_dict(ck["model"]); model.eval()
_, va, _ = T.build_loaders(cfg, kit)
ds = va.dataset
M, V, Y, S = [], [], [], []
with torch.no_grad():
    for wav, y, vel_y, style, _ in va:
        sid = style if model.genre is not None else None
        drive = model.encoder(wav)
        tonic = model.genre(sid) if sid is not None else None
        full, _ = model.rnn(drive, tonic=tonic, return_all=True)
        motor = full[:, :, model.rnn.motor_idx]
        n = min(drive.shape[1], y.shape[1])
        M.append(motor[:, :n].numpy()); V.append(vel_y[:, :n].numpy())
        Y.append(y[:, :n].numpy()); S.append(style.numpy())
M, V, Y, S = (np.concatenate(a) for a in (M, V, Y, S))
with torch.no_grad():
    H = model.decoder.velocity(torch.from_numpy(M)).numpy()
C, T_ = Y.shape[:2]
clip_id = np.repeat(np.arange(C)[:, None], T_, 1)
rows = getattr(ds, "rows", None)
names = [ (r.get("style"), r.get("midi_filename", r.get("id", ""))) for r in rows] if rows else None

def r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else float("nan")

def cluster_ci(h, t, cid, n=2000):
    rng = np.random.default_rng(0); u = np.unique(cid); out = []
    idx = {c: np.where(cid == c)[0] for c in u}
    for _ in range(n):
        pick = np.concatenate([idx[c] for c in rng.choice(u, len(u))])
        out.append(r(h[pick], t[pick]))
    out = np.array(out); out = out[~np.isnan(out)]
    return np.percentile(out, 2.5), np.percentile(out, 97.5)

def demean(x, cid):
    o = x.copy()
    for c in np.unique(cid):
        m = cid == c; o[m] -= x[m].mean()
    return o

print(ckp.parent.name)
for cname in kit.classes:
    c = kit.classes.index(cname)
    hit = Y[:, :, c] > 0.95
    if hit.sum() < 50: continue
    h, t, cid = H[:, :, c][hit], V[:, :, c][hit], clip_id[hit]
    te = test_slice(len(t))
    ht, tt, ct = h[te], t[te], cid[te]
    u = np.unique(ct)
    cm_h = np.array([ht[ct == k].mean() for k in u]); cm_t = np.array([tt[ct == k].mean() for k in u])
    lo, hi = cluster_ci(ht, tt, ct)
    print(f"{cname:<11} test rows {len(tt):>5} from {len(u):>2} clips | probe r {r(ht,tt):+.2f} "
          f"clip-boot [{lo:+.2f},{hi:+.2f}] | within-clip {r(demean(ht,ct),demean(tt,ct)):+.2f} "
          f"between-clip {r(cm_h,cm_t):+.2f} | ALL {len(np.unique(cid))} clips r {r(h,t):+.2f} "
          f"within {r(demean(h,cid),demean(t,cid)):+.2f}")
np.savez(ckp.parent.name + "_diag.npz", H=H, V=V, Y=Y, S=S)

