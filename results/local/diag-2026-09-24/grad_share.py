"""How much of each optimiser step reaches the two readouts?

Instruments a handful of real training steps from a checkpoint's own config
(fresh init, seed 0, same data order as train.py) and records, per step:
the pre-clip global grad norm, the clip coefficient, each group's grad norm,
and how far AdamW actually moved each group. Single-threaded so it can run
beside a training job without the oversubscription collapse.

    OMP_NUM_THREADS=1 .venv/Scripts/python.exe results/local/diag-2026-09-24/grad_share.py \
        configs/velocity_probe_cpu.yaml 6
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
import train as T                                   # noqa: E402
from build import build_model, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit                         # noqa: E402

torch.set_num_threads(1)
cfg = load_config(ROOT / sys.argv[1])
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 6
cfg["train"]["workers"] = 0
seed = cfg["train"].get("seed", 0)
torch.manual_seed(seed); np.random.seed(seed)
sg = get_subgraph(cfg)
kit = DrumKit.from_tier(cfg["kit"]["tier"])
tr, _, n_styles = T.build_loaders(cfg, kit)
model, kit = build_model(cfg, sg, n_styles=max(int(n_styles or 1), 1))
T.calibrate_encoder(model, tr.dataset, cfg, torch.device("cpu"))
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                        lr=cfg["train"].get("lr", 3e-3),
                        weight_decay=cfg["train"].get("weight_decay", 0.0))

groups = {
    "vel_head": [model.decoder.vel_readout.weight, model.decoder.vel_readout.bias],
    "onset_head": [model.decoder.readout.weight, model.decoder.readout.bias],
    "rnn": list(model.rnn.parameters()),
    "encoder": [p for p in model.encoder.parameters() if p.requires_grad],
}
if model.genre is not None:
    groups["genre"] = list(model.genre.parameters())
names = {id(p): g for g, ps in groups.items() for p in ps}
other = [p for p in model.parameters() if p.requires_grad and id(p) not in names]
if other:
    groups["other"] = other


def gnorm(ps):
    gs = [p.grad.detach().flatten() for p in ps if p.grad is not None]
    return float(torch.cat(gs).norm()) if gs else 0.0


rec = []
real_clip = torch.nn.utils.clip_grad_norm_
real_step = opt.step


def clip(params, max_norm, *a, **k):
    rec.append({"pre": {g: gnorm(ps) for g, ps in groups.items()}})
    total = real_clip(params, max_norm, *a, **k)
    rec[-1]["total"] = float(total)
    rec[-1]["coef"] = min(1.0, max_norm / (float(total) + 1e-6))
    rec[-1]["before"] = {g: [p.detach().clone() for p in ps] for g, ps in groups.items()}
    return total


def step(*a, **k):
    out = real_step(*a, **k)
    r = rec[-1]
    before = r.pop("before")
    r["moved"] = {g: float(torch.cat([(p.detach() - b).flatten() for p, b in zip(ps, before[g])]).abs().mean())
                  for g, ps in groups.items()}
    return out


torch.nn.utils.clip_grad_norm_ = clip
opt.step = step
T.run_epoch(model, list(itertools.islice(iter(tr), n_steps)), opt, cfg, torch.device("cpu"), train=True)

print(f"lr {opt.param_groups[0]['lr']}  grad_clip {cfg['train'].get('grad_clip', 1.0)}  "
      f"velocity_weight {cfg['train'].get('velocity_weight', 1.0)}")
for i, r in enumerate(rec):
    pre = "  ".join(f"{g} {v:.2e}" for g, v in r["pre"].items())
    mv = "  ".join(f"{g} {v:.1e}" for g, v in r["moved"].items())
    print(f"step {i}: total {r['total']:.2e} coef {r['coef']:.1e}\n   grad  {pre}\n   moved {mv}")
