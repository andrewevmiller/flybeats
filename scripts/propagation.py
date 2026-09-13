"""Can the subgraph carry a time-varying signal from the ears to the wings at all?

``scripts/diagnose.py`` answers "did this checkpoint learn anything"; this
answers the prior question, which turned out to be the blocking one: with the
encoder fixed and the rate regulariser no longer rewarding silence, the drive
into the JO afferents varies strongly (relative temporal variation ~2) and the
wing motor pool still barely moves (~0.006). Measured per hop, the loss is not
spread through the depth of the connectome -- it is a ~30x drop at the *first*
synapse, after which the neurons sit at softplus(0), i.e. receiving nothing.

The suspect is the global gain normalisation rather than the topology.
``model.gain_scale`` is set so the recurrent operator has spectral radius
``model.spectral_radius`` (0.9), which is what makes the Phase 4 arms
comparable. But rho is carried by a small, strongly connected hub subnetwork,
so pinning it at 0.9 divides every weight by that hub's gain and leaves the
typical neuron with a synaptic input far below its own bias.

This sweeps the target radius on an *untrained* model over real audio and
reports what survives at each hop, plus how close the network is to its
saturation clip. Forward passes only -- the point is what the architecture can
carry before any training, so that a Phase 3 run is not spent discovering that
the answer is "nothing".

    python scripts/propagation.py --config configs/v1_8piece_cpu.yaml \
        --radii 0.9 2 5 10 25 --clips 4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as train_mod  # noqa: E402
from build import build_model, device_of, get_subgraph, load_config  # noqa: E402
from decoder import DrumKit  # noqa: E402
from subgraph import hops_from  # noqa: E402


def relative_variation(rates: np.ndarray, mask: np.ndarray) -> float:
    """Mean over the masked neurons of (sd over time) / |mean|.

    Relative, because an absolute threshold calls a unit with sd 0.003 around a
    mean of 0.72 "varying" -- which is the mistake the first diagnostic made.
    """
    if not mask.any():
        return float("nan")
    sd = rates.std(axis=1).mean(axis=0)[mask]
    mu = np.abs(rates.mean(axis=(0, 1)))[mask]
    return float((sd / np.maximum(mu, 1e-12)).mean())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "v1_8piece_cpu.yaml")
    ap.add_argument("--radii", type=float, nargs="+",
                    default=[0.9, 2.0, 5.0, 10.0, 25.0])
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    device = device_of(cfg)
    sg = get_subgraph(cfg)
    kit = DrumKit.from_tier(cfg["kit"]["tier"])
    train_loader, _, n_styles = train_mod.build_loaders(cfg, kit)
    n_styles = max(int(n_styles or 1), 1)

    wav = torch.stack([torch.as_tensor(train_loader.dataset[i][0])
                       for i in range(min(a.clips, len(train_loader.dataset)))]).to(device)

    dist = hops_from(sg, sg.role("sensory").astype(np.int64))
    motor = np.zeros(sg.n_nodes, dtype=bool)
    motor[sg.role("motor").astype(np.int64)] = True
    hops = sorted({int(d) for d in np.unique(dist) if 0 <= d <= 4})
    clip = float(cfg["model"].get("state_clip", 20.0))

    print(f"{sg.summary().splitlines()[0]}   |   {wav.shape[0]} clips, {a.steps} steps")
    print("relative temporal variation of the firing rate, by hop from the JO afferents\n")
    cols = "".join(f"{'hop ' + str(h):>10}" for h in hops)
    print(f"{'radius':>8}{cols}{'motor':>10}{'mean rate':>11}{'at clip':>9}")
    print("-" * (8 + 10 * len(hops) + 30))

    for rho in a.radii:
        torch.manual_seed(cfg["train"].get("seed", 0))
        np.random.seed(cfg["train"].get("seed", 0))
        c = {**cfg, "model": {**cfg["model"], "spectral_radius": rho, "gain_scale": "auto"}}
        model, _ = build_model(c, sg, n_styles=n_styles)
        model = model.to(device).eval()
        train_mod.calibrate_encoder(model, train_loader.dataset, cfg, device)

        with torch.no_grad():
            drive = model.encoder(wav)[:, : a.steps]
            rates, _ = model.rnn(drive, return_all=True)
        r = rates.float().numpy()

        row = "".join(f"{relative_variation(r, dist == h):>10.5f}" for h in hops)
        # Saturation check: the state is clamped at +/- state_clip, and a rate
        # of softplus(clip) means that neuron is pinned and carries nothing.
        at_clip = float((r > np.log1p(np.exp(clip - 1.0))).mean())
        print(f"{rho:>8.2f}{row}{relative_variation(r, motor):>10.5f}"
              f"{r.mean():>11.4f}{at_clip:>9.1%}")

    print("\nA radius that leaves the motor column near zero cannot be trained out of:"
          "\nno gradient on the connectome's gains can create modulation that never"
          "\narrives. A radius whose 'at clip' column is large is saturated, which"
          "\nis the same failure from the other side.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
