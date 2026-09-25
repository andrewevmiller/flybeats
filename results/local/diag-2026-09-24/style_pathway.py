"""Where can a steady push enter the network and still move the drums?

Written on 24 Sep 2026 while asking why the style dial does nothing.
``scripts/measure_variety.py`` found that switching style changed 0% of hits.
The largest shift any style gave any drum's output was 0.001, on the 0..1 scale
where a hit needs 0.5.

Style is a tonic current into the 25 octopaminergic neurons, capped at +/-0.5
per neuron (``GenreModulation``). This script injects a constant current into
one group of neurons at a time and reports how far the drum outputs move. It
runs offline, the encoder then the recurrence then the decoder, on the first
8 s of one GMD test clip.

Found on runs/velocity_probe_cpu/best.pt, every neuron pushed the same way:

    push into          0.5      5       50     (max drum-output shift)
    octopaminergic   0.0004   0.0068  0.0363   (silencing them entirely: 0.0012)
    pC1              0.0020   0.1866  0.5300   (at 50, kick/snare/hats/toms play)
    random 25        0.0038   0.0830  0.4159
    motor            0.1741   0.7485  0.8804

Octopamine is not far from the motor neurons: median 2 hops, 18 direct
synapses. Training left its outgoing gains where they started (x1.00). Its
activity simply does not carry to the output at this operating point, and
random neurons carry a push further. Only the snare sits near the threshold
(0.47-0.70); every other drum stays at 0.45 or below.

    OMP_NUM_THREADS=1 .venv/Scripts/python.exe results/local/diag-2026-09-24/style_pathway.py \
        --checkpoint runs/velocity_probe_cpu/best.pt [--sign -1] [--groups octopaminergic pC1]

``--config`` builds an untrained model instead. Use it to check a new genre
target before spending a training run on it.
"""
import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--sign", type=float, default=1.0, help="+1 pushes up, -1 down")
    ap.add_argument("--amps", type=float, nargs="+", default=[0.5, 5.0, 50.0])
    ap.add_argument("--groups", nargs="+", default=["octopaminergic", "pC1", "random", "motor"])
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args(argv)
    torch.set_num_threads(1)

    from build import build_model, get_subgraph, load_config, role_index
    import measure_variety as M

    if a.checkpoint:
        ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
        cfg = ck["config"]
        sg = get_subgraph(cfg)
        model, kit = build_model(cfg, sg, n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
        model.load_state_dict(ck["model"])
    else:
        cfg = load_config(a.config)
        sg = get_subgraph(cfg)
        model, kit = build_model(cfg, sg, n_styles=18)
    model.eval()
    roles = role_index(sg)
    n = sg.n_nodes
    motor = np.asarray(model.rnn.motor_idx.cpu() if torch.is_tensor(model.rnn.motor_idx)
                       else model.rnn.motor_idx)

    root = ROOT / cfg["data"].get("root", "data/egmd/groove")
    song = M.pick_songs(root, 1, a.seconds, 0)[0]
    audio = M.load_audio(root / song["audio_filename"], cfg["audio"]["sample_rate"], a.seconds)
    with torch.no_grad():
        drive = model.encoder(torch.from_numpy(audio).unsqueeze(0))

    def run(tonic):
        with torch.no_grad():
            rates, _ = model.rnn(drive, state=None, tonic=tonic, return_all=True)
            prob = torch.sigmoid(model.decoder(rates[:, :, model.rnn.motor_idx])).squeeze(0).numpy()
        return prob, rates.squeeze(0).numpy()

    p0, r0 = run(None)
    rng = np.random.default_rng(0)
    others = np.setdiff1d(np.arange(n), np.concatenate([motor, roles["octopaminergic"]]))
    print(f"{song['audio_filename']}: drums at rest reach "
          + ", ".join(f"{c} {p0[:, i].max():.2f}" for i, c in enumerate(kit.classes)))
    print(f"{'push into':<18}{'amp':>7}{'drum shift':>12}{'group rate shift':>18}"
          f"{'motor rate shift':>18}   drums over 0.5")
    for g in a.groups:
        idx = rng.choice(others, 25, replace=False) if g == "random" else roles[g]
        for amp in a.amps:
            t = torch.zeros(1, n)
            t[0, torch.as_tensor(idx)] = a.sign * amp
            p, r = run(t)
            over = [c for i, c in enumerate(kit.classes) if (p[:, i] >= 0.5).any()]
            print(f"{g + f' ({len(idx)})':<18}{amp:>7g}{np.abs(p - p0).max():>12.4f}"
                  f"{np.abs(r[:, idx] - r0[:, idx]).mean():>18.4f}"
                  f"{np.abs(r[:, motor] - r0[:, motor]).mean():>18.4f}   {','.join(over)}")
    if model.genre is not None:
        with torch.no_grad():
            shifts = [np.abs(run(model.genre(torch.tensor([s])))[0] - p0).max()
                      for s in range(model.genre.embed.num_embeddings)]
        print(f"the model's own styles: largest drum shift {max(shifts):.4f}, "
              f"median {np.median(shifts):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
