"""Generate the starter drum kit that ships with the repo.

A sample backend is useless without samples, and a repo cannot ship an 808
without answering for where it came from. So the default kit is synthesised
from first principles -- a pitch-enveloped sine for the kick, filtered noise
for the snare and hats, a detuned pair for the toms. It is not a good kit. It
is a kit that exists, is tiny, is unambiguously ours to ship, and exercises
every path the sample player has: velocity layers, round-robin variants and
choke groups.

Point ``--out`` at a folder and swap in a real kit whenever you have one; the
layout is the documented one.

    python scripts/make_synth_kit.py              # -> kits/synth/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SR = 22_050

#: Three velocity layers, three variants each. Layers differ in brightness and
#: decay the way a harder-struck drum does, not just in level -- level alone is
#: what a gain knob already gives you.
N_LAYERS, N_VARIANTS = 3, 3


def _env(n: int, decay: float) -> np.ndarray:
    return np.exp(-np.arange(n) / (SR * decay)).astype(np.float32)


def kick(layer: int, rng) -> np.ndarray:
    n = int(SR * 0.35)
    t = np.arange(n) / SR
    f0, f1 = 110 + 25 * layer, 42 + 3 * layer
    # pitch sweep from the beater impact down to the body of the drum
    f = f1 + (f0 - f1) * np.exp(-t / (0.018 + 0.004 * (2 - layer)))
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(phase) * _env(n, 0.09 + 0.02 * layer)
    click = rng.standard_normal(n).astype(np.float32) * _env(n, 0.004) * (0.1 + 0.05 * layer)
    return (body + click).astype(np.float32)


def snare(layer: int, rng) -> np.ndarray:
    n = int(SR * 0.22)
    t = np.arange(n) / SR
    tone = (np.sin(2 * np.pi * 185 * t) + 0.7 * np.sin(2 * np.pi * 330 * t))
    tone = tone * _env(n, 0.035 + 0.01 * layer)
    wires = rng.standard_normal(n).astype(np.float32) * _env(n, 0.055 + 0.02 * layer)
    # a harder hit is brighter: less of the shell tone, more of the snares
    mix = 0.55 - 0.12 * layer
    return (mix * tone + (1 - mix) * wires * 1.3).astype(np.float32)


def hat(layer: int, rng, open_: bool = False) -> np.ndarray:
    n = int(SR * (0.55 if open_ else 0.07))
    x = rng.standard_normal(n).astype(np.float32)
    # crude high-pass: difference the noise, twice, then shape it
    x = np.diff(np.diff(x, prepend=x[:1]), prepend=x[:1])
    decay = (0.16 + 0.05 * layer) if open_ else (0.012 + 0.004 * layer)
    return (x * _env(n, decay) * 0.6).astype(np.float32)


def tom(layer: int, rng, f0: float) -> np.ndarray:
    n = int(SR * 0.3)
    t = np.arange(n) / SR
    f = f0 * (1 + 0.35 * np.exp(-t / 0.05))
    phase = 2 * np.pi * np.cumsum(f) / SR
    body = (np.sin(phase) + 0.4 * np.sin(phase * 1.51)) * _env(n, 0.075 + 0.015 * layer)
    skin = rng.standard_normal(n).astype(np.float32) * _env(n, 0.006) * 0.12
    return (body + skin).astype(np.float32)


def crash(layer: int, rng) -> np.ndarray:
    n = int(SR * 1.1)
    x = rng.standard_normal(n).astype(np.float32)
    x = np.diff(x, prepend=x[:1])
    shimmer = 1.0 + 0.25 * np.sin(2 * np.pi * 7.3 * np.arange(n) / SR)
    return (x * _env(n, 0.28 + 0.06 * layer) * shimmer * 0.5).astype(np.float32)


VOICES = {
    "kick": lambda l, r: kick(l, r),
    "snare": lambda l, r: snare(l, r),
    "hat_closed": lambda l, r: hat(l, r, open_=False),
    "hat_open": lambda l, r: hat(l, r, open_=True),
    "tom_low": lambda l, r: tom(l, r, 100.0),
    "tom_mid": lambda l, r: tom(l, r, 145.0),
    "tom_high": lambda l, r: tom(l, r, 200.0),
    "crash": lambda l, r: crash(l, r),
}

MANIFEST = """# The starter kit: synthesised, tiny, and ours to ship. Swap in a real one.
name: synth
sample_rate: 22050
layers: 3            # <layer>_<variant>.wav; layer 0 is the softest
choke_groups:
  hihat: [hat_closed, hat_open]
aliases:             # so a third-party kit's folder names still map
  kick: [bd, bassdrum, kik]
  snare: [sd, sn]
  hat_closed: [chh, hh, hihat_closed]
  hat_open: [ohh, hihat_open]
  crash: [cym, crash1]
"""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "kits" / "synth")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    import soundfile as sf

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "manifest.yaml").write_text(MANIFEST)
    rng = np.random.default_rng(a.seed)

    total = 0
    for cls, make in VOICES.items():
        d = a.out / cls
        d.mkdir(exist_ok=True)
        for layer in range(N_LAYERS):
            for v in range(N_VARIANTS):
                x = make(layer, rng)
                peak = np.abs(x).max()
                if peak > 0:
                    # normalise, then set the layer's level -- so layers differ
                    # in timbre as well as loudness rather than only in gain
                    x = x / peak * (0.45 + 0.22 * layer)
                sf.write(d / f"{layer}_{v}.wav", x.astype(np.float32), SR, subtype="PCM_16")
                total += 1
    size = sum(f.stat().st_size for f in a.out.rglob("*.wav"))
    print(f"{total} samples -> {a.out}  ({size / 1e6:.1f} MB, {len(VOICES)} classes, "
          f"{N_LAYERS} layers x {N_VARIANTS} variants)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
