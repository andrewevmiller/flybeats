"""numpy port of the engine's resampler (plugin/engine/src/dsp.cpp, class Resampler): the same prototype, the same
phase arithmetic, so the two agree to rounding. Used by the parity harness to make 16 kHz audio from host-rate
audio exactly as the plugin would, and by the engine tests.

Output k uses inputs q-H .. q+H around q = floor(k*M/L): a fixed delay of H input samples.
"""
import math

import numpy as np

HALF_TAPS = 96
BETA = 8.96
TRANSITION_HZ = 1300.0


def bessel_i0(x):
    s = term = 1.0
    y = x * x / 4.0
    for k in range(1, 200):
        term *= y / (k * k)
        s += term
        if term < 1e-17 * s:
            break
    return s


def prototype(host_rate, out_rate=16000, half_taps=HALF_TAPS, beta=BETA, transition_hz=TRANSITION_HZ):
    g = math.gcd(host_rate, out_rate)
    L, M = out_rate // g, host_rate // g
    fu = host_rate * L
    fc = out_rate / 2.0 - transition_hz / 2.0
    cut = 2.0 * fc / fu
    half = half_taps * L
    i = np.arange(-half, half + 1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        s = np.where(i == 0, cut, np.sin(np.pi * cut * i) / (np.pi * i))
    r = i / half
    i0b = bessel_i0(beta)
    w = np.array([bessel_i0(beta * math.sqrt(max(0.0, 1.0 - rr * rr))) / i0b for rr in r])
    return L, M, (L * s * w)


def resample(x, host_rate, out_rate=16000, half_taps=HALF_TAPS):
    """Offline: every output the streaming engine would have produced for input x (float32)."""
    x = np.asarray(x, np.float32)
    if host_rate == out_rate:
        return x.copy()
    L, M, h = prototype(host_rate, out_rate, half_taps)
    half = half_taps * L
    n = len(x)
    out = []
    k = 0
    while True:
        num = k * M
        q = num // L
        if q + half_taps > n - 1:
            break
        p = num - q * L
        acc = 0.0
        for j in range(-half_taps, half_taps + 1):
            tap = p + j * L
            if tap < -half or tap > half:
                continue
            idx = q - j
            if idx < 0:
                continue
            acc += float(x[idx]) * h[tap + half]
        out.append(np.float32(acc))
        k += 1
    return np.array(out, np.float32)


def resample_fast(x, host_rate, out_rate=16000, half_taps=HALF_TAPS):
    """The same outputs, vectorised per phase (sums in a different order: equal to rounding, not bit for bit)."""
    x = np.asarray(x, np.float64)
    if host_rate == out_rate:
        return x.astype(np.float32)
    L, M, h = prototype(host_rate, out_rate, half_taps)
    half = half_taps * L
    n = len(x)
    xp = np.concatenate([np.zeros(half_taps + 1), x])          # index shift so q - j >= -H maps inside
    ks = np.arange(0, (n * L) // M + 2)
    nums = ks * M
    qs = nums // L
    ks = ks[qs + half_taps <= n - 1]
    out = np.zeros(len(ks))
    for p in range(L):
        sel = np.flatnonzero((ks * M - (ks * M // L) * L) == p)
        if not len(sel):
            continue
        taps = np.array([p + j * L for j in range(-half_taps, half_taps + 1)])
        ok = (taps >= -half) & (taps <= half)
        js = np.arange(-half_taps, half_taps + 1)[ok]
        hv = h[taps[ok] + half]
        q = (ks[sel] * M) // L
        idx = q[:, None] - js[None, :] + half_taps + 1         # into xp; negatives were zeros
        out[sel] = (xp[idx] * hv[None, :]).sum(1)
    return out.astype(np.float32)
