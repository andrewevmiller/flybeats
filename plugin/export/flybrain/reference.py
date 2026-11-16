"""The numpy reference engine: the exported brain run frame by frame, with every Lab perturbation.

No PyTorch. scipy.sparse does the one sparse matrix-vector product per frame; its CSR product adds each row's
terms in index order, which the C++ core copies, so the two agree bit for bit in the same precision (checked by
the parity harness). PyTorch's dense product adds in another order, so against PyTorch the contract is a tolerance.

The engine is a streaming object: feed 16 kHz mono samples in blocks of any size; it emits one frame per 80
samples, then hit events one frame late (a peak needs the next frame). `run_clip` is the offline convenience.
"""
import numpy as np
import scipy.sparse as sp

from .brainfile import read_brain

NEG_INF = -np.inf


def sigmoid(x):
    """Numerically safe logistic (torch.sigmoid never overflows either)."""
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.result_type(x, np.float32))
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = e / (1.0 + e)
    return out


# ----------------------------------------------------------------------------------------------- ear

class Ear:
    """Mono 16 kHz samples -> one 16-band feature vector per hop. Bit-for-bit the same whatever the block size."""

    def __init__(self, brain, dtype=np.float64):
        m = brain.meta
        self.n_fft, self.hop, self.n_bands = int(m["n_fft"]), int(m["hop"]), int(m["n_bands"])
        self.dtype = dtype
        # torch.hann_window(N) is periodic: 0.5 - 0.5 cos(2 pi k / N)
        k = np.arange(self.n_fft)
        self.window = (0.5 - 0.5 * np.cos(2 * np.pi * k / self.n_fft)).astype(dtype)
        self.bands = brain["band_matrix"].astype(dtype)
        self.mean, self.std = brain["band_mean"].astype(dtype), brain["band_std"].astype(dtype)
        self.gain, self.offset = brain["band_gain"].astype(dtype), brain["band_offset"].astype(dtype)
        self.reset()

    def reset(self):
        self.buf = np.zeros(self.n_fft - self.hop, self.dtype)   # the last n_fft - hop samples, zeros at start
        self.pending = np.zeros(0, self.dtype)

    def push(self, samples, band_offset_ctl=None):
        """Returns (frames, 16) raw-normalised features and (frames, 16) ear features (gain and offset applied)."""
        x = np.concatenate([self.pending, np.asarray(samples, self.dtype)])
        n_frames = len(x) // self.hop
        raw = np.zeros((n_frames, self.n_bands), self.dtype)
        for t in range(n_frames):
            chunk = x[t * self.hop:(t + 1) * self.hop]
            frame = np.concatenate([self.buf, chunk])
            self.buf = frame[self.hop:]
            spec = np.fft.rfft(frame * self.window)
            power = (spec.real * spec.real + spec.imag * spec.imag)
            raw[t] = np.log1p(power @ self.bands)
        self.pending = x[n_frames * self.hop:]
        feats = (raw - self.mean) / self.std
        ear = feats * self.gain + self.offset
        if band_offset_ctl is not None:
            ear = ear + band_offset_ctl
        return feats, ear


# ----------------------------------------------------------------------------------------------- controls

class Controls:
    """The control block (section 6 of the plan). Every field at neutral gives the network as trained."""

    def __init__(self, brain):
        n, ne, nb = brain.n, len(brain["ear_idx"]), int(brain.meta["n_bands"])
        self.lesion = np.ones(n, np.float32)          # 0 silences a neuron (Lesion Brush); multiplies alive
        self.ear_mute = np.ones(ne, np.float32)       # 0 removes an ear neuron's drive (Deafen an Ear)
        self.gain_scale = 1.0                         # Gain to the Cliff: multiplies every weight
        self.tau_scale = 1.0                          # Tempo of Thought: multiplies tau before the clamp
        self.sign_flip = np.ones(n, np.float32)       # -1 flips every outgoing weight of a sender (Sign Flip)
        self.unsilence = np.zeros(n, np.int8)         # +1/-1 gives a silenced sender that sign (Wake the Modulators)
        self.band_offset = np.zeros(nb, np.float32)   # added to the ear features (Per-band Ear EQ)
        self.inject = []                              # [(neuron, current, frames_left)] current added to the drive
        self.threshold = float(brain.meta["hit_threshold"])

    def is_neutral(self, brain):
        return (np.all(self.lesion == 1) and np.all(self.ear_mute == 1) and self.gain_scale == 1.0
                and self.tau_scale == 1.0 and np.all(self.sign_flip == 1) and np.all(self.unsilence == 0)
                and np.all(self.band_offset == 0) and not self.inject
                and self.threshold == float(brain.meta["hit_threshold"]))


# ----------------------------------------------------------------------------------------------- core

class Core:
    """The sparse recurrent network, one step per frame."""

    def __init__(self, brain, dtype=np.float64):
        self.b, self.dtype = brain, dtype
        n = brain.n
        self.n = n
        self.indptr, self.indices = brain["w_indptr"], brain["w_indices"]
        self.w_values, self.w_unsigned = brain["w_values"].astype(dtype), brain["w_unsigned"].astype(dtype)
        self.sign = brain["sign"].astype(np.int8)
        self.alive0 = brain["alive"].astype(dtype)
        self.bias = brain["bias"].astype(dtype)
        self.tau = brain["tau_ms"].astype(np.float64)
        self.tau_min, self.tau_max = float(brain.meta["tau_min_ms"]), float(brain.meta["tau_max_ms"])
        self.dt = float(brain.meta["frame_ms"])
        self.ear_idx, self.ear_band = brain["ear_idx"], brain["ear_band"]
        self.configure(Controls(brain))
        self.reset()

    def configure(self, c):
        """Apply a control block (at a frame boundary). Rebuilds the matrix only when a weight control changed."""
        self.c = c
        # per-sender multiplier: gain x flip, and the sign override for silenced senders
        vals = self.w_values.copy()
        if np.any(c.unsilence != 0):
            for j in np.flatnonzero(c.unsilence):
                if self.sign[j] == 0:
                    vals[self.indices == j] = int(c.unsilence[j]) * self.w_unsigned[self.indices == j]
        pre_mult = (c.sign_flip.astype(self.dtype) * self.dtype(c.gain_scale))
        vals = vals * pre_mult[self.indices] if not (c.gain_scale == 1.0 and np.all(c.sign_flip == 1)) else vals
        self.W = sp.csr_matrix((vals, self.indices, self.indptr), shape=(self.n, self.n), dtype=self.dtype)
        alive = self.alive0.copy()
        alive[(c.unsilence != 0) & (self.sign == 0)] = 1.0
        self.alive = alive * c.lesion.astype(self.dtype)
        tau = np.clip(self.tau * c.tau_scale, self.tau_min, self.tau_max)
        # the model's alpha = dt / clamp(exp(log_tau)) in float32; at tau_scale 1 use the exported alpha bit for bit
        self.alpha = (self.b["alpha"].astype(self.dtype) if c.tau_scale == 1.0
                      else (self.dt / tau).astype(np.float32).astype(self.dtype))
        self.ear_mute = c.ear_mute.astype(self.dtype)

    def reset(self):
        self.v = np.zeros(self.n, self.dtype)
        self.r = np.zeros(self.n, self.dtype)

    def step(self, ear_in):
        """ear_in: (n_ears,) current per ear neuron (already the band features gathered per ear)."""
        drive = np.zeros(self.n, self.dtype)
        drive[self.ear_idx] = ear_in * self.ear_mute
        if self.c.inject:
            keep = []
            for j, cur, left in self.c.inject:
                if left > 0:
                    drive[j] += self.dtype(cur)
                    keep.append((j, cur, left - 1))
            self.c.inject = keep
        rec = self.W @ self.r
        t = -self.v + rec
        t = t + self.bias
        t = t + drive
        t = t * self.alpha
        self.v = self.v + t
        self.r = np.maximum(self.v, 0) * self.alive
        return self.r


# ----------------------------------------------------------------------------------------------- readout

class Readout:
    def __init__(self, brain, dtype=np.float64):
        self.dtype = dtype
        self.mn_idx, self.mn_drum = brain["mn_idx"], brain["mn_drum"]
        self.n_drums = int(brain.meta["n_drums"])
        self.mean, self.std = brain["mn_mean"].astype(dtype), brain["mn_std"].astype(dtype)
        self.w_hit, self.u_vel = brain["w_hit"].astype(dtype), brain["u_vel"].astype(dtype)
        self.c_hit, self.e_vel = brain["c_hit"].astype(dtype), brain["e_vel"].astype(dtype)
        self.blocks = [int(x) for x in brain["history_blocks"]]
        self.edges = np.cumsum([0] + self.blocks)
        self.back = int(self.edges[-1])
        self.p_hit, self.p_vel = brain["p_hit"].astype(dtype), brain["p_vel"].astype(dtype)
        self.onehot = np.zeros((len(self.mn_idx), self.n_drums), dtype)
        self.onehot[np.arange(len(self.mn_idx)), self.mn_drum] = 1
        self.reset()

    def reset(self):
        self.hist = np.zeros((max(self.back, 1), len(self.mn_idx)), self.dtype)   # ring of the last `back` z
        self.pos = 0
        self.frames = 0

    def step(self, rates):
        mn = rates[self.mn_idx]
        z = (mn - self.mean) / self.std
        feats = [z]
        for j, n_j in enumerate(self.blocks):
            # block j: frames t - edges[j+1] .. t - edges[j] - 1, i.e. lags edges[j]+1 .. edges[j+1]
            lags = np.arange(self.edges[j] + 1, self.edges[j + 1] + 1)
            idx = (self.pos - lags) % self.back
            feats.append(self.hist[idx].sum(0) / n_j)
        if self.back:
            self.hist[self.pos] = z
            self.pos = (self.pos + 1) % self.back
        ph, pv = self.onehot @ self.p_hit, self.onehot @ self.p_vel          # (n_mn, 1 + blocks)
        eff_hit = sum(f * ph[:, k] for k, f in enumerate(feats))
        eff_vel = sum(f * pv[:, k] for k, f in enumerate(feats))
        logits = (eff_hit * self.w_hit) @ self.onehot + self.c_hit
        vel = sigmoid((eff_vel * self.u_vel) @ self.onehot + self.e_vel)
        self.frames += 1
        return mn, z, sigmoid(logits), vel


# ----------------------------------------------------------------------------------------------- decode

class PeakDecoder:
    """loss.peaks, streaming: frame t is a hit if p[t] >= thr, p[t] >= p[t-1] and p[t] > p[t+1].
    p[-1] = -inf and, at the end of the stream, p[T] = -inf. A frame is decided when the next one arrives."""

    def __init__(self, n_drums):
        self.n = n_drums
        self.reset()

    def reset(self):
        self.prev = np.full(self.n, NEG_INF)
        self.cur = None
        self.cur_vel = None
        self.cur_frame = -1

    def push(self, prob, vel, frame, threshold):
        """Returns the decided (frame, hits (n_drums,) bool, vel (n_drums,)) for the previous frame, or None."""
        out = None
        if self.cur is not None:
            hits = (self.cur >= threshold) & (self.cur >= self.prev) & (self.cur > prob)
            out = (self.cur_frame, hits, self.cur_vel)
            self.prev = self.cur
        self.cur, self.cur_vel, self.cur_frame = np.array(prob), np.array(vel), frame
        return out

    def flush(self, threshold):
        if self.cur is None:
            return None
        hits = (self.cur >= threshold) & (self.cur >= self.prev)
        out = (self.cur_frame, hits, self.cur_vel)
        self.reset()
        return out


# ----------------------------------------------------------------------------------------------- engine

class RunawayGuard:
    """Clamp rates at a ceiling; if the mean rate stays above a limit for `frames` frames, trigger: the engine
    mutes, resets and raises a flag. Limits come from the brain's meta.runaway (set at export, per checkpoint)."""

    def __init__(self, brain):
        g = brain.meta.get("runaway") or {}
        self.ceiling = float(g.get("rate_ceiling", np.inf))
        self.mean_limit = float(g.get("mean_rate_limit", np.inf))
        self.frames = int(g.get("frames", 20))
        self.reset()

    def reset(self):
        self.over = 0
        self.triggered = 0

    def check(self, core):
        np.minimum(core.v, self.ceiling, out=core.v)
        np.minimum(core.r, self.ceiling, out=core.r)
        bad = not np.all(np.isfinite(core.v)) or float(core.r.mean()) > self.mean_limit
        self.over = self.over + 1 if bad else 0
        if self.over >= self.frames:
            self.over = 0
            self.triggered += 1
            return True
        return False


class Frame:
    __slots__ = ("index", "bands", "ear", "rates", "mn", "z", "prob", "vel", "hits", "reset_flag", "runaway")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k))


class ReferenceEngine:
    """Streaming: push(samples) at 16 kHz mono returns the decided frames (one hit decision per frame, one frame
    late). Frames also carry the snapshot data (bands, rates, probabilities) for that same frame."""

    def __init__(self, brain, dtype=np.float64, preroll_frames=0):
        self.b = brain
        self.dtype = dtype
        self.ear = Ear(brain, dtype)
        self.core = Core(brain, dtype)
        self.readout = Readout(brain, dtype)
        self.decoder = PeakDecoder(int(brain.meta["n_drums"]))
        self.guard = RunawayGuard(brain)
        self.controls = Controls(brain)
        self.preroll = preroll_frames
        self.pending = {}
        self.reset()

    def reset(self):
        self.ear.reset()
        self.core.reset()
        self.readout.reset()
        self.decoder.reset()
        self.guard.reset()
        self.frame = 0
        self.frames_since_reset = 0
        self.pending = {}
        self.reset_flag = True

    def set_controls(self, c):
        self.controls = c
        self.core.configure(c)

    def push(self, samples):
        feats, ear = self.ear.push(samples, self.controls.band_offset if np.any(self.controls.band_offset) else None)
        out = []
        for t in range(feats.shape[0]):
            ear_in = ear[t][self.core.ear_band]
            rates = self.core.step(ear_in)
            runaway = self.guard.check(self.core)
            mn, z, prob, vel = self.readout.step(self.core.r)
            f = Frame(index=self.frame, bands=feats[t].copy(), ear=ear[t].copy(), rates=self.core.r.copy(), mn=mn.copy(),
                      z=z, prob=prob, vel=vel, hits=None, reset_flag=self.reset_flag, runaway=runaway)
            self.reset_flag = False
            self.pending[self.frame] = f
            decided = self.decoder.push(prob, vel, self.frame, self.controls.threshold)
            self.frame += 1
            self.frames_since_reset += 1
            if decided is not None:
                out.append(self._finish(decided))
            if runaway:
                # soft mute and reset: the state goes to zero, the decoder forgets, the frame counter continues
                self.core.reset(); self.readout.reset(); self.decoder.reset(); self.ear.reset()
                self.frames_since_reset = 0
                self.reset_flag = True
        return out

    def flush(self):
        d = self.decoder.flush(self.controls.threshold)
        return [self._finish(d)] if d is not None else []

    def _finish(self, decided):
        idx, hits, vel = decided
        f = self.pending.pop(idx)
        muted = self.preroll and (idx - (self.frame - self.frames_since_reset)) < self.preroll
        f.hits = np.zeros_like(hits) if muted else hits
        return f


def run_clip(brain, audio16k, dtype=np.float64, controls=None, block=None, preroll_frames=0):
    """Offline: every frame of a 16 kHz mono clip. Returns dict of stacked arrays (frames first)."""
    eng = ReferenceEngine(brain, dtype, preroll_frames)
    if controls is not None:
        eng.set_controls(controls)
    audio16k = np.asarray(audio16k)
    frames = []
    if block is None:
        frames += eng.push(audio16k)
    else:
        for s in range(0, len(audio16k), block):
            frames += eng.push(audio16k[s:s + block])
    frames += eng.flush()
    frames.sort(key=lambda f: f.index)
    keys = ("bands", "ear", "rates", "mn", "z", "prob", "vel", "hits")
    out = {k: np.stack([getattr(f, k) for f in frames]) for k in keys}
    out["frame"] = np.array([f.index for f in frames])
    out["runaway"] = np.array([f.runaway for f in frames])
    return out


def hit_list(hits):
    """(frames, drums) bool -> sorted list of (frame, drum)."""
    f, d = np.nonzero(hits)
    return sorted(zip(f.tolist(), d.tolist()))


def load(path):
    return read_brain(path)
