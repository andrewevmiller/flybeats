"""Phase 5: streaming inference and MIDI out.

Ring-buffer audio in, stateful inference, MIDI out to a drum sampler. The
network is recurrent and its state carries across blocks, so the only thing
that has to be right is that each block is encoded with the same causal context
the training path used -- which is why ``AudioToJO.forward_window`` exists and
is shared between the two.

Latency budget from PLAN.md: ~20 ms buffer + ~10 ms inference. ``--benchmark``
measures the inference half on this machine so the buffer size can be chosen
against a real number rather than a hoped-for one.

``sounddevice``/``mido``/``python-rtmidi`` need system audio and MIDI libraries
and are imported lazily, so the benchmark and the offline renderer run without
them.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


class StreamingDrummer:
    """Stateful block-wise inference with per-class refractory gating."""

    def __init__(self, model, kit, sample_rate: int, step_ms: float,
                 threshold: float = 0.3, refractory_ms: float = 50.0,
                 device: torch.device | None = None):
        self.model = model.eval()
        self.kit = kit
        self.sample_rate = sample_rate
        self.step_ms = step_ms
        self.threshold = threshold
        self.refractory_steps = max(1, int(refractory_ms / step_ms))
        self.device = device or torch.device("cpu")

        self.hop = model.encoder.hop
        # Ring buffer holding the causal context the encoder needs, so the
        # first step of every block is encoded exactly as it would be offline.
        self.context = deque(maxlen=model.encoder.context_samples)
        self.context.extend([0.0] * model.encoder.context_samples)

        self.state = None
        self.step = 0
        self.last_fire = {c: -10_000 for c in kit.classes}
        self.prev = np.zeros(kit.n, dtype=np.float32)
        self.tonic = None

    def set_style(self, style_id: int | None) -> None:
        """Swap the groove without swapping checkpoints."""
        if self.model.genre is None or style_id is None:
            self.tonic = None
            return
        with torch.no_grad():
            self.tonic = self.model.genre(torch.tensor([style_id], device=self.device))

    @torch.no_grad()
    def push(self, block: np.ndarray) -> list[tuple[int, float, str]]:
        """Feed one audio block; return ``(midi_note, velocity, class)`` triggers."""
        ctx = np.fromiter(self.context, dtype=np.float32, count=len(self.context))
        self.context.extend(block.astype(np.float32).tolist())

        n_steps = len(block) // self.hop
        if n_steps == 0:
            return []

        wav = torch.from_numpy(np.concatenate([ctx, block.astype(np.float32)])).unsqueeze(0)
        wav = wav.to(self.device)
        start = len(ctx) // self.hop
        drive = self.model.encoder.forward_window(wav, start, n_steps)

        if self.state is None:
            self.state = self.model.rnn.initial_state(1, self.device, drive.dtype)
        rates, self.state = self.model.rnn(drive, state=self.state, tonic=self.tonic)
        prob = torch.sigmoid(self.model.decoder(rates)).squeeze(0).cpu().numpy()

        events = []
        for k in range(prob.shape[0]):
            step = self.step + k
            for c, name in enumerate(self.kit.classes):
                v = float(prob[k, c])
                rising = v >= self.threshold and v > self.prev[c]
                # still climbing -- wait for the actual peak. Only a lookahead
                # inside this block is available; at a block boundary the next
                # push sees it as a rising edge against the carried prev, which
                # is why prev must be updated on every path through this loop.
                climbing = k + 1 < prob.shape[0] and prob[k + 1, c] > v
                if (rising and not climbing
                        and step - self.last_fire[name] >= self.refractory_steps):
                    self.last_fire[name] = step
                    vel = int(np.clip(40 + 87 * (v - self.threshold) / max(1 - self.threshold, 1e-6),
                                      1, 127))
                    events.append((self.kit.notes[c], vel, name))
                self.prev[c] = v
        self.step += prob.shape[0]
        return events

    def reset(self) -> None:
        self.state = None
        self.step = 0
        self.prev[:] = 0
        self.last_fire = {c: -10_000 for c in self.kit.classes}


def benchmark(model, kit, cfg, block_ms: float = 20.0, n_blocks: int = 50) -> dict:
    """Measure real inference latency per block on this machine."""
    sr = cfg["audio"]["sample_rate"]
    d = StreamingDrummer(model, kit, sr, cfg["audio"]["step_ms"])
    block = np.zeros(int(sr * block_ms / 1000.0), dtype=np.float32)

    d.push(block)                                   # warm up
    times = []
    for _ in range(n_blocks):
        block = np.random.standard_normal(len(block)).astype(np.float32) * 0.1
        t0 = time.perf_counter()
        d.push(block)
        times.append((time.perf_counter() - t0) * 1000.0)

    times = np.array(times)
    return {
        "block_ms": block_ms,
        "inference_ms_mean": float(times.mean()),
        "inference_ms_p95": float(np.percentile(times, 95)),
        "realtime_factor": float(block_ms / times.mean()),
        "meets_budget": bool(np.percentile(times, 95) < block_ms),
    }


def run_live(model, kit, cfg, style: int | None = None, block_ms: float = 20.0,
             midi_port: str | None = None) -> None:
    import mido
    import sounddevice as sd

    sr = cfg["audio"]["sample_rate"]
    drummer = StreamingDrummer(model, kit, sr, cfg["audio"]["step_ms"])
    drummer.set_style(style)

    port = mido.open_output(midi_port) if midi_port else mido.open_output()
    print(f"MIDI -> {port.name}; listening at {sr} Hz, {block_ms} ms blocks. Ctrl-C to stop.")

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        for note, vel, _ in drummer.push(indata[:, 0]):
            port.send(mido.Message("note_on", channel=9, note=note, velocity=vel))
            port.send(mido.Message("note_off", channel=9, note=note, velocity=0))

    with sd.InputStream(channels=1, samplerate=sr,
                        blocksize=int(sr * block_ms / 1000.0), callback=callback):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nstopped.")
        finally:
            port.close()


def render_file(model, kit, cfg, wav_path: Path, out_mid: Path,
                style: int | None = None, block_ms: float = 20.0) -> int:
    """Offline: run a wav through the streaming path and write a MIDI file.

    Uses the same StreamingDrummer as live playback, so what this renders is
    what the live path would have produced -- no separate offline code to drift.
    """
    import soundfile as sf
    import pretty_midi

    sr = cfg["audio"]["sample_rate"]
    audio, in_sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if in_sr != sr:
        audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * sr / in_sr)),
                          np.arange(len(audio)), audio).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak

    drummer = StreamingDrummer(model, kit, sr, cfg["audio"]["step_ms"])
    drummer.set_style(style)

    block = int(sr * block_ms / 1000.0)
    pm = pretty_midi.PrettyMIDI()
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="flybeats")
    n = 0
    for a in range(0, len(audio) - block + 1, block):
        t = a / sr
        for note, vel, _ in drummer.push(audio[a: a + block]):
            inst.notes.append(pretty_midi.Note(velocity=vel, pitch=note,
                                               start=t, end=t + 0.05))
            n += 1
    pm.instruments.append(inst)
    out_mid.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_mid))
    return n


def load_checkpoint(path: Path, device: torch.device):
    from build import build_model, get_subgraph
    from decoder import DrumKit

    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    sg = get_subgraph(cfg)
    model, kit = build_model(cfg, sg, n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), DrumKit(ck["kit"]), cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--config", type=Path, default=None,
                    help="benchmark an untrained model straight from a config, to size "
                         "the subgraph against the latency budget before training it")
    ap.add_argument("--benchmark", action="store_true", help="measure inference latency and exit")
    ap.add_argument("--render", type=Path, help="wav in -> MIDI out, via the streaming path")
    ap.add_argument("--out", type=Path, default=ROOT / "runs" / "render.mid")
    ap.add_argument("--style", type=int, default=None)
    ap.add_argument("--block-ms", type=float, default=20.0)
    ap.add_argument("--midi-port", type=str, default=None)
    ap.add_argument("--lesion", type=str, default=None,
                    help="mute a confirmed population during playback, e.g. pIP10")
    ap.add_argument("--slider", nargs=2, action="append", metavar=("NAME", "VALUE"),
                    default=[], help="e.g. --slider drive 1.5 --slider pocket 1.2")
    a = ap.parse_args(argv)

    device = torch.device("cpu")
    if a.checkpoint:
        model, kit, cfg = load_checkpoint(a.checkpoint, device)
    elif a.config:
        from build import build_model, get_subgraph, load_config
        cfg = load_config(a.config)
        model, kit = build_model(cfg, get_subgraph(cfg))
        model = model.to(device).eval()
    else:
        ap.error("pass --checkpoint, or --config to benchmark an untrained model")

    from build import get_subgraph, role_index
    roles = role_index(get_subgraph(cfg))
    for name, value in a.slider:
        model.set_slider(name, float(value), roles)
        print(f"slider {name} = {value}")
    if a.lesion:
        from ablations import lesion
        lesion(model, roles, a.lesion, 0.0)
        print(f"lesioned {a.lesion} ({len(roles[a.lesion])} neurons)")

    if a.benchmark:
        r = benchmark(model, kit, cfg, a.block_ms)
        print(f"block {r['block_ms']:.0f} ms | inference {r['inference_ms_mean']:.1f} ms mean, "
              f"{r['inference_ms_p95']:.1f} ms p95 | {r['realtime_factor']:.2f}x realtime")
        print("within budget" if r["meets_budget"] else
              "OVER BUDGET: lower subgraph.max_nodes or raise audio.step_ms")
        return 0 if r["meets_budget"] else 1

    if a.render:
        n = render_file(model, kit, cfg, a.render, a.out, a.style, a.block_ms)
        print(f"wrote {n} notes -> {a.out}")
        return 0

    run_live(model, kit, cfg, a.style, a.block_ms, a.midi_port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
