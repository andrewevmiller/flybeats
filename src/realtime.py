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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Trigger:
    """One drum hit: what, how hard, and *when*.

    ``velocity`` is 0..1 here, not MIDI's 1..127 -- the MIDI scaling belongs at
    the MIDI edge, and a sample bank converting 1..127 back into a gain would be
    a lossy round trip through a unit it should never see. ``note`` rides along
    so MidiBank need not look it up again.

    ``t`` is seconds from the start of the stream, at step resolution. It used
    to be the enclosing audio block's start time, which quantised every hit to
    the block (20 ms) in a model whose whole claim is timing.
    """

    cls: str
    velocity: float
    t: float
    note: int


class StreamingDrummer:
    """Stateful block-wise inference with per-class refractory gating."""

    def __init__(self, model, kit, sample_rate: int, step_ms: float,
                 threshold: float = 0.3, refractory_ms: float = 50.0,
                 class_refractory_ms: dict[str, float] | None = None,
                 speed: float = 1.0,
                 device: torch.device | None = None):
        self.model = model.eval()
        self.kit = kit
        self.sample_rate = sample_rate
        self.step_ms = step_ms
        self.threshold = threshold
        # Speed: core updates per encoder frame, and the dial is continuous.
        # A fractional speed is not a fractional number of updates on any one
        # frame -- there is no such thing -- it is k or k+1 per frame with the
        # remainder carried in a phase accumulator, so the *average* comes out
        # at the dial setting. 2.5 alternates 2 and 3.
        self.speed = max(float(speed), 1e-6)
        self._phase = 0.0
        # What an integer dial resolves to, kept for the benchmark's report and
        # for anything that wants one representative number.
        self.substeps = max(1, int(round(self.speed)))
        # Per class and in milliseconds, not one count of steps. Per class
        # because that is what "hats flutter while the kick stays human" is;
        # in milliseconds because once the core sub-steps at a rate that varies
        # with the music, a refractory measured in steps no longer names a
        # fixed span of time.
        #
        # The default scales with speed, and it has to: at a fixed 50 ms every
        # class is capped at 20 hits/s, so sub-stepping generates peaks the
        # picker then throws away and the dial does nothing at all. Measured,
        # the first time: 12.5 -> 17.5 -> 14.2 -> 12.5 hits/s across speeds
        # 1, 2, 4, 8. Speed sets the resolution *and* the default allocation;
        # a per-class override is absolute, so naming one pins it back to a
        # human timescale while everything else runs fast.
        default_ms = refractory_ms / max(self.speed, 1e-6)
        self.refractory_ms = {c: float((class_refractory_ms or {}).get(c, default_ms))
                              for c in kit.classes}
        self.device = device or torch.device("cpu")

        self.hop = model.encoder.hop
        # Ring buffer holding the causal context the encoder needs, so the
        # first step of every block is encoded exactly as it would be offline.
        self.context = deque(maxlen=model.encoder.context_samples)
        self.context.extend([0.0] * model.encoder.context_samples)

        self.state = None
        self.step = 0           # core output rows emitted so far
        self.frame = 0          # encoder frames consumed so far
        self.last_fire = {c: -1e9 for c in kit.classes}     # seconds
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
    def push(self, block: np.ndarray) -> list[Trigger]:
        """Feed one audio block; return the hits it produced."""
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
        schedule = self._schedule(n_steps)
        rates, self.state = self.model.rnn(drive, state=self.state, tonic=self.tonic,
                                           substeps=schedule)
        prob = torch.sigmoid(self.model.decoder(rates)).squeeze(0).cpu().numpy()
        vhat = self.model.decoder.velocity(rates)
        vhat = None if vhat is None else vhat.squeeze(0).cpu().numpy()

        events: list[Trigger] = []
        times = self._step_times(schedule)
        for k in range(prob.shape[0]):
            t = times[k]
            for c, name in enumerate(self.kit.classes):
                v = float(prob[k, c])
                rising = v >= self.threshold and v > self.prev[c]
                # still climbing -- wait for the actual peak. Only a lookahead
                # inside this block is available; at a block boundary the next
                # push sees it as a rising edge against the carried prev, which
                # is why prev must be updated on every path through this loop.
                climbing = k + 1 < prob.shape[0] and prob[k + 1, c] > v
                if (rising and not climbing
                        and (t - self.last_fire[name]) * 1000.0 >= self.refractory_ms[name]):
                    self.last_fire[name] = t
                    if vhat is not None:
                        # How hard, read at the peak from the head trained on
                        # the drummer's own MIDI velocities.
                        vel = float(np.clip(vhat[k, c], 0.0, 1.0))
                    else:
                        # No velocity head: a model trained before there was
                        # one. Fall back to peak height above the threshold,
                        # normalised -- confidence standing in for dynamics,
                        # which is what every bundle exported before this
                        # change contains.
                        vel = float(np.clip((v - self.threshold) / max(1 - self.threshold, 1e-6),
                                            0.0, 1.0))
                    events.append(Trigger(cls=name, velocity=vel, t=t,
                                          note=self.kit.notes[c]))
                self.prev[c] = v
        self.step += prob.shape[0]
        self.frame += n_steps
        return events

    def _schedule(self, n_frames: int) -> list[int]:
        """How many core updates each frame of this block gets.

        The phase carries across blocks as well as frames, so a fractional dial
        does not quietly round itself off at every block boundary -- at speed
        2.5 and four frames per block, that would be 3,2,3,2 | 3,2,3,2 rather
        than a true alternation, and the average would drift with block size.
        """
        out = []
        for _ in range(n_frames):
            self._phase += self.speed
            k = int(self._phase)
            self._phase -= k
            out.append(max(1, k))
        return out

    def _step_times(self, schedule: list[int]) -> list[float]:
        """Real time for every row the core produced, in seconds.

        The *true* frame duration, not the nominal step_ms. The encoder hop is a
        whole number of samples -- 110 at 22.05 kHz, which is 4.9887 ms, not 5 --
        so a clock built from step_ms drifts 0.23% against the audio it is
        supposed to be locked to: 9 ms over a four-second clip, 2.3 s over an
        album side.

        Each frame's sub-steps subdivide that frame's own span. With a uniform
        k this is exactly the old ``step_index * hop / (sr * k)`` grid; with a
        varying k there is no uniform grid to be on, which is the whole reason
        hits carry a timestamp rather than a step count.
        """
        frame_s = self.hop / self.sample_rate
        times = []
        for f, k in enumerate(schedule):
            t0 = (self.frame + f) * frame_s
            times.extend(t0 + j * frame_s / k for j in range(k))
        return times

    def reset(self) -> None:
        self.state = None
        self.step = 0
        self.frame = 0
        self._phase = 0.0
        self.prev[:] = 0
        self.last_fire = {c: -1e9 for c in self.kit.classes}


def benchmark(model, kit, cfg, block_ms: float = 20.0, n_blocks: int = 50,
              speed: float = 1.0) -> dict:
    """Measure real inference latency per block on this machine.

    Speed costs linearly: k core updates per frame is k times the recurrence.
    Measure before trusting a fast setting live -- the offline renderer has no
    such ceiling.
    """
    sr = cfg["audio"]["sample_rate"]
    d = StreamingDrummer(model, kit, sr, cfg["audio"]["step_ms"], speed=speed)
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
        "speed": float(speed),
        "substeps": d.substeps,
        "inference_ms_mean": float(times.mean()),
        "inference_ms_p95": float(np.percentile(times, 95)),
        "realtime_factor": float(block_ms / times.mean()),
        "meets_budget": bool(np.percentile(times, 95) < block_ms),
    }


def fit_speed_to_budget(model, kit, cfg, speed: float, block_ms: float = 20.0,
                        n_blocks: int = 20) -> tuple[float, dict]:
    """Find the fastest setting at or below ``speed`` that this machine can hold.

    Speed costs linearly -- k core updates per encoder frame is k times the
    recurrence -- so a dial that is fine offline will overrun the audio callback
    live, and an overrunning callback does not degrade gracefully: it drops
    buffers and clicks. The offline renderer has no such ceiling and should not
    be capped, which is why this is called from the live path only.

    Halves the dial until p95 fits the block budget, and never returns less than
    1.0: below that the model is not keeping up with real time at all, and the
    honest thing is to say so rather than to pretend a slower dial fixes it.
    """
    tried = []
    s = float(speed)
    while True:
        r = benchmark(model, kit, cfg, block_ms=block_ms, n_blocks=n_blocks, speed=s)
        tried.append(r)
        if r["meets_budget"] or s <= 1.0:
            return s, r
        s = max(1.0, s / 2.0)


def drummer_for(model, kit, cfg, threshold: float | None = None,
                speed: float = 1.0,
                class_refractory_ms: dict[str, float] | None = None) -> "StreamingDrummer":
    """Build the streaming drummer at the threshold this model was scored at.

    ``evaluate`` sweeps the peak-picking threshold and reports the one that
    suits the model's own output scale; a checkpoint carries it and a bundle
    passes it through. Ignoring that and picking at a fixed 0.3 is not a small
    difference -- on the 25-epoch model it is 154 notes in four seconds against
    a musical handful.
    """
    thr = threshold if threshold is not None else cfg.get("eval", {}).get("threshold", 0.3)
    return StreamingDrummer(model, kit, cfg["audio"]["sample_rate"],
                            cfg["audio"]["step_ms"], threshold=float(thr),
                            speed=speed, class_refractory_ms=class_refractory_ms)


def run_live(model, kit, cfg, style: int | None = None, block_ms: float = 20.0,
             midi_port: str | None = None, threshold: float | None = None,
             speed: float = 1.0, class_refractory_ms: dict[str, float] | None = None) -> None:
    import mido
    import sounddevice as sd

    sr = cfg["audio"]["sample_rate"]

    # Measure before opening the stream, and cap rather than glitch. An audio
    # callback that overruns its block does not slow down gracefully; it drops
    # buffers, and the result is clicks that sound like a broken model rather
    # than a machine that is out of headroom.
    safe, r = fit_speed_to_budget(model, kit, cfg, speed, block_ms)
    if safe < speed:
        print(f"speed {speed:g} needs {r['inference_ms_p95']:.1f} ms p95 against a "
              f"{block_ms:g} ms block on this machine -- capping to {safe:g}. "
              f"The offline renderer (--render) has no such ceiling.")
        speed = safe
    elif not r["meets_budget"]:
        print(f"WARNING: even speed 1 misses the block budget here "
              f"({r['inference_ms_p95']:.1f} ms p95 against {block_ms:g} ms). "
              f"Expect dropouts; use --render instead, or a smaller subgraph.")

    drummer = drummer_for(model, kit, cfg, threshold, speed, class_refractory_ms)
    drummer.set_style(style)

    from soundbank import MidiBank
    port = mido.open_output(midi_port) if midi_port else mido.open_output()
    bank = MidiBank(kit, port=port)
    print(f"MIDI -> {port.name}; listening at {sr} Hz, {block_ms} ms blocks. Ctrl-C to stop.")

    def callback(indata, frames, time_info, status):
        if status:
            print(status)
        for hit in drummer.push(indata[:, 0]):
            bank.trigger(hit.cls, hit.velocity, hit.t)

    with sd.InputStream(channels=1, samplerate=sr,
                        blocksize=int(sr * block_ms / 1000.0), callback=callback):
        try:
            while True:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nstopped.")
        finally:
            port.close()


def render_file(model, kit, cfg, wav_path: Path, out_path: Path,
                style: int | None = None, block_ms: float = 20.0,
                threshold: float | None = None, bank=None, speed: float = 1.0,
                class_refractory_ms: dict[str, float] | None = None) -> int:
    """Offline: run a wav through the streaming path and write the result.

    Uses the same StreamingDrummer as live playback, so what this renders is
    what the live path would have produced -- no separate offline code to drift.
    With a sample bank, the mixer runs block by block exactly as it would in an
    audio callback and the output is a wav; otherwise the output is MIDI.
    """
    import soundfile as sf
    import pretty_midi

    from soundbank import MidiBank

    sr = cfg["audio"]["sample_rate"]
    audio, in_sr = sf.read(wav_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if in_sr != sr:
        audio = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * sr / in_sr)),
                          np.arange(len(audio)), audio).astype(np.float32)
    peak = np.abs(audio).max()
    if peak > 0:
        audio /= peak

    drummer = drummer_for(model, kit, cfg, threshold, speed, class_refractory_ms)
    drummer.set_style(style)

    block = int(sr * block_ms / 1000.0)
    audible = bank is not None and bank.produces_audio
    out_blocks: list[np.ndarray] = []
    # 960 ticks/beat puts the MIDI grid at ~0.5 ms, below the 5 ms step the
    # model resolves. The default 220 quantises to 2.3 ms, which would throw
    # away timing the model actually has.
    pm = pretty_midi.PrettyMIDI(resolution=960)
    inst = pretty_midi.Instrument(program=0, is_drum=True, name="flybeats")
    n = 0
    for a in range(0, len(audio) - block + 1, block):
        for hit in drummer.push(audio[a: a + block]):
            n += 1
            if audible:
                bank.trigger(hit.cls, hit.velocity, hit.t)
            else:
                inst.notes.append(pretty_midi.Note(
                    velocity=MidiBank.to_midi_velocity(hit.velocity), pitch=hit.note,
                    start=hit.t, end=hit.t + 0.05))
        if audible:
            out_blocks.append(bank.mix(block))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if audible:
        # let the tails ring out rather than truncating mid-cymbal
        for _ in range(int(sr * 2.0) // block):
            out_blocks.append(bank.mix(block))
        import soundfile as sf
        mixed = np.concatenate(out_blocks) if out_blocks else np.zeros(1, dtype=np.float32)
        peak = float(np.abs(mixed).max())
        if peak > 1.0:
            mixed = mixed / peak * 0.98        # only on overload; never quietly
        sf.write(out_path, mixed.astype(np.float32), sr)
    else:
        pm.instruments.append(inst)
        pm.write(str(out_path))
    return n


def load_checkpoint(path: Path, device: torch.device):
    from build import build_model, get_subgraph
    from decoder import DrumKit

    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck["config"]
    # As load_bundle does: play at the threshold the eval sweep chose, not the
    # config's fixed one. An explicit --threshold still wins in drummer_for.
    if ck.get("best_threshold") is not None:
        cfg.setdefault("eval", {})["threshold"] = float(ck["best_threshold"])
    sg = get_subgraph(cfg)
    model, kit = build_model(cfg, sg, n_styles=max(int(ck.get("n_styles", 1) or 1), 1))
    model.load_state_dict(ck["model"])
    return model.to(device).eval(), DrumKit(ck["kit"]), cfg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--bundle", type=Path, default=None,
                    help="a self-contained model file from scripts/export_bundle.py. "
                         "Needs no data/ directory: no connectome, no corpus")
    ap.add_argument("--config", type=Path, default=None,
                    help="benchmark an untrained model straight from a config, to size "
                         "the subgraph against the latency budget before training it")
    ap.add_argument("--benchmark", action="store_true", help="measure inference latency and exit")
    ap.add_argument("--render", type=Path, help="wav in -> drums out, via the streaming path")
    ap.add_argument("--out", type=Path, default=None,
                    help="default: runs/render.wav with --sound-source samples, "
                         "runs/render.mid otherwise")
    ap.add_argument("--sound-source", choices=["midi", "samples"], default="midi",
                    help="'samples' mixes a kit here and writes/plays audio, so no "
                         "external sampler is needed")
    ap.add_argument("--kit-dir", type=Path, default=ROOT / "kits" / "synth",
                    help="sample kit for --sound-source samples")
    ap.add_argument("--style", type=int, default=None)
    ap.add_argument("--block-ms", type=float, default=20.0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="core updates per encoder frame: 1 is as trained, 4 is the "
                         "fly's own wingbeat timescale, above that is faster than the "
                         "animal. Costs linearly; check --benchmark --speed first")
    ap.add_argument("--class-speed", nargs="+", default=[], metavar="CLASS=SPEED",
                    help="per-class speed, e.g. --speed 4 --class-speed kick=1 snare=1 "
                         "to let the hats flutter while the kick stays human. The core "
                         "runs at --speed for everyone; this decides who may use it")
    ap.add_argument("--threshold", type=float, default=None,
                    help="peak-picking threshold; default is whatever the model's own "
                         "eval sweep chose (see eval.threshold)")
    ap.add_argument("--midi-port", type=str, default=None)
    ap.add_argument("--lesion", type=str, default=None,
                    help="mute a confirmed population during playback, e.g. pIP10")
    ap.add_argument("--slider", nargs=2, action="append", metavar=("NAME", "VALUE"),
                    default=[], help="e.g. --slider drive 1.5 --slider pocket 1.2")
    a = ap.parse_args(argv)

    device = torch.device("cpu")
    roles = None
    if a.bundle:
        # The self-contained path. Nothing below may reach for the subgraph:
        # a bundle is meant to run on a machine that has never downloaded the
        # connectome, so its roles come with it.
        from bundle import load_bundle
        model, kit, cfg, roles = load_bundle(a.bundle, device)
    elif a.checkpoint:
        model, kit, cfg = load_checkpoint(a.checkpoint, device)
    elif a.config:
        from build import build_model, get_subgraph, load_config
        cfg = load_config(a.config)
        model, kit = build_model(cfg, get_subgraph(cfg))
        model = model.to(device).eval()
    else:
        ap.error("pass --bundle or --checkpoint, or --config to benchmark an "
                 "untrained model")

    if roles is None:
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
        r = benchmark(model, kit, cfg, a.block_ms, speed=a.speed)
        print(f"block {r['block_ms']:.0f} ms | inference {r['inference_ms_mean']:.1f} ms mean, "
              f"{r['inference_ms_p95']:.1f} ms p95 | {r['realtime_factor']:.2f}x realtime")
        print("within budget" if r["meets_budget"] else
              "OVER BUDGET: lower subgraph.max_nodes or raise audio.step_ms")
        return 0 if r["meets_budget"] else 1

    # A class's own speed is expressed as its refractory: the core runs once, at
    # --speed, and each class decides how much of that resolution it takes.
    # Running a core per class is not on the table -- the connectome is a single
    # coupled network, with no per-class subnetwork to run at its own rate.
    class_refractory = {}
    for item in a.class_speed:
        name, _, value = item.partition("=")
        if name not in kit.classes:
            ap.error(f"--class-speed: {name!r} is not in this kit ({', '.join(kit.classes)})")
        try:
            cs = float(value)
        except ValueError:
            ap.error(f"--class-speed: {item!r} should look like kick=1")
        if cs <= 0:
            ap.error(f"--class-speed: {name} needs a positive speed, got {cs}")
        class_refractory[name] = 50.0 / cs
    if class_refractory:
        print("class speeds: " + ", ".join(
            f"{k} {50.0 / v:g}x ({v:.1f} ms)" for k, v in class_refractory.items()))

    if a.render:
        from soundbank import make_bank
        bank = (make_bank("samples", kit, cfg["audio"]["sample_rate"], a.kit_dir)
                if a.sound_source == "samples" else None)
        if bank is not None:
            missing = bank.validate(kit.classes)
            if missing:
                # silence, not a crash: lesion mode already needs "some classes
                # do not sound" to be an ordinary outcome
                print(f"warning: {a.kit_dir} has no samples for {', '.join(missing)} "
                      f"-- those classes will be silent")
        out = a.out or (ROOT / "runs" / ("render.wav" if bank is not None else "render.mid"))
        n = render_file(model, kit, cfg, a.render, out, a.style, a.block_ms,
                        a.threshold, bank, a.speed, class_refractory or None)
        print(f"wrote {n} hits -> {out}")
        return 0

    run_live(model, kit, cfg, a.style, a.block_ms, a.midi_port, a.threshold, a.speed,
             class_refractory or None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
