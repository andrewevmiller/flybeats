# flybeats: Speed Control — Implementation Plan

A user dial from a human drummer's tempo up to the fly's own. Inference-time
only: no retraining, and at speed 1 the output is bit-identical to today's.

## What "fly speed" means, concretely

The dial has a real anchor rather than a taste one. The model's membrane time
constant is initialised at 20 ms and its floor is `tau_ms_min: 5.0`; a fly beats
its wings at ~200 Hz, a 5 ms period. So:

| speed | effective τ | what it is |
|---|---|---|
| 1 | 20 ms | as trained — a human drummer's timescale |
| 2 | 10 ms | quick; fills start to flutter |
| **4** | **5 ms** | **full fly: the wingbeat period, and the model's own τ floor** |
| 8–16 | 2.5–1.25 ms | faster than the animal. A musical effect, not a biological one |

The dial should be labelled that way: **normal → full fly** over 1–4, with 4–16
available but marked as beyond anything the connectome's timescales justify.

At step 5 ms the core already runs at exactly 200 updates per second — one per
wingbeat period. Speed 1 is not arbitrary; it is the fly's clock rate already.

## The mechanism: sub-step the core, hold the drive

Run `k` recurrent updates per encoder frame, reusing that frame's drive.

```python
for frame in range(t):                 # encoder frames, 5 ms apart
    for _ in range(k):                 # k core updates inside each
        r = self.rate(v)
        v = v + alpha * (-v + self.recurrent(w, r) + inp + base)
        out.append(...)                # output grid is now step_ms / k
```

Why this and not the obvious alternatives:

* **`alpha` does not change.** `alpha = step / tau`; speeding up means dividing
  both by `k`, so the discrete update is *exactly* the one that was trained.
  Nothing about stability, saturation or the spectral-radius normalisation
  shifts. The network runs the same trajectory, `k` times further per second of
  music.
* **Scaling τ alone saturates.** `alpha` is clamped at 1.0 and τ at 5 ms, so
  τ-only speed-up tops out near 4× and then silently stops doing anything. The
  existing `pocket` slider is exactly this, and is correctly limited to small
  nudges.
* **Time-warping the output desyncs it.** Resampling event times afterwards
  changes the tempo of the drums relative to the music they are playing over.
  That is a different feature (and a much easier one).

Usefully, sub-stepping with a held drive is *identically* `repeat_interleave` on
the drive tensor, which makes the equivalence test a one-liner and means the
offline path can be validated against the streaming one exactly.

## What else has to scale, or the dial does nothing

* **Peak-picker refractory.** `refractory_steps = refractory_ms / step_ms` caps
  the event rate at 20 hits/s/class. Keep it constant *in steps* so it shrinks
  in milliseconds as the grid gets finer. Leave it in ms and the faster dynamics
  produce peaks that are then thrown away.
* **Event timestamps.** The output grid is `step_ms / k`; `StreamingDrummer`
  must place notes on it or everything lands early.
* **Nothing in the encoder.** The DSP hop, the envelope kernel and the
  standardisation affine are all tied to the trained `step_ms`. Moving the ear
  off its grid invalidates the calibration saved in the checkpoint.

That last point is the honest limitation of this design: **at high speed the
drummer improvises faster, it does not listen faster.** The drive is a staircase
at the finer grid with no content above the encoder's ~100 Hz Nyquist, so the
extra hits come from internal dynamics, not from hearing finer detail. Making it
hear faster means a finer encoder grid, which means recalibration and a retrain
at that step — out of scope here, and worth stating rather than blurring.

## Measured, before implementing

Whole-clip forward on the 25-epoch bundle, 4 s of audio, threshold 0.5,
refractory held at 10 steps:

```
 speed  core steps/s  eff tau ms   hits   hits/s   xRT (whole-clip, 2 threads)
     1           200       20.00     37      9.2    5.12
     2           400       10.00     68     17.0    2.73
     4           800        5.00    116     29.0    1.44
     8          1600        2.50    147     36.8    0.72
    16          3200        1.25    279     69.8    0.34
```

Hit rate rises with speed but sub-linearly — 16× the compute buys 7.6× the hits,
because the per-class refractory and the peak shapes both bite. Cost is linear
in `k`, exactly as the loop implies.

**The live path tops out around speed 3–4** on the 10k tier. These numbers are
whole-clip batches, not the block-wise streaming path, so re-measure with
`--benchmark --speed` before trusting them live; offline rendering has no
ceiling beyond patience.

## Surface

```bash
python src/realtime.py --bundle m.fb --render song.wav --speed 4     # full fly
python src/realtime.py --bundle m.fb --benchmark --speed 4           # can it?
```

* `ConnectomeRNN.forward(..., substeps: int = 1)` — the only model change.
* `StreamingDrummer(..., speed: float = 1.0)` — derives `substeps`, keeps
  `refractory_steps` fixed, places events on the finer grid.
* `--speed` on `realtime.py`, passed through render, live and benchmark.
* Not a `set_slider` entry: speed changes the loop structure, not a parameter,
  so it belongs to the runner. `pocket` stays as the small τ nudge for feel.

## Order

0. **Done.** A time-based, per-class peak picker. Both features need it before
   either can work: with `k` varying per frame a refractory measured in *steps*
   stops naming a fixed span of time, and "hats flutter while the kick stays
   human" *is* a per-class refractory. `push` now returns `Trigger(cls,
   velocity, t, note)` with velocity in 0..1 and `t` the hit's own time.

   Two timing bugs fell out of writing it, both of which were throwing away
   resolution the model already had: every hit was stamped with its enclosing
   audio block's start time (20 ms quantisation), and the step clock used the
   nominal `step_ms` rather than the encoder's true hop -- 110 samples at
   22.05 kHz is 4.9887 ms, so the drum track drifted 0.23% against the music,
   9 ms over a four-second clip. Onsets now land on the true step grid within
   0.25 ms, which is the MIDI tick grid and nothing else.

1. `substeps` in `ConnectomeRNN.forward`, plus the equivalence test against
   `repeat_interleave`. Nothing else moves yet.
2. `StreamingDrummer` speed: substeps, refractory in steps, event timing.
3. `--speed` on the offline render. This is the whole feature for a user who
   renders files.
4. `--benchmark --speed` and a live-path guard: measure p95 against the block
   budget, warn and cap rather than glitch.
5. Fractional speeds: accumulate a phase and run `k` or `k+1` updates per frame,
   so the dial is continuous rather than 1/2/4/8.
6. Ramping: changing `k` between blocks is safe (state carries, no parameter
   jump), so a live dial needs no crossfade — but test for clicks anyway.

## Tests

* **Speed 1 is bit-identical** to the current output. Non-negotiable regression.
* **Sub-stepping equals `repeat_interleave`** on the drive, exactly.
* **`alpha` is unchanged** across speeds — the guard against someone later
  "simplifying" this into a τ rescale that silently changes the dynamics.
* **Hit rate rises monotonically** with speed on a fixed clip.
* **State stays bounded** at speed 16 (no drift into the clip at either end).
* **Event timestamps** land on the `step_ms / k` grid and the last event is
  inside the clip.

## Per-class speed, and speed on fills only

These are the two the dial exists to serve, and they compose rather than
compete: **global sub-stepping generates temporal resolution; per-class rules
allocate it.** Shortening the hi-hat's refractory cannot invent peaks the
dynamics never produced, so per-class speed does nothing until the core can run
fast -- and a fast core without per-class rules is just a buzz roll.

* **Per-class speed** = one core pass at the highest requested `k`, then a
  per-class refractory (step 0, done) and optionally a per-class threshold.
  Not one core per class: the connectome is a single coupled network and there
  is no per-class subnetwork to run at its own rate.
* **Speed on fills only** = let `k` vary per frame, driven by the pC1
  population's own activity rather than a static dial. pC1 is already the
  `drive` slider's target and already means "fill density, intensity", so the
  signal is there: read its mean rate per frame, map it through a curve to
  `k(t)`, and the drummer speeds up exactly where it is already playing harder.
  This is why `t` has to be real time -- with `k` varying, there is no uniform
  grid to count steps on.

## Open questions

* **Is it musical above 4?** 70 hits/s across 8 classes is a buzz roll. Per-class
  speed may be the better instrument — hats flutter at fly speed while the kick
  stays human — which the decoder's per-class structure already allows.
* **Speed on fills only**, driven by the `drive` (pC1) slider, would be more
  musical than a global dial, and is closer to what the population actually does.
* **The spiking model** is where fly-rate dynamics stop being a metaphor. If the
  surrogate-gradient LIF version lands, speed becomes a property of the neurons
  rather than a loop count.
