# Proposal: a readout that sees the last 200 ms

Status: **proposal, not locked.** Nothing in `config/locked.yaml` or the code has changed. It is separate from the
prereg-v6 readout-scaling proposal (`reports/prereg-v6-readout-proposal.md`), which is also not locked.

**Timing matters.** prereg-v6's stopping rule B says that if memorisation fails under prereg-v6, the pipeline is not
changed again for it. So this proposal has to be accepted or rejected **before** any prereg-v6 run. If accepted, it
goes into the same tag as the scaling refit (one prereg-v6 with both changes), or becomes its own tag ahead of it.

Written 30 Sep 2026 from `scripts/trace_information.py` (`reports/trace_information.json`), run on the prereg-v5
memorisation weights of the real network and control 1.

## Why this proposal exists

The locked readout reads each frame's hit and loudness from **that frame's** motor-neuron activity alone: one weight
per readout motor neuron, applied to the present moment (`network.py`: `z * w_hit`, summed over the drum's own motor
neurons).

On the 8 memorisation clips, the best straight-line readout of the hits from the trained networks' motor neurons
(average precision, fitted and scored on the same frames; kick / snare / closed hi-hat; chance 0.020 / 0.016 / 0.036):

| What the readout sees | Real | Control 1 |
|---|---|---|
| The present moment, every readout motor neuron | 0.176 / 0.074 / 0.136 | 0.324 / 0.198 / 0.248 |
| The last 200 ms (4 blocks of 50 ms), the motor neurons' 64 main patterns | 0.303 / 0.191 / 0.361 | 0.318 / 0.259 / 0.411 |

- **Seen over the last 200 ms, the real network's motor neurons hold nearly as much as control 1's.** Seen at the
  present moment only, they hold about half.
- **So the real network's information reaches its motor neurons, but spread over time,** in a form the locked
  readout cannot use. Control 1's lines up more with the present moment.
- **Other explanations were checked and ruled out** in the same run:
  - no single stage of the network loses the information (ears, interneurons 1 hop, 2+ hops, motor neurons were
    within a few hundredths of control 1 at each);
  - the information isn't simply late (letting the readout see up to 100 ms past each moment helped both networks
    about equally);
  - the learned time constants are similar (median 10-21 ms at every stage in both);
  - the silenced neurons don't cut the routes (removing or keeping them changes ear-to-motor flow by about 1%);
  - lowering the starting gain doesn't help (0.5 or 0.8 x gain: no more independent signals, no more information).

The two rows are not exactly like for like (all motor neurons against 64 main patterns; one moment against block
averages). The size of the difference for the real network (snare 0.074 against 0.191) is what makes it worth
proposing.

### A biological reason, independent of these results

A drum hit is a movement, and a movement is a muscle's force, not a motor neuron's instantaneous rate. A muscle
smooths its motor neurons' activity over tens of milliseconds (twitch rise and decay). A readout that weighs each motor
neuron's recent history is a simple stand-in for the muscle the motor neuron drives. The locked readout treats each
motor neuron as if its rate at one 5 ms frame were the movement.

## What this means for integrity, stated up front

- **It is a change made after seeing results,** and the trigger was a real-against-control difference on memorisation
  (the real network gains more from history than control 1). It is listed in the changes-after-results table below.
- **It is expected to help the real network more than the controls.** That is the reason for it, and also the risk:
  a readout change tuned to one network's weakness. Against that:
  - it applies to every network alike (the real one and all 5 controls, memorisation and training);
  - it is the same small, fixed form for all: what each network delivers to its motor neurons still decides the
    result;
  - the main test is on unseen songs, which this result says nothing about.
- **It does not touch the connectome.** Wiring, signs, synapse counts, the ear, which motor neurons play which drum,
  and the disjoint readout (each drum reads only its own motor neurons) are all unchanged.
- **Readout power grows a little:** 80 new numbers in total (below), against about 41,000 network dials. The present
  moment stays one of the inputs, so the locked readout is a special case of the new one.

## Proposed change

**Each drum's readout weighs its motor neurons' last 200 ms through a short learned time profile, shared by that
drum's motor neurons.**

1. **History:** for each readout motor neuron, at each frame t, five numbers from its scaled activity z (the same z
   the locked readout uses):
   - the present frame, z(t);
   - the means over frames t-1 to t-10, t-11 to t-20, t-21 to t-30 and t-31 to t-40 (5-50, 55-100, 105-150 and
     155-200 ms back).

   Causal: nothing after frame t. At the start of a clip, frames before the first count as z = 0 (each motor
   neuron's average activity). They fall inside the 2 s warm-up, which the loss and scoring ignore.
2. **Readout:** hit logit for drum d at frame t = the sum over d's motor neurons of w_hit(neuron) x (the profile of
   drum d applied to that neuron's five numbers) + c_hit(d). The profile is five numbers per drum; loudness has its
   own five per drum, through u_vel and e_vel in the same way. So a drum's motor neurons share one time profile, and
   each keeps its own weight.
3. **New parameters:** 5 (hit) + 5 (loudness) per drum x 8 drums = **80 numbers.** (40 if loudness keeps reading the
   present moment only; see Alternatives.)
4. **Starting point:** each profile starts as (1, 0, 0, 0, 0), so the untrained readout is exactly the locked one.
   Learning moves the profile only if history helps.
5. **Learning:** the profiles are trained with everything else, by the locked optimiser, learning rate, schedule and
   gradient clipping. No new settings.
6. **Scaling:** unchanged. The readout normalisation (and prereg-v6's refit, if accepted) acts on z before the
   history is formed, so the refit stays output-preserving: the profile multiplies the already-rescaled z.

## Alternatives considered

- **A free 40-tap kernel per motor neuron** (182 x 40 numbers). The most flexible, but over 7,000 readout numbers
  would let the readout do part of the network's job. Rejected.
- **The same 5-tap shape per motor neuron** (182 x 5 numbers each for hit and loudness). More flexible than one
  profile per drum, but 5x the readout's numbers. Kept as the fallback if a reviewer objects that a shared profile
  forces one timing on muscles of different speed.
- **Fixed muscle filter** (each motor neuron smoothed by a fixed exponential, time constant set from the
  literature). Nothing learned, so no extra readout power, and the strongest biological story. Rejected for now
  because the right time constants differ across the wing and leg muscles in this readout and we have no locked
  source for them; it could be the next version if one is found.
- **Loudness on the present moment only** (40 new numbers). Simpler; the trace measured hits, not loudness. A fair
  choice if Andrew prefers the smallest change.
- **Change which motor neurons play snare and closed hi-hat** (the trace found narrow wiring from the ears to the
  wing power-muscle motor neurons those drums read). Not proposed: reassigning drums because the real network did
  badly on them would be tuning to the result.

## Checks before any run

1. **Unit tests:**
   - with profiles at (1, 0, 0, 0, 0), every output equals the locked readout's to the last digit;
   - causality: changing the audio after frame t changes no output at or before t;
   - each drum still reads only its own motor neurons;
   - with prereg-v6's refit, outputs are unchanged across a refit.
2. **CUDA graph:** `network.py` changes, so the code fingerprint changes and `scripts/check_cuda_graph.py` must pass
   again (all three parts) before the recorded step is used.
3. **Speed:** the history is a running buffer of the last 40 frames for 182 neurons: small. Measured with
   `scripts/profile_step.py` and reported; expected under 5% per step.

## How it is judged

The same preregistered memorisation pair as before (real network, then control 1), with the locked pass rule (F1 >=
0.9 on kick, snare and closed hi-hat). prereg-v6's stopping rule B applies: if memorisation still fails, the pipeline
is not changed again, and training of the real network and all 5 controls goes ahead as locked, with the failure
reported up front.

**Recorded either way, for the write-up:** each drum's learned profile, real against control 1. If the real
network's profiles put clearly more weight on the past than control 1's, that supports the reading above.

## Changes made after seeing results

| Version | Change | What triggered it | Disclosure |
|---|---|---|---|
| (proposed) | Readout weighs each motor neuron's last 200 ms through a learned 5-number profile per drum | `trace_information.py` on prereg-v5 memorisation weights: the real network's motor neurons hold about half control 1's hit information at the present moment, but nearly as much over the last 200 ms | **Triggered by a real-against-control result, on memorisation only.** Expected to help the real network more. Applies to every network alike; the connectome and the readout map are unchanged. The untrained readout equals the locked one. |

## If accepted: the work

1. **`locked.yaml`:** under `model`, `readout_history: {present: true, blocks_ms: [50, 50, 50, 50],
   profile: per_drum, init: present_only, velocity: same}`, in the same tag as the scaling refit or its own.
2. **`network.py`:** the history buffer and per-drum profiles in `run`.
3. **Tests:** the four above.
4. **`check_cuda_graph.py`** run again; **`profile_step.py`** run again.
5. **The runs:** the memorisation pair, about 5.3 hours.

### Draft CHANGELOG line

> 2026-09-30 - prereg-v6 (or v7): the readout weighs each readout motor neuron's last 200 ms (the present frame and
> four 50 ms block means) through a 5-number time profile per drum, one for hits and one for loudness (80 new numbers),
> starting at present-only so the untrained readout equals the locked one. Triggered after results: on the prereg-v5
> memorisation weights, the real network's motor neurons held about half control 1's hit information at the present
> moment but nearly as much over the last 200 ms (scripts/trace_information.py). Applies to every network alike; the
> wiring, ear, readout map and disjoint readout are unchanged. Disclosed as a change after a real-against-control
> result on memorisation.
