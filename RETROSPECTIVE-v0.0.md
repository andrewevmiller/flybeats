# flybeats v0.0 retrospective — and the restart as v0.1

Sep 26, 2026 · @andevmiller

## Summary

**flybeats v0.0 was written entirely by Claude, and it failed.** From 12 to 25 September 2026, Claude wrote the plan, the code, the training runs, the diagnostics and the handoff notes.

The result plays **snare only**. Its style and drive controls do nothing. Its best timing score (0.32 onset F-measure) came from one seed, one drummer, and held only on the drum kit it trained on.

The central question, whether the real fly connectome helps a network play drums, was never answered. **v0.1 starts over from scratch, with me verifying every piece of Claude's code and operating it by hand.** v0.0 is archived for reference, not as a base to build on.

## What v0.0 set out to do

v0.0 aimed to show that the real fruit-fly wiring diagram is a useful starting point for a network that listens to audio and plays drums. The plan (PLAN.md, 12 September) set out the method:

- **Wiring fixed by the connectome.** A 10k–30k neuron slice of MaleCNS v1.0, from the ear (Johnston's organ) to the wing and leg motor neurons. Only connection strengths were trained, never which neurons connect.
- **Thin encoder and decoder,** so the connectome had to do the work.
- **Training on E-GMD / GMD,** real drum audio with aligned MIDI.
- **Required controls:** randomly rewired graphs, shuffled signs, a plain GRU and a bypass shortcut. The plan was explicit that without these there is no evidence the connectome helped.
- **Extras on top:** genre injected through octopamine neurons, biological sliders (drive, tightness, pocket), lesion mode and real-time MIDI playback.

## What happened

Claude made 169 commits over 14 days. Almost half landed in the first three days, before any model had been trained at full scale.

| Date | What happened |
| --- | --- |
| 25 Sep | Style-dial overnight experiment; v0.0 stopped and archived |
| 24 Sep | A′ queue runs overnight. A probe bug is found and "kick and toms backwards" is retracted. Standardisation tops timing at 0.32, then fails the other-kit check. Every run so far turns out to be drummer1 only |
| 21 Sep | A′ loudness-fix queue prepared. Two of five control arms still cannot run a forward pass |
| 16–17 Sep | Release roadmap, test plan, DAW / FL Studio guide and operation manual written, with no working model yet |
| 14–15 Sep | Local GPU bring-up (RTX 2060 Max-Q, 6 GB). The probe was found to score different audio on every run. "eGMD extracted" corrected: it was not |
| 13 Sep | Scheduled overnight session fails to start (no messaging tool, no CLI) |
| 12 Sep | PLAN.md written; 28 commits of scaffolding, kits and demo audio |

**Where v0.0 lives:** the `claude/quirky-turing-e735dk` branch holds 12–14 September (59 commits). The work from 15–25 September is on `cuda-path-verified` (also pushed as `claude/style-dial-2026-09-25`). Local copies of plans and demos are in `.00\`.

## Why it failed

**The one experiment that mattered never ran.** The controls (random rewiring, shuffled signs, GRU, shortcut) were "Phase D, still hardware-blocked" when v0.0 stopped. As late as 21 September, two of the five arms could not run a forward pass. Without them, no result says anything about the connectome.

**The model does not play drums.** On six test songs the best-examined model played snare only, at 10 hits/s against the human's 8.6. The style dial changed 0% of hits, and the drive slider did nothing. Tightness was the only live control, and at 1.5 it silenced the model. Style entered through octopamine neurons that do not reach the motor neurons at all.

**The numbers were weak and narrow.** Timing peaked at 0.32 onset F-measure, on one seed. Every 256-clip run trained on one drummer, and 227 of those clips were fills. The standardisation gain vanished on any kit other than GMD's own.

**The subgraph was not selected by anatomy.** Three hops from the ear reach 154,853 of 162,517 neurons, so a trimming heuristic, not the wiring, chose the 10k neurons trained.

**Claims had to be walked back repeatedly.** The 24 September handoff alone retracted seven results, including:

- "Kick and low tom confidently backwards" was a probe artifact (a mostly-drummer7 slice, resampling time steps).
- "A′1 worked" came from the old probe on the cloud; locally it put kick backwards.
- "Crash is right" was a starting-weights artifact; the heads barely moved.
- "256 of 897 clips" hid that all 256 were one drummer.

**Effort went to the wrong layer.** While the core question stayed open, Claude produced plans for UI, visualizer, soundbank, speed, DAW integration, a release roadmap and a test plan. It also built demo pages, an operation manual and two kit files, and spent days on loudness-head fixes.

**The process could not run unattended.** Scheduled sessions failed to start, stalled on permission prompts, or were killed by Windows Update restarts. Two runs sharing the machine slowed one epoch from 76 s to 1,415 s.

## What carries forward

No v0.0 code or results carry into v0.1. What survives is a set of facts and lessons.

- **Data facts.** GMD's 256-clip subset is drummer1 only. E-GMD is 141 GB unpacked: the same performances replayed on 43 kits, so it adds sound variety, not playing variety.
- **Connectome facts.** A few hops from the ear reach nearly the whole brain, so the slice needs a real anatomical rule, fixed in advance. Octopamine neurons do not reach the motor neurons, so any control input needs a verified path.
- **Hardware.** An RTX 2060 Max-Q with 6 GB VRAM, Turing architecture (compute 7.5); memory is the binding limit. Never run two trainings at once.
- **Method.** Decide the metric, the controls and the win rule before training. Use several seeds, and treat a gap smaller than the seed spread as no result. Probe and score on fixed, seeded data.
- **Operations.** Pause Windows Update before overnight runs, and do not rely on unattended sessions starting themselves.

## v0.1: starting over

**v0.1 is a fresh start. Claude still writes the code, but I verify it and run it by hand.** It lives in a new repository in `.01\`, separate from the v0.0 history. v0.0 stays archived on its branches as a record of what not to repeat.

How v0.1 differs from v0.0:

- **Design locked before any training.** The draft pre-registration (`config\locked.yaml`, tag `prereg-v1`) fixes the connectome slice rule, the kit, the scorecard and the win rule up front.
- **The controls come first.** The real connectome must beat all five degree-preserving random rewirings on beat alignment by more than the seed spread. If it does not, that is the result.
- **New data.** Slakh2100 (redux, 16 kHz) replaces GMD / E-GMD, with per-song splits and stem dropout.
- **No product work until the question is answered.** No UI, DAW, soundbank or release plans until the win rule has been evaluated on the test split, once.
- **A human checks and runs every step.** In v0.0 Claude ran largely on its own, including unattended overnight sessions. In v0.1 nothing runs until I have verified it, and I start and watch every run myself.
