# Roadmap: what happens next, and in what order

The design this project implements is [PLAN.md](PLAN.md); what it currently
does and does not do is the README's
[Results](README.md#results-and-what-they-are-not) and
[What is not done](README.md#what-is-not-done). This file is narrower: the
order of the next few pieces of work, what each costs, and what would count as
finishing it.

**Two releases, because two different things are "done":**

- **v0.1 — playable.** Someone installs, takes a bundle, and gets drums out of
  their own audio. Gated on code and CPU only, so it is reachable now.
- **v1.0 — the claim.** The Phase 4 arms across five seeds, answering whether
  the connectome's topology earns its place. Gated on a GPU, not on code.

**The rule that sets the order:** anything that changes the loss lands before
anything expensive. That is why the velocity head went in ahead of the Phase 4
arms, and it is why the velocity fixes below go ahead of the full-corpus run.
Getting this backwards costs the run, not the fix.

---

## Where the model actually is

One number frames the next phase: the last run trained on **256 of the 897
available GMD training clips**, and the README already records this model as
limited by data rather than by epochs. Three and a half times the corpus is
sitting unused.

| | |
|---|---|
| onset F (GMD, 12 epochs, 10k nodes) | 0.30 |
| velocity, per class | one class right, two significantly backwards, the rest noise |
| epoch cost, 256 clips | 495 s |
| epoch cost, 897 clips | ~29 min (projected) |

---

## Phase A′ — make the velocity head use what it already receives

The head is not starved. `scripts/probe_velocity.py` puts hit strength in the
motor pool (tom_low 0.39, snare 0.23, crash 0.30) and the head reads it
backwards on two classes. That is a decoder and optimisation problem, and these
are the candidates, cheapest first.

| | change | why | cost |
|---|---|---|---|
| A′1 | raise `velocity_weight` above 1.0 | At 1.0 the detection BCE dominates the gradient, and both heads read the same motor pool through the same scale | 100 min |
| A′2 | linear head with the target centred, instead of the sigmoid | The sigmoid's gradient is flattest exactly where GMD's velocities cluster | 100 min |
| A′3 | score only near the onset peak, not across the whole kernel support | That is where the streaming path reads velocity; training flat across the support teaches the head to average | 100 min |

Each is a retrain at 256 clips plus `python scripts/probe_velocity.py --seed 0`.

### A′1 result: it worked, and the head was simply outvoted

`velocity_weight` 1.0 → 5.0, everything else held. Head correlation at the onset
peak, baseline → A′1, with 95% bootstrap intervals:

```
class          baseline                    velocity_weight 5.0
snare          -0.098  [-0.19, -0.01]      +0.062  [-0.01, +0.13]
tom_low        -0.284  [-0.41, -0.14]      +0.306  [+0.13, +0.45]
tom_mid        -0.140  [-0.29, +0.02]      +0.231  [+0.09, +0.37]
crash          +0.473  [+0.33, +0.60]      +0.349  [+0.17, +0.50]
kick, hats     span 0                      span 0
```

**No class is significantly negative any more**, and three are significantly
positive. tom_low is the telling one: it was the class carrying the most unused
motor signal (0.39 at the motor units, −0.28 at the head), and it flipped to
significantly right. Head spread rose from 0.084 to 0.105 against a target
spread of 0.272 — still compressed, so this is "reading dynamics weakly", not
"solved".

Two things this does **not** establish. Onset F came in at 0.2761 against the
baseline's 0.3006, which is either a real detection cost from crowding the loss
or run-to-run noise; one run with no seeds cannot tell those apart, and it
needs seeds before it is quoted. And a weight of 5 is the first value tried,
not a tuned one.

### A′2 result: the linear head fixes a different fault, and fails the gate

Linear head, `velocity_weight` left at 1.0. Head correlation at the onset peak,
with 95% bootstrap intervals, against the baseline and A′1:

```
class          baseline                A′1 weight 5.0        A′2 linear head
snare          -0.098 [-.19,-.01]      +0.062 [-.01,+.13]     -0.138 [-.22,-.06]
tom_low        -0.284 [-.41,-.14]      +0.306 [+.13,+.45]     +0.433 [+.32,+.54]
tom_mid        -0.140 [-.29,+.02]      +0.231 [+.09,+.37]     +0.283 [+.13,+.43]
hat_open       +0.117 spans 0          -0.039 spans 0         +0.172 [+.02,+.29]
crash          +0.473 [+.33,+.60]      +0.349 [+.17,+.50]     +0.120 spans 0
head sd        0.084                   0.105                  0.154
onset F        0.3006                  0.2761                 0.2831
```

**A′2 fails the gate**: snare is still significantly negative, which A′1 had
fixed. But it is not a failure to discard, because it is clearly better on the
axis A′1 barely moved — head spread 0.154 against 0.105, on a target spread of
0.272, and the strongest tom_low of the three.

So the two fixes address different faults and neither does both. The weight
change stops the head being outvoted by the detection BCE, which is what
cleared the negative classes. The linear head stops the sigmoid squashing the
output, which is what widened the spread. **`velocity_lin_w5_cpu` runs the
composition**, which is the obvious thing neither experiment alone tested.

**Done when** at least one class's head correlation is significantly positive
without pushing another negative. Compare bootstrap intervals, never point
estimates: the same analysis read hat_open as 0.37, then 0.18, then 0.26 on one
checkpoint before its sampling was seeded, and a pooled correlation reported
~0.35 while the per-class picture was incoherent.

**Never run two of these at once.** Torch takes a thread per core in each
process and the oversubscribed threads spin rather than progress — one epoch
went from 76 s to 1,415 s. Sequential, or set `OMP_NUM_THREADS`.

---

### The seed result, which undermines every per-class claim above

`velocity_probe_cpu_s1` is the baseline config at seed 1. Nothing else differs
— same data, same hyperparameters, same 12 epochs. Against seed 0:

```
class          baseline seed 0         baseline seed 1
snare          -0.098 [-.19,-.01]      -0.068 [-.14,+.01] spans 0
tom_low        -0.284 [-.41,-.14]      +0.315 [+.17,+.44]
tom_mid        -0.140 [-.29,+.02]      +0.359 [+.22,+.48]
hat_closed     (not significant)       +0.158 [+.06,+.26]
crash          +0.473 [+.33,+.60]      -0.228 [-.38,-.05]
head sd        0.084                   0.100
onset F        0.3006                  0.2981
```

**tom_low moves from significantly negative to significantly positive, and
crash from significantly positive to significantly negative, on a seed change
alone — with non-overlapping intervals in both cases.**

The bootstrap intervals are therefore not what they were being read as. They
resample *rows within one trained model*, so they answer "given this
checkpoint, how precisely is its correlation estimated" — and they answer it
correctly. They say nothing about how reliably the *configuration* produces
that correlation, and that is the quantity every comparison in this phase
actually needs.

So the gate — at least one class significantly positive, none significantly
negative — cannot be decided from a single run. A′1 "passing" it and A′2 and
A′3 "failing" it are one draw each from a distribution wide enough to change
the sign of individual classes. **Read the three result sections above with
that caveat: their per-class claims are not established.** Specifically, "the
velocity head was outvoted, not blind" does not survive a swing of this size.

Two things that did survive, and one that got sharper:

- ~~**onset F is stable across seeds**: 0.3006 and 0.2981 … the detection cost
  of `velocity_weight` 5.0 looks real rather than noise.~~ **Withdrawn — see
  below.** That was written with two baseline seeds and one A′1 seed, and the
  baseline's tight pair was itself a draw. A′1's own two seeds are 0.2761 and
  0.2992, a range of 0.0231. Two points do not measure a spread, and this is
  what it costs to forget that.
- **The head is still predicting close to the mean.** Every arm sits at
  0.08–0.15 head sd against a target sd of 0.272. That number is not sign-
  sensitive and it has not moved much in any condition.
- **`velocity_w5_cpu_s1` is now the most valuable run in the queue.** It is A′1
  at seed 1, and it answers directly whether A′1's gate pass replicates or was
  a draw.

The methodological fix, when this is picked up: the gate has to be evaluated on
a per-class correlation averaged over seeds, with the spread taken *across*
seeds rather than across rows. That is three runs per arm rather than one, and
it is the difference between a result and an anecdote.

### What replicates: A′1 clears the gate at both seeds, at no detection cost

With two seeds on each of the baseline and A′1:

```
                onset F              head sd            gate
baseline s0     0.3006               0.084              fail (snare, tom_low neg)
baseline s1     0.2981               0.100              fail (crash neg)
A′1 s0          0.2761               0.105              PASS
A′1 s1          0.2992               0.137              PASS
```

Three things follow, and only these three:

1. **The gate verdict replicates even though its per-class content does not.**
   A′1 passes twice, the baseline fails twice. *Which* classes are positive
   changes completely between A′1's seeds — tom_low, tom_mid and crash at seed
   0; only tom_low at seed 1 — but "no class significantly backwards" held
   both times, and "some class backwards" held both times for the baseline.
   The property is stable; the identities are not.

2. **No detection cost is observed, and the standing caveat resolves that way.**
   A′1's two onset F values, 0.2761 and 0.2992, straddle both baseline values.
   The difference of means (~0.012) is half A′1's own seed-to-seed range
   (0.023). Whatever `velocity_weight` 5.0 costs detection, it is smaller than
   this design can see.

3. **A′1's head spread beats the baseline's at both seeds**: 0.105 and 0.137
   against 0.084 and 0.100, non-overlapping. Still 38–50% of the target's
   0.272, so the head remains closer to the mean than to the drummer.

That is the phase's result, and it is narrower than the three single-run
sections above suggested: **turning the velocity weight up makes the head stop
producing significantly-inverted classes, and widens what it produces, without
a measurable detection cost — and nothing can yet be said about which drum it
learns.**

`velocity_probe_cpu_s2` (third baseline seed) is running and will tighten (1)
and (2). A third A′1 seed would be worth more than either, and is not queued.

---

### A′3 result: the composition is worse than either half of it

Linear head *and* `velocity_weight` 5.0 — the obvious next experiment, on the
reasoning that the two fixes addressed different faults. It does not combine
them. It fails the gate harder than A′2 did:

```
class          A′1 weight 5.0        A′2 linear head        A′3 both
kick           +0.028 spans 0        (not significant)      -0.129 [-.25,-.02]
snare          +0.062 spans 0        -0.138 [-.22,-.06]     +0.138 [+.06,+.21]
hat_open       -0.039 spans 0        +0.172 [+.02,+.29]     -0.162 [-.29,-.01]
tom_low        +0.306 [+.13,+.45]    +0.433 [+.32,+.54]     +0.403 [+.29,+.52]
tom_mid        +0.231 [+.09,+.37]    +0.283 [+.13,+.43]     +0.369 [+.23,+.49]
crash          +0.349 [+.17,+.50]    +0.120 spans 0         -0.081 spans 0
head sd        0.105                 0.154                  0.113
onset F        0.2761                0.2831                 0.2690
```

**Two classes significantly backwards, where A′2 had one**, and the spread it
was supposed to inherit from A′2 did not come with it: 0.113 against 0.154,
barely above A′1's 0.105.

What it did do is *move* which classes are backwards rather than reduce them.
Snare, which A′2 got wrong, is now correctly positive — consistent with the
weight change doing there what it did in A′1. Kick and hat_open went negative
in exchange. The toms are the strongest of any arm. So the composition is not
inert; it redistributes the error rather than removing it, which is what a head
still fitting something other than hit strength would look like.

**A′1 remains the only arm that clears the gate.** On the rule — a
significantly positive class with none significantly negative, then widest head
spread among those that pass — A′1 wins by being the only candidate, at 38% of
the target's spread.

Read that with the caveat the whole phase carries: these are single runs, and
the intervals are bootstrap over rows *within* one run. They say nothing about
run-to-run variation, which is exactly what the three seed runs still in the
queue are for. A gap smaller than the across-seed spread is not a result, and
that spread is still unmeasured.

---

## Phase B′ — the full-corpus run

One long run at every training clip the corpus can actually supply, which is
**846, not 897**: GMD indexes 897 train rows but ships 51 of them MIDI-only,
with `audio_filename` blank. The loader drops those, so 846 is what a
limit-free run trains on. (Checked rather than assumed — no *named* audio file
is missing from disk, in any split, so the download is complete; the shortfall
is the dataset's, not ours.)

The ~29 min/epoch, ~6 hours for 12 recorded here needs re-deriving. Phase A′
runs measure 300–343 s/epoch on 256 files, and only the training portion scales
with the corpus (validation stays at 120 clips), so 846 files extrapolates to
roughly 15–20 min/epoch and 3–4 hours for 12. That is an extrapolation from a
different config, not a measurement; the first epoch of the real run settles it.

This is the run that might move onset F off 0.30, and it produces the model
worth shipping. It must come after A′ or it gets redone.
`configs/v1_full_corpus.yaml` is ready for it — its `_base_` line is the Phase
A′ decision and has to be pointed at whichever arm actually won.

**Before starting it, read this — it is the binding constraint on this box.**

*This container runs only while the session is active.* It starts when
something wakes the session and stops shortly after the turn ends. Measured on
14 Sep: training attempts began at 12:40:51, 12:51:31 and 13:14:15, each within
seconds of a wake, and `uptime` read 0 min at every check. An epoch takes ~7
minutes, so a turn that starts training and returns gives it about one minute.
Epochs 3 and 4 of `velocity_lin_w5_cpu` completed only because a turn happened
to be holding a wait loop open; between 08:23 and 12:24 — four hours, with the
queue "running" the whole time — not one epoch finished.

There is no unattended training here. A background job is not background; it is
a foreground job that dies with the turn. Anything that takes longer than a
turn has to be held open deliberately, an epoch at a time.

Per-epoch resume is what makes that survivable: a run picks up from its last
completed epoch, so a batch can be advanced across many short sessions. What it
cannot do is rescue a run whose *epoch* outlasts the session, because nothing is
ever checkpointed.

Phase B′ extrapolates to **15–20 minutes an epoch**. That needs a held session
of at least that long per epoch, 12 times over, and any interruption inside an
epoch loses it entirely. Options, cheapest first:

- Run it somewhere that stays up — a machine whose processes outlive a chat
  turn. This is the honest answer, and Phase D wants a GPU anyway.
- Checkpoint inside the epoch (every N batches, saving the batch index with the
  optimiser state) so progress survives a restart that lands mid-epoch. A
  contained change to `src/train.py`, and the only option that makes *this* box
  viable for long runs.
- Shorten the epoch by splitting the corpus, which changes what an epoch means
  and breaks comparability with everything in A′.

Do not simply launch it and hope: the failure mode is silent, and looks
identical to a run that is merely slow.

**Done when** onset F has either moved or provably stopped moving with the data
limit lifted — which turns "undertrained" from an assumption into a finding
either way.

---

## Phase C — ship v0.1

None of this needs the CPU, so it overlaps the runs above.

- **Publish a bundle as a release.** The README's headline path names
  `flybeats-8piece.fb`, and no such file exists anywhere in the repo — the
  quick start is currently unfollowable by anyone who has not trained their own
  model. `scripts/export_bundle.py` takes seconds and verifies the bundle
  reproduces the checkpoint exactly before writing it. This is the cheapest
  thing on this page that changes whether the project is usable.
- **Verify the quick start** end to end from a clean checkout against the
  published bundle, on a machine with no `data/` directory.
- **Tag it**, with release notes that say plainly what it does and does not do:
  it follows the music's energy rather than its groove, and its velocity is not
  yet dynamics.

**Done when** someone with no connectome, no corpus and no sampler can go from
`git clone` to a wav of drums over their own audio.

---

## Phase D — the experiment that answers the question

Still hardware-blocked:

```bash
python src/ablations.py --config configs/v1_8piece.yaml --lesion --epochs 40 --seeds 5
```

`--seeds` is not optional — a gap smaller than the across-seed spread is not a
result. The harness was already seed-aware; what running it found instead were
two things that would each have cost a rented GPU day:

- **It crashed on the fourth arm.** `FlyBeats.forward` passes `substeps=` to
  whichever core is installed, and the two arms that replace the core outright,
  `gru` and `shortcut`, never gained the argument when the speed work added it.
  So the harness trained `real`, `rewired` and `sign_shuffled` — the expensive
  part — and then died. Nothing tested the control cores against the real
  core's calling convention, so nothing caught it. Fixed, and they are now
  tested against `ConnectomeRNN`'s signature itself rather than a copy of it,
  so the next parameter added there fails in CI instead of mid-run.
- **The spread it reports is over random graphs only.** Weight init and data
  order are pinned to a single draw for every arm and every repeat. That makes
  the arms exactly comparable — topology is the only thing that differs — but
  it means a gap clearing the reported spread is a gap *at that one init*.
  `--vary-init` repeats every arm across inits too, so each arm carries its own
  spread and the comparison is between distributions rather than between a
  distribution and a point. It costs `--seeds` times as many runs on the
  deterministic arms, which is why it is opt-in, and it is where a published
  number should come from.

Whether the second point changes any conclusion is a quantity this project does
not have yet — and the Phase A′ seed runs are measuring exactly it. If onset F
moves more across inits than the arms differ by, the matched design cannot
carry the claim by itself.

Both halves of the command have now been run end to end on a smoke config —
the two arms that had never executed at all, and the `--lesion` sweep, which
produces its table. That says the machinery works, and nothing about the
biology: one epoch of synthetic audio is not a model to interpret.

---

## Parked, with the reason

- **`SoundFontBank`** ([SOUNDBANK_PLAN.md](SOUNDBANK_PLAN.md) step 7) — needs a
  machine with a sound card. FluidSynth is not installed in the working
  environment and there is no audio device, so it could not be exercised at
  all, and the repo already carries "live audio has never been run" as a
  hazard.
- **Speed on fills only** ([SPEED_PLAN.md](SPEED_PLAN.md)) — driving `k(t)`
  from pC1's own activity per frame, so the drummer speeds up where it is
  already playing harder. The genuinely interesting one left, and the per-frame
  schedule the fractional dial needed is exactly the mechanism it takes.
  Nothing gates it.
- **The Phase 1 trim claim** — 3 hops from JO reaches 154,853 of 162,517
  neurons, so the subgraph is selected by `_trim` rather than by anatomy. A
  real methodological weakness, but replacing the criterion invalidates the
  subgraph cache and every model trained on it, so it belongs between
  experiment campaigns rather than mid-flight.

  One measurement to add to it: after the trim, **49 of the 405 sensory
  neurons have no path to the motor pool at all** in the 10k subgraph, and 41
  of 405 in the 30k. They receive, and they cannot reach the readout, so they
  can be lesioned with no possible effect — a null for those neurons means the
  trim cut their path, not that the biology does not use them. Every other
  confirmed population reaches the motor pool in full, which is why the
  exact-zero `aPN1` and `vPN1` rows in a smoke lesion sweep are a symptom of an
  untrained model rather than of a disconnected one.
- **The 162k render tier** and **the spiking model** — neither blocks anything.
