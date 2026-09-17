# Release roadmap: from a research repo to an installable instrument

[ROADMAP.md](ROADMAP.md) orders the *next few* pieces of work and stops at
"v1.0 — the claim". This file covers the rest of the distance: what it takes
to hand flybeats to someone who is not its author and have them install it,
open a window, and play it — with the training campaign finished rather than
parked.

It is a companion to, not a replacement for, the existing plans.
[PLAN.md](PLAN.md) is the science, [UI_PLAN.md](UI_PLAN.md) and
[VISUALIZER_PLAN.md](VISUALIZER_PLAN.md) are the screens,
[SETUP.md](SETUP.md) is today's install path and [RUNBOOK.md](RUNBOOK.md) is
the GPU machine. This file is the order those become one product, and the
gates between.

---

## The honest starting position

Four facts set every date below. None of them is a guess; each is already
recorded in the repo.

**1. There is no shippable model yet, and no shipped anything.** The last run
trained on 256 of 897 GMD clips and reached onset F 0.30. The velocity head
reads dynamics weakly and was, until Phase A′1, reading two classes backwards.
No `.fb` bundle exists anywhere in the repo, so [SETUP.md](SETUP.md)'s "play
path" — the five-minute one — is currently unfollowable by anybody.

**2. There is no UI code.** `viz/` does not exist. `src/masking.py` does not
exist. `live_server.py`, the event schema, the recorder, the frontend: all
planned in detail, none written. What does exist is
`flybeats_demo_September142026.html` — a static simulation of the PR-mode
screen. That is a design fixture, not a running surface, and its real value is
as the visual target the first real shell has to match.

**3. There is no installable package.** No `pyproject.toml`, no `setup.py`, no
console entry points. Every documented invocation is `python src/<file>.py`
from the clone root, and installation is a sixteen-section manual. That is a
reasonable state for a research repo and a disqualifying one for a release.

**4. Live audio has never been run.** Not once, on any machine. The streaming
path is benchmarked (5.4 ms per 20 ms block at 10k nodes, 14.2 ms at 30k — 1.4×
realtime at the shipping tier) and rendered offline, but `sounddevice` has never
opened a device and no MIDI has ever left the process. Every latency claim below
is a projection from the offline benchmark until R1 closes this.

The consequence worth stating plainly: **the UI track and the training track do
not block each other, and should not be sequenced as if they do.** The recorder
in Phase V2 is what decouples them — once fixtures exist on disk, every screen
in [UI_PLAN.md](UI_PLAN.md) can be built, reviewed and tested with no GPU, no
neuPrint connection and no trained model. Getting that gate early is worth more
than anything else on this page.

---

## Five releases

| | Release | Gate | The one sentence |
|---|---|---|---|
| **R0** | `0.1` — installable | CPU | `pip install flybeats`, a published bundle, drums out of your own wav |
| **R1** | `0.2` — audible | CPU + a sound card | Live audio in, MIDI and samples out, on real hardware, measured |
| **R2** | `0.3` — visible | CPU | The two stage surfaces, fixture-driven: brain view and avatar |
| **R3** | `0.4` — operable | CPU | The full shell: PR mode, Presentation mode, the installer wizard |
| **R4** | `1.0` — the claim | GPU | Finished training, the Phase D ablation table, the Bench, signed installers |

Training finalization is not a phase at the end. It runs as a track from R0 to
R4 and each release re-cuts the bundle from the best checkpoint available at
that point. The version number of the *software* and the version number of the
*model* are separate and both go in the release notes.

---

## R0 — `0.1`, installable

**Gate:** none. Reachable today, on CPU, in a cloud session.
**Why first:** every other release is measured by someone else running it, and
right now nobody can.

### R0.1 — Make it a package

| | change | why |
|---|---|---|
| a | `pyproject.toml`, package `flybeats`, `src/` → `src/flybeats/` | `python src/realtime.py` is not an install story, and the flat `src/` layout makes `pip install` ambiguous about what ships |
| b | Console entry points: `flybeats play`, `train`, `ablate`, `doctor`, `bundle`, `serve` | One binary, subcommands, `--help` that lists them. The CLI is the product's real API and the GUI shells out to it |
| c | Extras: `[realtime]` (sounddevice, mido, rtmidi), `[connectome]` (neuprint), `[dev]` | The play path should not drag in the build path's dependencies. Today `requirements.txt` comments them out, which is a different thing from declaring them optional |
| d | Pin a lower bound *and* a tested upper bound on torch | A silent CPU-wheel install is already the repo's most-documented failure |

The move to `src/flybeats/` touches every import and every documented command.
Do it once, here, before there are users with muscle memory — not later.

### R0.2 — `flybeats doctor`

`scripts/diagnose.py` and `scripts/bootstrap_local.py` already contain most of
this logic and nothing surfaces it as a first-class command. Promote it:

- Python version, platform, venv state
- torch build, CUDA availability, the CPU-wheel-when-a-card-is-present case
  (already exits non-zero in `bootstrap_local.py` — keep that)
- Audio devices present, MIDI ports present, FluidSynth present
- Which assets are on disk: connectome, corpus, subgraph cache, bundles, kits
- Thread oversubscription warning (the 13× penalty is real and undocumented at
  the point of failure)
- **Machine-readable `--json`**, because the installer wizard in R3 consumes it

**Done when** a user with a broken install can paste one command's output and
have the problem be legible from it.

### R0.3 — Publish a bundle

This is [ROADMAP.md](ROADMAP.md) Phase C and it stays the cheapest
high-value item on any of these pages. `scripts/export_bundle.py` already
verifies the bundle reproduces the checkpoint exactly before writing.

- Cut `flybeats-8piece.fb` from the best checkpoint that exists at R0 —
  which is a 256-clip, onset-F-0.30 model, and the release notes say so
- Attach it to a GitHub release; `flybeats doctor --fetch-bundle` pulls it
- **Ship a model card with it** (see the training track below). A bundle
  without one invites exactly the claim the project keeps refusing to make

### R0.4 — Verify the play path from cold

On a machine with no `data/`, no corpus, no connectome: `pip install flybeats`,
fetch the bundle, render drums over a wav. Any step that fails is R0 work.

**R0 done when** someone with no connectome, no corpus and no sampler goes from
a package manager to a wav of drums, and the only document they needed was a
quick start that fits on one screen.

---

## R1 — `0.2`, audible

**Gate:** a machine with a sound card and, ideally, a MIDI device. Per
[RUNBOOK.md](RUNBOOK.md), that is the local Windows machine, not a cloud
session.

The whole of Phase 5 is written and none of it has ever touched a device. This
release is where the projection becomes a measurement.

- **Run the live path.** `sounddevice` input, ring buffer, streaming inference,
  MIDI out via `mido`/`python-rtmidi`, samples out via `soundbank.py`. Record
  what actually happens, including the failures
- **Measure end-to-end latency**, not inference latency: capture → encoder →
  forward → peak pick → MIDI on the wire. The 20 ms budget in
  [PLAN.md](PLAN.md) covers the buffer and the inference; the parts nobody has
  measured are device buffering and driver round-trip, and on Windows that
  usually means the difference between WASAPI shared and ASIO
- **Underrun behaviour.** Design rule 2 of [VISUALIZER_PLAN.md](VISUALIZER_PLAN.md)
  says a stalled client must never cause an audio underrun. That rule is
  currently untested because there has never been an audio stream to under-run
- **Exercise `SoundFontBank`**, the one parked piece of
  [SOUNDBANK_PLAN.md](SOUNDBANK_PLAN.md), which needed exactly this machine
- **Device selection and fallback** as a real code path: no default input, no
  MIDI ports, a device that disappears mid-session. These become UI states in
  R3, so they need to be enumerable now

**R1 done when** the latency table in the README reports a measured
capture-to-MIDI number on named hardware, and the repo no longer carries "live
audio has never been run" as a hazard.

---

## R2 — `0.3`, visible

**Gate:** none for most of it. Phase V0's skeleton fetch needs a neuPrint token
and runs overnight, so it starts on day one of this release and everything else
proceeds around it.

This is [VISUALIZER_PLAN.md](VISUALIZER_PLAN.md) V0 → V4, built in its stated
order, with one change of emphasis: **V2's recorder is the release's real
deliverable**, and it should land before any frontend work at all.

### The order, and what each unlocks

| | Phase | Unlocks |
|---|---|---|
| a | **V0** skeleton fetch + cache | Everything anatomical. Resumable, rate-limited, overnight. Ship the coverage report — truncated EM fragments have no skeleton and silent dropping is the failure mode |
| b | `src/masking.py` | The shared primitive under both the Phase 4 ablations and the live lesion toggles. Written once, two menus over it. It is small and it is load-bearing for R4 |
| c | **V2** event schema + `live_server.py` + **recorder** | The decoupling gate. After this, frontend work needs no model |
| d | Fixtures committed | At least: a clean take, a lesion mid-take, a degradation event, a schema-mismatch case. These are the frontend's test corpus and the Presentation-mode kiosk's content |
| e | **V1** static 3D explorer | Offline, and it validates the subgraph extraction independently of training |
| f | **V3** brain HUD | Soma point cloud, recolored per frame by group activity |
| g | **V4** avatar + kit | The surface that survives degradation, and the one that reads at distance |

### The two correctness items, not styling items

Both are named in [UI_PLAN.md](UI_PLAN.md) and both are easy to defer into a
bug:

1. **Per-theme activity polarity.** `activity_channel: brightness` inverts
   meaning between light and dark stages. On white, maximum activity becomes
   invisible. Light mode drives darkness *and* saturation.
2. **Lesion is not an activity value.** Silent and lesioned must be
   distinguishable in both themes, so lesioned groups render as hollow outline
   — structurally absent, not temporarily quiet. The lesion moment is the best
   demo the project produces and it fails outright if the room cannot tell the
   two apart.

Write these as a contrast test over the committed fixtures, run in CI. "Verify
silent, lesioned and peak are mutually distinguishable in both themes" is a
pass/fail check, not a design review.

### The performance question this release has to answer

[VISUALIZER_PLAN.md](VISUALIZER_PLAN.md) open question 1 — whether
`grouping: type_bilateral` stays under ~150 groups on the shipping subgraph —
is answerable from `data/verified_types.json` and the extracted subgraph today,
with no rendering at all. Answer it before the payload shape is fixed, because
the per-tick payload and the frontend update path both change if it does not.

**R2 done when** both surfaces render a committed fixture at 50 Hz on a laptop
with no GPU, and the degradation ladder (50 → 25 Hz → coarsen grouping → drop
tweens → drop brain view) is exercised by a fixture rather than by hope.

---

## R3 — `0.4`, operable

**Gate:** R2's fixtures. No GPU, no model.

This is the release the request is really about: the shell around everything,
for installation *and* for operation.

### R3.1 — One shell, one decision

**Decision: a desktop shell hosting the local Python service, with the frontend
served from it — and every screen in that shell backed by a CLI command it
shells out to.**

The second half of that sentence is the load-bearing half. It gives three
things at once: the GUI can never do something the CLI cannot (so the headless
path stays first-class and CI-testable), the GUI's actions are reproducible and
pasteable when a user asks for help, and the installer wizard becomes a thin
front end over `flybeats doctor --json` rather than a second implementation of
the same probing logic.

`flybeats_demo_September142026.html` is the visual target for PR mode. It
already resolves the layout, the density control and the theme behaviour. Treat
it as the acceptance reference: the real shell should be diffable against it.

### R3.2 — The installation interface

This is new work with no plan document behind it yet, so it gets specified
here. It is a first-run wizard, not a separate installer binary — the download
is the app, and the app walks you through what it still needs.

**Step 0 — What do you want to do?** The single most important screen, because
the two paths differ by about 6 GB and four hours:

| Path | Needs | Time |
|---|---|---|
| **Play** | Python, the package, a bundle, a kit | ~5 min, ~1 GB |
| **Build** | the above + connectome (1.1 GB) + corpus (5.4 GB) + a GPU worth using | an evening |

[SETUP.md](SETUP.md) already draws this distinction in prose at the top of the
document. The wizard makes it a fork rather than a paragraph somebody skims.

**Step 1 — Environment check.** Rendered from `flybeats doctor --json`. Each
row is pass / warn / fail with a specific remedy, and the CPU-wheel-with-a-card
case is called out loudly because it is the failure that silently invalidates
everything downstream.

**Step 2 — Assets.** Resumable downloads with real progress and real sizes:
bundle, sample kit, and on the build path the connectome and corpus. Verify
checksums. Show what is already on disk rather than re-fetching it.

**Step 3 — Devices.** Pick audio in, audio out, MIDI out. Test-tone button,
input meter. This is where R1's device enumeration and fallback paths surface.

**Step 4 — First sound.** Play a bundled demo clip through the model and show
the avatar hitting the kit. The wizard does not end on a checklist; it ends on
the thing working.

**Obligations, from the rest of the project's rules:**
- Every wizard step prints the equivalent CLI command it ran
- Kit tiers the confirmed motor-neuron count cannot support are shown disabled,
  never offered and then rejected
- An uninstall path that names every directory it created (SETUP.md §15 already
  has the content; the wizard should be able to act on it)

### R3.3 — The operation interface

[UI_PLAN.md](UI_PLAN.md)'s v1 cut, built:

- **PR mode** with Brain / Split / Avatar stage toggle, Split as the default
- **Play / Inspect / Deep** density — one control, no reload, no state loss
- **Three-field slider callouts** anchored to the populations they drive
  (modulates / acts through / live rate), with receptor labels *derived from the
  neurotransmitter table*, not hardcoded, and Pocket's receptor field reading
  not-applicable rather than being filled for symmetry
- **The rail** — mapping and theme keys, hot by definition
- **Curated lesion set** over `masking.py`, audible within one buffer tick
- **Hot / cold separation** with cold controls behind one collapsed
  Configuration section with an explicit Apply and a reconnect label
- **Source indicator** — live model vs. fixture replay — persistent in the top
  bar, with the biological sliders visibly inert during replay
- **Health strip** — latency, dropped frames, schema version, with a real error
  state for a schema version the frontend does not know
- **Degradation indicator** — announced, never silent, visible at every density
- **Presentation mode** as a second window on the same event stream: avatar-led,
  brain reduced to a ribbon, two touch-sized controls (genre pad, rhythm gate),
  theme pinned rather than following the host machine, and **runnable from a
  fixture with no GPU, no neuPrint and no model** — a shipping requirement, not
  a side effect

**Out of R3, explicitly:** the touchscreen build, hardware controller mapping,
skin packs beyond `schematic` plus one theme, and the v2 performance modes
(continue, call-and-response, accompany) — stub buttons with tooltips, visibly
disabled.

**R3 done when** a person who has never seen the repo installs from a download,
finishes the wizard, and lesions pIP10 mid-take from a window — and when the
same build runs unattended from a fixture on a machine with nothing else on it.

---

## R4 — `1.0`, the claim

**Gate:** a GPU. This is the only release that is hardware-blocked, and by this
point it is the *only* thing blocking.

- **Phase D**, the ablation campaign that is the reason the project exists:
  real subgraph vs. degree-matched rewiring vs. sign shuffle vs. parameter-
  matched GRU, plus the encoder→decoder shortcut floor. Five seeds. A gap
  smaller than the across-seed spread is not a result
- **The Bench** — [UI_PLAN.md](UI_PLAN.md)'s third surface: ablation compare,
  V1 explorer, checkpoint diffs. A separate window with no real-time obligation,
  which is exactly why it was evicted from the live modes
- **Signed, checksummed installers** per platform, and the release notes that
  say plainly what the thing does and does not do
- **The Phase 4 table published alongside every headline result**, per
  [PLAN.md](PLAN.md)'s own instruction, including if it comes out against the
  hypothesis

Open question to settle before the Bench is built:
[UI_PLAN.md](UI_PLAN.md) open question 2 — four conditions live in parallel, or
precompute and scrub. Scrubbing is cheaper and the Bench has no latency budget;
live parallel is what makes the lesion demo work there too. The answer changes
the Bench's architecture, so it is not a detail to discover during
implementation.

**R4 done when** the central claim has an answer, with error bars, in a table a
stranger can read — and the software around it is something they can install
without reading sixteen sections of anything.

---

## The training track

Running the whole length of the above, because the model is a deliverable with
its own version number and its own gates.

### T1 — Finish Phase A′ (CPU, ~100 min each, sequential)

Two candidates remain after A′1 raised `velocity_weight` to 5.0 and flipped
three classes to significantly positive:

- **A′2** — linear head with the target centred, instead of the sigmoid whose
  gradient is flattest exactly where GMD's velocities cluster
- **A′3** — score only near the onset peak, where the streaming path actually
  reads velocity, rather than flat across the kernel support

Both configs already exist (`configs/velocity_lin_cpu.yaml`,
`configs/velocity_peak_cpu.yaml`). **Never run two at once** — torch takes a
thread per core per process and one epoch went from 76 s to 1,415 s when they
overlapped. Sequential, or set `OMP_NUM_THREADS`.

### T2 — Settle the A′1 detection cost (needs seeds)

A′1 came in at onset F 0.2761 against a 0.3006 baseline. One run with no seeds
cannot distinguish a real detection cost from run-to-run noise, and the number
must not be quoted until it can. This is a small seeded re-run and it gates
whether `velocity_weight: 5.0` ships.

### T3 — Phase B′, the full-corpus run (GPU)

897 training clips instead of 256. ~29 min/epoch projected on CPU; the point of
the GPU is to make that a different number. This is the run that either moves
onset F off 0.30 or establishes that the data limit was not the binding
constraint — a finding either way, and the first model genuinely worth shipping.

Must come after T1/T2, per the rule that governs the whole roadmap: **anything
that changes the loss lands before anything expensive.**

### T4 — Promote and document

Each release re-cuts `flybeats-8piece.fb` from the best available checkpoint.
Every cut ships a **model card** stating, without softening:

- What it was trained on: corpus, split, clip count, epochs, seeds
- Onset F per class with intervals, not a pooled point estimate
- Velocity: per-class head correlation at the onset peak with bootstrap
  intervals, and the plain statement that this is weak dynamics rather than
  dynamics
- What it does musically: follows the music's energy rather than its groove
- The Phase 4 ablation table, once it exists, or its absence marked as such
- Provenance: connectome version, subgraph cache hash, config, commit

### T5 — Phase D (R4)

Covered above. The one addition worth making *before* the hardware exists:
make `src/ablations.py` seed-aware and one command, so a GPU day turns into
results rather than into a day of harness setup.

### Parked, and why — unchanged

`SoundFontBank` unparks at R1 (it needed a sound card, which R1 brings).
**Speed on fills only** ([SPEED_PLAN.md](SPEED_PLAN.md)) — driving `k(t)` from
pC1's own per-frame activity — is nothing's blocker and the most interesting
thing left; it is a good R3-or-later addition because it is a control surface as
much as a model change. **The Phase 1 trim claim** — 3 hops from JO reaches
154,853 of 162,517 neurons, so the subgraph is selected by `_trim` rather than
by anatomy — is a real methodological weakness whose fix invalidates the
subgraph cache and every model trained on it. It belongs *between* campaigns:
after R4's Phase D result is banked, not during it.

---

## What runs in parallel, and what genuinely blocks

```
R0 packaging ──┬─> R1 audio (needs a sound card)
               │
               ├─> R2 V0 skeleton fetch (overnight, needs a token)
               │        └─> V2 recorder + fixtures ──> V1/V3/V4 ──> R3 shell
               │
               └─> T1 velocity fixes ──> T2 seeds ──> T3 full corpus (GPU)
                                                          └─> T5 Phase D ──> R4
```

Three real blockers, and only three: a sound card for R1, a neuPrint token and
an overnight for V0, and a GPU for T3/T5. Everything else is work, not waiting.

The sequencing mistake available here is building frontend surfaces before the
V2 recorder exists, which couples every screen to a running model and turns
every UI change into a training-environment problem. The recorder is cheap and
it is the gate. Build it first.

---

## Risks

- **Frontend scope creep.** The skin and theme system absorbs unbounded effort.
  Ship `schematic` plus one theme; skin packs are an extension point, not a
  deliverable. This is already flagged in
  [VISUALIZER_PLAN.md](VISUALIZER_PLAN.md) and it is the risk most likely to
  eat R3.
- **The wizard becoming a second implementation.** If installer logic diverges
  from `flybeats doctor`, there are now two probe paths and only one is tested.
  The rule — every screen shells out to a command — exists to prevent this and
  needs enforcing in review, not in intent.
- **Shipping a bundle that over-promises.** A 0.30-onset-F model in a polished
  shell reads as a finished instrument. The model card is the mitigation and it
  is not optional.
- **Cloud sessions cannot do the long work.** Documented in
  [RUNBOOK.md](RUNBOOK.md): the container runs only while a turn is active, and
  four hours of nominally-running queue produced zero finished epochs. Nothing
  on the training track should be scheduled into one.
- **Sync drift over long sessions.** If the visual and audio clocks diverge,
  resync on `trigger` timestamps rather than free-running the animation clock.

---

## Open questions this roadmap does not settle

1. **Desktop shell technology.** A native wrapper (Tauri/Electron) around the
   local service, or a browser tab pointed at `localhost`? The browser is
   dramatically cheaper and gets remote display for free; the wrapper is what
   makes "download and run" true for a non-technical user and is what the
   installer wizard implies. Decide before R3 starts, because it determines
   whether R3 needs a per-platform build and signing pipeline at all.
2. **Genre pad layout** — a fixed 2D projection of the style embedding, or one
   recomputed as genres are added? A moving pad breaks muscle memory between
   sessions, which matters more for performance than for research.
3. **Second-screen Presentation window** — same process with a shared stream, or
   a separate client connecting to `live_server.py` like any other? The second
   is cleaner and gets remote display for free, and it interacts with question 1.
4. **Does the shipping bundle carry the visualizer's anatomy?** `soma_layout.json`
   makes a bundle self-contained for the brain view; it also makes the ~10 MB
   number wrong. Either the play path's brain view needs a separate download or
   Presentation mode is fixture-only in the play path — both are defensible, and
   the choice changes what R0's bundle contains.
