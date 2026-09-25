# Release roadmap: from a research repo to an installable instrument

[ROADMAP.md](ROADMAP.md) orders the *next few* pieces of work and stops at
"v1.0 — the claim". This file covers the rest of the distance: what it takes to
hand flybeats to someone who is not its author and have them install it, open a
window, and play it — with the training campaign finished rather than parked,
and with each release's own test gate as a shipping condition rather than a
follow-up.

Companions: [PLAN.md](PLAN.md) is the science, [UI_PLAN.md](UI_PLAN.md) and
[VISUALIZER_PLAN.md](VISUALIZER_PLAN.md) are the screens,
[TEST_PLAN.md](TEST_PLAN.md) is the detail behind every test gate below,
[SETUP.md](SETUP.md) is today's install path and [RUNBOOK.md](RUNBOOK.md) is
the GPU machine.

---

## How to read the token estimates

Every task carries an estimate of what it costs an AI agent to *execute*, in
tokens. They exist for budgeting sessions, not for scheduling people.

**Method.** Measured sizes of the files each task must read and write
(chars ÷ 4), then:

```
session ≈ 15k baseline  +  3 × (tokens read)  +  4 × (tokens written)
```

The baseline is system prompt and tool definitions. The 3× on reading covers
exploration that turns out to be unnecessary — greps, files opened and
discarded. The 4× on writing covers the edit / run / fix cycle, which re-reads
its own output two to four times.

**For scale**, the whole repository measures: `src/` 48.8k, `tests/` 25.2k,
`scripts/` 25.3k, docs 48.3k, configs 1.9k. So a task that genuinely touches
every module is reading ~50k before it writes anything.

**Bands are ±50%, and skew high.** They assume an agent that already has repo
context and does not take a wrong architectural turn. They exclude re-work after
a rejected review.

**The number that actually matters is the session count.** A context window is
~200k. Anything estimated above ~150k needs splitting before it starts, or it
will compact mid-task and lose the thread. Those tasks are marked
**multi-session** with a suggested split.

---

## The honest starting position

Four facts set everything below. Each is already recorded in the repo.

**1. There is no shippable model, and no shipped anything.** The last run
trained on 256 of 897 GMD clips and reached onset F 0.30. Those 256 are all
from drummer1, because `max_files` keeps the first rows of an info.csv ordered
by drummer. No `.fb` bundle exists
anywhere, so SETUP.md's five-minute "play path" is unfollowable by anybody.

**2. There is no UI code.** `viz/` does not exist. `src/masking.py` does not
exist. The event schema, the recorder, the frontend: planned in detail, none
written. `flybeats_demo_September142026.html` is a static simulation of the
PR-mode screen — a design fixture, and the visual target the real shell must
match.

**3. There is no installable package.** No `pyproject.toml`, no entry points.
Every documented invocation is `python src/<file>.py` from the clone root.

**4. Live audio has never been run.** Not once, on any machine. Every latency
claim is a projection from the offline benchmark until R1 closes it.

**5. The suite tests the library thoroughly and the product not at all.**
Thirteen test files, 139 passing — and `subprocess` appears nowhere in
`tests/`, so no test runs a documented command. Three gaps cost real money:
`build_arm` is never called (so no arm is proven to forward-pass), `deep_merge`
is untested (so config inheritance is unverified — which is how `device: auto`
reached the `*_cpu.yaml` configs unnoticed), and `train.evaluate` is never
called (so the function producing every quoted onset F is unexercised).

**The consequence:** the UI track and the training track do not block each
other. The V2 recorder is what decouples them — once fixtures are on disk, every
screen builds with no GPU, no neuPrint and no model.

---

## The testing rule

**No release ships on a suite that does not cover its own new surface.** Test
gates below are shipping conditions, not follow-ups. Two consequences worth
stating:

- The Tier 1 tests from [TEST_PLAN.md](TEST_PLAN.md) land in **R0**, before any
  GPU time. They are the ones whose failure costs a campaign.
- The cheap CLI safety net (`--help` exits 0, unknown flag exits non-zero) lands
  **before** the packaging refactor, not after. It is what makes the refactor
  safe.

---

## Five releases

| | Release | Gate | The one sentence | Tokens |
|---|---|---|---|---|
| **R0** | `0.1` installable | CPU | `pip install flybeats`, a published bundle, drums from your own wav | ~590k–930k |
| **R1** | `0.2` audible | sound card | Live audio in, MIDI and samples out, measured on real hardware | ~210k–345k |
| **R2** | `0.3` visible | CPU | The two stage surfaces, fixture-driven | ~850k–1.33M |
| **R3** | `0.4` operable | CPU | The full shell: PR mode, Presentation mode, installer wizard | ~880k–1.42M |
| **R4** | `1.0` the claim | GPU | Finished training, the Phase D table, the Bench, signed installers | ~550k–880k |

Training finalization runs as a track from R0 to R4, not as a phase at the end.
Each release re-cuts the bundle from the best checkpoint it has, with a model
card. Software version and model version are separate and both go in the notes.

---

## R0 — `0.1`, installable

**Gate:** none. Reachable today, on CPU.

### Tasks

| | Task | Tokens | Notes |
|---|---|---|---|
| R0.1a | `pyproject.toml`, extras (`[realtime]`, `[connectome]`, `[dev]`), console entry points | 40k–65k | Do before the move |
| R0.1b | `src/` → `src/flybeats/`, rewrite imports across 15 modules, 13 tests, 13 scripts | 150k–250k | **multi-session** |
| R0.1c | Update path references across docs (48.3k of markdown) | 60k–95k | Mechanical, verify links |
| R0.2 | `flybeats doctor`, promoting `diagnose.py` (2.7k) and `bootstrap_local.py` (4.4k), with `--json` | 60k–90k | Wizard consumes the JSON |
| R0.3 | Cut and publish a bundle | 25k–40k | **GPU machine only** — see below |
| R0.4 | Verify the play path from a cold machine | 30k–50k | Mostly running, not writing |
| R0.5 | Preflight guards on `train`: CPU-wheel case, thread oversubscription, missing corpus | 40k–60k | Replaces the training GUI — see the decision record |

**Split R0.1b as:** `src/` layout move plus import rewrite (~100k) → test-suite
import fixes and green suite (~80k) → scripts and CI (~50k).

**R0.3 is not a cloud task.** `runs/` is gitignored and no checkpoint is
committed anywhere, so only a machine that has trained one can cut a bundle. In
practice that is the box in [RUNBOOK.md](RUNBOOK.md) — an errand for the next
session there.

### Test gate — Tier 1, plus the refactor's safety net

| | Test | Tokens | Why here |
|---|---|---|---|
| ~~T1.1~~ | ~~Every arm in `ARMS` builds and forward-passes, parametrized~~ | **landed 21 Sep** | And it caught the thing it was written for, live: `substeps` had reached `ConnectomeRNN` and `FlyBeats` but not `GRUCore`/`ShortcutCore`, so both raised `TypeError` on every call through `FlyBeats.forward`. Same failure as the precedent, different keyword |
| ~~T1.2~~ | ~~`deep_merge` plus a resolution table per shipped config~~ | **landed 21 Sep** | The `_cpu` family was resolving to `device: auto`. Pinned at the root of the family; the table now covers all ten configs and a config with no row fails |
| ~~T1.3~~ | ~~`train.evaluate` against known activations~~ | **landed 21 Sep** | Held |
| ~~T1.4~~ | ~~Seed determinism across arms~~ | **landed 21 Sep** | Held, including the one worth pinning: an arm moves with the seed iff it is in `STOCHASTIC_ARMS` |
| T3.2 | Argument-surface smoke tests for every entry point | 25k–40k | **Land before R0.1b.** Cheapest possible refactor insurance |
| T3.3 | A small `.fb` fixture built from `subgraph_2k` | 25k–40k | Gives the play-path test something to run without training |

**Gate total: ~50k–80k remaining.** Tier 1 is done — 137 CPU tests before it,
192 after — and it paid for itself on the first one. See
[results/local/2026-09-21-tier1-and-the-broken-venv.md](results/local/2026-09-21-tier1-and-the-broken-venv.md),
which also records a Phase D confound found in passing: the `gru` and
`shortcut` arms discard the genre tonic entirely, so they differ from the real
arm in two ways rather than one. That needs a decision, not a commit.

**R0 done when** someone goes from a package manager to a wav of drums, the only
document they needed fits on one screen, and the suite proves every ablation arm
runs.

---

## R1 — `0.2`, audible

**Gate:** a sound card, and ideally a MIDI device. The local Windows machine,
not a cloud session.

| | Task | Tokens | Notes |
|---|---|---|---|
| R1.1 | Run the live path for the first time; record what happens | 30k–50k | Mostly hardware, low token |
| R1.2 | End-to-end latency: capture → encoder → forward → peak pick → MIDI | 30k–50k | Device buffering and driver round-trip are the unmeasured parts |
| R1.3 | Device selection and fallback as real code paths | 40k–70k | No default input, no MIDI ports, device lost mid-session |
| R1.4 | Exercise `SoundFontBank` | 25k–40k | The one parked piece of SOUNDBANK_PLAN |

### Test gate

| | Test | Tokens |
|---|---|---|
| — | Device fallback paths against a fake/loopback device, so the three failure states are covered without hardware in CI | 50k–80k |
| T3.1 | The play path as a command: `--render` in, non-silent wav of expected duration out | 35k–55k |

`realtime.render_file`, `load_checkpoint` and `drummer_for` are all currently
untested, and they are that path.

**R1 done when** the README reports a measured capture-to-MIDI number on named
hardware, and "live audio has never been run" leaves the hazard list.

---

## R2 — `0.3`, visible

**Gate:** V0's skeleton fetch needs a token and an overnight. Start it day one;
everything else proceeds around it.

Build order is [VISUALIZER_PLAN.md](VISUALIZER_PLAN.md)'s, with one change of
emphasis: **the V2 recorder is the release's real deliverable** and lands before
any frontend work.

| | Task | Tokens | Notes |
|---|---|---|---|
| R2.a | `scripts/fetch_skeletons.py` — resumable, rate-limited, coverage report | 60k–90k | Write day one so it runs night one |
| R2.b | `src/masking.py` — one primitive, two consumers | 50k–80k | Small and load-bearing for R4 |
| R2.c | Event schema + `live_server.py` + **recorder** | 120k–180k | The decoupling gate |
| R2.d | Commit fixtures: clean take, lesion mid-take, degradation, schema mismatch | 40k–60k | The frontend's test corpus |
| R2.e | V1 static 3D explorer | 80k–120k | Validates subgraph extraction independently |
| R2.f | V3 brain HUD | 150k–250k | **multi-session** |
| R2.g | V4 avatar + kit | 150k–250k | **multi-session** |

**Answer before the payload shape is fixed:** whether `grouping: type_bilateral`
stays under ~150 groups on the shipping subgraph. It is answerable from
`verified_types.json` and the extracted subgraph today, with no rendering
(~20k), and both the per-tick payload and the frontend update path change if it
does not.

### Test gate

| | Test | Tokens | Why |
|---|---|---|---|
| T4.1 | Schema contract: known version accepted, unknown produces a *named* error, every fixture validates | 40k–60k | A subtly wrong render is worse than a refusal |
| T4.2 | `masking.py` parity — live lesion and ablation paths produce identical model state | 40k–65k | "One primitive, two menus" is only true if a test says so |
| T4.3 | Contrast: silent, lesioned and peak mutually distinguishable in **both** themes | 60k–90k | Protects the project's best demo |
| T4.4 | Fixture replay determinism, including sliders inert during replay | 40k–60k | Presentation mode is a fixture-only shipping requirement |

**Gate total: ~180k–275k.**

**R2 done when** both surfaces render a committed fixture at 50 Hz on a laptop
with no GPU, and the degradation ladder is exercised by a fixture rather than by
hope.

---

## R3 — `0.4`, operable

**Gate:** R2's fixtures. No GPU, no model.

### The one architectural decision

**A desktop shell hosting the local Python service, with every screen backed by
a CLI command it shells out to.** The second half is load-bearing: the GUI can
never do what the CLI cannot, its actions are pasteable when a user asks for
help, and the installer wizard becomes a thin front end over
`flybeats doctor --json` rather than a second implementation of the same probing.

Settle the shell technology before starting — native wrapper versus a browser
tab on `localhost` — because it decides whether R3 needs per-platform builds and
signing at all.

| | Task | Tokens | Notes |
|---|---|---|---|
| R3.1 | Shell scaffold and the local service it hosts | 80k–130k | |
| R3.2 | Installer wizard: path fork → environment → assets → devices → first sound | 200k–350k | **multi-session**, one per step pair |
| R3.3 | PR mode: stage toggle, density, callouts, rail, lesion set, hot/cold, source, health, degradation indicator | 250k–400k | **multi-session**; diff against the demo HTML |
| R3.4 | Presentation mode as a second window on the same stream | 100k–150k | |
| R3.5 | Read-only training monitor: loss curve, per-epoch val onset F, ETA, stop | 80k–120k | Launching stays on the CLI |
| R3.6 | `train --resume` | 50k–80k | The monitor over-promises without it |

**The wizard's step 0 is the important screen**: play path (~5 min, ~1 GB) versus
build path (an evening, ~6 GB). SETUP.md draws that line in prose at the top of
the document; the wizard makes it a fork rather than a paragraph people skim.

**Two rendering rules are correctness, not styling**, and both are in the R2 test
gate above: per-theme activity polarity, and lesion as its own channel rendered
as outline.

### Test gate

| | Test | Tokens |
|---|---|---|
| T3.4 | `doctor --json` shape: keys the wizard reads exist; the three exit-code cases from the 14 Sep bring-up hold | 30k–45k |
| — | Wizard/CLI parity: every step's underlying command runs headless and produces the same result | 50k–80k |
| — | Hot controls change nothing requiring reconnect; cold controls are unreachable outside Configuration | 40k–60k |

**Gate total: ~120k–185k.**

**R3 done when** a person who has never seen the repo installs from a download,
finishes the wizard, and lesions pIP10 mid-take — and the same build runs
unattended from a fixture.

---

## R4 — `1.0`, the claim

**Gate:** a GPU, and by now the only thing blocking.

| | Task | Tokens | Notes |
|---|---|---|---|
| R4.1 | Make `ablations.py` seed-aware and one command | 60k–90k | Do before the hardware exists |
| R4.2 | Run Phase D: five arms, five seeds | 40k–70k | Mostly wall-clock; tokens are launch and read |
| R4.3 | The Bench: ablation compare, V1 explorer, checkpoint diffs | 200k–300k | **multi-session** |
| R4.4 | Signed, checksummed installers per platform | 80k–150k | Scope depends on the R3 shell decision |
| R4.5 | Release notes and the published Phase 4 table | 40k–70k | Including if it comes out against the hypothesis |

**Settle before building the Bench:** four conditions live in parallel, or
precompute and scrub. Scrubbing is cheaper and the Bench has no latency budget;
live parallel is what makes the lesion demo work there too. It changes the
Bench's architecture, so it is not a detail to discover mid-implementation.

### Test gate

| | Test | Tokens |
|---|---|---|
| T2.1 | `beat_alignment_error` and `groove_similarity` against analytically known answers | 35k–55k |
| T2.2 | `feel.aggregate` weighted pooling | 25k–40k |
| T2.3 | Velocity probe statistics are seeded and reproducible | 30k–45k |
| — | Golden-value regression: `evaluate` on a fixed checkpoint and fixture, to tolerance | 40k–60k |

**Gate total: ~130k–200k.** Tier 2 lands here because this is where the numbers
get published and have to be defensible.

**R4 done when** the central claim has an answer with error bars in a table a
stranger can read, and the software around it installs without reading sixteen
sections of anything.

---

## The training track

| | Task | Tokens | Gate |
|---|---|---|---|
| T1 | Finish Phase A′ — A′2 and A′3, sequential | 20k–40k | CPU, ~4 h each on the local box |
| T2 | Seed the A′1 detection cost before quoting it | 30k–50k | CPU |
| T3 | Phase B′, the full-corpus run | 40k–70k | **GPU** |
| T4 | Promote each bundle and write its model card | 40k–60k | Per release |
| T5 | Phase D | 60k–100k | **GPU**, see R4.1 |

Token cost here is low because the work is wall-clock, not authorship: launch,
wait, read results, write them up. **Track total: ~190k–320k.**

Two standing rules. Anything that changes the loss lands before anything
expensive — that is why A′ precedes B′. And never run two training jobs at once:
the documented collapse is 76 s → 1,415 s per epoch.

**Every model card states**, without softening: corpus, split, clip count,
epochs, seeds; per-class onset F with intervals rather than a pooled point
estimate; per-class velocity correlation at the onset peak with bootstrap
intervals and the plain statement that this is weak dynamics, not dynamics; that
it follows the music's energy rather than its groove; the Phase 4 table once it
exists, or its absence marked as such; and provenance — connectome version,
subgraph hash, config, commit.

---

## Decision record: no training GUI for end users

Asked directly, and recorded because it will come back: should an end user get a
graphical way to train on their own hardware?

**No — and the reason is a data problem, not a UI one.**

`src/dataset.py` binds to the GMD / E-GMD layout: an `info.csv` pointing at audio
with **sample-aligned MIDI**. That alignment exists because the corpus was
captured on an electronic kit emitting MIDI while it recorded. A user's own audio
has no aligned MIDI and nothing here produces it. So "train it on your hardware"
resolves to recomputing the same model the project publishes as a bundle — fewer
clips, untuned, unseeded, six hours, for a result worse than a 10 MB download.

The heaviest argument against is not cost. This project's discipline is bootstrap
intervals over point estimates and seeds over single runs, after `hat_open` read
0.37, then 0.18, then 0.26 on one checkpoint before its sampling was seeded. A
window that ends a run by printing `onset F: 0.31` hands a stranger that number
with none of the scaffolding, and they will quote it.

There is also no `--resume`: a six-hour job on a laptop that sleeps, behind a
window implying robustness, is a promise the code does not keep.

**Instead:** preflight guards on `train` now (CPU-wheel case, thread
oversubscription, missing corpus — ~40k–60k, most of the GUI's real value), and
at R3 a **read-only training monitor** in the shell already being built — loss
curve, per-epoch validation onset F, ETA, stop button (~80k–120k), with
`--resume` alongside it (~50k–80k). Launching stays on the CLI.

The feature users actually want is "make it play like *my* kit." That needs an
e-kit MIDI capture path or adaptation from unlabeled audio — a product decision,
not a screen.

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

Three real blockers: a sound card for R1, a token and an overnight for V0, a GPU
for T3 and T5. Everything else is work, not waiting.

The available sequencing mistake is building frontend surfaces before the V2
recorder exists, which couples every screen to a running model. The recorder is
cheap and it is the gate.

---

## Total budget

| Track | Tasks | Test gate | Total |
|---|---|---|---|
| R0 | 405k–650k | 185k–280k | 590k–930k |
| R1 | 125k–210k | 85k–135k | 210k–345k |
| R2 | 670k–1.05M | 180k–275k | 850k–1.33M |
| R3 | 760k–1.23M | 120k–185k | 880k–1.42M |
| R4 | 420k–680k | 130k–200k | 550k–880k |
| Training track | 190k–320k | — | 190k–320k |
| **Total** | | **~700k–1.08M** | **~3.3M–5.2M** |

**Test work is ~20% of the total** — 700k–1.08M of 3.3M–5.2M. That is a
defensible proportion for a project whose central output is a claim about a
result, and it is concentrated where failure is expensive: R0's gate is 30% of
that release, because those are the tests whose absence costs a GPU campaign.

### What makes these wrong

- **A wrong architectural turn costs more than the task.** The R3 shell decision
  and the Bench's live-versus-scrub question are the two places where deciding
  late doubles the estimate.
- **Frontend work is the least predictable.** R2.f, R2.g and R3.3 are new code
  in a language the repo does not yet contain, against a design that exists only
  as a static HTML mock. Treat their upper bounds as the likely case.
- **They exclude review cycles.** A rejected approach re-runs the task.
- **They assume the fixtures hold.** If `type_bilateral` blows past ~150 groups,
  R2.c and everything downstream of the payload shape gets re-scoped.

---

## Risks

- **Frontend scope creep.** Ship `schematic` plus one theme; skin packs are an
  extension point, not a deliverable. Most likely thing to eat R2 and R3.
- **The wizard becoming a second implementation.** Every screen shells out to a
  command; enforce it in review, not in intent.
- **Shipping a bundle that over-promises.** A 0.30-onset-F model in a polished
  shell reads as a finished instrument. The model card is the mitigation and it
  is not optional.
- **Cloud sessions cannot do the long work.** The container runs only while a
  turn is active; four hours of nominally-running queue produced zero finished
  epochs.
- **Sync drift over long sessions.** Resync on `trigger` timestamps rather than
  free-running the animation clock.

---

## Open questions this roadmap does not settle

1. **Desktop shell technology** — native wrapper or a browser tab on
   `localhost`? Decide before R3 starts; it determines whether a per-platform
   build and signing pipeline exists at all.
2. **Genre pad layout** — a fixed 2D projection of the style embedding, or one
   recomputed as genres are added? A moving pad breaks muscle memory between
   sessions.
3. **Second-screen Presentation window** — shared process or a separate client
   on `live_server.py`? The second is cleaner and gets remote display free.
   Interacts with question 1.
4. **Does the shipping bundle carry the visualizer's anatomy?** `soma_layout.json`
   makes a bundle self-contained for the brain view and makes the ~10 MB number
   wrong. Either the play path's brain view needs a separate download, or
   Presentation mode is fixture-only there. The choice changes what R0's bundle
   contains.
