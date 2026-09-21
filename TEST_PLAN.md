# Test plan

The suite is not starting from nothing: thirteen files, 139 passed and 18
skipped on a clean checkout, with CI running everything that does not need a
card. This plan is about the gaps in it, and about the surfaces
[RELEASE_ROADMAP.md](RELEASE_ROADMAP.md) is about to add.

It is ordered by **cost of failure**, not by coverage percentage. That is this
repository's own standard — several tests here exist because something "fails
in CI rather than mid-run" — and it puts the work in the right order, because
a gap that wastes a GPU day is worth more than four gaps that waste a minute.

---

## What is already right, and should not be rebuilt

- **The fixtures are real.** `tests/fixtures/subgraph_2k.npz` is induced from a
  genuine MaleCNS extraction rather than synthesised, so Dale's law and the
  confirmed populations are actually exercised. `conftest.py` prefers a locally
  built cache where one exists. This is the right design and everything below
  builds on it.
- **Skips are honest.** The 18 skips need the 1.1 GB connectome or the 5.4 GB
  corpus, and the guards that skip them are what keeps a fresh clone runnable.
- **The `gpu` marker** keeps `-m "not gpu"` meaning exactly "what CI can run".
- **The numerically hard things are covered**: the hand-written `SparseSpMM`
  backward against a dense reference with duplicate edges forced, gradient
  checkpointing transparency, encoder windowed-vs-whole-clip agreement.

---

## Tier 1 — failures that cost a GPU campaign

These are the ones to write first, and all four are cheap.

> **All four landed on 21 September** (`tests/test_ablations.py`,
> `tests/test_config.py`, `tests/test_evaluate.py`), taking the suite from 137
> CPU tests to 187. The sections below are kept as the reasoning, not as a
> to-do list; each now ends with what writing it found. Two of them found
> something. See
> [results/local/2026-09-21-tier1-and-the-broken-venv.md](results/local/2026-09-21-tier1-and-the-broken-venv.md).

### 1.1 Every ablation arm must build and do a forward pass

**The gap:** `tests/test_ablations.py` tests the graph *transforms* — rewiring
preserves degrees, sign shuffle leaves topology identical, the GRU hits its
parameter budget. It never calls `build_arm`, and `GRUCore` and `ShortcutCore`
are never instantiated anywhere in the suite.

**Why it is first:** this exact failure has already happened. The CI workflow's
own header records that "the Phase 4 ablation harness had two arms that could
not do a forward pass, because a signature changed in `model.py` and the
replacement cores in `ablations.py` did not. A test existed for neither." CI
was added in response — but CI runs a suite that still does not build an arm,
so the same signature drift would pass today.

Phase D is a GPU day. A broken arm fails after the money is spent.

```python
@pytest.mark.parametrize("arm", ablations.ARMS)   # real, rewired,
def test_every_arm_builds_and_runs(arm):          # sign_shuffled, gru, shortcut
    model = ablations.build_arm(arm, cfg, small_graph, n_styles=3, seed=0)
    out = model(wav_batch)
    assert out.shape == expected and torch.isfinite(out).all()
```

Add a backward pass and a finite-gradient assertion to the same test. Roughly
thirty lines, and it retires the only failure mode on this page with a known
precedent.

**Found:** the precedent had already recurred. `substeps` went into
`ConnectomeRNN.forward` and `FlyBeats.forward`; `GRUCore` and `ShortcutCore`
did not follow, and both raised `TypeError` on every call through
`FlyBeats.forward` — playback, transcribe, bundle export, and two of the five
Phase D columns. Training stayed green because `run_epoch` calls `model.rnn`
directly. Fixed by giving the speed schedule one definition
(`model.substep_schedule`, `ablations.hold_drive`); the test now also pins the
row count, a per-frame schedule, and rejection of a bad one.

### 1.2 `build.deep_merge` — config inheritance

**The gap:** four lines implementing `_base_` inheritance, untested, and every
config in the repo depends on them.

**Why it matters, concretely:** this session found that `velocity_probe_cpu.yaml`
inherits `device: auto` and `bf16: true` from `v1_8piece.yaml` — so the configs
named `_cpu` stop running on CPU the moment a CUDA wheel is present. That is a
resolution nobody checked. Tests should pin what a merged config actually
resolves to, per config, so the inherited surface is visible:

- a nested override replaces only its own key, leaving siblings intact
- a scalar overriding a dict, and a dict overriding a scalar, behave as
  intended rather than as an accident
- **a resolution table**: for each shipped config, assert the fully merged
  `train.device`, `train.bf16`, `subgraph.max_nodes` and `data.max_files`. A
  test that prints the resolved config is the cheapest defence against the
  class of trap above.

**Found:** the trap was live. All five `_cpu` configs resolved to
`device: auto`. `device: cpu` is now pinned in `v1_8piece_cpu.yaml`, the root
of that family, so the four velocity configs inherit it; `device_of` is
asserted separately from the table, because the `_cpu` name is a promise about
where a run happens and no row edit can satisfy it. The table covers all ten
configs, and a config with no row fails a completeness test. One addition to
the plan: each Phase A′ arm is checked to differ from the baseline by exactly
one knob, since an arm carrying two changes attributes both to one.

### 1.3 `train.evaluate` — the function every headline number comes from

**The gap:** never called in the suite. It returns `onset_f` and
`best_threshold`, which are the numbers quoted in the README, in ROADMAP.md and
in every model card the roadmap promises.

Assert against hand-constructed activations where the answer is known: a perfect
copy scores 1.0, silence scores 0.0, and the swept `best_threshold` agrees with
`onset_f_sweep` computed independently. `test_pipeline.py` already does exactly
this for `onset_f_measure` — the gap is that `evaluate`'s own aggregation and
threshold sweep sit above it, untested.

**Found:** nothing wrong. Seven tests, driven by a stub model returning
prescribed activations so the expected answer is not computed by the code under
test. Beyond the plan: the headline equals the curve's argmax, a fixed
threshold off the grid is added to the sweep, clips average rather than pool,
and a velocity head predicting each class's mean scores r = 0 with an MAE under
0.25 — the trap per-class correlation exists to catch.

### 1.4 Seed determinism across the arms

Phase D's result is a comparison of arms across five seeds, so the harness must
be reproducible in the way the claim requires: same seed and same arm gives the
same weights; different seeds give different ones; and — the one worth pinning
— **a stochastic arm's seed must not change the real arm's graph.**
`STOCHASTIC_ARMS` exists in `ablations.py`; nothing tests that the distinction
holds.

**Found:** it holds. Stated as an iff, since membership is what decides whether
an arm gets repeats or one run quoted with no error bar. The real arm is also
checked to survive the null draws intact — every arm is built from one shared
`SubGraph`, so a rewiring that wrote through it would contaminate the baseline
the nulls are compared against.

---

## Tier 2 — failures that ship a wrong number

### 2.1 The untested metrics

`metrics.beat_alignment_error` and `metrics.groove_similarity` are both named in
[PLAN.md](PLAN.md)'s Metrics section as things reported alongside every headline
result, and neither is tested. `match_onsets` is only exercised indirectly.

Each has an analytically known answer that makes a good test: onsets placed
exactly on the grid give zero alignment error, a constant offset gives that
offset, and groove similarity is 1.0 against itself and known-low against a
shifted pattern.

### 2.2 `feel.aggregate`

`feel.analyse` has one test (a deliberate lag is measured as a lag). `aggregate`
— which pools per-clip reports weighting each class by its onset count — has
none, and it is what turns per-clip measurements into the number someone quotes.
Weighted pooling is easy to get subtly wrong and impossible to notice: assert
that a class with ten onsets outweighs one with a single onset, and that pooling
one report returns that report.

### 2.3 A regression guard on the velocity statistics

The repo has been misled by this analysis once already: the same checkpoint read
`hat_open` as 0.37, then 0.18, then 0.26 before its sampling was seeded, and a
pooled correlation reported ~0.35 over an incoherent per-class picture.
`scripts/probe_velocity.py` is now seeded, and that seeding should be pinned by
a test — same checkpoint and same seed gives the same intervals, twice.

---

## Tier 3 — failures a user hits, which nothing currently catches

**The structural gap: there are no CLI tests at all.** `subprocess` does not
appear anywhere in `tests/`. Every test imports a module; nothing runs a
command. So argparse wiring, the documented invocations in
[SETUP.md](SETUP.md), and the entry points R0 is about to introduce are all
untested — and an `src/` → `src/flybeats/` move is precisely the change that
breaks them silently.

### 3.1 The play path, end to end, as a command

This is [RELEASE_ROADMAP.md](RELEASE_ROADMAP.md) R0.4 expressed as a test:

```
python -m flybeats play --bundle <fixture>.fb --render <short>.wav --out drums.wav
```

Assert it exits 0 and writes a non-silent wav of the expected duration. Mark it
`slow` if it needs to be, but have it, because it is the only path most users
will ever run. `realtime.render_file`, `load_checkpoint` and `drummer_for` are
all currently untested, and they are that path.

### 3.2 Argument-surface smoke tests

For each entry point: `--help` exits 0, an unknown flag exits non-zero, and
mutually exclusive flags are actually exclusive. Trivial to write, and they are
what makes a packaging refactor safe.

### 3.3 A tiny bundle fixture

`test_bundle.py` covers the round trip well but needs a model. Commit a small
`.fb` built from `subgraph_2k` so the play path above has something to run
without training. This also gives the eventual `flybeats doctor --fetch-bundle`
something to verify against offline.

### 3.4 `doctor --json` schema

Once R0.2 lands, the installer wizard consumes that JSON. Pin its shape: the
keys the wizard reads exist, and the three exit-code cases from the 14 September
bring-up hold — card present and torch blind exits 1, card present and visible
exits 0, no card exits 0.

---

## Tier 4 — the surfaces that do not exist yet

Write these as the code lands, not after.

### 4.1 The event schema is a contract, so test it as one

[VISUALIZER_PLAN.md](VISUALIZER_PLAN.md) versions the schema and the frontend
rejects a manifest whose `schema_version` it does not know. That needs three
tests: a known version is accepted, an unknown one produces a *named* error
rather than a subtly wrong render, and every committed fixture validates against
the schema it declares.

### 4.2 `masking.py` parity

R2 introduces one masking primitive with two consumers — the Phase 4 ablation
list and the live lesion toggles. The test that matters is that they agree:
lesioning a population through the live path produces the same model state as
the ablation path. One primitive with two menus is only true if a test says so.

### 4.3 The two rendering-correctness rules

Both are named in [UI_PLAN.md](UI_PLAN.md) as correctness rather than styling,
and both are checkable without a human:

- **Activity polarity.** In light mode, activity drives darkness and saturation;
  peak activity must not converge on the background colour.
- **Silent vs. lesioned vs. peak must be mutually distinguishable in both
  themes.** Lesioned renders as outline, which is a different channel from
  activity entirely.

Implement as a contrast assertion over the committed fixtures. This is the test
that protects the project's best demo.

### 4.4 Fixture replay determinism

Presentation mode must run from a fixture with no GPU, no neuPrint and no model
— a shipping requirement. So: replaying a fixture twice produces identical
frame output, and replay with the biological sliders moved produces *identical*
output too, because in replay they are inert.

---

## Kinds of test the suite does not yet use

- **Property-based** (`hypothesis`) fits three places well: `deep_merge` over
  arbitrary nested dicts, `peak_pick` refractory behaviour over random
  activations, and `resolve_class` over arbitrary kit tiers. Each is a pure
  function with an invariant that is easier to state than to enumerate.
- **Golden-value regression** on the numeric path: pin `evaluate`'s output on a
  fixed checkpoint and fixture to a tolerance. This catches "still runs, now
  computes something else", which is the failure mode every result depends on
  and no current test covers.
- **Coverage measurement.** `pytest-cov` is not installed and is not in
  `requirements-dev.txt`. Add it, but read it as a map of what is unexercised
  rather than as a target — a percentage goal on this codebase would push effort
  toward `format_table` and away from Tier 1.

---

## Tooling and CI

- Add `pytest-cov` and `hypothesis` to `requirements-dev.txt`.
- Add markers alongside `gpu`: `slow` (the CLI end-to-end tests) and `corpus`
  (the 18 that need a download), so the 18 skips become nameable rather than
  merely counted.
- Keep CI at `-m "not gpu"`. Once R0 packages the project, add a matrix over
  Python 3.11/3.12 and — this is the one worth the runner minutes — **Windows**,
  because the queue-script bug found this week (`.venv/bin/python` and `pgrep`,
  neither of which exists there) is exactly what a Windows job catches and a
  Linux job never will.
- The GPU tests stay the local machine's job. That asymmetry is deliberate and
  documented in [RUNBOOK.md](RUNBOOK.md); the counts to expect are recorded
  there too.

---

## Deliberately not tested

- **Model quality.** No test asserts an onset F floor. A threshold here would
  either be so low it never fires or would fail on legitimate research changes,
  and this project's whole discipline is that a single unseeded number is not a
  result. Quality belongs in the model card, not in the suite.
- **Exact numerical agreement across devices.** Different kernels reduce in
  different orders; `test_cuda.py` already asserts agreement past float32 noise
  rather than bitwise, which is correct.
- **The full-graph render tier and the spiking model.** Neither exists in
  runnable form; tests would be fiction.

---

## Order of work

1. Tier 1 — four tests, all cheap, and 1.1 has a precedent (before any GPU time)
2. Tier 3.2 and 3.3 — the cheap safety net for R0's packaging refactor
3. Tier 2 — as the model card takes shape and its numbers need defending
4. Tier 3.1 — with R0.4, since they are the same verification
5. Tier 4 — alongside each surface, never after
