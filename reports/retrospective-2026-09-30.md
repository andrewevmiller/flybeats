# Retrospective: 30 Sep to 1 Oct 2026, and where to go from here

Written 1 Oct 2026. Sources: `reports/session-2026-09-30.md`, `log-2026-09-30.md`, `session-2026-10-01.md`, the
commits from 1374d67 to f14f85a.

**The goals (Andrew, 1 Oct):** first, whether the real fly wiring plays drums along with the band better than random
wirings, by the preregistered test; ultimately, how to make the fly wiring play drums.

## 1. Where we started, and where we are

**Start of 30 Sep:** prereg-v5 locked, the memorisation check about to run for the real network and control 1, then
training. No network had been trained beyond memorisation; the scorecard and win rule were names without measures.

**Now:**
- Memorisation fails for every network tried. The real wiring trails the random control by a wide margin.
- The cause is better understood (below), and one readout change helps the real wiring partly.
- The scorecard and the win rule are fully defined, locked before any scoring, and implemented with tests.
- The main test's power is estimated, and it is limited.
- The recorded training step takes about 3 s (the ordinary step 5.3 s, down from 9.6), with the readout history at
  the same cost.
- **Still no network has been trained on the training set.** `runs/` is empty.

## 2. What we learned

### About the real wiring against random wiring (memorisation, 8 fixed clips)

| Run | Kick | Snare | Closed hi-hat | All drums F1 | False hits |
|---|---|---|---|---|---|
| Real, locked readout (prereg-v5) | 0.623 | 0.352 | 0.520 | 0.521 | 676 |
| Real, history readout (trial) | 0.724 | 0.338 | 0.601 | 0.575 | 518 |
| Real, wider answer key (trial, rejected) | 0.214 | 0.000 | 0.000 | 0.095 | 92 |
| Control 1, locked readout | 0.855 | 0.752 | 0.880 | 0.812 | 213 |

- **The random control memorises far better** than the real wiring, at every check, with the same kinds of error.
  This is on training clips only, with one control; it says nothing yet about the main question, which is about
  unseen songs.
- **Why, as far as the evidence goes** (`trace_information.py`):
  - The information reaches the real network's motor neurons, but spread over the last ~200 ms instead of lined up
    with the present moment. The history readout was built on this, and it helped kick and closed hi-hat.
  - The real wiring's routes from the ears into the motor neurons for **snare and closed hi-hat (the wing power
    muscles)** carry about 60% of every control's capacity. Snare is the drum no readout change has moved.
  - Not the cause: silencing of modulatory/unknown neurons, the starting gain, the collapse into a few signals
    (control 1 collapses more and still does better), late information.
- **The readout map is a design choice, not biology,** and it matters: we put the two most timing-critical drums on
  the muscles whose wiring is narrowest in the real fly.

### About the task (playing along to unseen songs)
- **The band predicts the drums only weakly across songs.** A straight-line probe barely beats chance. An
  unrestricted model with the same ear (`feasibility.py`) scores 0.204 on kick, snare and closed hi-hat when
  hearing its own song, against 0.117 when hearing another song's band.
- **On the locked scorecard,** the gap from hearing the right song is +5.5 points on beat alignment (the main rule)
  and +8 to +9 on tempo following, listening and groove.
- **Under-playing is the biggest drag on beat alignment:** that model only ever fires kick and closed hi-hat, and
  the density balance punishes playing too few hits.

### About the test itself
- **Power is limited, and set by the test songs, not by training.** On beat alignment, the noise from the choice of
  151 test songs is about 1.1 points whatever the material per song. An 80% chance of a win needs the real network
  to beat the controls by roughly 2/3 to all of what hearing the band is worth to an unrestricted model.
- **More material per song or 10 controls doesn't fix it** (`power_options.py`).
- **Two reasons it is probably pessimistic:** the ceiling model was weak, and the noise came from two very different
  predictors rather than similar networks on the same songs.
- **False wins stay rare:** about 6% (seed spread as standard deviation, locked), which is the rule's purpose.

## 3. What we built and locked

| What | Where |
|---|---|
| Controls within neuron classes, reciprocal pairs kept | prereg-v4 |
| Memorisation on training's schedule, diagnostics, connection gradient once per clip | prereg-v5 |
| "Seed spread" defined: sample sd of the 5 controls; real > best control + spread | prereg-v5.1, `scoring/win_rule.py` |
| Scorecard: five facets, floors, rescaling, supporting rule "all 3" | prereg-v5.2, `scoring/scorecard.py` |
| Readout after the frame loop; recorded step 2.96 s; history readout (off by default) at 2.99 s | 750b6a1 |
| Proposals, not locked: prereg-v6 readout refit, history readout | `reports/*-proposal.md` |

## 4. What went well
- **Locking before looking.** The seed spread and every facet were defined and checked on reference parts before
  any network was scored, and revised only on reference checks (groove changed from cosine; style fit made report
  only).
- **Checks caught real bugs before they cost anything.** A memory leak and a precision failure in the gradient
  shortcut were caught. My own groove binning and drum-merging bugs were caught by the reference checks, and two
  test-fixture errors by the unit tests.
- **Changes after results are written down with their triggers** (the tables in the prereg-v6 and history-readout
  proposals), and stopping rule B is set before the next memorisation run.
- **Speed:** the readout move made the ordinary step 44% faster and made the history readout trainable.

## 5. What went wrong (mostly mine)
- **I reasoned for most of an afternoon as if the drums were in the input.** They aren't. The swing claim, the 4 kHz
  ear explanation and an isolated-drum ear test had to be retracted, and one wrong answer went to a collaborator.
- **Run-time estimates were badly off twice:** the trace took 25 minutes against my 60-90; the history trial was
  heading for 20 hours against my 3.5-4.5. The second came from designing the readout inside the frame loop without
  thinking about per-frame cost.
- **The prereg-v4 prototype moved synapse counts between senders,** and its eigenvalue claim had to be corrected.
- **Scope drift.** I kept proposing diagnostics and side projects, up to a stronger ceiling model. Each was
  reasonable alone; together they drifted from the question. Several design changes came from real-against-control
  differences, each adding a forking path to discount.
- **Two GPU jobs at once stalled or slowed both**, three times (the paired runs, the CUDA-graph check against the
  wide-target trial, nearly the history trial).
- **Smaller:** the CPU limits silently not applying until set through UXTU with auto-reapply; shell quoting mangling
  backslashes in my edit scripts (cost time, no damage).

## 6. Integrity ledger: design changes made after seeing results

| Change | Trigger | Real-against-control? |
|---|---|---|
| prereg-v4 controls within classes | Old control 1 memorised better; ear-to-motor shortcuts found | Yes (memorisation) |
| prereg-v5 training fixes | Real network's memorisation failure | No |
| prereg-v6 refit (proposed) | Readout drift trigger set in advance, fired | No (pre-set rule) |
| History readout (proposed) | Trace: real network's information spread over time, unlike control 1 | Yes (memorisation) |
| Scorecard revisions (groove, style fit) | Reference parts only, before any scoring | No |

None touches the main metric, which has never been computed for a real-against-control pair. The supporting rule
and every facet were fixed before any network was scored with them.

## 7. Where we stand on the two goals
- **Goal 1, the test:** the pipeline is ready to train, but the test is under-powered, and the real wiring starts
  from behind (on memorisation). A plausible outcome is that the real wiring does not win. By the project's own rule
  ("if it does not, that is the result"), that is a valid and reportable answer, not a failure.
- **Goal 2, making the fly play drums:** we now have concrete leads: the readout map (the narrow power-muscle routes),
  reading the motor neurons' recent history, and playing the right number of hits. None of them belongs in the
  preregistered test unless set in advance on grounds other than its results.

## 8. Plan: two tracks, kept apart

### Track A: finish the preregistered test (the critical path)
| Step | What | GPU time (measured basis: 2.96-2.99 s a step) |
|---|---|---|
| A1 | Decide prereg-v6: the refit (its pre-set trigger fired) and the history readout (disclosed); lock | none |
| A2 | Memorisation pair under prereg-v6, real then control 1; stopping rule B from here | about 2.5 h each, 5 h |
| A3 | BabySlakh training, real and all 5 controls, 1,000 steps each (locked): a full rehearsal of training, validation, the silence test and scoring | about 1 h each, 6-7 h |
| A4 | Set the full-set step count from A3's measured cost, and lock it. The full set has 80 h of training audio (about 2,250 steps a pass at 8 clips) | e.g. 10,000 steps: about 8-9 h a network, about 2-2.5 days for all 6 |
| A5 | Score the 5 controls on validation songs (control spread); then the test split once; report with the power limit stated | minutes, on the CPU |

### Track B: make the fly wiring play drums (exploratory; never feeds back into Track A)
After Track A is locked through A4, on training and validation songs only, clearly labelled as exploration:
- **B1 Readout map:** put timing-critical drums on motor neurons with wide routes from the ears (steering, legs), or
  let training choose the assignment; the trace says the power-muscle routes are narrow in the real fly.
- **B2 Richer readout history:** per-neuron profiles (the fallback in the proposal).
- **B3 Density:** getting the real wiring to play about the right number of hits.
- **B4 Beyond the slice:** more hops from the ears, other motor groups, the slow-modulation arm (dopamine).

## 9. Decisions needed
1. **A1:** adopt both (recommended; disclose that the memorisation result can't separate them, which matters little
   because under stopping rule B it gates nothing), the refit only, or test the refit alone first (+2.5 h).
2. **A4:** a GPU budget for full training (the step count follows from it).
3. **Track B's timing:** after the test is locked (recommended), or interleaved on spare GPU time with strict
   separation.
