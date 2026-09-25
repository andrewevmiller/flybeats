# Plain-language run reports

Non-technical reports on the Phase A′ loudness (velocity) experiments of
23–24 September 2026, written as each run finished. The logs and probe
output they quote are under `runs/`, which is not in git.

**Terms.** A drum is *right* when the model's loudness guess rises with how
hard the drummer hit, with a 95% range (resampled by song) entirely above
zero; *backwards* when the range is entirely below zero; *can't tell* when it
spans zero. *Timing* is the best validation onset F. The *refit* recomputes
the loudness readout exactly after training (`src/refit.py`), leaving the rest
of the model alone.

## Current, in reading order

1. `00-overview.txt`: the four overnight arms, rescored with the fixed probe.
2. `why-kick-and-low-tom-read-backwards.txt`: what was wrong with the probe.
3. `why-the-loudness-guess-ignores-the-signal.txt`: why the trained loudness
   readout barely leaves its random start.
4. `refit-loudness-readouts.txt`: the exact refit, all four arms.
5. `arm-A2-seed2.txt`, `arm-baseline-seed2.txt`, `00-seed2-verdict.txt`: second
   seeds, and whether A2 beats the baseline.
6. `arm-standardised-readouts.txt`: standardised motor rates, first seed.

## Superseded: kept as written, not to be quoted

- `arm-A1-loudness-5x.txt`, `arm-A2-linear-output.txt`,
  `arm-A3-graded-at-hit.txt`, `arm-baseline.txt`,
  `00-overview-old-probe-2026-09-24.txt`: their drum verdicts come from the old
  probe, which scored a one-drummer slice of the test songs and overstated its
  certainty. "Kick and low tom backwards" was an artefact of that. Report 1
  above has the corrected verdicts. Their run details and timing scores
  still stand.
- `00-overview-stopped-2026-09-24.txt`: the queue's state after the 23 Sep
  Windows Update restart. The two missing arms were rerun.
- Any verdict on the *trained* readout, including "crash right in every arm"
  in report 1: it depends on the random start, so runs are compared on the
  refit verdicts instead (report 4 onwards).
