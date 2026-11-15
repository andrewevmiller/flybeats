# Decision: the memorisation gate becomes a diagnostic

Date: 11 Nov 2026. Decided by Andrew, following the Fable recommendation (option c).

## What was decided
- The memorisation gate (8 fixed clips, hit F1 ≥ 0.9 on kick, snare and closed hi-hat at step 3,000) is no longer a pass/fail check for the fly network.
- It stays as a **reported diagnostic**: every memorisation number is published, with the locked loss and locked peak rule as the yardstick.
- The main test becomes training on unseen songs, real wiring against controls, under the locked settings.
- The square-root drum-weight loss is **not** adopted. Its trial results are reported as a trial only.

## Why (the record, in order)
1. The gate was locked on 30 Sep 2026, before any memorisation result.
2. Nothing passed it under the locked loss: the fly network (real and control 1) and the unrestricted GRU at both sizes (`reports/gate_calibration/*/result.json`, `reports/memorise/*/memorise.json`).
3. Calibration rule written before it ran: a "mis-set" outcome means neither model reaches the gate (`scripts/gate_calibration.py`, verdict lines).
4. With square-root weights, only the 157,200-parameter GRU passed (`reports/gate_calibration/large-loss-sqrt/`). The matched-size GRU (41,744 parameters) did not, and the fly network failed badly under the same weights (`reports/memorise/real-loss-sqrt-stopped-step1660/log.jsonl`).
5. Re-setting the gate after these results would fit the threshold to the outcome, so it was not done.

## Disclosure (for any write-up)
"The memorisation gate was locked on 30 September 2026, before results. No network passed it under the locked loss, including unrestricted networks. Its role was changed to a diagnostic on 11 November 2026, after those results. Every memorisation number is reported. Control 1 scored ahead of the real wiring on the gate. The real and control networks also differ in free parameters (real 41,186; controls about 83,600)."

## Still to do (Andrew's steps, not made here)
- `config/locked.yaml`: change the memorisation section to say diagnostic, not pass/fail. New tag.
- `config/CHANGELOG.md`: entry with the disclosure above, dated.
- Training on unseen songs: a separate command, per Andrew.

## Not affected
- The connectome (wiring, signs, synapse counts, cell-type tying) is unchanged.
- The loss, peak rule and threshold are unchanged.
- Dial counts are unchanged.
