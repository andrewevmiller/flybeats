# Next-step options after Fix 04 (saved for later)

Status: options, not decisions. Written 11 Nov 2026 after the Fix 04 review. Nothing locked has changed.

## Where things stand

- **Gate (memorisation, 8 clips, F1 ≥ 0.9 on kick, snare, closed hi-hat at step 3,000):** no model has passed it.
- **Fix 04 (square-root drum weights):** failed. On the real network, recall collapsed as precision rose, and the model stopped firing on snare and closed hi-hat. Its run was stopped early and not finished.
- **Min-gap rule (spec 5):** rescored on saved weights only. It raises F1 a little but does not close the real-vs-control gap. Not adopted.
- **Connectome:** no fix or option so far changes the wiring, signs, synapse counts or cell-type tying. The readout map and the gate decision are the two items that bear on the connectome's use or test.

## Options, ranked

### 1. Re-run gate calibration with extra read-outs (in progress)
- **Tests:** whether any unrestricted GRU passes the gate under the locked rule and under the 10-frame minimum gap.
- **Changes:** records min-gap F1 beside locked F1, saves weights, and can add a square-root arm.
- **Cost:** GRU models, about 6 min per size on the GPU; CPU timing to be measured.
- **Decision rule:** if the large GRU passes with the minimum gap, the peak rule blocks the gate. If nothing passes under either rule, the gate is unattainable as set.
- **Integrity:** rules written into the script before the run. Changes nothing locked. Adopting a gap still needs a tag and disclosure.
- **Connectome impact:** none.

### 2. Decide the gate's role, then train on unseen songs
- **Tests:** generalisation to songs not seen in training. Never measured.
- **Cost:** about 1 h per network (1,000 steps × 6 networks), per the retrospective's step A3.
- **Decision needed:** keep, re-set, or demote the gate. Changing it is a locked-config change: new tag, disclosed as made after results.
- **Connectome impact:** none to the wiring. Dropping the gate removes the memorisation test of real against control.

### 3. Housekeeping before any long run (code only)
- Fix `memorise.py --pair --loss-weight sqrt` so the real run no longer overwrites `reports/memorise/real/` (its `final.pt` is the only copy and is git-ignored).
- Save weights at every check.
- Re-time the CUDA graph check on an unthrottled GPU (needs a GPU go). The current result looks throttled.
- **Connectome impact:** none.

### 4. Spec 3, separate learning rates (real, then control 1)
- **Cost:** about 5.4 h of GPU.
- **Decision rule:** real gains at least +0.05 mean F1 and control 1 does not gain as much.
- **Odds:** low. The loss rise it targets also appears in control 1.
- **Connectome impact:** none.

### 5. Spec 6, refit alone (real, then control 1)
- **Cost:** about 5.4 h of GPU.
- **Odds:** lowest. Control 1 drifts more and memorises well.
- **Connectome impact:** none.

## Not recommended now
- Finishing the square-root run. Kick was still rising, but snare and closed hi-hat sat at zero.

## Open questions for later
1. Keep the gate, re-set it with a new tag, or treat it as a diagnostic only?
2. Should the minimum gap be adopted? It changes the scorecard's input and needs a tag.
3. Is the overnight stop confirmed as a Windows restart, and is a restart guard in place before the next long run?
4. Where did the earlier 0.08 s per step figure come from? No source found; the GRU measures 0.10–0.13 s per step on the GPU.
5. Should the readout map (snare and closed hi-hat on narrow power-muscle routes) be considered now, or stay in the exploratory track? It touches how the connectome is used, not its wiring.
6. Is real-vs-control memorisation still worth GPU time, given controls have about twice the dials, or go straight to training on unseen songs?
