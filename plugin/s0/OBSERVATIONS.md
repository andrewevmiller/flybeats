# S0 observations (fill in; one copy per operating system)

Operating system and version: ______________________  FL Studio version: __________  Sample rate: ________  Buffer: ________

Build used: commit ____________  built on ____________ (date)  by ____________

| # | Observation | Step in README | Result | Notes (what you saw and heard) |
|---|---|---|---|---|
| 1a | An effect on a mixer insert hears another track routed into the insert (main input meter moves; clicks follow it) | Obs. 1, steps 1-2 | yes / no | |
| 1b | The same effect hears a track assigned to its sidechain input in the wrapper's Processing tab (sidechain meter moves; with Listen = Sidechain the clicks follow it) | Obs. 1, step 3 | yes / no | |
| 2 | An instrument (generator) channel hears no mixer audio on its input bus (input bus disabled by the host, or enabled but always silent) | Obs. 2 | hears none / hears audio | input bus shown as: enabled / disabled |
| 3a | FL routes MIDI out from the **effect** (a synth on the same port sounds on each onset) | Obs. 3 | yes / no | |
| 3b | FL routes MIDI out from the **instrument** (a synth on the same port sounds on each beat) | Obs. 3 | yes / no | |
| 4 | FL's delay compensation follows the reported latency (with PDC on, the 200 ms delayed pass-through lines up with the grid; clicks land 200 ms late) | Obs. 4 | yes / no / partly | latency shown by the wrapper: ______ samples; PDC setting: ______ |
| 5 | The macOS build loads natively on Apple Silicon (not bridged) | Obs. 5 | yes / no / n.a. on Windows | |

Anything unexpected (crashes, a bus missing from the wrapper, the sidechain not offered, MIDI port options absent):

_____________________________________________________________________________

_____________________________________________________________________________

Conclusions the plan needs from these:
- Main build as an effect with main + sidechain inputs is viable in FL Studio: yes / no (from 1a, 1b)
- MIDI out from the effect is usable in FL Studio: yes / no (from 3a); if no, the built-in sampler is the only output path in FL and the instrument build carries MIDI out (3b)
- Live mode can rely on the host's delay compensation: yes / no (from 4)
