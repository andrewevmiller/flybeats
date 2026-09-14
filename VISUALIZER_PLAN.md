# FlyDrums Visualizer — Execution Plan

Companion to `PLAN.md`. Covers two rendering surfaces driven by one event
stream:

- **Brain view** — real neuPrint anatomy, lit by per-cell-type firing rate
- **Performance view** — schematic fly avatar striking a drum kit, both
  reacting to decoder output

Everything user-facing is config-driven, following the same rule `PLAN.md`
applies to the model: expose it in `configs/*.yaml`, not as scattered
constants.

---

## Design rules (apply to every phase)

1. **One event stream, two consumers.** The brain view and the performance
   view subscribe to the same socket. Never build a second real-time pipeline.
2. **Nothing in the render path touches the audio path.** Aggregation happens
   inline after the forward pass (microseconds); all socket I/O happens on a
   dedicated thread reading a bounded queue. A stalled or absent client must
   never cause an audio underrun.
3. **Cell-type names come from `data/verified_types.json`.** Same rule as the
   rest of the project — no hardcoded type strings anywhere in `viz/`.
4. **The frontend never imports Python concepts.** It consumes a versioned
   JSON schema and a recorded fixture. This is what makes it independently
   developable and independently skinnable.
5. **Real anatomy over abstract layout**, everywhere it is affordable.
6. **Visuals are instrumentation, not decoration.** Every visual channel maps
   to a named, confirmed population or a real decoder output. If a thing on
   screen isn't driven by model state, it shouldn't move.

---

## Phase V0 — Skeleton fetch + cache (gate)

Depends on: `verified_types.json` (Phase 0), extracted subgraph (Phase 1).

`neuprint-python`'s `fetch_skeleton` is per-body-ID; there is no bulk skeleton
download equivalent to the connectivity feather files. For a 10k–30k node
subgraph that is 10k–30k HTTP calls, so this is a one-time, resumable,
rate-limited batch job — never something run at render time.

**`scripts/fetch_skeletons.py`**
- Input: body IDs from the live subgraph
- Thread pool with exponential backoff; respects neuPrint per-account limits
- Resumable: skip body IDs already in cache, safe to Ctrl-C and rerun
- Writes `data/skeletons_cache/` (one array per body ID, or a single parquet
  keyed by body ID)
- Also fetches synapse XYZ for the subgraph's edges — used only by the
  offline explorer

**Two derived artifacts from one fetch:**

| Artifact | Fidelity | Consumer |
|---|---|---|
| `skeletons_full.parquet` | full branching geometry + synapse points | Phase V1 explorer |
| `soma_layout.json` | one representative XYZ per neuron | Phase V3 brain HUD |

**Gate:** report coverage before proceeding. EM reconstructions include
truncated fragments, and some bodies will have no skeleton. Emit an explicit
`missing_skeletons.json` rather than silently dropping neurons — a neuron
absent from the layout is a neuron whose firing is invisible, which is a
correctness problem for a tool whose purpose is showing you what fired.

---

## Phase V1 — Static 3D subgraph explorer (offline)

Highest payoff per hour of work, and the source of the project's hero image.
No real-time constraints, so use the full-fidelity artifact.

**`viz/anatomy.py`** — shared load/transform layer (also used by V3)
- Reads the skeleton cache + `verified_types.json`
- Provides: type→neuron index, neuron→XYZ, edge list with NT sign and synapse
  counts, and the reduced soma layout

**`viz/subgraph_explorer.py` / `scripts/render_subgraph_3d.py`**
- Plotly → standalone interactive HTML, no server needed
- Default coloring: cell class (JO afferent / AMMC-WED / pC1-pC2 / pIP10 /
  wing MN / other)
- Alternate colorings selectable by flag: per-edge learned gain, NT sign,
  hemisphere
- Synapse contact points drawn as real XYZ rather than centroid-to-centroid
  lines

**Checkpoint-diff mode:** render the same view colored by learned gain at
several training checkpoints. The drift away from the `log(synapse_count)`
initialization is the visual evidence that gradient descent is doing something
to the connectome — a picture instead of a table.

**Acceptance:** the render visibly reproduces the expected pathway
(JO → AMMC/WED → pIP10 → wing MNs). If it doesn't, the k-hop extraction is
wrong and this caught it — which is the main reason this phase comes first.

---

## Phase V2 — Event transport + recording

The plumbing both live views depend on. Build and test it headless.

**Producer side, inside `realtime.py`:**
- After each forward pass, aggregate neuron state → per-cell-type groups via a
  precomputed sparse assignment matrix (one `index_add`, negligible against
  the ~10 ms inference budget)
- At the existing MIDI dispatch point, emit a `trigger` event with the same
  timestamp as the note-on — audio, MIDI, and visuals all originate from one
  clock reading
- Both go onto one bounded, non-blocking queue. **Queue full → drop oldest and
  increment a counter.** Never block.

**`viz/live_server.py`:**
- Dedicated thread; only job is draining the queue to a websocket
- Mostly I/O-wait, negligible CPU, no meaningful GIL contention
- Serves the manifest on connect, then streams frames

**Recording:** a `--record session.jsonl` flag dumps manifest + event stream to
disk, and the server can replay it. This is the single highest-leverage piece
of the whole visualizer plan: the entire frontend can then be built, restyled,
and demoed without a GPU, without neuPrint, and without a trained model.
Build the recorder before the frontend.

**Acceptance:** a 60-second recorded session replays frame-accurately; dropped
frame count is zero under normal load; killing the websocket client mid-run
produces no audio glitch.

---

## Phase V3 — Brain HUD (live)

**Rendering:** soma point cloud from `soma_layout.json`, recolored per frame by
group activity. ~100–150 groups at 50 Hz is comfortably within plain SVG/DOM
update budget — no WebGL needed for this view alone.

**Controls panel.** The four sliders from `PLAN.md` (drive, tightness, pocket,
master rhythm gate) render adjacent to a highlight of the population each one
actually drives (pC1/P1, global GABA gain, membrane τ scale, pIP10). The
mapping is the point: it should read as biological control, not four unlabeled
knobs.

**Lesion toggles.** Per-cell-type mute switches wired to the shared
`masking.py` primitive. Muting a type changes its rendering on screen while
the beat audibly degrades — the best demo this project produces, and the
reason lesion mode and the brain view should ship in the same release.

**Lesion is not an activity value — it gets its own channel.** Rendering a
lesioned group by darkening it collides with the activity encoding: a silent
group and a lesioned group both render dim. On a light stage the same
collision runs in reverse, with both rendering pale. The states are not the
same and the distinction matters most at exactly the moment the demo depends
on it.

Render lesioned groups as **hollow outlines** — filled means live, outline
means masked. Outline carries no luminance meaning, so it survives both
themes and any `activity_channel` setting unchanged. A lesioned population
then reads as structurally absent rather than temporarily quiet, which is
what it actually is. This is a correctness constraint under design rule 6,
not a style choice, so it is fixed rather than exposed in the config surface.

**Acceptance:** lesioning pIP10 produces near-silence (near-binary gate, per
`PLAN.md`) and a visibly outlined region, within one buffer tick of the
toggle. Separately: silent, lesioned, and peak-activity groups are mutually
distinguishable in both light and dark themes. Verify against a recorded
fixture, in both themes, before the view ships.

---

## Phase V4 — Avatar + kit (live)

**`viz/limb_map.py`** — the reverse lookup the kit config doesn't yet have:
`{kit_piece: (mn_population, limb_segment, strike_arc)}`. Scoped to the shipping
kit tier. v1 is 8-piece with bilateral split, so only two wings need
articulation; leg segments land if and when leg mode ships.

**Rendering technology:** canvas (PixiJS) rather than SVG. Simultaneous sprite
transforms across multiple limbs and pads at up to 50 Hz will cause DOM reflow
jank; sprite batching will not.

**Animation:** critically-damped spring tweens — rotation for limb segments,
scale/opacity for pads. No physics engine.

**Lesion treatment matches V3.** A limb segment whose driving MN population is
masked renders as outline rather than fading out or going still — opacity
reads as "not currently playing" and would be indistinguishable from a limb
that simply has no note this bar. Pads whose population is masked take the
same treatment. One visual vocabulary across both surfaces: filled is live,
outline is masked.

**Strike timing — the one non-obvious detail.** A naive "begin the swing at
note-on" lands the contact frame tens of milliseconds *after* the sound, which
reads as the fly chasing its own drums. Since the sound cannot be delayed,
schedule the **contact frame at the trigger timestamp**: compress or skip the
windup, and put the follow-through after. The visual event path has no
equivalent of the audio path's latency budget, so this is free.

**Velocity mapping:** decoder magnitude drives arc amplitude and pad bounce
together, through a configurable response curve.

**Style assumption (override if you disagree):** schematic, not mascot. Only
segments backed by a real motor population are articulated; everything else is
static line work. This keeps the avatar consistent with the "real anatomy, real
control signal" identity of the rest of the project, and avoids committing to a
rigged-character art pipeline. A cartoon mascot is a substantially larger asset
effort and worth deciding before anyone draws sprites.

**Acceptance:** replay a recorded session against the MIDI render of the same
session; strikes and hits are perceptually simultaneous.

---

## Phase V5 — Customization layer

Customization is threaded through V1–V4, but the preset/theme system is worth
one consolidation pass so it isn't reinvented per view.

**Four layers, separately configurable:**

1. **Data** — which groups exist, and at what granularity
2. **Layout** — where things sit in space
3. **Mapping** — how model values become visual values
4. **Theme** — palette, typography, background

**Live vs. restart.** Anything in Mapping or Theme changes live, from the
panel, mid-performance. Data and Layout changes require a manifest rebuild and
a reconnect. Make this distinction visible in the UI — a knob that silently
does nothing until restart is worse than no knob.

### Activity polarity is derived from theme, not configured

`activity_channel: brightness` encodes firing rate as luminance, which
inverts between themes. On dark, brighter means more active. On light, a
bright group is a white dot on a white field — peak activity becomes
invisible and the channel silently reverses meaning. Under design rule 6
that's instrumentation that lies.

So polarity is derived from the resolved theme rather than exposed as a
separate key: on light, activity drives darkness **and** saturation together,
since pure luminance ramps have less usable range on white than on black.
`both` is the safer default for light, because size carries the signal where
the luminance range is compressed.

The stage follows the theme by default rather than pinning dark in the DAW
convention — a permanently dark canvas would mean light mode never touches
anything but the chrome. `stage_follows_theme: false` restores the dark
canvas for anyone who wants it.

Open: whether `decay_ms` wants a higher default on light. Rapid dark-to-pale
flicker across a bright field may read harsher at 50 Hz than the inverse does
on dark. Check against a recorded fixture rather than guessing.

### Config surface (`configs/viz_*.yaml`)

| Layer | Key | Options | Live? |
|---|---|---|---|
| Data | `grouping` | `type` / `class` / `type_bilateral` | no |
| Data | `visible_types` | list, or `all` minus exclusions | yes |
| Layout | `view` | `anatomical` / `dorsal` / `sagittal` / `exploded_by_class` | no |
| Layout | `camera_presets` | named orientations, cycleable at runtime | yes |
| Mapping | `activity_channel` | `brightness` / `size` / `both` | yes |
| Mapping | `response_curve` | `linear` / `log` / `gamma(n)` | yes |
| Mapping | `decay_ms` | visual persistence after a firing peak | yes |
| Mapping | `velocity_curve` | strike amplitude vs. decoder magnitude | yes |
| Kit | `layout_preset` | `gm_8piece` / `articulated` / `bilateral_split` | no |
| Kit | `pad_skin`, `reaction_tween` | skin pack, spring stiffness/damping | yes |
| Avatar | `style_pack` | `schematic` (default) / user-supplied | no |
| Avatar | `articulated_limbs` | derived from kit tier, overridable | no |
| Theme | `palette`, `background`, `accent` | named packs or inline hex | yes |
| Theme | `mode` | `system` (default) / `light` / `dark` | yes |
| Theme | `stage_follows_theme` | default true; false pins the stage dark | yes |

**Skin packs** are a directory of sprites plus a manifest. Because the frontend
consumes recorded fixtures, a new skin can be authored and reviewed against a
saved session with nothing else running.

Every skin pack ships both theme variants, or declares in its manifest which
it supports and fails loudly at load rather than rendering illegibly on the
background it wasn't drawn for. The resolved theme is recorded in the session
manifest, so a fixture replays looking the way it did when it was captured.

---

## Event schema (the contract)

Versioned. Frontend rejects a manifest whose `schema_version` it doesn't know,
rather than rendering something subtly wrong.

**Handshake, once per connection** — keeps per-tick frames tiny:

```jsonc
{
  "type": "manifest",
  "schema_version": 1,
  "groups": ["JO-B_L", "JO-B_R", "pIP10_L", ...],   // fixed order
  "layout": { "JO-B_L": [x, y, z], ... },
  "kit": { "pieces": [...], "limb_map": {...} },
  "config": { /* resolved viz config */ }
}
```

**Per tick (~50 Hz, ~500 bytes):**

```jsonc
{ "type": "activity", "t": 12.482, "v": [0.11, 0.03, ...] }  // matches manifest order
```

**On drum trigger (sparse):**

```jsonc
{ "type": "trigger", "t": 12.485, "class": "snare",
  "velocity": 0.82, "population": "wing_MN_R" }
```

Per-neuron streaming is deliberately excluded: ~120 KB/tick vs. ~500 bytes, and
it would force the brain view into WebGL for no interpretive gain. If
per-neuron detail is ever wanted, it belongs in the offline explorer.

---

## Repo additions

```
flydrums/
  viz/
    anatomy.py            # skeleton cache load, shared layout (V1 + V3)
    subgraph_explorer.py  # V1 static 3D render -> standalone html
    ablation_plot.py      # Phase 4 bar charts
    limb_map.py           # kit piece -> MN population -> limb segment
    live_server.py        # V2 websocket broadcaster + recorder/replayer
    schema.py             # event schema + version constant
    static/
      index.html
      brain.js            # V3 point cloud + controls
      avatar.js           # V4 limb renderer + spring tween
      kit.js              # V4 pad renderer
      skins/              # skin packs
  scripts/
    fetch_skeletons.py    # V0 gate
    render_subgraph_3d.py
  configs/
    viz_default.yaml
  src/
    masking.py            # shared: Phase 4 shortcut floor + V3 lesion toggles
```

---

## Performance budget

Per-tick, live tier:

| Item | Budget |
|---|---|
| Group aggregation (inline) | < 0.5 ms |
| Queue push | non-blocking, < 0.1 ms |
| Socket write (separate thread) | off the critical path |
| Frontend frame | 20 ms @ 50 Hz |

**Degrade in this order** when over budget: drop visual frame rate to 25 Hz →
coarsen `grouping` to `class` → disable avatar follow-through tweens → disable
the brain view, keep the performance view. Audio and MIDI never degrade; they
are not downstream of any of this.

---

## Build order

V0 → V1 → V2 (**including the recorder**) → V3 → V4 → V5.

V1 is worth doing early even though it's offline, because it validates the
subgraph extraction. V2's recorder is the gate that decouples all frontend work
from the model, so nothing in V3–V5 should start before it exists.

---

## Risks

- **Skeleton coverage gaps** — truncated EM fragments have no skeleton;
  mitigated by the explicit coverage report in V0 rather than silent dropping.
- **Fetch time** — 10k–30k rate-limited calls may take hours. Resumable by
  design; run it overnight against the first stable subgraph.
- **Frontend scope creep** — the skin/theme system can absorb unbounded effort.
  Ship `schematic` + one theme; skin packs are an extension point, not a v1
  deliverable.
- **Sync drift over long sessions** — if visual and audio clocks diverge,
  resync on `trigger` timestamps rather than free-running the animation clock.

---

## Open questions

1. Does `grouping: type_bilateral` stay under ~150 groups for the shipping
   subgraph, or does the confirmed type count push it high enough that the
   per-tick payload and the SVG update path both need rethinking?
2. Should the checkpoint-diff render (V1) read gains from W&B artifacts or from
   local checkpoint files — i.e. is training-run provenance something the
   visualizer should track, or does it just take a path?
3. Schematic vs. mascot avatar — worth settling before any sprite work starts.
4. Does the lesion toggle set need to be identical to the Phase 4 ablation
   type list, or is the live set a curated subset chosen for audible effect?
