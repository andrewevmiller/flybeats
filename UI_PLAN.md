# Flybeats UI Plan

Companion to `PLAN.md` and `VISUALIZER_PLAN.md`. Covers the desktop
application shell: how the two rendering surfaces, the biological controls,
and the ablation tooling are arranged for a user.

Supersedes the earlier three-mode draft (presentation / performance /
research), which split by audience. That was the wrong axis — see below.

---

## Design rules

1. **Split by compute state, not by audience.** You are frequently performer
   and researcher in the same minute: lesioning pIP10 mid-take is both the
   best demo the project produces and a real observation. What actually
   divides screens is whether inference is running live.
2. **Nothing reachable inside a live mode may be non-real-time.** This is the
   membership test for every feature. If it can't hold the latency budget in
   `PLAN.md` Phase 5, it lives in the Bench.
3. **Controls sit next to what they drive.** Per `VISUALIZER_PLAN.md` V3 — a
   slider anchored to the population it modulates reads as biological
   control; the same slider in an anonymous rail reads as a synth knob.
4. **Live vs. restart is visible, always.** A control that silently does
   nothing until reconnect is worse than no control.
5. **Type names come from `data/verified_types.json`.** Same rule as the rest
   of the project. Kit tiers the confirmed motor-neuron count can't support
   are shown disabled, never offered and then rejected.

---

## Two modes, one Bench

| Surface | Live? | For |
|---|---|---|
| **PR mode** | yes | Playing, inspecting, lesioning — the working screen |
| **Presentation mode** | yes (or replay) | Spectators, kiosk, second screen, VJ output |
| **Bench** | no | Ablation compare, V1 explorer, checkpoint diffs |

The Bench is a separate window, not a mode. Ablation compare runs four models
at 4x inference cost and cannot hold the audio budget; it was the single
feature dragging an entire "research mode" into existence around itself.
Evicting it is what let the other two merge.

---

## The stage

Both live modes render the same two surfaces from `VISUALIZER_PLAN.md`, fed by
one websocket event stream:

- **Brain view** (V3) — soma point cloud, recolored per frame by group activity
- **Performance view** (V4) — schematic fly avatar striking the kit

Two surfaces can't both be the anchor, so the stage carries a
**Brain / Split / Avatar** toggle. It's live and cheap, and Avatar-only
doubles as the manual form of the first degrade step.

### Which surface leads, by mode

**PR mode defaults to Split.** You need the brain view to see what a lesion
did and the performance view to see whether the strike landed on the beat.

**Presentation mode leads with Avatar.** Two reasons. The degrade order in
`VISUALIZER_PLAN.md` sheds the brain view before the performance view, so the
avatar is what survives a bad frame budget and should therefore be what an
audience is already watching. And a fly hitting a drum reads instantly at
distance where a point cloud does not. The brain view is reduced to a
horizontal activity ribbon rather than dropped — it's the thing that makes
the demo legible as neuroscience rather than as an animation.

---

## PR mode

### Detail density — Play / Inspect / Deep

One three-position control, not three modes. Same layout, same components,
same running model; only density changes. No reload, no state loss, safe to
change mid-song.

- **Play** — rails minimal, numbers hidden, stage dominant
- **Inspect** — sliders, feel readouts, lesion state visible (the default)
- **Deep** — full timing distributions, per-group rate traces, lesion list
  always open, drop-frame and queue counters exposed

### Biological sliders as anchored callouts

The four sliders from `PLAN.md` render as callouts on the brain pane, each
with a leader line to the population it drives:

Each callout carries **three fields**, not one. What the slider modulates,
the receptor or mechanism it acts through, and what is firing right now. The
first two are static labels; the third is live state.

| Slider | Modulates | Acts through | Live readout |
|---|---|---|---|
| Drive / density | pC1 / P1 population | cholinergic input (nAChR) | pC1 rate, now |
| Tightness | inhibitory synaptic gain, global | GABA-A (Rdl); GluCl on Glu edges | inhibitory drive, now |
| Pocket | membrane time constant | intrinsic — no receptor | mean τ, current value |
| Master rhythm gate | pIP10 | pIP10's ACh output to wing MNs | pIP10 rate, now |
| Genre pad | OA-VPM / VUM tonic bias | octopamine receptors | OA drive, now |

**Do not invent a receptor for Pocket.** It scales an intrinsic membrane
property and has no receptor in the loop. The field should read as not
applicable rather than being filled for visual symmetry — a made-up label
here would misrepresent the mechanism to exactly the audience most likely to
read it closely.

**Receptor labels are derived, not hardcoded.** They follow the same rule as
cell-type names: read the neurotransmitter prediction table used to assign
edge signs in `PLAN.md` Phase 1, and label from that. A slider whose
underlying edges are Glu-dominated should not say GABA-A because a constant
somewhere says so. Where the NT call is low-confidence, the callout says so
rather than asserting a receptor.

**The live field is a readout, never a control.** Same rule as the feel
metrics — it displays, it doesn't drag. Style it as text, not as a second
handle, so nobody mistakes current firing rate for something they can set.

At Play density the callouts collapse to bare handles. At Inspect they show
the modulated population and the live rate. At Deep all three fields are
visible plus the group's recent rate trace.

### The rail

Controls with no spatial anchor on the brain — the live Mapping and Theme
layers from `VISUALIZER_PLAN.md` V5: `activity_channel`, `response_curve`,
`decay_ms`, `velocity_curve`, palette. These sit in a horizontal strip below
the stage. Everything in the rail is hot by definition.

### Lesion

Per-cell-type toggles wired to the shared `masking.py` primitive. Toggling
darkens the group on the brain pane and degrades the beat audibly within one
buffer tick.

**PR mode carries a curated subset, not the full Phase 4 ablation list.**
(This answers `VISUALIZER_PLAN.md` open question 4.) The live set is chosen
for audible effect and fits on screen without scrolling; the Bench gets the
exhaustive list. One primitive, two menus over it.

### Source

Live model vs. replaying a recorded session (`--record session.jsonl`, V2).
This changes what every other control means, so it sits in the top bar as a
persistent indicator, not buried in a menu. In replay, the biological sliders
are inert and shown as such — the events are already on disk.

### Health

Top bar carries latency, dropped-frame count, and schema version. The
frontend rejects a manifest whose `schema_version` it doesn't know, so that
rejection needs a real error state: name the version mismatch, don't render
something subtly wrong.

---

## Hot vs. cold controls

This is the same distinction `VISUALIZER_PLAN.md` V5 draws as live vs.
restart. One vocabulary for both documents:

**Hot** — instant, safe mid-performance. Biological sliders, genre pad
position, lesion toggles, stage toggle, detail density, all Mapping and Theme
keys, `visible_types`, `camera_presets`.

**Cold** — requires a manifest rebuild and reconnect. Kit tier, bilateral
split, Live ↔ Render compute tier, `grouping`, `view`, `layout_preset`,
`style_pack`.

Cold controls live behind a single collapsed **Configuration** section with an
explicit Apply and a "requires reconnect" label. They are never mixed into the
rail. A performer must be able to touch anything visible without risking a
stall.

---

## Degradation

The order is set in `VISUALIZER_PLAN.md`: 50 → 25 Hz, coarsen `grouping` to
`class`, drop avatar follow-through tweens, drop the brain view. Audio and
MIDI never degrade.

UI obligations:

- Degradation is **announced, not silent** — a small persistent indicator
  naming the current step. A demo that quietly halved its frame rate looks
  broken in a way nobody can diagnose from the room.
- The indicator is visible at every density including Play, because that's
  the density most likely to be running in front of an audience.
- Steps are reversible and the UI shows when headroom returns.

---

## Light and dark mode

Follows the OS setting by default, with a manual override that persists.
Three states, not two: `system` (the default), `light`, `dark`. Theme is a
Theme-layer key in `VISUALIZER_PLAN.md` V5, so it is hot — it changes live,
mid-performance, without a reconnect.

Implementation: `prefers-color-scheme` with a `data-theme` attribute
override. Watch the media query rather than reading it once, so a machine
that flips at sunset flips the app too, mid-session.

### The part that isn't just a palette swap

`activity_channel: brightness` encodes firing rate as luminance. That
inverts between themes. On a dark stage, brighter means more active and
reads correctly. On a light stage, a bright group is a white dot on a white
background — maximum activity becomes invisible, and the map silently
reverses meaning.

So the Mapping layer needs a per-theme polarity: in light mode, activity
drives *darkness* and saturation rather than luminance. This is a rendering
correctness issue, not a styling preference — under design rule 6 in
`VISUALIZER_PLAN.md`, a channel that inverts its meaning between themes is
instrumentation that lies.

`activity_channel: size` and `both` are unaffected by polarity, and `both` is
the safer default for light mode — size carries the signal even where the
luminance range is compressed.

### Stage vs. chrome

**Decision: the stage follows the theme.** Both panes render light on a light
system and dark on a dark one — not the DAW convention of a permanently dark
canvas. Keeping the stage dark in light mode would sidestep the polarity
problem, but it would also mean light mode never touches anything except the
chrome, which isn't worth building. A real light stage is the version that
looks right on a bright desk, which is where most of the work happens.

`stage_follows_theme` stays as a config key, defaulting true, for anyone who
wants the dark-canvas convention back.

Presentation mode pins to an explicitly configured theme rather than
following the system — a kiosk or projector should not change appearance
because the host machine hit sunset. Default dark, since that surface is
usually projected into a dim room, but it's a config value, not a constant.

### Lesion needs its own channel

A true light stage exposes a collision that a dark-only stage was hiding. If
activity maps to darkness in light mode, then a silent group is pale — and a
lesioned group is also pale. The two states become indistinguishable at
exactly the moment the distinction matters most.

The same collision exists in reverse on a dark stage: silent and lesioned
both render dim. It was always there; committing to both themes is what
makes it impossible to ignore.

So **lesion is not an activity value.** It gets a channel of its own that
carries no luminance meaning in either theme — render lesioned groups as
hollow outlines, filled groups being live ones. A lesioned population then
reads as structurally absent rather than temporarily quiet, which is what it
actually is. The same treatment applies to the avatar: a lesioned limb
segment goes to outline rather than fading out.

This matters for the demo more than anywhere else. The lesion moment is the
best thing this project does; it fails if the audience can't tell
"switched off" from "not playing right now."

### Light-mode specifics

- Activity drives darkness **and** saturation together, not darkness alone.
  Pure luminance ramps on white have less usable range than on black.
- Silent baseline is a visible pale outline, never invisible — a group at
  zero still needs to hold its position so you can see it come in.
- Lit drum pads in the performance view use saturated fill rather than
  brightness, for the same reason.
- Check whether `decay_ms` wants a higher default on light: rapid dark-to-pale
  flicker across a bright field may read harsher at 50 Hz than the inverse
  does on dark. Verify against a recorded fixture rather than guessing.

### Obligations

- Every skin pack ships both variants, or declares which it supports; a skin
  that only works on one background should fail loudly at load rather than
  render illegibly.
- Theme is captured in the recorded session manifest, so a fixture replays
  looking the way it did when it was recorded.
- Contrast-check the semantic states that must stay readable in both themes:
  lesioned (outline), silent, peak activity, degraded indicator, replay
  indicator, schema mismatch error.
- Verify silent, lesioned, and peak are mutually distinguishable in both
  themes before shipping either. This is a correctness test, not a design
  review.

---

## Presentation mode

Avatar-led, brain as ribbon, no labels, no numbers. Only two exposed
controls: the genre pad and the master rhythm gate, both sized for touch.

Because the frontend consumes recorded fixtures, this mode runs with no GPU,
no neuPrint connection and no trained model. That makes it the natural build
for a kiosk, an unattended loop, or a second screen at a show. Treat "runs
from a fixture" as a shipping requirement for this mode, not a side effect.

When driven live from PR mode, Presentation opens as a second window sharing
the same event stream rather than as a mode switch — so you can play from the
working screen while the audience sees the clean one.

---

## Touchscreen roadmap (post-v1)

The tablet build is Presentation mode, not a port of PR mode.

1. Genre pad and rhythm gate already work as touch targets — bigger hit
   areas, no redesign.
2. The four biological sliders are the first candidates for physical knob
   mapping. MIDI CC out from the same hot-control bus that drives the
   on-screen callouts, so hardware and software read one source of truth.
3. Lesion, Bench and Deep density stay desktop-only. They're dense and
   numeric by nature; the touch build's value is being the opposite of that.
4. Source indicator matters more here, not less — a kiosk running a fixture
   should say so somewhere discoverable, or someone will eventually claim the
   fly is improvising when it's replaying.

---

## v1 cut

PR mode with Split stage · Play/Inspect/Deep density · brain view (V3) ·
performance view (V4, schematic avatar) · three-field slider callouts
(modulates / acts through / live rate) · curated lesion set · hot/cold
separation with Configuration gate · source indicator with fixture replay ·
degradation indicator · light/dark following system with a true light stage,
per-theme activity polarity, and lesion rendered as outline · Presentation
mode · Bench with ablation compare and the V1 explorer.

Out: touchscreen build, hardware controller mapping, skin packs beyond
`schematic` plus one theme, and the v2 performance modes (continue, call and
response, accompany) — stub buttons with tooltips, visibly disabled.

---

## Open questions

1. Brain pane at PR-mode width: does the point cloud stay legible when the
   stage is split, or does Split need to be an uneven ratio rather than 50/50?
   Related to `VISUALIZER_PLAN.md` open question 1 — if `type_bilateral`
   pushes past ~150 groups, the pane gets crowded before the payload does.
2. Does the Bench run its four ablation conditions live in parallel, or
   precompute and scrub? Scrubbing is cheaper and the Bench has no real-time
   obligation, but live parallel makes the lesion-style demo possible there
   too.
3. Genre pad: fixed 2D projection of the style embedding, or a layout
   recomputed as genres are added? A moving pad breaks muscle memory between
   sessions, which matters more for performance than for research.
4. Second-screen Presentation window — same process with a shared stream, or
   a separate client connecting to `live_server.py` like any other? The
   second is cleaner and gets remote display for free.
