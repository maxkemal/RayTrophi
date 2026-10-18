# Physics-driven fluid synthesis continuation

Planning checkpoint: 2026-10-18. This document defines the next implementation
slice; it does not claim new runtime code, builds or live acceptance.

Subsequent source checkpoint: [continuous physical generation](PHYSICAL_AUDIO_GENERATION.md)
implements file-free liquid/gas flow and DEM friction/rolling textures through the
existing shared service. F0 has a first runtime source delivery, not accepted
completion; control-history/replay/export, tails and measured events remain open.
The user corrected work order: physical sound core first, combined UI pass last.

## Starting point

Concurrent asset voices, shared runtime world poses, automatic fluid activity
control, spectral direct-path filtering and sparse specular/diffuse playback are
in source. Bounded user-built fluid-room checks passed; long playback, cache replay,
16/32-source performance, acoustic export parity and listening quality remain open.
See [the acoustics roadmap](SCENE_ACOUSTICS_ROADMAP.md) and
[diffuse acceptance](SCENE_ACOUSTICS_DIFFUSE.md).

Source inspection establishes three constraints:

- `MatterAudioTelemetry.cpp` publishes gross kinetic transport activity as
  `0.5 * mass * speed^3 / voxel_length`, plus a weighted world position. Its W
  dimension does not make it radiated acoustic power. It publishes no liquid
  impact, bubble radius or splash event stream. End-state samples do not supply
  integrated work (`energy_measured` remains false).
- `AudioReaction` already owns activity envelopes, signal expiry and scene-clock
  handling. `AudioSourceRuntime` resolves pose/gain/pitch for acoustic queries and
  playback. Extend these paths rather than adding a competing controller.
- `AudioService.cpp` currently prepares asset-backed miniaudio sounds and capture
  looks up an asset. A runtime generator needs an explicit source-kind contract;
  a generated loop file alone would not finish runtime synthesis or export.

## F0: continuous flow generator

First deliver a mono procedural flow texture controlled by the existing
`fluid_flow` channel. Model it explicitly as an artistic activity-driven source,
not calibrated fluid acoustics. Do not infer splashes from frame-to-frame changes
in the scalar activity or label noise bursts as measured impacts.

Use deterministic seeded noise, bounded filters and smooth activity modulation.
Reuse reaction input calibration and envelopes; generator brightness/texture
controls shape timbre without redefining the activity unit. Keep authored gain,
distance law, bus and pitch semantics consistent with existing sources. Automatic
start, stale-input fade and Stop All suspension retain existing transport rules.

Introduce a versioned procedural recipe on an emitter, with an explicit source
kind. Existing scenes default to asset playback. An emitter selects one input
kind; reject ambiguous asset-plus-generator configuration before mutation.
Changing kind releases the previous runtime voice safely and preserves the
emitter ID, spatial settings and routing. Undo/redo and project load use the same
validated recipe; no temporary WAV becomes hidden scene authority.

Put DSP, recipe validation and generator data-source lifetime into focused new
modules. `AudioService.cpp` receives minimal integration wiring because it exceeds
the workspace file-size limit. Reuse the existing mixer/acoustic node after the
generator, so dry, spectral, specular and diffuse processing share one signal path.

The device callback consumes fixed-size published controls and preallocated DSP
state. No JSON, geometry access, telemetry polling, allocation, file I/O, blocking
locks or unbounded event processing may run there. Define ownership for controls,
reset and destruction; stop/detach the node before freeing callback-visible state.
Retain the current 32-resident-voice limit; generators do not gain a second budget.

## Shared authoring contract

Extend existing source create/update/get and discovery schemas rather than add an
unrelated generator registry. The final parameter names must be settled with F0
implementation, but all surfaces must expose the same operations:

| Operation | Required behavior |
|---|---|
| Configure source | Select asset or versioned flow recipe; validate atomically |
| Inspect source | Return authored recipe, input kind and existing stable ID |
| Inspect runtime | Report readiness, input age, generator state and bypass/error |
| Play/pause/stop/all | Use existing transport and automatic suspension rules |
| Save/load/undo | Preserve recipe and seed; exclude transient DSP state |
| Render/capture | Evaluate the same generator and identify propagation coverage |

UI in the existing contextual inspector, `rt.audio` bindings and external
`audio.*` IPC must call one service implementation. Reject unknown recipe versions,
unknown keys, nonfinite/out-of-range controls and invalid domain/channel bindings
before changing sources or transport. Errors leave source state and preview
revision unchanged. Telemetry remains explicit opt-in; configuring a generator
must not start simulation or enable collection.

## F1: scene clock, capture and export

Define generator samples by scene sample index, recipe version and seed; output
must not depend on UI polling frequency or callback block partition. A publication
contains simulation time, generation/revision and sequence, allowing repeated
samples, stale data and rewind to be distinguished. Do not replay the same control
publication as a new event on every UI frame.

Forward cache replay must supply the same control history as simulation. Seek or
rewind invalidates stale controls and acoustic requests, resets generator state,
and reconstructs from an explicit checkpoint or bounded preroll. If history is
unavailable, report unavailable rather than silently resynthesize from current
activity. Paused timeline mode must not advance procedural time; continuous mode
must have an explicit clock owner.

Native capture and offline render evaluate the same DSP with a sampled control
history. Static recipes alone cannot reproduce a changing simulation. Define
history storage/versioning, missing-history errors, frame-to-sample mapping and
resampling before claiming export parity. Existing live acoustic spectral/echo
processing also lacks export parity; a dry procedural export must report that
boundary explicitly. Full acoustic parity remains a separate roadmap gate.

Automatic inactivity should fade input excitation, then drain bounded generator
and acoustic tails before releasing the voice. Explicit Stop All retains immediate
user control. Define maximum tail duration and distinguish draining from active
excitation in shared runtime status. Do not leave silent voices resident forever.

F0 is a source checkpoint; physics-driven synthesis is not complete until F1 and
its script/IPC, persistence and acceptance work are delivered.

## F2: measured splash/contact events

Add liquid contact/event telemetry only where the solver owns reliable contact
data. Required event provenance includes stable identity/sequence, simulation
time, world position, participating substance and a defined measured impulse or
dissipation quantity with units. Specify coalescing, capacity, overflow counters,
cache storage and replay deduplication before consuming events as sound.

No extra render-geometry scan or per-face Triangle facade is an event authority.
Any geometry needed by this work uses flat TriangleMesh/DNA data. Do not estimate
bubble resonances from particle count or claim bubble sound without a radius/model
input. Until this stage exists, the continuous source remains flow texture only.

## Delivery order and acceptance

1. F0: recipe/DSP/data-source modules, minimal service integration, shared
   UI/script/IPC validation, project/history migration and focused regressions.
2. F1: deterministic clock/history/replay, capture/offline generator parity and
   bounded tail ownership, with documented acoustic-export limitations.
3. F2: measured event contract and solver publication, bounded event synthesis and
   replay, followed by listening/material calibration.
4. Close outstanding acoustic lifecycle/convergence/performance acceptance;
   proceed to broader transport and UI work in their existing roadmaps.

Non-build checks for each source delivery: inspect callback allocation/ownership,
project registration, schema/discovery parity, migration defaults, validation
atomicity and deterministic PCM assertions. Reuse existing synthesis/reactive
acoustics tests where practical; keep raw captures and bulk measurement output
temporary rather than adding large fixtures to Git.

After the user builds Studio, run external IPC tests from a separate process in
a disposable scene. F0 requires finite/nonzero PCM, silence at zero activity,
deterministic same-seed output across block sizes, invalid-config atomicity,
expiry/Stop All/resume, remove/rebind, and simultaneous asset/procedural voices.
F1 additionally requires pause, rewind, cache replay, save/reload, undo/redo,
live/offline dry PCM equivalence and bounded tail cleanup. Exercise in-flight
acoustic updates and scene changes without callback lifetime errors.

Measure 1/8/16/32 sources with dry, spectral, specular and diffuse configurations;
record device block size/sample rate, callback time versus its block deadline,
underruns, main-thread/control cost, GPU query latency and memory. Result age is
not GPU execution time. Agree target hardware and budgets before marking
performance accepted. Listen separately to steady flow, start/stop, rapid control
changes and moving sources; meter output alone does not establish water quality.

Codex must not build or launch Studio for this work. Shader compilation is needed
only if a later implementation changes shader sources/ABI; F0 does not inherently
require a shader change. No live tests were run for this planning checkpoint.
