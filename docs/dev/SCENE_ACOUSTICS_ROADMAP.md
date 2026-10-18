# Scene acoustics: shared-material direction

## Current continuation and UI follow-up — 2026-10-17

2026-10-18 source delivery: [scoped transport and acoustic A/B](AUDIO_SCOPED_CONTROLS.md)
adds explicit selected/group/scene operations, staged acoustic batch configuration
and recipe-preserving dry/processed bypass through shared UI/Python/IPC services.
Group membership is a transient target list, not persistent project authoring.
Existing playback acceptance was extended without a new script or live run.
User C++ build and consolidated acceptance remain pending; no shader change.
This supersedes the queued scope/batch/A-B item below only to this bounded extent.

2026-10-18 continuation planning: [physics-driven fluid synthesis](AUDIO_FLUID_SYNTHESIS_PLAN.md)
defines continuous flow generation first, then scene-clock/replay/capture closure,
then measured splash/contact events. Existing flow activity does not measure
splashes or acoustic power. This is a source-audited implementation plan, not a
new runtime delivery or acceptance result; queued UI work below remains open.

2026-10-18 next source slice: [single-bounce diffuse transport](SCENE_ACOUSTICS_DIFFUSE.md)
adds opt-in Lambert source-ray quadrature, receiver visibility, eight delay-energy
bins and approximate live auralization, using the existing scattering split.
UI/Python/IPC share the same configuration and source validation. After the user's
build, bounded fluid-room acceptance passed zero/full/half scattering energy,
moving-fluid reports, type/cone rejection and native PCM output without overload.
Long-run, convergence and performance acceptance remain open. Multi-bounce / late
reverb, diffraction and phase-resolved wave physics remain future layers.

Source delivery: [physics-driven spatial sources](SCENE_AUDIO_PHYSICS_SOURCES.md)
connect existing reactive asset voices to shared runtime pose, spectral filtering,
GPU early reflections and diagnostics. Moving-query polling and source isolation
are included. The user rebuilt; real-fluid automatic start/stop, runtime position
following and Stop All/resume passed. Scene filter/echo acceptance remains open
because live queries encountered undefined acoustic materials/AS coverage limits.
Real-time fluid/fire/fracture sound generation is not part of this slice.

Core continuation: transient simulation visuals are now outside the explicitly
reported authored-flat-environment boundary scope. Raster TLAS slots remain stable
and direct/reflection/debug shaders filter simulation display geometry consistently.
CPU trace shares object classification. This source change needs a C++ rebuild and
both acoustic shaders rebuilt; the saved [fluid room](FLUID_AUDIO_ROOM.md) remains
the acceptance fixture. No new live success is claimed before that build.

2026-10-18 rebuild verification: the fluid room now completes both previews across
eight real solver steps without the reproduced revision error; seven post-warmup
samples retained five paths each. A temporary authored-wall move changed the
reflection distance and restoration returned it to baseline. See the room document
for evidence and limits. Continuous timeline/cache replay and PCM/performance
acceptance remain open; this closes the bounded step-driven revision regression.

User-built early-reflection playback is audible in the echo/voice rooms; see
[the live lab](ECHO_ROOMS_LAB.md). This does not close full physics-driven playback,
export parity, multibounce/late reverb or 16/32-source performance acceptance.
The current source slice connects automatic/reactive sources to the same acoustic
processing path; after acoustic acceptance, continue with fluid-flow/splash-driven
faucet sound generation. Source preview eligibility now includes spatial reactive
sources. Real-fluid control acceptance is recorded in the physics-sources document;
manual-room echo acceptance and real-fluid control acceptance are separate checks.

User-requested UI work is queued alongside that continuation, not implemented by
this planning checkpoint:

- Reorganize the general shared Substance editor into logical tabs and a bounded
  body across object properties, domains, flow sources and colliders, not only
  audio. Preserve one field registry/service/editor and context-specific binding
  semantics; see [shared Substance UI direction](SHARED_SUBSTANCE_UI_ROADMAP.md).

- Compact the left main-panel asset section as libraries grow: fixed/bounded
  list height with independent scrolling, search/type filters and collapsed
  categories; use clipped/virtualized rows for large lists. Avoid an endlessly
  growing vertical asset list, keep selection/details contextual, and preserve
  the viewport-first left rail/right dock/bottom editor structure. This is a
  recorded UI follow-up, not part of the physics-source implementation slice.

- Clarify shared transport: existing `audio.play_all` / **Play all sources** starts
  bound scene sources, including those with an enabled acoustic preview. It does
  not enable acoustic processing on every source. Existing `stop_all` stops them
  together. File/asset audition is a separate exclusive preview.
- Modernize value entry in both the main panel and contextual inspector with
  consistent audio controls: sliders paired with precise numeric entry, visible
  units/ranges, appropriate logarithmic mapping for frequency/time where useful,
  and a clearly stated linear-gain/dB display convention (including mute at zero).
  Preserve keyboard entry, reset-to-default and shared validation; avoid knobs or
  sliders that conceal the actual value or change stored units implicitly.
- Consolidate scattered Play/Pause/Stop controls for asset audition, scene sources
  and preview into a consistent transport layout in the existing Audio editor.
  Always show the target/scope and currently active source or audition. Use
  persistent text/icon states for playing, paused, stopped, timeline-held and
  automatic-suspended, with enabled/disabled actions and selected scope visible;
  do not rely on color alone. Distinguish acoustic processing enablement from
  playback state and show when exclusive asset audition stops scene playback.
  Main panel and inspector must reflect the same shared service state, with no
  competing UI-local transport flags. State/scope changes and any new canonical
  operations require scripting/IPC parity and documented transition tests.
- Provide explicit selected/group/scene playback scope, batch acoustic processing
  enable/disable, and a dry/processed A/B for the same scope. A room demo's script
  batch is not a finished generic group authoring system. Preserve each source's
  authored gain/loop/binding and clearly show playing/held, processing enabled,
  pending/ready/bypassed/error states. Validate eligibility and budgets for a
  whole batch before mutation; define atomic failure semantics.
- Keep viewport labels compact: short source name or readable fallback, a bounded
  screen width with ellipsis, and full source/asset names on hover or in the
  contextual inspector. Avoid appending long asset filenames to every visible
  marker. Offer full detail for selection without flooding the viewport, and
  handle overlapping labels. Display truncation must not rename assets or sources
  or alter stable IDs/picking. This refines the first combined-label fix in
  [audio gizmo visibility](AUDIO_GIZMO_VISIBILITY.md).
- Use the existing Audio bottom editor/contextual inspector and shared core
  operations through UI/Python/IPC. No permanent shelf or UI-only batch logic.
  Deliver scope/validation documentation and relevant batch regressions with
  implementation; user build plus manual label/transport checks remain required.

## Earlier checkpoints

2026-10-17 source update: [live spectral playback, sparse verified specular echoes
and concurrent source transport](SCENE_ACOUSTICS_PLAYBACK.md) are wired through
UI/Python/IPC. This revision requires both acoustic shader and Studio builds;
listening, GPU lifecycle and performance acceptance remain open. Full reflected
spatial arrival, export parity and broader reflection discovery remain roadmap work.

2026-10-16 surface-debug source update: Render Settings view 15 now provides a
selected-source **Acoustic Energy (solid proxy)** surface field on Vulkan RT,
replacing the screen-space information card. Shared source selection and render
settings are available through UI/Python/IPC. This first field uses distance/cone
attenuation and solid geometry visibility, not the six-band transmission solver or
reflection physics. The user confirmed the built surface view and source-selection focus fix; quantitative
rendered acceptance remains open. See
[visualizer contract and checklist](SCENE_ACOUSTICS_DEBUG.md).

Next design checkpoint after neighborhood measurements:
[relevance, bbox/SDF proxies and debug views](SCENE_ACOUSTICS_PROPAGATION_DEBUG_PLAN.md).
Bounds are candidate filters or explicit approximations; they do not silently
become exact acoustic walls. Phase-aware wave work remains distinct from the
current six-band energy/direct-path preview.

2026-10-16 acceptance update: the user-built mask-cast correction passed all
13 Vulkan RT external GPU/native PCM cases after the separately accepted
RayFusion run. See [GPU acceptance record](SCENE_ACOUSTICS_GPU.md). Small-scene
direct-path acceptance is recorded for both; lifecycle stress, large-scene
performance and subsequent propagation stages remain open.

2026-10-15: user extended the audio target to object interaction, obstruction,
transmission and physical echoes. Shared object materials precede propagation;
see OBJECT_SUBSTANCE_BINDINGS.md for the implemented source checkpoint. Test
scripts remain deferred to the final acceptance stage by user instruction.

Latest source checkpoint: [direct-path diagnostic](SCENE_ACOUSTICS_TRACE.md) and
[session listening preview](SCENE_ACOUSTICS_PREVIEW.md) connect bounded flat-mesh
traces to source volume through UI/Python/IPC. The first scene acceptance checklist
is ready for the user's build. Object-barrier mode counts a named object once as
an explicit assembly approximation; closed-wall thickness, spectral filtering,
background solving and reflections remain incomplete. The 2026-10-16 external IPC/native PCM acceptance passed for the direct-path
preview; this does not validate the remaining physics stages.

2026-10-16 next implementation slice:
[GPU-first bounded solver](SCENE_ACOUSTICS_BACKGROUND_PLAN.md). Reuse existing GPU
AS/hit infrastructure with asynchronous result collection, explicit acoustic
coverage and stale-result rejection. Do not force lazy CPU BVH updates for live
audio. This supersedes the CPU-worker proposal. The first
[GPU source delivery](SCENE_ACOUSTICS_GPU.md) is implemented; user shader/Studio
build and final external/manual acceptance remain pending.

2026-10-16 follow-up: user-built RayFusion GPU/native PCM acceptance passed
13 cases; see [GPU acceptance checkpoint](SCENE_ACOUSTICS_GPU.md). Vulkan RT,
in-flight stress, large-scene performance and remaining propagation stages are
still separate acceptance gates.

1. **Shared material authority:** object -> SubstanceLibrary profile and explicit
   overrides; domain-independent binding, MSF compatibility, project/history and
   UI/Python/IPC. Acoustic band data and energy-conserving response are in source.
   No measured built-in acoustic table or live acceptance is claimed yet.
2. **Geometric direct paths:** immutable flat TriangleMesh/DNA geometry snapshots,
   bounded acceleration structure and source/listener segment queries. Report
   blockers, material provenance, unknown responses and path length/delay. Treat
   thin surfaces and closed/layered walls explicitly; do not multiply entry/exit
   losses as if they were independent walls. Moving/deforming geometry needs
   revision ownership and bounded updates before it can be live authority.
3. **Early reflections:** validated source -> surface -> listener paths, visibility
   on every segment, band response, reflection order/path budgets and delay. Begin
   with specular paths; diffuse scattering needs energy accounting and deterministic
   sampling. A material name alone cannot produce a room response.
4. **Audible propagation:** shared filtered direct path and delayed reflection /
   impulse-response processing. Keep render/analysis work outside the device
   callback; smoothly replace responses with cancellation/latest-result ownership.
   UI, Python, IPC and offline export must use the same acoustic model and timing.
5. **Broader physical coverage:** air absorption and environment-derived sound
   speed, source directivity/listener orientation, motion/Doppler, diffraction at
   openings/edges, multilayer transmission and temperature/wetness-dependent
   materials. Geometric reflection rays are not a complete wave solver; explicitly
   identify approximations and the frequency/scene scales they support.
6. **Late reverberation and closure:** bounded diffuse/multiple-bounce response,
   deterministic project/export, reference scenes, physical-unit calibration,
   listening/performance budgets and the final combined script/manual acceptance.

No per-face Triangle collection may become geometry authority. Acoustic state must
not silently enable domain telemetry or simulation, infer physical data from PBR
appearance, duplicate SubstanceLibrary records or add a permanent UI shelf.
