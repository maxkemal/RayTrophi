# Independent live volume outputs: implementation boundary and render audit

Requested outcome: one physical Matter domain, separate named live VDB records
for solid Fog, vapor and individual gas-source contributions. Independent volume
materials control density/scattering/emission and multiple scattering. Physical
temperature is always the common simulated field. SurfaceSDF remains on Domain
Output and is not added to the VDB volume list.

## Verified blockers in current source

1. `FluidDomainFogVolume.cpp::syncDomainFogVolume` adds carrier Fog density to
   gas-grid density and binds a single `desc.fluid_fog_shader`. Source identity
   has already been lost at that point; splitting materials afterward cannot
   recover density ownership.
2. `ParticleSystemObject` stores only one `domain_fog_vdb_ids` entry and one
   Fog hittable per domain. Cleanup, rewind/rebind, removal and live packet
   publication currently assume that cardinality.
3. `shaders/volume_overlap_selection.glsl` explicitly supports one primary and
   one companion. A third simultaneous medium is ignored. Companion emission
   and its material program are not evaluated. This is not a general mixture.
4. `shaders/volume_closesthit.rchit` enables `volumeRandomWalk` only when there
   is no companion, no active material program, no emission and no internal
   layered surface. Splitting coincident boxes alone silently disables the
   solid Fog's requested path-traced multiple scattering. Simply deleting the
   guard would omit the companion from flight and shadow sampling.
5. GPU gas inventory separates vapor/condensate/rain/foreign mass and shared
   energy, not emitter identities. `gas_source_density` etc. are aggregate
   injection scratch fields. `solid_gas_band` is a collider emission band, not
   an advected emitter-provenance field. Material security stubs named
   `gas.get_source_materials` do not establish an implemented source split.

No renderer or physical solver was edited during this audit. Previous phase
output changes remain intact. No build, shader compilation, application launch
or live scene mutation was performed for this audit.

## Required architecture

### 1. Stable render stream registry

Use structured keys scoped to the particle-system/domain stable identity:
`{domain_uid, substance_tag, phase, producer_uid}`. A display name is not an
identity. Producer 0 is anonymous/aggregate; gas source UIDs are the authored
stable `timeline_uid`, not names or positional indices. Solid/liquid parcel
streams can also retain their birth-source UID where it exists.

Each stream owns one VDB manager ID, one live volume object and an independent
volume shader binding. Its authoring is persisted separately from transient
density buffers. Empty streams retain authoring and stable identity; no per-frame
delete/create loop. Domain removal and project close release ALL stream IDs.
Rewind/cache replay rebinds the same stream keys. Hiding a VDB entry must not be
overwritten to visible by each simulation sync.

Keep the ordinary one-volume fast path and existing imported VDB/cloud behavior.
Use the existing VDB list and material editor; no new permanent UI surface.
Expose list/get/set stream appearance through one service used by UI/script/IPC.
Fields describe appearance, not authored physical temperature.

### 2. Separate field producers before any sum

Solid/liquid carrier Fog gathers by substance + current phase, using existing
canonical particle state. Gas/vapor renders its appropriate physical field.
No density sample is copied into multiple streams as though each owns all gas.

For individual gas emitters add passive, advected contribution fields at source
injection. They use the SAME resolved gas velocity, geometry, boundary rules and
time intervals. They do not solve extra pressure/velocity/temperature fields or
duplicate physical mass/energy. Account for background, generated vapor and
unattributed material explicitly. Specify how reaction/phase-transfer changes
contributions; do not assume the emitting source owns all downstream products.

Render contributions must partition the authoritative target density field,
with nonnegative weights and a defined zero/unknown fallback. Arbitrary mixing
does not justify selecting the nearest emitter or retaining only the dominant
source. Cache snapshots and rewind must capture the contribution state; live
GPU residence must be retained. Apply explicit memory/resource-budget checks.

Temperature sampled by gas streams is the common physical gas temperature.
Solid parcel Fog reads its canonical parcel temperatures, not a second manually
authored temperature and not the flame shader's temperature channel.

### 3. General mixture integration, preserving ordinary render paths

Independent overlapping VDBs need one consistent medium evaluation, not repeated
full coincident box marches. At each point evaluate all active contributions:

- Extinction is additive, component-wise for colored coefficients.
- Scattering, phase function and source emission are evaluated per material.
- Light attenuation includes every active stream.
- Scattering events select a contributor with the appropriate physical weight;
  multiple-scattering eligibility must not silently disappear because vapor
  overlaps solid Fog. Define and test mixed random-walk/march behavior explicitly.
- Preserve current gas-to-SDF handoff and bounce accounting, volume masks,
  opaque geometry termination and single-VDB/cloud fast paths.
- No third-medium truncation and no suppressed secondary-emitter blackbody.
  Report actual supported capacity/resource failure instead of silently dropping
  a medium. Keep behavior coherent across supported render backends.

Do not change the renderer traversal/bounce sequence merely to add VDB records.
Put mixture evaluation in focused modules; existing >2,000-line files get only
minimal integration calls. A safe implementation still needs to extend medium
evaluation; claiming that the current shader can remain entirely unchanged is
incorrect.

## Acceptance gates before calling the feature complete

1. Snow Fog + vapor Fog, coincident bounds: independent material edits; no
   blackbody on Snow, no density sharing, Snow multiple scattering remains active.
2. Two gas sources with different source appearance, crossing/mixing: scalar
   temperature is shared and matches solver probes; source weights advect and
   sum to the authoritative density. No independent source temperatures.
3. Three overlapping media, including two emissive ones: no omitted medium,
   emission or material. Check transmittance against analytic uniform-medium
   extinction and order independence when VDB registration order changes.
4. Repeat 1/3 with the current melted SurfaceSDF and an opaque collider. No lost
   surface, double march, bounce consumption change or incorrect handoff.
5. Isolated Snow multiple scattering is unchanged. Mixed results retain the
   requested capability rather than taking an undocumented lower-quality path.
6. Ordinary imported VDB, atmospheric cloud, single gas domain and existing
   SDF/splat scenes retain their established path and appearance.
7. Empty/reborn streams, hide/show, material edit, emitter rename/delete,
   domain removal, undo/redo, save/reopen and cached playback preserve stream
   identity, clean resources and never reset physics for appearance edits.
8. Script/IPC transactional validation and readback of appearance use the same
   service. Static checks plus user builds and explicit external IPC/live render
   acceptance; Codex does not build or launch Studio under workspace rules.

Implementation is not complete. This audit establishes why a material-only or
VDB-list-only patch would violate the user's render-preservation requirement.

## Step 1 implemented: isolated field preparation (not connected to rendering)

`Fluid/MatterFogFields.h/.cpp` now provides an internal field builder. It reads
canonical particle SoA through the current FluidViewPlan, groups carrier Fog by
substance and current solid/liquid/unresolved state, and keeps aggregate gas in
a separate field. Untagged seed carriers can resolve to an explicit default
substance tag supplied by the caller. Existing density/temperature splats are
reused with an optional read-only selection mask; legacy callers leave it null.

Gas input requires an explicitly authoritative snapshot and explicit temperature
units (Kelvin or the existing normalized-3000 grid convention). Missing physical
temperature remains unavailable; the builder does not invent an ambient or an
artistic temperature. It performs no solver readback itself, and a future caller
must actually materialize a current snapshot before asserting authoritative.
Gas-source provenance is not available yet: the output is honestly labelled
GasAggregate, not fictitious separate source fields.

The builder has explicit stream and dense-buffer/scratch budgets, rejects invalid
gas sizes/units/densities and leaves the prior output unchanged on refusal. This
budget excludes preexisting input/output storage and small stream metadata. No
physical arrays are mutated. No phase conversion, emitter, VDB ID or shader is
created. No material, registration, cache lifecycle or renderer integration is
enabled in this step. Existing live scenes continue using the current publisher.

Validation so far is limited to static source checks; the field builder has not been exercised at runtime. The unregistered C++ test source was removed because it had no executable test target. Runtime integration must verify independent temperatures, density conservation, unchanged physical state and atomic refusal through a runnable validation path.

Next step: stable live VDB records and independent material bindings, using the
existing panel/lifecycle, with UI/script/IPC parity. Renderer integration remains
gated on the overlap and multiple-scattering acceptance above; no claim of a
working multi-volume render is made by this field-builder step.

## Live field inspection added before VDB integration

`rt.fluid.inspect_fog_fields(domain)` and IPC `fluid.inspect_fog_fields` call the
same on-demand service in `RtApiMatterFogInspection.cpp`. This exercises the real
field builder against initialized Matter state without advancing simulation,
editing routing, registering VDBs or changing the renderer. Returned fields
include substance tags, physical phases, density sums, active cells and physical
temperature ranges. Missing/unwritten temperatures are not reported as cold.
The carrier maximum absolute density error compares the separate fields with
the existing aggregate splat. Use the reference maximum density to judge relative
floating-point error. Gas remains aggregate; emitter provenance is not fabricated.

Vulkan-resident gas takes an explicit on-demand dense readback and rejects a
failed density read. Host simulation uses its current grid. The builder has a
128 MiB budget and the diagnostic bounds its additional snapshot/reference/mask
storage to 128 MiB. No per-frame diagnostic allocation or readback is added.
Invalid/missing domains, uninitialized state, excessive allocations and unknown
IPC parameters reject. This is a diagnostic surface, not a material authoring UI.
`render_connected=false` explicitly records that the legacy publisher still draws
the scene. Independent VDB records/materials remain pending the render gates above.

After the user's next build, load the melting scene and pause at a frame with
surviving solid Fog and vapor. From a separate PowerShell terminal run:

```powershell
python -c 'import sys,json; sys.path.insert(0,"scripts/test"); from rt_ipc import RtIpc; c=RtIpc(); d="Physics Domain 1"; before=c.call("fluid.state_digest",domain=d); r=c.call("fluid.inspect_fog_fields",domain=d); assert c.call("fluid.state_digest",domain=d)==before; assert r["carrier_max_abs_density_error"] <= 1e-5*max(1,r["carrier_reference_max_density"]); print(json.dumps(r,indent=2)); c.close()'
```

Expected: carrier solid and aggregate gas appear as independent rows, temperature
ranges describe their own physical fields, density error is within tolerance and
the physical state digest is unchanged. If all solid has already melted, there
will be no solid Fog row. This validates preparation, not overlapping rendering.

Live check on 2026-10-24: the currently running application answered IPC with
frame 0, paused and `fluid.list_domains = {"domains": []}`. No scene was loaded,
reset or stepped by Codex. The new diagnostic requires the next user build;
its C++ runtime acceptance has not yet been performed.

### First live inspection: temperature mapping correction

The user's rebuilt melting scene was inspected at paused frame 28 with 87,552
carriers. The new inspection rejected the gas temperature with `Gas Fog
temperature must be finite and nonnegative`; repeating the inspection left
`fluid.state_digest` unchanged. Existing particle probes found solid Snow at
263.15 K and fluid carriers up to 441.88 K. Existing gas probes reported air
temperatures around 676--808 K in the sampled hot/vapour cells.

The diagnostic incorrectly assumed `gas.temperature * 3000` meant physical
Kelvin. Matter gas actually stores a signed virtual-temperature field in the
world thermal scale. `RtApiMatterFogInspection.cpp` now converts the snapshot
using domain ambient, world normalized ambient and `kelvin_per_unit`, followed
by the existing `airKelvinFromVirtual` humidity inversion using the authoritative
vapour inventory. This matches the existing gas-cell probe/heat-coupling
convention. The field builder receives explicit Kelvin values. Physical state
and existing rendering are unchanged. The additional diagnostic memory budget
now also accounts for the six inventory sidecars. Nonfinite input still rejects.

At this first inspection, the correction still needed a user build. The next
inspection below supersedes that pending status; the first runtime check caught
the conversion bug before independent VDB publication.

### Rebuilt live inspection passed: paused frame 28

External `scripts/test/rt_ipc.py` calls exercised the rebuilt C++ diagnostic on
the user's scene without stepping, resetting or changing routes:

- Grid: 50 x 50 x 50; state version 67; gas snapshot read from the device.
- Solid Snow field: 78,300 carriers, 13,612 density-active cells, density sum
  9787.49998169278, sampled physical temperature 263.1498--273.1501 K.
- Aggregate gas field: 27,100 density-active cells, density sum
  4008.6925775324835, physical air temperature 191.0557--749.7097 K.
- Carrier maximum absolute density error versus the legacy aggregate splat:
  exactly 0.0 (reference maximum density 2.566215753555298).
- Builder peak dense byte bound: 3,087,552.
- Physical state digest and simulation control state unchanged across inspection;
  repeated inspection returned identical results.
- Missing domain and unknown parameters (`temperature`, `unexpected`) rejected;
  physical state digest remained unchanged after all refusals.

This passes the live field-preparation/density-conservation check for this
Snow-plus-gas scene. The liquid phase is routed to SDF, so it correctly has no
carrier Fog row. Temperature ranges only describe this snapshot, not a separate
thermodynamic accuracy/calibration acceptance. Independent VDB material records,
general overlapping-media rendering and per-emitter gas provenance remain
unimplemented. `render_connected` remains false.

## Independent Fog publication and Vulkan mixture: source integration completed

The next implementation connects the prepared fields to the existing volume
pipeline. This section supersedes earlier statements that publication is absent;
the new C++/GLSL changes still require the user's build and visual acceptance.

- `MatterFogSnapshot.cpp` owns physical snapshot preparation, shared by the
  inspector and publisher. Diagnostics and rendering use the same Kelvin/humidity
  conversion and field partition. Simulation arrays are never rewritten.
- `MatterFogStreams.cpp` owns stable field IDs, independent material initialization
  and appearance persistence. Carrier defaults clone the old appearance with
  emission disabled; aggregate gas clones it with random walk disabled. The two
  shader objects are independent. Subsequent user edits are preserved.
- `MatterFogPublisher.cpp` publishes one live NanoVDB/hittable per field, through
  the existing VDB manager and backend packet builder. Empty fields retain their
  appearance and previous registration. Pause uses state-version gating; rewind
  invalidates/rebinds every stream, and domain deletion releases every ID/object.
  The old combined Matter Fog slot is retired. Ordinary Fluid Fog is unchanged.
- Both project serialization paths retain `matter_fog_streams` appearances.
  Sixteen total registered volumes and 32 persistent appearances are bounded;
  overflow is reported before creating additional stream registrations.
- VDB shows named Snow/solid and Vapor/Gas entries with independent material and
  visibility controls. SurfaceSDF is excluded from that list. Output links open
  the corresponding Fog editor. Simulation-owned entries cannot be deleted as
  imported files or switched into SDF through the generic VDB mode selector.
- Existing `volume.list_materials/get_material/set_material` and `rt.volume`
  expose stable `matter:<domain>:<field>` targets. Their optional `visible` boolean
  shares the same service with UI. Materials use existing atomic validation.
  Legacy `gas:<domain>` targets remain accepted as aliases for aggregate gas;
  gas shader APIs/UI now address only that gas appearance. Ambiguous Matter
  `fluid.get_fog_shader/set_fog_shader` reject with migration guidance to the
  independent volume targets. Ordinary Fluid Fog APIs retain their behavior.
- Inspector rows now report stable material target, live VDB ID, name, visibility
  and publication state. `render_connected` reflects populated live bindings;
  it is not a statement that image-quality acceptance has passed.

`volume_mixture.glsl` handles joint integration for the new Matter Fog path in
Vulkan RT. Source type 6 marks these NanoVDB contributions without changing the
packet layout. Ordinary imported/cloud/single-volume paths remain on their old
entry points when no new Matter mixture participates.

All participating ordinary Fog fields are sampled together (up to the existing
16-slot capacity). Extinction adds component-wise; each material program and
emission source is evaluated separately. Shadows integrate the same sum. Without
random walk, a single joint Beer-Lambert march integrates source terms. If a
component requests random walk, joint free flights select a scattering contributor
by its scattering rate, retain colored absorption weights and sample that
component's phase function. A component with random walk disabled contributes
single scattering and attenuation without requesting further walk events. The
existing per-material exact-event/similarity threshold remains available.
Flights and shadows include all contributors, including media entered after the
first segment, and probe opaque/SDF boundaries before continuation. Primary volume
AOVs and transparent versus diffuse bounce types are maintained.

RayFusion and CPU receive the independent VDB/material records through their
existing volume paths. The new joint stochastic integrator is Vulkan RT only;
RayFusion retains its existing bounded scattering/overlap approximation. Full
cross-backend overlap parity is not established by this change. Per-emitter gas
provenance is still absent: this release separates carrier phase/substance Fog
from aggregate gas, without fabricating source weights.

Validation completed without building:

- 10 existing RT gas optical contract tests passed.
- Existing Matter phase routing/seed/cache/project-registration audit passed.
  Its seed-call assertion was made whitespace-independent after the existing
  source formatting had made that string check fail.
- IPC capability/discovery audit passed (689 dispatched methods, 670 documented).
- Project/filter XML, new module registration, shader instance-field names,
  lexical delimiter checks and `git diff --check` passed.

Required next build/live acceptance, on the same paused melting scene:

1. Confirm `fluid.inspect_fog_fields` reports published Snow and gas rows with
   distinct VDB IDs and `render_connected=true`; carrier density error stays
   within the prior tolerance and physical digest stays unchanged.
2. Read both stable targets with `volume.get_material`. Change Snow density,
   scattering/random_walk, gas emission and visibility independently; confirm
   other target settings and physical state stay unchanged. Reject unknown keys
   and non-boolean visibility atomically. Restore appearance after probes.
3. In Vulkan RT verify Snow random walk with vapor overlap, separate blackbody,
   visible collider and refractive melted SDF. Check for black boxes, missing
   surfaces, double attenuation, self-intersection loops and unacceptable cost.
4. Hide/show, rewind, empty/rebirth, save/reopen and domain removal must preserve
   authoring and release/rebind all live IDs. Compare isolated Snow and ordinary
   imported VDB/cloud scenes. Check three overlapping Matter Fog materials and
   registration-order independence. These image/lifecycle checks remain pending.

### First mixture shader build: compiler process crash

The user's shader build reported `glslc exit -1073741819` for
`volume_closesthit.rchit`. No shader compilation was run by Codex. The first
compatibility correction removes dynamically indexed arrays of NanoVDB accessors,
maps and buffers passed through `inout` calls. Per-sample scalar NanoVDB handles
now follow the ordinary fog path. The returned optical struct also uses a numeric
similarity flag instead of a bool member. The mixture/material/RW paths remain
enabled. This targets suspicious aggregate access/optimization, not a confirmed
root cause: a successful user rebuild is still required. Reinitializing the scalar
accessor sacrifices cross-contributor leaf caching; performance must be measured
after the compiler failure is resolved.

The next user attempt reported `-1073741571` (`0xC00000FD`, stack overflow).
`compile_shaders.ps1` now selects `-O0` only for `volume_closesthit.rchit`, instead
of its usual aggressive `-O`. Other modules retain `-O`. This isolates the heavy
SPIR-V optimization path without disabling mixture features or changing failed
output/publication guards. Optimizer attribution is still a hypothesis until
the user rebuilds. Driver pipeline compilation time and render performance of
this module need measurement after successful compilation; this is not evidence
that the final render has passed acceptance.

### User build and independent material IPC checks passed

The user reports a successful, much faster shader build with the module-specific
`-O0` flag. At paused frame 28, the rebuilt application publishes independent
Snow and aggregate-gas Fog records (VDB IDs 6 and 7), with melted SDF on ID 8.
`render_connected=true`; carrier maximum absolute density error remains 0.0.
The user also confirms two separately editable entries in the VDB panel.

External reversible IPC probes passed all eight checks: Snow density edits leave
gas settings unchanged; gas emission edits leave Snow unchanged; hiding gas keeps
Snow visible; invalid visibility and unknown material keys reject atomically;
original appearances are restored; physical state digest and simulation control
state are unchanged. No reset, stepping or project save was performed.

Both materials currently have random walk disabled. Vulkan pipeline status at
inspection was `rt_pipeline_state=none`, `build_count=0`; therefore Vulkan RT
pipeline creation, mixed random-walk images and runtime performance remain
unverified despite successful SPIR-V compilation and live VDB publication.

### Live RT regression: two-visible-volume cost and dark SDF

The user measured roughly 18 FPS with one volume visible and 1.2 FPS with both
visible, plus overly dark SDF water. At paused frame 59 both Fog materials had
random walk disabled, shadow_steps=8 and shadow_stride=4. A short external IPC
instrumentation window recorded 284,224 paths, 1,286,757 volume rays and
448,048,325 density samples. Instrumentation was restored to disabled afterward;
simulation was not advanced. The old mixture counter incorrectly classified its
shadow lookups as density samples (shadow_density_samples remained zero), so
post-fix comparisons must use density_samples + shadow_density_samples.

Source audit found a concrete multiplicative cost: every camera step evaluated
lighting separately for every scattering contributor; each light march used
camera step spacing instead of the authored shadow-step budget and re-sampled
all contributors. The compatibility workaround also recreated NanoVDB accessors
on every lookup. New light visibility additionally treated refractive SDF as an
opaque blocker, unlike the existing fog light mask. The mixture's primary AOV
could override the SDF AOV on a boundary handoff.

Corrections, pending the next user build/live comparison:

- One scattering-contributor sample per camera step, weighted by its sampling
  probability; material phase/color remain separate.
- A ray-constant sampled light with shared light/sky visibility caches using the
  authored shadow stride (sky at least eight), matching the existing march's
  reuse policy. Random-walk events do not reuse this spatial camera cache.
- Each medium's own shadow_steps bounds its light chord (up to 64); ordinary
  shadow_strength blending is restored. Random-walk shadows remain physical.
- Two fixed scalar NanoVDB accessor caches cover the common Snow+gas case;
  further media retain a scalar fallback. No dynamic accessor arrays are used.
- No opaque-SDF test on light visibility; camera and walk SDF handoffs remain.
- No duplicate camera solid probe after the parent already clipped the segment.
  The parent avoids its secondary-ray probe gate for companion intervals.
- Boundary handoff preserves the SDF/solid primary AOV; ordinary Fog AOV records
  actual transmission and retains the established thin-fog threshold.
- Shadow samples now have their own instrumentation counter.

The ten existing optical contract checks and focused structural checks passed.
Shader compilation, recovered FPS and corrected water appearance are not yet
verified. `-O0` remains in place for the volume shader; no material-density edits
or simulation changes were used to hide the regression.

### Next RT comparison: 5 FPS both, about 20 FPS single

After the user rebuild, reported performance improved to 5 FPS with both fields,
about 20 FPS with one, and about 50 FPS with Fog off. A reversible external
visibility/counter comparison at paused frame 28 recorded:

| Visibility | Paths | Volume rays | Density samples | Shadow samples |
| --- | ---: | ---: | ---: | ---: |
| Both | 6,505,071 | 2,588,165 | 176,804,099 | 69,777,382 |
| Snow | 7,481,610 | 6,056,123 | 78,126,753 | 18,275,120 |
| Gas | 7,203,564 | 4,705,651 | 73,211,217 | 28,984,254 |

These short windows are diagnostic counts, not a controlled equal-spp timing
benchmark. Pass-capped paths were 2/6/0 respectively; repeated-pass exhaustion
does not explain the dominant cost. Original visibility/counter enablement was
restored and physical state digest remained unchanged. RT pipeline was ready;
both fields had random walk and emission disabled, with matching phase/sampling
settings, shadow_steps=8 and shadow_stride=4.

The next source changes remove a second optical gather by retaining a weighted
reservoir contributor during the first gather. Camera sampling reuses the
canonical `volumeNanoDensity` index-space ray for the two scalar caches, avoiding
repeated world-to-index transforms. Joint sparse skipping takes the minimum safe
advance across every participating field, checks future box entries, and only
uses proven live zero-background producers. Graphs, density noise, visible
remapped background and unknown producers conservatively disable skipping.

A conservative common-case selector now retains the existing optimized companion
march for exactly two non-emissive, non-random-walk, graph-free media with matching
phase lobes and camera step/budget. Independent density curves/colors/absorption
remain per medium. Differing phase, emission, graph, camera sampling, random walk,
or more than two media use the general mixture. This avoids invoking the general
integrator where the established two-medium path already represents the inputs.

Only shader sources changed in this correction. Ten existing optical contract
tests plus structural/field-name checks passed. The next user shader build and
actual FPS comparison are still required; recovered speed is not claimed yet.

### Follow-up: exact raw-cell reuse for single and overlapping fog

User reported 14 FPS with both fields and solid atmosphere, versus 10 FPS with
physical atmosphere, then authorized further RT optimization without changing
behavior. This step caches raw NanoVDB interpolation corners within a shader
invocation, keyed by grid address and integer cell. Two fixed scalar slots keep
the common two-grid case independent; additional grids use ordinary reads.
Only raw field values are cached. Remapping, cutoff, material density and
interpolation weights are still evaluated for every sample. No frame-to-frame
shadow cache or interpolated light-volume approximation has been introduced.

The companion camera march now prepares its affine NanoVDB ray once, matching
the primary path. General-mixture shadow chords similarly prepare a ray once
for each of the two resident grids; noise, material programs and other grids
keep their original fallback. Sample positions, shadow budgets/stride, phase,
random-number consumption, emission and SDF handoff policies are unchanged.
Affine reassociation may produce normal floating-point rounding differences.

Validation: all 10 existing RT optical contract tests, Matter phase contracts,
shader delimiter/cache-key checks and targeted diff whitespace checks passed.
No shader/application compilation was run. GPU register cost and actual cache
hit benefit still require the user's build and live comparison; no speedup is
claimed from these source checks. Compare the same paused frame, camera,
resolution and RT settings: gas only, snow only, both, each with solid and
physical atmosphere. Also inspect SDF water brightness, overlapping fog edges
and density/material edits for immediate updates.

### Natural Vulkan-loop lighting probe after user build

External named-pipe probe at paused frame 28 confirmed the lighting cost with
both fields and Snow alone. Instrumented progressive sample throughput:

| Atmosphere | Scene lights | Both fields samples/s | Snow only samples/s |
| --- | --- | ---: | ---: |
| Solid | Point | 15.1 | 21.3 |
| Solid | Point + directional | 8.6 | 16.2 |
| Physical | Point + directional | 6.7 | 13.9 |
| Physical | Point | 9.0 | 16.0 |

These are short natural-loop windows with counters enabled and adaptive
sampling, not uninstrumented UI FPS or equal-spp GPU timestamp benchmarks.
Both-field shadow density reads per path were respectively 18.18, 25.66,
31.57 and 26.89; Snow-only values were 4.13, 6.25, 7.39 and 5.75. Path cap
exhaustion was negligible. The main cost increase is consistent with optical
sampling rather than runaway handoffs.

Source audit confirms the common companion camera path disables sparse skipping
with `!hasCompanion`, while the physical-sky path adds its sun and sky lighting
marches alongside scene-light sampling. Next candidates are a conservative
joint-empty proof for the common companion path and reuse of identical shadow
queries where direction, chord and sampling policy actually match. Removing a
lighting contribution or increasing shadow stride would change behavior and is
not part of the authorized optimization goal.

Measurement caveat: `viewport.render_frames` currently invokes the CPU
`Renderer::render_progressive_pass` even when viewport status says Vulkan. Its
returned timings were discarded; the saved probe uses only the natural Vulkan
loop. Hiding a directional also leaves it in the host light list, so the final
comparison actually added/deleted only the temporary Sun. No source fix to
these separate diagnostic/light-list issues was attempted during measurement.
Original world/light settings, Fog visibility, frame and physical field
statistics were restored; transient VDB IDs/state versions changed on resync.
Detailed evidence: `rt_gas_lighting_probe_20261024.json`.

### Joint-empty traversal and identical sun-shadow query reuse

Added `volume_companion_empty.glsl` to the existing common two-medium camera
path. It requires both scalar NanoVDB paths to be eligible (no density graph or
noise), both producers to guarantee zero background, and near-zero source
values to remain optically invisible after each material's remap/cutoff. It
then asks the existing conservative topology/leaf-bound helper for each field,
takes the minimum span, bounds it by the current overlap segment and advances
only whole steps on the original jittered lattice. A visible companion always
prevents a primary-only skip. No camera or shadow sample budget was reduced.

Added invocation-local `volume_sun_shadow_cache.glsl`: a freshly evaluated scene
directional shadow can satisfy the physical-sky sun query at exactly the same
position, direction, camera step and primary chord. The directional's finite
limit must also contain the entire companion chord. Approximate direction
matches, different positions/steps and truncated chords retain the original
march. Both radiance contributions, their phase weights and RNG consumption
are preserved. This benefits single Fog and the common companion path; the
general-mixture/random-walk paths keep their existing lighting implementation.

Both modules are registered in the project and filters; the large closest-hit
source receives only include/call integration. Existing optical contract suite
now runs 12 tests, including independent occupied-support/lattice comparison
and exact sun-query guards, with its release script mirror synchronized.
All 12 tests, Matter phase contracts, shader delimiter checks, project XML
registration and targeted whitespace checks passed. No compilation was run.
Pending user shader build: repeat the same frame/camera comparisons above,
check vapor-only regions alongside Snow, SDF-water brightness and sun/shadow
appearance. Performance and GPU visual equivalence remain unverified until
the new shaders are built and loaded.

### Fixed-sample measurements after the joint-empty/sun-query build

External IPC observed the natural Vulkan loop reaching progressive samples 16
and 64 after each accumulation reset. Timing runs had volume counters OFF;
separate instrumented runs gathered optical work. Frame 28, camera and authored
optical settings were held fixed. Medians of two or three repeated runs:

| Configuration | First 16 samples, seconds | First 64 samples, seconds |
| --- | ---: | ---: |
| Physical, point + directional, Snow + vapor | 1.99 | 6.88 |
| Solid, point + directional, Snow + vapor | 2.15 | 7.02 |
| Solid, point only, Snow + vapor | 1.33 | 4.29 |
| Physical, point + directional, Snow only | 0.91 | 3.28 |
| Solid, point only, Snow only | 0.68 | 2.20 |
| Physical, point + directional, both Fog hidden | 0.28 | 1.04 |
| Solid, point only, both Fog hidden | 0.56 | 2.06 |

Diagnostic ablation, NOT a proposed quality setting: setting both Fog materials'
shadow_steps to zero reduced the first 64 samples to 2.90 seconds for physical
point+directional and 2.45 seconds for solid point-only. Restoring the original
eight shadow steps restored the authored rendering. This strongly implicates
nested optical shadow sampling, but the ablation also changes illumination,
adaptive pixel activity and secondary paths; it is not an isolated GPU-stage
percentage. With the same directional present, physical atmosphere added no
clear timing penalty in this scene/window; the directional's added work is the
dominant difference from the solid point-only setup.

Separate instrumented 64-sample runs counted roughly 123M shadow density samples
for solid point-only, 228M for solid point+directional and 250M for physical
point+directional. Pass caps were 6/2/2, so runaway free-pass exhaustion remains
unsupported. Instrumented 64-sample times were 5.75/9.22/9.01 seconds and are
excluded from the timing table: counters measurably affect runtime. These runs
do not expose sun-cache hit counts, so matching timing is not proof of cache hit.

Limitations: externally polled wall time, not GPU timestamps; threshold polling
brackets are retained in the JSON. Adaptive sampling was intentionally preserved
and active pixel counts differ between configurations. No old/new SPIR-V binary
A/B was performed, so no optimization speedup is established by this run.
`getMillisecondsPerSample()` currently returns a constant zero in the Vulkan
adapter; `viewport.render_frames` invokes CPU rendering and was not called.
All authored materials, world, full light properties, camera, timeline frame
and physical Fog statistics were restored. Detailed evidence is saved in
`rt_gas_fixed_sample_probe_20261024.json`; no render source changed for this probe.

### Cube visibility isolates a surface/volume interaction cost

User observed approximately 70 -> 50 ms/sample when hiding the cube collider.
At frame 28 with solid world and the existing point light, toggled only the
Default_Cube render visibility through its UI power button (no generic mesh
visibility IPC operation exists). Timing/counters remained external IPC probes,
never the embedded script workspace or CPU render_frames. Cube started hidden
and was returned to hidden. Matched camera/frame/world/lights were verified.

| First 64 samples, counters off | Cube hidden, repeat | Cube visible |
| --- | ---: | ---: |
| Snow + vapor visible | 2.89 s | 5.03 s |
| Both Fog hidden, SDF water retained | 0.87 s | 0.96 s |

The first hidden baseline was 2.87 s with Fog, consistent with the repeat. Its
Fog-off timing was 1.58 s and is not used in the comparison above; the repeated
hidden control after the UI toggle was 0.87 s. These are short polled wall-time
measurements with adaptive sampling, not isolated shader timestamps.

Separate instrumented 64-sample runs, both Fog visible:

| Counter | Cube hidden | Cube visible |
| --- | ---: | ---: |
| Paths traced | 7,716,741 | 8,181,232 |
| Volume rays | 3,905,796 | 6,535,867 |
| Solid probes | 3,905,796 | 6,536,346 |
| Solid probe hits | 0 | 2,968,884 |
| Gas-to-surface handoffs | 0 | 2,968,405 |
| Primary density samples | 80,234,114 | 107,649,654 |
| Shadow density samples | 79,568,724 | 123,621,125 |
| Charged diffuse | 0 | 429,259 |
| Charged specular | 1,692,382 | 3,489,003 |
| Pass caps | 0 | 0 |

This isolates a large interaction increment with Fog, rather than the cube's
ordinary 12-triangle drawing cost. Source audit matches the counters: gas finds
the nearest solid, hands the path to its surface, and later surface/specular
paths can traverse Fog again. The aggregate does not prove those traversals
are redundant or that the probe itself costs the entire increment. They may
be necessary secondary paths; removing them would alter rendering. Next audit
should distinguish repeated integration of an already consumed interval from
legitimate surface/SDF secondary paths before modifying traversal. No runtime
code was changed during this probe.

Evidence: `rt_cube_hidden_probe_20261024.json`,
`rt_cube_visible_probe_20261024.json`, `rt_cube_hidden_repeat_probe_20261024.json`.
Full Fog appearances and physical field statistics were restored and verified;
no verification screenshots were written into the repository.

### Leaf-local optical reads and final solid handoff traversal

Code audit after the cube probe confirmed three avoidable costs, while retaining
the legitimate surface/secondary Fog interactions:

* Each optical interpolation called the NanoVDB accessor for all eight corners.
  A shadow query nests these reads inside each illuminated camera sample, for
  each light/sky term and each medium, so shadow_steps is not a once-per-ray
  budget. `volumeNanoCorners` now resolves the first corner with its tree level.
  If all eight corners stay in that same 8^3 leaf, remaining values use the
  public PNanoVDB leaf addressing helper directly. If the resolved level is a
  constant internal tile/background, one raw read supplies all eight corners.
  Leaf-boundary cells retain the original per-corner accessor path. Raw values,
  interpolation order, remap, cutoff, sample lattice/budget and RNG are unchanged.
* The triangle any-hit loaded instance and geometry records before recognizing
  a distance-only volume probe. Its flags/sentinel branch now returns before
  those two SSBO reads. The incoming-ray flag guard is essential: ordinary
  camera rays must never read the aliased location-1 probe payload. Sphere
  dispatch still precedes triangle reads; nearest-hit acceptance remains intact.
* The final SDF-to-solid handoff knew the nearest solid distance but retraced
  the whole original interval. It now returns a conservative lower bound and
  raygen uses it as tMin on that one final trace. Origin/direction remain exactly
  unchanged. Positive float bits are rounded down with a further 256-ULP bin
  guard; the actual nearest hit stays in range. Reuse of the existing traversal
  word preserves the 22-dword payload: low byte stays the input cull mask, upper
  bits carry the one-shot outbound bound, and the next trace restores the real
  mask before shader entry. Only skipVolumeAABBs consumes the bound. Gas-to-SDF
  retains the original interval, preserving water/foam absorption before solids.

SDF's own solid probe was not included in solid_probe_runs/hits previously;
it now calls the existing instrumentation helper too. Subsequent instrumented
counts therefore include both gas and SDF probes and must not be interpreted as
a new increase in ray count relative to the older partial counter definition.

Validation: 15 RT optical contract tests and 10 volume-material checks passed,
as did Matter phase contracts, delimiter checks and the unchanged 22-dword
payload layout check. Broad cloud contracts reported five assertions (packet
size and relocated density texture/linked-layer logic); the identical failures
were reproduced against HEAD shader sources with untouched input files. They
are pre-existing cloud-test failures and were not edited in this task.

No application/shader compilation or live performance claim. User must compile
the shader set, including volume_closesthit, shadow_anyhit and raygen, then
compare the same first-64-sample cube-visible/hidden and shadow_steps=8 cases.
Also check water absorption/foam in front of the cube, thin/cutout geometry,
splat/hair visibility and front-face shading when close to a surface.

### Opt-in GPU timing foundation after no observed optimization gain

User rebuilt and reported no improvement (possibly a small regression), so no
speedup is claimed for the leaf/probe/handoff changes above. User also reported
desktop/mouse lag during initial samples, low observed CPU load and nearly full
GPU utilization. Added a default-off Vulkan timestamp capture rather than more
unmeasured render optimizations. Detailed scope, disabled-path guarantees,
UI/Python/IPC controls and user build checklist are in `RT_GPU_TIMING_20261024.md`.

This first layer measures total camera RT dispatch GPU ms for at most 64
dispatches plus separate existing synchronous-submit and async-slot host waits.
It does NOT claim per-solid/SDF/shadow shader execution time. The ordinary
shader path is unchanged by this instrumentation layer. Six new executed
non-build timing contracts, 15 optical tests, 10 material checks, phase contracts,
IPC capability/descriptor audits and source checks passed. Application build
and live query results remain pending the user's build.

### General shadow traversal and solid-query optimization

**Withdrawn after the next user build reported a substantial performance
regression. This section records the attempted change, not the active path.**

No further live measurement loop was started. Source audit confirms the actual
surface BSDF interaction still consumes its diffuse/transmission/specular bounce
budget after a straight gas/SDF handoff. Making those transport handoffs spend
a bounce would cut real surface lighting and is not an optimization used here.
Existing Russian-roulette policy is left unchanged in this correction.

Changes:

* `nanoEmptyIndexStep` now computes the available geometric skip distance before
  reading leaf flags/extrema/remap curves (or internal tile activity). If even an
  empty leaf cannot advance a full camera/shadow step, it returns immediately.
  This targets the particularly wasteful case of long shadow steps in short
  leaves. Both the geometry proof and optical proof remain required; returned
  skip distances and quadrature/sample budgets are unchanged. Majorant query
  counters now count only extrema tests that can yield a useful advance.
* `lightMarchAcc` resolves shadow-strength blend before any density lookup and
  returns unity immediately for an exactly zero blend. The ambient override of
  one still performs the original calculation, and shadow_steps is not lowered.
* Added an inline ray-query solid probe variant for gas and SurfaceSDF. Hardware
  opaque triangle intersections commit without launching the separate any-hit
  shader; nonopaque triangle candidates are explicitly confirmed, matching the
  current distance-only probe's acceptance policy. Query traversal completes
  to the nearest candidate rather than terminating at the first visited hit.
  A procedural AABB candidate aborts the inline attempt and runs the original
  full masked probe, preserving spheres, splats and hair intersection behavior.
  Original masks, bounds, volume-to-surface handoffs and bounce policy remain.

The shader plan builds both `volume_closesthit.spv` and
`volume_closesthit_rayquery.spv`. Runtime selects the latter only when rayQuery
was actually enabled on the Vulkan device and the variant file exists; the
ordinary module is retained for compatibility. No payload/volume ABI growth,
new authoring setting or scene-geometry mutation was introduced. Procedural-rich
scenes may pay an inline attempt before falling back; benefit is not claimed
for that case without a user build/live check.

Validation: 18 optical contracts, six GPU timing contracts, ten material checks,
Matter phase contracts, delimiter/XML registration checks passed. Shader plan
ValidateOnly passed without invoking a compiler. No application/shader compile
or new live benchmark was run. User must build application and shaders (including
the new rayquery SPV), check cube/SDF water and procedural splat/hair visibility,
then perform one equal 0..63 GPU timing comparison with diagnostic atomics OFF.

### Rollback of regressed inline-query/shadow changes

User reported much higher runtime cost after building the preceding change.
Read-only IPC confirmed pipeline ready, no build/install pending, GPU timestamp
capture disabled/uninitialized and volume atomic counters disabled. No render
or scene-visibility experiment was started; enabled diagnostic work does not
explain this reported regression.

Reverted the entire last optimization batch: removed inline ray query module,
runtime auto-selection helper and optional SPV build entry; restored the previous
majorant-proof order and shadow-strength calculation. Runtime again loads only
volume_closesthit.spv. Next user shader build retires the obsolete rayquery SPV
through the existing stale-artifact cleanup, so an old file cannot silently
reactivate the path. Removed tests tied only to withdrawn implementations;
retained the bounce-budget audit and added a stale-variant regression contract.
The optional timing tools, independent Fog materials, phase outputs and earlier
render behavior remain intact.

No exact GPU root cause is claimed: inline ray-query register/stack pressure and
procedural fallback overhead are possibilities, not measured diagnoses. Source
rollback alone does not change the currently running executable/SPV; application
and shader rebuild by the user is required to return to the preceding code path.
No new optimization is layered onto the regressed build during this rollback.

### Surface-side hidden Fog shadow work (mesh and splat consumer)

User confirmed rollback restored normal performance, then reported expensive
solid interaction even with splat output and gas disabled. Static comparison
with RayFusion found a distinct, previously overlooked shadow integrator in
closesthit.rchit::computeVolumeShadowTransmittance. It is called by mesh/splat
direct lighting and shared BSDF light evaluation, not the main gas self-shadow
loop that previous optimizations targeted.

Concrete defects in that integrator:

1. SurfaceSDF source_type 4 entered the participating-medium Fog integral. Thus
   a liquid/solid surface reconstruction could be repeatedly sampled as Fog
   using volume density/scatter knobs, despite being surface geometry. RayFusion
   explicitly excludes type 4 from its Fog shadow integrator/deep atlas.
2. shadow_steps 0 was clamped to 1 and still sampled at least the midpoint;
   disabling gas self-shadow did not disable this surface-side work. The final
   clamp also used lower=3 with upper=1 or 2 for small authored budgets.
3. NanoVDB header/accessor setup ran before volume/ray interval rejection.
4. Its midpoint/integration reads were missing from shadow_density_samples.

Read-only scene inventory, without a benchmark or scene mutation, showed frame
27, gas=hidden, solid=follow, liquid=sdf, render_mode=particles: 78,622 body splats,
8,894 SDF carriers and 36 spray Fog carriers. Both render backends reported two
volume entries: carrier Fog plus SDF, not a live vapor grid. This is consistent
with surface-side SDF work remaining when the user disabled gas.

Extracted the existing surface shadow function into volume_surface_shadow.glsl;
the >2000-line closest-hit file receives only an include. Added early SDF and
disabled-shadow rejection; moved accessor setup behind interval checks; made
small-budget clamps well-defined. Kept Fog quadrature, strength blend, valid
positive budgets, physical bounce handling and the retired ray-query path out
of this change. Surface/SDF proxy exclusion removes an erroneous Fog darkening
term and can brighten surfaces: this is a membership/setting correctness fix,
not a bit-identical shading optimization. It does not implement a new native RT
SDF surface-shadow solution; that remains a separate limitation.

Metrics report v9 now includes the previously uncounted mesh/splat surface Fog
shadow samples; counts may increase because coverage expanded, not because new
marches were added. Recording uses one aggregate counter update per completed
surface query, gated by the existing opt-in instrumentation; no timing/readback
or benchmark loop was added. The report buffer was enlarged for the scope note.

Validation: 19 optical contracts (including surface membership, zero-shadow
rejection, accessor order, small budgets and accounting), ten material checks,
six timing checks, Matter phase and descriptor checks passed. No application or
shader compile, and no speedup claim. Next user build should compare splat + SDF
with gas hidden, then two Fog fields with positive/zero shadows; inspect any
removed false SDF Fog darkening separately from genuine surface shadows.


### General-mixture shadow sparse traversal (2026-10-24)

User isolation: cube-only and splat-only are cheap with Fog/SDF disabled.
Cost grows with Fog coverage and shadow/scatter work. Treat solid geometry as
an amplifier of volume lighting paths, not a demonstrated standalone bottleneck.

Source audit: the common companion path marches the camera interval once and
reads both fields at each sample. Its per-light transmittance is the product of
separate field integrals. Different phase functions, step policies, emission,
programs or random-walk settings select the general mixture integrator instead.
That integrator already had camera sparse skipping but its mfShadow loop read
every authored midpoint of each intersecting field, including proven-empty cells.
Thus two fields can increase cost independently of their total bounding volume.

Changes in volume_mixture.glsl:
- Reuse nanoEmptyIndexStep for shadow chords of the two cached live NanoVDB fields.
  Require producer provenance, no noise/program, and background below the existing
  optical cutoff. Leaf support/bounds remain governed by the existing helper.
- Advance by whole original midpoint steps; retain the authored budget, locations,
  weights, material strengths and independent field integration. Nonzero weak
  density is not discarded and no visibility threshold was added.
- Reuse the already clipped shadow interval instead of repeating mfContains at
  every prepared NanoVDB shadow sample. The canonical density sampler still checks
  its local bounds.
- No new profiling, settings, payload fields or build variants.

Executed non-build checks: 21 optical contracts including randomized midpoint/
optical-depth equivalence for proven-zero prefixes; 10 volume material contracts;
phase routing/cache/API/project contracts. User shader compilation and runtime
image/performance verification remain pending. No speedup has been measured.


### Prepared extinction for mixture light chords (2026-10-24)

After user compilation, sparse-shadow skipping produced no meaningful scene
speedup. Do not treat empty topology as the demonstrated dominant cost.

The general-mixture mfShadow loop called mfEvaluate at every midpoint but used
only extinction. That function refetched the instance and material-program state,
initialized full scattering/emission terms, normalized absorption, and applied
random-walk coefficient policy on every sample. For the already eligible prepared
NanoVDB path these coefficients are constant across the light chord.

Hoisted scattering, normalized RGB absorption and the exact existing random-walk
reduction policy before the midpoint loop. Each retained midpoint now reads the
same canonical density and forms only extinction. The fallback still calls the
full evaluator for noise/program/other producer paths. Existing sample locations,
shadow budgets, strength, early termination and sample counters are preserved;
no new cutoff, lower quality, scattering model, API or profiler was introduced.

Executed: 22 optical contracts, including 2000 coefficient cases covering colored
and neutral absorption plus random-walk event limits; 10 material contracts and
phase contracts. These are non-build source/numerical checks, not GPU compilation
or runtime benchmarks. The user performs compilation; speedup remains unmeasured.
