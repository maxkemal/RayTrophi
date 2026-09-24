# Production Particle System and GPU Simulation Roadmap

> **Durum:** **AKTIF / ACTIVE** — canonical implementation plan for the general-purpose
> particle system, high-quality low-cost effects, GPU-first physics, CPU
> fallback, multi-geometry rendering, and optional domain coupling.
>
> **Last updated:** 2026-09-24 (review pass: vertical slice after Phase 1,
> load-only legacy migration, Surface State Deposit output, field-masked
> surface sources, kinematic collider proxies moved to their own note,
> deterministic event ordering, RT instance budget, mud/snow and volcanic
> scenario boundaries)

## 1. Purpose

RayTrophi needs one coherent particle architecture that can serve two different
classes of work without confusing them:

1. Low-cost but production-quality effects such as flames, smoke puffs,
   explosions, sparks, debris, water streams, droplets, foam, dust and trails.
2. Particles that act as sources or carriers for the existing gas and fluid
   domain solvers.

A domain is not required to create or render a particle effect. Domain coupling
is an optional output of the same emitter and simulation state. The existing
particle-to-domain density, temperature, fuel and fluid-source concepts remain
valid and must not be replaced by a separate competing emitter model.

"Low cost" describes simulation cost, not visual quality. The target is a
high-quality GPU-driven effect path with soft intersections, temporal stability,
proper lighting, material-aware shading and scalable LOD. Gas and fluid domains
remain the high-fidelity solution when pressure, incompressibility, obstacle
flow, combustion transport or grid-scale vortices are required.

### 1.1 Two tiers, one input

Many target scenes exist in two tiers:

| Scene | High-fidelity tier (render/film) | Real-time tier (viewport/game) |
|---|---|---|
| Mud, snow, sand | Granular solver: cohesion, compaction, footprints emerge | Particle clumps + Surface State Deposit + heightfield/MSF |
| Water, splashes | APIC fluid | Ribbon stream, droplets, splats, foam |
| Smoke parting | Gas domain | Lit impostors + force fields |
| Fire | Gas combustion + MSF | Flipbook/blackbody flame + light proxy |

This roadmap owns the real-time tier and the optional hand-off into the
high-fidelity tier. Both tiers read the **same** collider input, including
animated characters; that input is defined in `KINEMATIC_COLLIDER_SOURCES.md`,
not here. Switching tier must never require re-authoring what the scene's
bodies are doing.

## 2. Current implementation truth

Agents must begin from these facts rather than assuming the system is either
fully CPU or fully GPU:

- `ParticleSimulationSystem` owns the authoritative particle SoA and already
  supports independent emitters, colliders, force fields, per-emitter domain
  deposits and multiple systems.
- The current Vulkan particle compute path dispatches
  `sim_particle_force_integrate`, synchronizes, downloads velocity, and then
  performs lifetime, position integration, scene collision, over-life visual
  interpolation and self-collision on the CPU. It is a partial accelerator, not
  a GPU-resident particle simulation.
- Current self-collision uses a CPU sorted neighbor grid. Scene collision
  supports planes, spheres, capsules, AABB, OBB, SDF, convex decomposition and
  mesh BVH, but resolution is CPU-side for ordinary particles.
- Current appearance is copied from each emitter into every particle as
  start/end size, opacity and RGB values. The CPU linearly interpolates those
  values every step. Raster billboards consume the interpolated values, while
  ray-traced instances use one shared material, normally derived from the first
  emitter's start color. The render paths therefore do not share one appearance
  contract.
- Multiple scene geometry sources already exist as
  `ParticleRenderSettings::mesh_sources`, including weights, UI editing,
  serialization and weighted render-source selection. This is useful
  groundwork, not the finished feature: it is system-level, labelled WIP in the
  UI, and source choice is derived from a stable pool-slot hash instead of being
  sampled and stored when a particle is born.
- Scene mesh extraction in `ParticleRenderBridge.cpp` already reads the
  canonical flat `TriangleMesh` / DNA SoA. That rule remains binding. Per-face
  `Triangle` facade collections must never become authoritative scene geometry.
- New systems now have independent creation and point-emitter authoring. The
  directional emitter gizmo is selectable, translatable and rotatable in the
  full SDL viewport coordinate system.
- OptiX is frozen according to `OPTIX_DONDURULDU.md`. This roadmap must preserve
  existing OptiX behavior, but new GPU-driven rendering work targets Vulkan and
  must not expand the OptiX feature surface.

### 2.1 Code entry points for implementers

| Area | Current source of truth |
|---|---|
| Particle descriptors, SoA and runtime interface | `RayTrophiStudio/source/include/ParticleSimulation.h` |
| Simulation, CPU collision and partial GPU dispatch | `RayTrophiStudio/source/src/Physics/ParticleSimulation.cpp` |
| Visible mesh-instance bridge and weighted sources | `RayTrophiStudio/source/src/Physics/ParticleRenderBridge.cpp` |
| Particle-system scene object and runtime registration | `RayTrophiStudio/source/include/scene_data.h` |
| Compute abstraction | `RayTrophiStudio/source/include/SimulationCompute.h` |
| Vulkan kernel registry/backend | `RayTrophiStudio/source/src/Device/SimulationComputeVulkan.cpp` |
| Current authoring panel integration | `RayTrophiStudio/source/src/UI/scene_ui_forcefield.hpp` |
| Serialization | `RayTrophiStudio/source/src/Utils/SceneSerializer.cpp` and `RayTrophiStudio/source/src/Core/ProjectManager.cpp` |
| Shared script service facade | `RayTrophiStudio/source/src/Api/RtApiParticle.cpp` |
| Python and IPC bindings | `RayTrophiStudio/source/src/Api/RtPython.cpp` and `RayTrophiStudio/source/src/Api/RtIpc.cpp` |

These are navigation points, not invitations to keep growing the largest files.
Section 15 defines the required module split.

## 3. Product contract

The finished system must satisfy all of the following:

- An emitter creates simulation particles; it does not own a private material
  implementation or a private renderer.
- A particle may feed zero or more outputs at the same time.
- A particle effect can run and render without a gas or fluid domain.
- Domain deposit is an optional output, not a system-wide mode switch.
- GPU compute is the primary execution path for high particle counts, collision
  and constraints. CPU is the reference implementation and runtime fallback.
- CPU and GPU use the same parameter names, clamping, seeds, state transitions,
  curve data and error semantics.
- Normal GPU operation keeps particle state resident on the GPU. Full position
  or velocity buffers are not downloaded every frame.
- Appearance is evaluated from reusable profiles and GPU curve/ramp LUTs. No
  material object is created per particle or per lifetime sample.
- Geometry is selected from reusable multi-entry geometry sets. Selection is
  made at spawn and remains stable for that particle's lifetime.
- UI, Python scripting and IPC call the same core services. No phase is complete
  with a panel-only implementation.
- Save/load, undo where applicable, timeline bake/scrub and deterministic seed
  behavior are part of the feature, not follow-up polish.

## 4. Target object model

```text
ParticleSystem
  execution_policy
  simulation_profile
  emitters[]
  appearance_profiles[]
  geometry_sets[]
  outputs[]
  collision_profile
  event_rules[]
  domain_links[]

Emitter
  source_shape / source_binding
  transform / parent binding
  rate / bursts / lifetime
  initial velocity / spread
  simulation_profile override (optional)
  appearance_profile id
  geometry_set id
  output mask

Output
  Billboard / Flipbook
  Ribbon / Trail
  Mesh Instance
  Procedural Primitive
  Surface Splat
  Volume Blob / Volume Splat
  Light Proxy
  Domain Deposit
  Surface State Deposit
```

`Surface Splat` and `Surface State Deposit` are deliberately different:

- **Surface Splat** is an *event* with a visual lifetime (a decal-like mark,
  a splash). It fades or is recycled.
- **Surface State Deposit** *accumulates* into the Material State Field
  (moisture, heat, soot/dirt channels on the texel/vertex MSF layers). Mud on a
  character's legs, soot on a wall, a surface drying out are state, not events.
  The MSF owns decay, drying and rendering; the particle only deposits.

### 4.0 Emitter sources

`source_shape / source_binding` includes, besides point, shape and mesh
surface/volume:

- **Field-masked surface source:** emit from a mesh surface only where a
  surface field passes a threshold (MSF temperature > T, moisture > M, char
  mask, painted mask). Rate may scale with the field value. This is how
  "smoke rises only where the floor is burning" and "steam rises where lava
  touches water" are authored without per-frame script glue.
- The field is **sampled**, never copied into emitter state; a missing field
  reports an unresolved binding rather than emitting everywhere or nowhere.

Outputs are composable. A burning debris emitter may render mesh instances,
spawn a spark trail, contribute a small light and deposit temperature/fuel into
a gas domain from the same particles.

### 4.1 Runtime particle state

The baseline runtime state is data-oriented and backend-neutral:

```text
position, previous_position
velocity
age, lifetime
inverse_mass, collision_radius
rotation, angular_velocity
seed, spawn_id
emitter_id
appearance_profile_id
geometry_source_id
output_mask, flags
```

Optional streams are allocated only when required by active modules:

```text
temperature, soot, density
foam, wetness
trail_id, trail_coordinate
custom scalar/vector channels
```

Current RGB, opacity and size arrays are transitional. The target renderer
derives them from normalized age and profile LUTs. CPU debug/readback may
materialize evaluated values on demand, but they are not authoritative state.

## 5. Appearance profiles

`ParticleAppearanceProfile` replaces emitter-owned start/end material behavior.
It contains:

- shading model: Flame, Smoke, Water, Spark, Dust, Debris or Custom;
- color ramp or temperature-to-blackbody ramp;
- opacity, size, emission, roughness and distortion curves;
- flipbook/texture array and frame-blending settings;
- lighting and shadow mode;
- soft-particle intersection distance;
- random variation ranges driven by the stable particle seed;
- renderer-specific quality levels that preserve the same artistic intent.

Curves and ramps compile into shared GPU LUTs. Raster, RayFusion and supported
Vulkan RT paths read the same normalized profile data. Water does not fake its
look through an RGB lifetime ramp; it uses absorption, transmission, IOR,
thickness and foam. Flame color is temperature/emission driven. Smoke uses
soot/density, scattering and lighting. A generic color ramp remains available
for motion-graphics effects.

### Legacy migration

Load-time conversion only (CLAUDE.md rule 5 — no second live code path):

- start/end size become a two-key size curve;
- start/end opacity become a two-key opacity curve;
- start/end RGB become a two-stop generic color ramp;
- the legacy fields are **read by the loader only**. They are never written,
  the runtime never reads them, and the start/end authoring UI is removed in
  the same phase that introduces profiles (Phase 2), not at the end of the
  roadmap;
- the new fields carry new names, so an old reader can never misread new data
  as the old meaning;
- migration is idempotent and covered by save-load-save tests.

## 6. Geometry sets

The current weighted `mesh_sources` vector is promoted into a reusable
`ParticleGeometrySet`:

```text
ParticleGeometrySet
  id, name
  selection_mode: WeightedRandom / Sequential / ShuffleBag / EmitterControlled
  entries[]
    stable scene node id + name fallback
    enabled
    weight
    material policy: PreserveSource / AppearanceOverride / ExplicitMaterial
    scale range
    rotation range
    alignment: Random / Velocity / SurfaceNormal / EmitterAxis
    local pivot and axis correction
    LOD policy
```

Requirements:

- A system may own several geometry sets.
- Each emitter may select a different set.
- An output may override the emitter's geometry set.
- Geometry choice is sampled from `spawn_id + seed`, stored in
  `geometry_source_id`, and remains stable until death.
- Reusing a dead slot creates a new spawn id and therefore a new valid sample.
- The original material assignment can be preserved per source mesh.
- Invalid or deleted scene nodes are reported explicitly; they do not silently
  fall back to an unrelated mesh.
- Geometry snapshots are produced from flat `TriangleMesh` / DNA SoA data.
- The long-term Vulkan path reuses scene mesh/BLAS handles for instancing instead
  of copying source triangles into particle-owned geometry.

## 7. GPU-first execution model

### 7.1 Backend policy

```text
Auto         Prefer a compatible GPU backend; fall back to CPU explicitly.
GPURequired  Fail with a surfaced error if the requested GPU path is unavailable.
CPU          Run the reference implementation.
```

Vulkan Compute is the first production target. The dispatch contract remains
backend-neutral so CUDA may implement the same kernels later, but a missing CUDA
kernel must fall back or fail according to policy rather than pretending GPU
execution succeeded.

### 7.2 GPU-resident frame pipeline

1. Consume emitter commands and allocate from a GPU dead-slot/free list.
2. Initialize spawn state, profile ids, geometry ids and stable seeds.
3. Advance age and kill expired particles.
4. Evaluate forces and velocity modules.
5. Integrate predicted positions.
6. Build collision broadphase data.
7. Resolve scene collision and continuous collision detection.
8. Resolve particle constraints/self-collision when enabled.
9. Apply event rules into bounded append buffers.
10. Commit positions and velocities.
11. Deposit optional domain channels.
12. Compact visibility lists, reduce bounds and create indirect draw arguments.
13. Render directly from device-resident state.

Normal frames read back only bounded counters, telemetry and explicitly
requested events. Editor inspection may request a snapshot; requesting it must
be visible in transfer statistics so debugging tools cannot accidentally become
a permanent synchronization point.

### 7.3 Residency contract

Every particle stream records one of three states:

- Host authoritative
- Host/device equal
- Device authoritative

This is the same class of contract already learned by the gas solver. A boolean
"GPU data valid" flag is insufficient because it cannot express device-only
newer state. Consumers must declare which streams they read and write before a
stage can remove a transfer.

### 7.4 CPU fallback and failure recovery

- A simulation step is committed by one backend. Silent half-GPU/half-CPU
  execution after a failed kernel is forbidden.
- Under `Auto`, a GPU allocation or dispatch failure performs one full state
  download from the last valid device state, marks the system CPU-resident and
  continues on CPU from the next uncommitted step.
- Under `GPURequired`, the same failure stops the system and reports the exact
  missing capability or failed stage.
- Backend switching at an authored frame boundary is supported through an
  explicit state transfer.
- CPU and GPU share seed/hash functions, constants, LUT generation and clamp
  rules. Exact floating-point equality is not required; physical tolerances are
  specified per validation scenario.
- Timeline caches store backend-independent particle state and schema versions.

## 8. Collision and constraint architecture

### 8.1 Collision tiers

| Tier | GPU representation | Intended use |
|---|---|---|
| Fast | plane, sphere, capsule, AABB, OBB | flames, sparks, distant effects |
| Balanced | SDF and bounded convex pieces | droplets, debris, dense effects |
| High | flattened mesh BVH plus swept tests | fast debris and precision impacts |
| Particle | spatial hash plus PBD/XPBD constraints | granular particles and cohesive droplets |

### 8.2 Broadphase

The GPU implementation uses a uniform spatial hash:

1. Generate cell keys and particle indices.
2. Sort keys or use count/prefix/scatter depending on measured backend cost.
3. Build cell ranges.
4. Visit the 27 neighboring cells with an authored neighbor cap.
5. Record overflow counters rather than silently dropping contacts.

The algorithm chosen for Vulkan and CPU must produce equivalent contact sets
within the documented neighbor cap. The CPU sorted-grid implementation remains
the reference path.

### 8.3 Continuous collision detection

High-speed particles use `previous_position -> predicted_position` swept sphere
tests. Discrete penetration correction alone is not acceptable for explosion
debris because it tunnels through thin geometry. Restitution, static/dynamic
friction, adhesion and kill/stick/bounce response are common material/contact
properties, not collider-type-specific UI inventions.

### 8.4 Scene geometry

Collider acceleration structures are built from canonical flat scene geometry.
Static meshes reuse cached SDF/BVH data. Dynamic object transforms update
instance transforms without rebuilding local geometry. Deforming geometry has
an explicit recook/refit policy and telemetry; it must not silently rebuild a
large SDF every frame.

### 8.5 Animated characters and kinematic bodies

Characters are **not** a particle-system feature. Bone-attached collider proxies
are defined in `KINEMATIC_COLLIDER_SOURCES.md` as a solver-neutral source that
fluid, gas, granular and particles consume through the existing shared collider
path. This roadmap consumes them like any other rigid collider:

- per-proxy linear and angular velocity feed contact response and friction;
- proxies participate in `OnCollision` events (foot strike → mud clumps);
- the GPU collision snapshot (Phase 6) includes proxies with no special case.

The particle system must not grow its own skeleton sampling, bone lookup or
character-specific collider type.

### 8.6 Settling and hand-off

Particles that come to rest on a surface (dust, ash, mud clumps, snow) must not
stay alive forever. A **settle rule** (speed below threshold for N steps while
in contact) removes the particle and optionally writes it into a persistent
consumer: Surface State Deposit, a heightfield accumulation layer or a
granular/fluid domain. Settled-and-removed counts are telemetry, so pool growth
from resting particles is visible.

## 9. Events and sub-emitters

Supported event sources:

- OnBirth
- OnDeath
- OnCollision
- OnSpeedThreshold
- OnEnterDomain / OnExitDomain
- authored custom condition

GPU stages write compact event records into bounded append buffers. Every rule
has a per-particle cooldown, minimum impulse where applicable, maximum triggers
per particle and a global event budget. Overflow is counted and exposed.

Sub-emitters consume these event records in the next safe spawn stage. They do
not recursively modify the active particle array during collision dispatch.

**Deterministic ordering.** GPU append buffers filled with atomics have no
stable order. Before consumption, event records are sorted by
`(event_type, spawn_id, per-particle event index)`. Sub-emitter spawn seeds are
derived from that key, never from the record's position in the buffer. Without
this step the replay/scrub acceptance in Phase 7 passes or fails at random.
When the event budget overflows, the kept subset is chosen by the same key, so
which events survive is also deterministic.

Example:

```text
Debris OnCollision
  minimum speed: 3 m/s
  cooldown: 0.15 s
  spawn: 4-12 sparks + one dust puff + one surface splat
```

## 10. High-quality low-cost outputs

### 10.1 Flame and smoke

- texture-array flipbooks with adjacent-frame blending;
- blackbody-driven flame core and separate smoke/soot response;
- depth-aware soft intersections;
- GPU UV/noise distortion without changing simulation positions;
- stable seed-based variation;
- lit smoke impostors or equivalent bounded lighting approximation;
- weighted blended OIT or another measured order-independent solution where
  ordinary alpha sorting fails;
- motion vectors or a defined temporal strategy so animation does not shimmer;
- optional light-proxy and optional domain-deposit outputs.

### 10.2 Explosion

An explosion is an authored composite, not one oversized emitter:

- flash;
- expanding fireball;
- shockwave ribbon or mesh;
- smoke puffs;
- sparks;
- weighted multi-mesh debris;
- collision-driven secondary dust/sparks;
- optional gas-domain impulse, fuel and temperature deposit.

The composite shares one root transform and timeline control while keeping each
layer independently editable.

### 10.3 Water flow

- GPU ribbon/trail for the coherent stream;
- droplets created by breakup rules;
- GPU scene collision;
- collision splats and foam events;
- thickness-aware transmission, IOR and absorption;
- optional cohesion/self-collision quality mode;
- optional transfer into an APIC fluid domain when true liquid accumulation is
  required.

The ribbon is a render/output representation, not an alternate authoritative
physics state. Its control points are derived from particle or trail state.

### 10.4 Mud and snow (real-time tier)

Example: a character walks on a muddy road.

| Part | Owner |
|---|---|
| Footprint / ground giving way | Granular solver (high tier) or heightfield/MSF accumulation (real-time tier) — **not** particles |
| Clumps thrown on foot strike | This roadmap: `OnCollision` from a kinematic foot proxy → sub-emitter, Debris/Mud appearance, optional XPBD cohesion |
| Mud sticking to legs and shoes | This roadmap: Surface State Deposit into the character's MSF moisture/dirt channels |
| Clumps landing | Settle rule (§8.6) → deposit into the ground's surface state |

Snow is the same structure with a different appearance profile, lower
cohesion and a powder puff sub-emitter.

### 10.5 Volcanic eruption

| Part | Owner |
|---|---|
| Lava bombs (ejecta) | This roadmap: multi-mesh debris, `temperature` stream cooling over lifetime, blackbody → rock appearance, light proxy while hot, swept collision |
| Impact dust and sparks | This roadmap: `OnCollision` sub-emitters |
| Ash column | This roadmap: lit smoke impostors; optional density/temperature/momentum deposit into a gas domain for transport |
| Ash fall on the ground | Settle rule → Surface State Deposit |
| Steam where lava meets water | Field-masked surface source (§4.0) |
| Pyroclastic flow (dense ground-hugging current) | Gas domain — out of scope for particles |
| **Flowing lava** | APIC + substance chain (temperature-dependent viscosity, crust, solid phase) — **out of scope** for particles; a cheap heightfield lava flow is a separate, unplanned feature |

## 11. Core service, UI, scripting and IPC

The implementation must introduce one non-UI service layer for all mutations.
The exact names may be adjusted before Phase 1 freezes the contract, but the
required operation families are:

```text
particle.system.*
particle.emitter.*
particle.appearance.*
particle.geometry_set.*
particle.output.*
particle.collision.*
particle.event_rule.*
particle.backend.*
particle.stats / particle.capabilities
```

Python mirrors the same operations. Validation is performed once in the core
service and produces the same error semantics in UI, Python and IPC. New panels
must not mutate descriptors directly after the service exists.

Authoring remains viewport-first: compact system controls in the existing
Simulation context, contextual detail in the right dock, and timeline/event
editing in the bottom editor. This roadmap does not authorize a large permanent
particle shelf over the viewport.

## 12. Telemetry and validation

Required live statistics:

- selected/actual backend and fallback reason;
- alive, spawned, killed and recycled counts;
- per-stage CPU and GPU time;
- upload/download bytes and synchronization count;
- broadphase cells, maximum occupancy and overflow;
- contact and swept-hit counts;
- solver iterations and constraint residual;
- event count and event-buffer overflow;
- visible/cull counts per output;
- geometry-source distribution;
- NaN/Inf rejection count;
- domain deposits landed/dropped, preserving existing diagnostics;
- surface state deposits landed/dropped and settled-and-removed counts;
- ray-traced instance count per output against its budget, and how many
  particles were demoted to billboard/impostor by the budget.

Reference validation scenarios:

1. Ballistic free fall with no collision.
2. High-speed particles crossing a thin wall.
3. Sphere/capsule/OBB/SDF/mesh-BVH collision parity.
4. Dense self-collision pile with a fixed seed.
5. Backend switch at a known frame.
6. Forced GPU dispatch failure and CPU continuation.
7. Multi-geometry weighted distribution and slot recycling.
8. Save/load/save migration of legacy appearance fields.
9. Flame/smoke visual parity across Solid, RayFusion and supported Vulkan RT.
10. Explosion event-budget stress.
11. Water ribbon breakup, splat and optional APIC transfer.
12. Optional domain deposit with no visible carrier, visible carrier, and both
    outputs enabled together.
13. **Muddy walk (real-time tier).** Animated character with kinematic foot
    proxies on a surface with MSF. Measure over IPC: clump spawn events occur
    at foot strikes only (count ≈ steps × clumps per strike), leg MSF
    moisture/dirt rises below the knee and stays near zero above it, settled
    clumps leave no living particles after they rest.
    *Most insidious failure:* clumps spawn continuously while walking — the
    body is being seen as one rigid collider, not as feet.
14. **Field-masked source.** A plane with a painted MSF temperature patch;
    spawn positions lie inside the patch only, and rate follows the field.
15. **Deterministic events.** Same seed, two runs and one scrub-back: identical
    sorted event keys and identical sub-emitter spawn positions.
16. **RT instance budget.** Debris count above the budget: TLAS instance count
    stays at the budget, the excess renders as billboards, nothing disappears.

Performance gates are relative to Phase 0 measurements on the same reference
hardware. The roadmap does not invent hardware-independent millisecond claims.
GPU production status requires a meaningful speedup at representative counts
and no routine full-state readback; a kernel that runs quickly but synchronizes
and downloads every frame does not pass.

## 13. Implementation phases

Each phase is complete only when its observable acceptance gates pass. Code
existence or successful compilation alone is not completion.

### Phase 0 — Baseline, contracts and probes

**Deliverables**

- Record CPU and current partial-GPU performance for ballistic, scene-collision
  and self-collision scenarios at several particle counts.
- Add transfer/synchronization accounting to the particle stats surface.
- Add CPU/GPU state comparison and NaN/Inf probes.
- Freeze schema names, backend policy semantics and migration rules.

**Acceptance**

- A script can report which stages ran on which backend.
- Current GPU velocity download cost is visible rather than hidden inside an
  integration timer.
- Fixed-seed CPU baselines are stored for later phases.

### Phase 1 — Core authoring service and schema split

**Deliverables**

- Focused modules for particle system, emitter, profile, geometry-set, output,
  collision and event operations.
- Serialization versioning and stable ids.
- UI, Python and IPC routed through the same services.

**Acceptance**

- Every canonical mutation can be executed without opening the particle panel.
- Invalid references and ranges return the same error in Python and IPC.
- Save/load preserves ids and references.

### Phase 1.5 — Vertical slice: campfire

Purpose: prove the frozen contracts end-to-end on one visible scene before the
long parallel phases begin. A single person cannot validate contracts for
months without seeing them render; contract mistakes must surface here, not in
Phase 12.

**Deliverables** (thinnest possible version of each, no polish)

- One Appearance Profile per layer (Flame, Smoke, Spark) compiled to LUTs and
  read by the raster path (Phase 2 minimum).
- Output list with Billboard + Light Proxy (Phase 4 minimum).
- Ballistic spawn/age/kill/integrate resident on Vulkan for this scene only;
  the CPU path is still the reference (Phase 5 minimum).
- Script `scripts/tests/particle_campfire_slice.py` building the scene through
  the Phase 1 service only.

**Acceptance**

- The scene is built, rendered and saved/reloaded purely through IPC.
- Transfer telemetry shows no full-state download per frame for this scene.
- Any contract change the slice forces is written back into Phase 1's frozen
  schema **before** Phases 2–5 continue. The slice code is then either promoted
  into those phases or deleted; it is never kept as a second path.

### Phase 2 — Appearance profiles and legacy migration

**Deliverables**

- Curve/ramp assets and GPU LUT compilation.
- Flame, Smoke, Water, Spark, Dust, Debris and Custom shading models.
- Load-time migration from start/end appearance fields.
- Removal of per-step CPU RGB/opacity/size interpolation from the production
  render path.
- Removal of the start/end authoring UI and of every writer of the legacy
  fields, in this phase.

**Acceptance**

- An old project looks equivalent after migration within documented tolerance.
- Saving a migrated project writes no legacy appearance field.
- Two emitters can share one profile without duplicating material objects.
- Raster and Vulkan-supported RT consumers read the same profile definition.

### Phase 3 — Geometry sets and spawn-stable source selection

**Deliverables**

- Reusable multi-entry geometry sets.
- Per-emitter and per-output geometry-set references.
- Spawn-time weighted/sequential/shuffle selection stored in particle state.
- Source material preservation and explicit override modes.

**Acceptance**

- A recycled slot may select a new source, while a living particle never changes
  source unexpectedly.
- Measured source frequencies match weights within statistical tolerance.
- Deleting one source reports an unresolved entry without corrupting other
  sources.
- Geometry continues to originate from flat `TriangleMesh` / DNA SoA data.

### Phase 4 — Output foundation and GPU-driven presentation

**Deliverables**

- Composable output list and output masks.
- GPU visibility compaction, bounds reduction and indirect draw arguments.
- Billboard/flipbook, mesh-instance, primitive and light-proxy foundations.
- Direct Vulkan consumption of device-resident particle state.
- Ray-traced instance budget per output. Mesh particles reuse scene BLAS
  handles, but each is still one TLAS instance, and tens of thousands of
  instances updated every frame are a real TLAS cost. Past the budget (and by
  distance/screen size) particles demote to billboard/impostor instead of
  adding instances.

**Acceptance**

- One emitter can drive mesh, billboard and light outputs simultaneously.
- Exceeding the RT instance budget demotes, it does not drop particles or grow
  the TLAS unboundedly.
- Hidden/disabled outputs do not destroy stable pools or force repeated
  structural rebuilds.
- Normal presentation does not download the full particle state.

### Phase 5 — GPU-resident core simulation

**Deliverables**

- GPU free list, spawn, lifetime, kill and stable spawn ids.
- Forces, integration, angular state and device-resident commit.
- Three-state residency tracking and explicit backend transitions.
- CPU implementation using the same contracts.

**Acceptance**

- Ballistic systems remain on device from spawn through rendering.
- CPU and Vulkan trajectories stay within the Phase 0 tolerance.
- Forced backend fallback resumes without particle duplication or time loss.
- There is no per-frame velocity or position download in the healthy GPU path.

### Phase 6 — GPU collision and constraints

**Deliverables**

- GPU primitive, SDF and mesh-BVH collision snapshots.
- Swept sphere/continuous collision for high-speed particles.
- GPU spatial hash and PBD/XPBD self-collision.
- Shared contact material and response rules.
- Kinematic collider proxies (`KINEMATIC_COLLIDER_SOURCES.md`) included in the
  GPU collision snapshot as ordinary rigid colliders.
- Settle rule and settled-and-removed telemetry (§8.6).

**Acceptance**

- Thin-wall tunneling test passes at the documented speed/radius range.
- Contact overflow is reported and never silently corrupts memory.
- CPU/Vulkan contact outcomes match within scenario-specific tolerances.
- Dense collision is meaningfully faster on GPU than the CPU reference on the
  same reference hardware.

### Phase 7 — Events, sub-emitters and trails

**Deliverables**

- Bounded GPU event buffers and event rules.
- Event key sort before consumption and key-derived sub-emitter seeds (§9).
- Safe next-stage sub-emitter consumption.
- Trail ids/history and GPU ribbon generation.

**Acceptance**

- Collision-triggered sparks and splats obey threshold, cooldown and budgets.
- Event overflow is observable.
- Scrubbing or replay with a fixed seed reproduces event counts and ordering
  within the documented deterministic contract.

### Phase 8 — Production flame and smoke

**Deliverables**

- Flipbook interpolation, soft particles, blackbody flame, lit smoke,
  distortion, temporal strategy and scalable LOD.
- Optional light and domain-deposit outputs.

**Acceptance**

- A convincing close-up flame and smoke effect runs without any authored
  domain.
- Enabling a gas domain adds transport fidelity without replacing or duplicating
  the emitter.
- Panel intersections, depth edges and camera motion do not produce hard cards
  or unstable flicker under the supported quality profile.

### Phase 9 — Explosion composite

**Deliverables**

- Composite effect asset with flash, fireball, shockwave, smoke, sparks and
  weighted multi-mesh debris.
- Collision-driven secondary effects and optional domain impulse/deposit.

**Acceptance**

- Layers are independently editable but controlled by one root timeline event.
- Event and particle budgets degrade through LOD rather than failing abruptly.
- Debris preserves source geometry/material policy and passes swept collision.

### Phase 10 — Water stream, droplets, splats and foam

**Deliverables**

- GPU ribbon stream, breakup rules, droplet collision, surface splats and foam.
- Water appearance model with thickness-aware absorption/transmission.
- Optional cohesive/self-collision quality and APIC transfer output.

**Acceptance**

- A high-quality tap/hose stream is possible without a fluid domain.
- Droplets collide and create bounded splat/foam events.
- Enabling APIC transfer allows accumulation without changing the authored
  emitter or appearance profile.

### Phase 11 — Domain interoperability

**Deliverables**

- Domain Deposit as a first-class output with per-emitter/profile overrides.
- Explicit density, temperature, fuel, momentum and fluid-transfer channels.
- Common diagnostics for landed/dropped deposits.

**Acceptance**

- Visible output, carrier-only output and combined output all work from one
  emitter definition.
- Domain absence is valid and does not generate errors unless the user marks the
  link required.
- Required links fail clearly when their target domain is missing.

### Phase 11b — Surface state and field-masked sources

**Deliverables**

- Surface State Deposit output into MSF moisture/heat/dirt channels.
- Field-masked surface emitter source (§4.0).
- Settle rule hand-off into surface state (§8.6).

**Acceptance**

- Validation scenarios 13 (muddy walk) and 14 (field-masked source) pass.
- A deposit onto an object without MSF reports "no surface state", it does
  not silently succeed.
- Deposited state survives save/reload and timeline replay; its decay/drying
  belongs to the MSF, not to particle code.

### Phase 12 — Production hardening

**Deliverables**

- LOD, distance budgets, output culling and memory budgets.
- Long-run stability, cache, save/load and backend-failure tests.
- Documentation, examples and replacement of WIP labels.

**Acceptance**

- Long simulations produce no unbounded pool, event or material growth.
- GPU/CPU capability and fallback status are visible in UI, Python and IPC.
- All reference scenarios pass after save/reload and timeline replay.
- Legacy projects still load (the legacy UI and writers were already removed in
  Phase 2).

## 14. Phase dependency graph

```text
Phase 0
  -> Phase 1
      -> Phase 1.5 (campfire slice; may amend Phase 1 contracts)
          -> Phase 2 -> Phase 4 -> Phase 8 -> Phase 9
          -> Phase 3 -> Phase 4 -----------^    |
          -> Phase 5 -> Phase 6 -> Phase 7 -----+
                                  |              |
                                  +-> Phase 10 <-+
Phase 5 + Phase 4 -> Phase 11
Phase 6 + Phase 7 + Phase 11 + MSF -> Phase 11b
KINEMATIC_COLLIDER_SOURCES (separate note) -> Phase 6 proxies, scenario 13
All phases -> Phase 12
```

Phase 2 and Phase 3 may proceed in parallel after Phase 1.5 has confirmed the
contracts.
Phase 5 must not be declared complete by moving only the existing force kernel;
residency through render consumption is the gate. Phase 8 may prototype visuals
before Phase 6, but production fire/explosion/water acceptance waits for the
required collision and event phases.

## 15. File and ownership strategy

`ParticleSimulation.cpp`, `scene_ui_forcefield.hpp`, `scene_ui_gizmos.cpp` and
`scene_data.h` are already large. They may receive minimal integration wiring,
but new implementation belongs in focused modules such as:

```text
ParticleCoreService
ParticleExecutionPolicy
ParticleGpuPipeline
ParticleGpuCollision
ParticleAppearanceProfile
ParticleGeometrySet
ParticleOutputGraph
ParticleEventSystem
ParticleBillboardOutput
ParticleRibbonOutput
ParticleDomainOutput
```

Names are illustrative; ownership boundaries are mandatory. GPU ABI structs
must have static size assertions and one documented binding table per kernel.
Generated shader binaries are build artifacts and are verified only after the
user performs the project build/shader compilation.

## 16. Explicit non-goals

- Replacing the existing gas or APIC fluid domain solvers.
- Making every cheap flame physically simulate combustion.
- Creating one material per particle, color bucket or lifetime sample.
- Using ImGui drawing as production particle rendering.
- Making `Triangle` facade collections authoritative scene geometry.
- Expanding the frozen OptiX feature set.
- Hiding GPU failures behind an unreported CPU path.
- Calling a synchronized per-frame upload/dispatch/download loop
  "GPU-resident."
- Owning skeleton sampling or character colliders. Those live in
  `KINEMATIC_COLLIDER_SOURCES.md` and serve every solver.
- Ground deformation and footprints. They are granular-solver output (high
  tier) or heightfield/MSF accumulation (real-time tier), never particle state.
- Flowing lava, pyroclastic flows or any other continuum that needs pressure,
  viscosity or phase change (§10.5).
- Keeping legacy appearance fields as a second live path after Phase 2.

## 17. First implementation milestone

The first implementation batch is Phase 0 plus the contract portion of Phase 1:

1. Add honest particle backend/transfer telemetry.
2. Freeze stable ids and service-level descriptor types for Appearance Profile,
   Geometry Set, Output and Execution Policy.
3. Expose read-only capability/state inspection through Python and IPC.
4. Record CPU and current partial-GPU baselines before changing execution.

The second batch is the Phase 1.5 campfire slice. It is the first point where
the contracts are judged by a rendered image rather than by their text.

Do not begin by adding another preset. Presets become small compositions of the
new core assets after the contracts exist.
