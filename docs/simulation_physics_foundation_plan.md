# RayTrophi Simulation Physics Foundation Plan

## Direction

RayTrophi physics will be a custom CPU/GPU compute simulation stack, not a wrapper around a third party physics engine. The core idea is field-first simulation: force fields, surface flow, gas, hair, wet paint, terrain, particles, cloth, and future rigid bodies all read from the same scene-level simulation context.

## Core Principle

CPU is the reference and debugging path. GPU is the production path.

Every solver should keep a CPU implementation for correctness, deterministic inspection, and artist debugging. CUDA kernels should mirror the CPU data model closely enough that a module can be compared step by step.

The public simulation layer must stay backend independent. Systems talk to `SimulationComputeContext`; CUDA, Vulkan compute, or any future backend sits below that API. OptiX remains a render/ray-query acceleration path, not the general simulation compute contract.

## M1 - Central Simulation World

Goal: create the shared scheduler and context without changing existing solver behavior yet.

- Add `SimulationWorld`
- Add `SimulationContext`
- Add `ISimulationSystem`
- Store current frame, time, fixed timestep, accumulator, substeps, and backend
- Expose scene `ForceFieldManager` through the context
- Expose backend-independent `SimulationComputeContext` through the context - started
- Upload packed force-field snapshots into a shared compute buffer - started
- Keep gas and hair on their current paths while the central API settles

Acceptance:

- SceneData owns one `SimulationWorld`
- Systems can register with deterministic order
- `advance()` produces stable substeps
- Systems can allocate/upload/download compute buffers through one API
- Systems can read one shared packed force-field compute buffer
- The context can be passed to gas, hair, wet, particle, cloth, and rigid modules later

## M2 - Unified Force Field Snapshot

Goal: stop every solver from uploading/evaluating force fields in its own shape.

- Build a CPU snapshot of enabled force fields once per simulation frame - started
- Build a GPU-friendly packed snapshot once per simulation frame - started
- Upload the packed snapshot through the backend-independent compute API - started
- Track dirty state when force fields change
- Let gas and hair consume the same snapshot - gas CPU/CUDA upload path started, hair snapshot path started
- Add affect masks for gas, hair, particles, cloth, rigid, wet, terrain

Acceptance:

- Gas consumes the central force-field snapshot before falling back to legacy manager data
- Hair samples the same packed field data
- CPU and CUDA evaluation agree on core force types

## M3 - Surface Flow Foundation

Goal: lift wet brush downhill flow into a reusable physics module.

- Extract mesh triangle to tangent/UV flow cache - started
- Store slope, tangent flow, flow length, and seam links - flow sample started, seam links remain in wet cache
- Add `SurfaceFlowField` as a reusable physics asset - started
- Let wet brush use the shared surface-flow module - started
- Add force-field projected flow: `project(gravity + forceField, surfaceTangent)` - started

Acceptance:

- Wet paint behavior stays intact
- Same surface-flow cache can serve rain, mud, erosion, particles, and cloth sliding
- Scene force fields can steer wet surface runoff

## M4 - GPU Particle Solver

Goal: create the simplest high-volume testbed for the new compute stack.

- Add SoA particle buffers - started
- Add CPU reference integration - started
- Connect particle system to `SceneData` and `SimulationWorld` - started
- Add CUDA integration
- Sample force fields - started
- Add lifetime, spawn, drag, collision plane, and debug draw - lifetime/spawn/drag/collision/debug overlay started
- Add emitter source modes for point, force field, and object origins - started
- Add emitter spawn modes for center and object AABB surface - started
- Add mesh surface emitter sampling with area-weighted triangle spawn - started
- Add collider list for debug planes and selected mesh AABB colliders - started
- Add object OBB collider mode for rotated mesh collision - started
- Add explicit sphere collider mode for fast object/manual collision volumes - started
- Add capsule collider mode for character/strand-friendly collision volumes - started
- Add selected-object primitive proxy generation and fit workflow - started
- Serialize particle settings, emitter descriptors, and collider descriptors with project files - started
- Add scene-level Particle System object model with hierarchy selection and multiple-system foundation - started
- Add particle physics profiles for spark, granular, fluid, and gas solver evolution - started
- Add realtime/preview/offline particle quality profiles for game and DCC render paths - started
- Add CPU spatial hash neighbor grid for particle self-collision and future fluid/gas domain coupling - started
- Add swept particle-vs-thin-collider crossing tests to reduce tunneling without forcing large thickness - started
- Add shared `SurfaceMeshCache` foundation for wet brush, surface emitters, and future mesh proxy colliders - started
- Add simulation grid domain descriptors with object-bound box domains for future gas/fluid solvers - started
- Add non-renderable, transformable, selectable domain proxy workflow - started
- Add domain flow sources that inject density/temperature/fuel/velocity into grid domains - started
- Add optional bake cache

Acceptance:

- 100k+ particles run on CUDA
- CPU and GPU paths produce comparable motion for simple fields
- Particles can be driven by scene force fields

## M5 - Hair/Cloth Solver Unification

Goal: generalize strand and constraint concepts.

- Share constraint iteration concepts between hair and cloth
- Add GPU strand/cloth buffer layout
- Add pin/root constraints
- Add bend/stretch damping
- Add collision hooks
- Add simulation cache for DCC timeline

Acceptance:

- Hair remains compatible with existing groom workflow
- Cloth can reuse the same scheduler, force fields, and GPU backend pattern

## M6 - Rigid Body Core

Goal: build our own rigid foundation after field/scheduler/gpu paths are proven.

- Rigid body state buffers
- Sphere, box, capsule, plane colliders
- Broadphase grid or sweep-and-prune
- Contact manifold
- Sequential impulse solver
- Force field acceleration
- CPU first, CUDA broadphase/solve later

Acceptance:

- Dynamic bodies respond to scene force fields
- Basic stacking and collisions are stable at fixed timestep
- Scene transforms can be written back after solve

## M7 - Bake, Cache, and Timeline

Goal: make the same simulation useful for game runtime and DCC production.

- Frame-addressable simulation cache
- Rewind/scrub support
- Partial invalidation when fields or emitters change
- Export/import cache metadata
- Per-system bake policies

Acceptance:

- Gas, particles, wet flow, hair, and cloth can be baked and replayed
- Timeline scrubbing does not require realtime resimulation unless requested

## Immediate Implementation Order

1. Add `SimulationWorld` and `SimulationContext`
2. Connect `SceneData` to own the central world
3. Add a force-field pointer into the context
4. Add minimal system registration and ordered stepping
5. Move gas update behind a simulation-system adapter
6. Move hair restyle/sim hooks behind the same context
7. Extract wet surface flow into `SurfaceFlowField`
8. Build the first shared GPU force-field snapshot

## Particle/Domain Optimization Notes

Status snapshot:

- 4k particles with one object collider, ground collider, and moving attractor force field were usable on CPU.
- Before collider resolve caching, object collider integration cost was around 30 ms.
- After frame-level resolved collider caching, the same case dropped to around 8 ms integrate / 9 ms total step with self collision off.
- With self collision on, total step was around 21 ms, with self collision around 11 ms.
- Emit and upload are currently cheap in this test case: roughly 0.1 ms emit and 0.05 ms upload.

Completed optimization:

- Object-bound collider resolution was moved out of the per-particle loop.
- A per-frame `resolved_colliders_` cache now resolves object AABB/OBB/sphere/capsule data once per collider before integration.
- Particle collision then uses cached resolved collider data for all particles in that frame.
- This remains transform-correct because the cache is rebuilt at the start of each simulation step.

Next optimization targets, later:

1. Add cheap collider broadphase before expensive swept OBB/capsule tests.
   Skip particles that are clearly outside a collider's expanded world AABB.
2. Split integrate stats into force/integrate/collider/cache timings.
   The current `Integrate` number includes collider cache refresh and collision solve.
3. Split self-collision stats into neighbor-grid build/sort and pair solve timings.
4. Optimize self collision after domain work resumes:
   use cheaper bucket traversal, reduce rebuilds where possible, then consider CPU parallel chunking.
5. Keep domain/gas work as the main line for now:
   standalone/bound domain object workflow, transparent/wire domain gizmo, particle-to-grid, then gas solver migration.

## Domain Flow Source Roadmap

Goal: make grid domains useful as gas/fluid containers without requiring particles first. A domain flow source is a non-render simulation source that writes directly into a selected grid domain.

First implementation:

- Add `SimulationFlowSourceDesc` data model owned by particle/grid simulation.
- Attach flow sources to domains by domain index for now.
- Add object-based flow source creation from the selected mesh bounds.
- Inject density, temperature, fuel, and velocity into the domain grid each simulation step.
- Keep the source independent from render visibility and backend geometry.

Next steps:

1. Add UI controls under the selected domain for source list, strength, radius, temperature, fuel, and velocity.
2. Support source types: point, object bounds center, object surface, and future mesh volume.
3. Add viewport gizmo/selection for flow sources after the domain proxy is stable.
4. Move legacy gas solver math onto this flow-source/grid-domain path (see Grid Fluid Solver Migration).
5. Add GPU compute kernels mirroring the CPU injection path.

## Grid Fluid Solver Migration (from legacy gas)

Goal: turn grid domains into a real Blender-like gas/fluid solver driven by flow sources and particles, reusing the *math* from the legacy `GasSimulator` while dropping its problem areas. Once this path is solid, the gas system in the VDB panel will be removed.

What was wrong with the legacy gas system:

- Runaway combustion. The whole domain ignited almost immediately. Root causes: heat-release fed back into temperature everywhere, and a neighbor-flame term spread fire cell-to-cell like a cellular automaton. `processCombustion` ended up as a stack of ad-hoc gates (pilot/autoignite/flame-memory/oxygen-proxy/throttle) that never actually fixed the feedback loop.
- Emitter system was too simple and is now redundant. The new architecture already has particles plus flow sources (point, object bounds, and mesh), so emission does not need the legacy emitter at all.

Migration decisions:

- Reuse `FluidSim::FluidGrid` as-is (MAC staggered grid, trilinear sampling, sparse 8^3 tiles, world<->grid conversion). Replace the collocated SoA in `SimulationGridDomainState` with it.
- Lift the solver *logic* but rewrite it as a backend-independent `GridFluidSolver` module taking `(FluidGrid&, settings, force snapshot)` as free functions: advection (SemiLagrangian/MacCormack/BFECC), buoyancy, vorticity, curl-noise turbulence, pressure solve + projection, dissipation, boundaries, CFL adaptive timestep.
- Drop the legacy `Emitter` struct entirely. Emission comes from flow sources and particle-to-grid splat.
- Drop `processCombustion` entirely. Do not port it.
- Consume the shared `SimulationForceFieldSnapshot` already present on `SimulationContext` instead of the gas system's own GPU force-field upload.

Combustion (fire), rewritten and opt-in:

- Build pure smoke/gas first (advect + buoyancy + pressure + dissipation). Fire is OFF by default, so "everything ignites" is impossible by construction.
- No neighbor-flame spreading. Ignition only happens where fuel exists and is heated by the source/incoming hot air; the flame field is advected with the flow, never propagated cell-to-cell.
- Bounded one-way heat release: `burned = burn_rate * fuel * dt`, `temperature += heat_release * burned` (clamped to max_temperature), `smoke += smoke_gen * burned`. No gate stack.
- Fuel is an advected scalar injected by flow sources; ignition is driven by the source temperature, not the domain's own accumulated heat.

Solid CPU first, GPU compute second (matches Core Principle):

- Stage 1: full CPU reference solver on grid domains, deterministic and debuggable.
- Stage 2: CUDA/Vulkan compute kernels mirroring the CPU data model step by step, through `SimulationComputeContext`, validated against the CPU path.

Implementation order:

1. `SimulationGridDomainState` -> `FluidGrid` - done. Domain state now owns a `FluidSim::FluidGrid` (MAC staggered). Flow-source injection, particle splat, decay, and gas buoyancy write to the grid (scalars cell-centered, velocity on faces). Channel mask now gates use, not allocation. Hand-rolled vertical advection removed; gas only buoys until step 2.
2. `GridFluidSolver` module: inject -> advect -> buoyancy -> force fields -> project -> dissipate (no fire) - done. New `RayTrophiSim::GridFluid` namespace (`GridFluidSolver.h/.cpp`), backend-independent CPU reference. Stages: semi-Lagrangian advection (velocity + density/temperature/fuel), wall BCs (Open/Closed/Periodic), buoyancy along -gravity, force-field accel, vorticity confinement, per-channel dissipation, velocity clamp, SOR pressure projection (warm-started, boundary-aware). MacCormack/BFECC + CFL substepping remain follow-ups. Our own math, no GPL Mantaflow code.
3. `stepGridDomains` calls the solver instead of the decay-only path - done. Order: synchronize -> inject sources -> particle splat -> per-domain `GridFluid::step`. Idle domains (no content, no source, no particles) are skipped.
4. Wire the force-field snapshot into the solver - done. `SimulationContext::force_snapshot` passed through and evaluated per cell as `SimulationSystemKind::Gas`.
5. Realtime render bridge - done. Each grid domain with content is mirrored as a transient (never serialized, hidden from hierarchy) `VDBVolume` hittable in `vdb_volumes` + `world.objects`, bound via `bindLiveVolume` to a live NanoVDB volume rebuilt each step from the FluidGrid density (`VDBVolumeManager::registerOrUpdateLiveVolume`). This reuses the existing VDB render path (TLAS instance + volume pass) on OptiX/Vulkan/viewport with **no backend edits**. Bridge lives in `SceneData::syncSimulationRenderVolumes` (called after `stepOnce`), keeping the sim layer render-agnostic. Density-only smoke for now (temperature/blackbody when fire lands). Important: volumes MUST be `VDBVolume`/`GasVolume` hittables in `world.objects` — both backends build volume TLAS instances by scanning `world.objects` and match the SSBO by `vdb_id`; a side render list does not work.
6. Opt-in fire reaction with the clean rules above - done. `GridFluid::processCombustion` runs after advection, before buoyancy: where `fuel > 0` and `temperature >= ignition`, burn a bounded fraction (`burn_rate * fuel * dt`), add `heat_release * burned` to temperature (capped at `max_temperature`), add `smoke_generation * burned` to density, and light the `interaction` flame field (decays each step). NO neighbor-flame spread — fire moves only by advecting hot temperature into fuel, so the legacy "whole domain ignites" runaway is impossible. Per-domain fire settings live on `SimulationGridDomainDesc` (serialized) with a "Fire / Combustion" panel that also turns on the Fuel + Temperature channels. The render bridge uploads a Kelvin-scaled temperature grid when the domain shader uses blackbody/channel-driven emission, so fire color works through the existing volume shader. Default OFF.
7. Timeline integration (frame stepping + scrub) and bake/cache - done (memory cache). Driver `SceneData::updateSimulationTimeline(frame, playing, dt, fps)` called each tick: **free-run** interactive preview by default (`sim_timeline_frame_ < 0`); **playing** the timeline switches to a deterministic bake from frame 0 with fixed dt (1/fps), caching each frame's grid state in memory; **scrub** restores from the cache instantly, or resimulates the gap from the nearest cached frame (capped at 8 steps/tick so big jumps don't hang). "Reset Simulation (Free-run)" button clears the cache and returns to live preview. Cache invalidated on project clear and on particle-system add/remove. Particles are not cached yet (grid/gas only) — fine for flow-source-driven gas; mixed particle→grid scenes desync particles on scrub-back. Disk bake / VDB sequence is the follow-up.
8. VDB export - done. `VDBVolumeManager::exportDenseGridToVDB` writes a dense FluidGrid (density/temperature/fuel/flame, world transform = origin+voxel) to a `.vdb`. `SceneData::exportDomainVDB` writes the selected domain's current frame; `exportDomainVDBSequence` deterministically bakes 0..end and writes start..end as `base_####.vdb` (blocking, returns to free-run after). UI "VDB Export" panel: directory + base name + Export Current Frame / Export Sequence. Sequence is synchronous for now (threaded bake is a follow-up).
9. GPU compute parity (CUDA kernels mirroring the CPU solver) - in progress (approach B: through `SimulationComputeContext`). Phased because GPU kernels can't be validated without on-machine builds:
   - **Phase 1 - done**: extended the compute API with real kernel dispatch — `ComputeDispatch` (kernel name + groups + bound buffers + push-constants), `ISimulationComputeBackend::dispatch/supportsDispatch/nativeBufferPtr`, and `SimulationComputeContext` wrappers. CPU backend leaves dispatch unimplemented (CPU solver keeps its direct path).
   - **Phase 2 - done**: CUDA compute backend `CudaSimulationComputeBackend` (`source/src/Device/SimulationComputeCuda.cu`) — `cudaMalloc`/`cudaMemcpy` buffers, `nativeBufferPtr` (device ptr), `supportsDispatch()=true`, a `dispatch` registry keyed by `cmd.kernel`, and a `sim_scale` validation kernel. `selfTestCudaSimulationCompute()` runs an alloc→upload→dispatch→download round-trip and logs OK/FAILED (called once from `SceneData::syncSimulationWorld`). The global backend is NOT switched yet (no behavior change) — installation + grid integration is Phase 3. Added to the build as a `CudaCompile` item.
   - **Phase 3**: keep each grid domain's channels in compute buffers; add a GPU path in `GridFluidSolver`/`stepGridDomains` that dispatches the stages (advect, buoyancy, force fields, vorticity, dissipation, divergence, SOR pressure, subtract-gradient, combustion) mirroring the CPU math; download density (+temperature) for the render bridge. CPU path stays as the reference.
   - **Phase 4**: validate CPU vs GPU agreement and tune.
10. Remove the gas system from the VDB panel once parity is reached.

### Render bridge — fixes landed (validated by user)

- **Vulkan RT frozen smoke** — Vulkan re-uploads the live grid buffer only when `g_gas_volumes_dirty` is set (Main.cpp gates the per-frame `updateVDBVolumes`). The bridge now sets it on every content change; OptiX was unaffected (reads the CUDA device grid directly).
- **Simulation drive mode** — global `g_sim_timeline_mode` (default **Timeline**) + a "Simulation Mode" selector in the domain panel. Timeline = play bakes into the memory cache, scrub restores, a stopped timeline is frozen/idle (cheap — render converges, loop sleeps). Live Update = continuous free-run interactive preview (heavier; always simulating + resetting accumulation). Accumulation reset / `start_render` only fire while the gas is actually changing (`simulation_render_updated`), so a frozen Timeline costs nothing. (Replaced the earlier `g_sim_live_render_update` toggle.)
- **OptiX hang on emitter edits** — the bridge does CUDA work (`registerOrUpdateLiveVolume`) + mutates `world.objects`; doing this while a backend rebuild is in flight poisoned the CUDA context. `syncSimulationRenderVolumes` now early-outs while `g_optix_rebuild_in_progress` / `g_viewport_rebuild_in_progress`.
- **Resolution / perf** — realtime cell budget dropped 2,000,000 -> 512,000 (full solver runs every cell each step); default `max_auto_resolution` 256 -> 128; the Resolution slider now also derives `voxel_size` so it is honored under Preserve Voxel Size.
- **Gas-only domains were frozen (no particles)** — `ParticleSimulationSystem::enabled()` returned false without particles/emitters, so `SimulationWorld` skipped the step and the redraw loop slept. It now also returns true when a grid domain has an enabled flow source or residual density (`hasActiveGridSimulation()`), so flow-driven gas advances and renders without any particles.

### Known limitations / open items

- CPU render path shows no live volume — `registerOrUpdateLiveVolume` keeps only a NanoVDB handle, not an OpenVDB grid, so `sampleDensityCPU` returns 0 (predates migration; affects legacy gas too).
- Deleting a domain while in render mode can stall the solid-mode switch (CUDA free + rebuild around an active render); mitigated, not fully root-caused.
- Flow sources with density 0 (fuel/temperature only) produce no visible smoke until the opt-in fire/combustion stage lands.

### Per-domain volume shader (migrated from the VDB/gas panel)

- Each `SimulationGridDomainDesc` now carries its own `std::shared_ptr<VolumeShader>` (host material data; created lazily, defaults to `createSmokePreset()`). It travels with the domain, so it is index-safe across add/remove and serialized with the domain (`serializeDomain`/`parseDomain` in ProjectManager).
- The render bridge binds `domains[d].shader` to the domain's live `VDBVolume` each frame, so edits apply live (the renderer re-reads the shader every volume sync).
- The domain panel shows a "Volume Shader" section reusing `SceneUI::drawVolumeShaderUI` (the same editor the VDB/gas panels use). Density/scatter/absorption/quality work now; emission/blackbody/color-ramp need the temperature grid, which arrives with fire.

### Where we are

Steps 1–8 of the migration plus per-domain shaders are done and working in OptiX and Vulkan RT viewport + render — grid fluid solver, fire/combustion, per-domain shaders, timeline bake/scrub (memory cache), and VDB export (single frame + sequence). The new path now has feature parity with the legacy gas for smoke/fire authoring. Next: **GPU compute parity** (CUDA/Vulkan kernels mirroring the CPU solver through `SimulationComputeContext`), then **remove the legacy gas system from the VDB panel**. Optional polish: threaded sequence bake, particle-aware timeline cache, timeline fps wiring (currently fixed 24).
