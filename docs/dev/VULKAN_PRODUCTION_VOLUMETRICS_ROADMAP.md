# Vulkan Production Volumetrics Roadmap

> **Durum:** AKTIF — Uretim hacim yolunun urun sozlesmesi ve kapilari.

## Product contract

RayTrophi's supported volumetric path is:

`Simulation Domain -> Vulkan Compute -> GPU-resident fields -> Vulkan RT`

The retired `GasVolume/GasSimulator` runtime is not a fallback. CPU execution is
kept only as a correctness reference and an explicit diagnostic mode.

Every production feature must satisfy five gates:

1. **Safety:** bounded allocation, validated dispatch dimensions, deterministic
   fallback, no unsafe resource replacement while RT is reading.
2. **Cost:** explicit VRAM, step, shadow and frame-time budgets.
3. **Quality:** reference scenes and measurable image/solver acceptance criteria.
4. **Speed:** no avoidable CPU/GPU round trips on the live path.
5. **Usability:** Interactive, Preview and Final profiles give coherent defaults;
   advanced controls remain available under Custom.

## User model

The normal workflow exposes five profiles:

| Profile | Purpose | Grid ceiling | Default domain budget | Memory strategy |
|---|---|---:|---:|---|
| Interactive | authoring and playback | 96 | 512 MB | RAM cache |
| Preview | lighting and motion review | 192 | 1 GB | RAM cache |
| Final | production bake/render | 512 | Dynamic (~25% System RAM, 4-16 GB) | RAM cache |
| Cinema | offline film-grade render | 1024 | Unlimited (RAM limit lifted) | Disk bake mandatory (`.volcache`) |
| Custom | expert control | 2048 (manual) | manual | manual |

Profiles are starting points, not hidden modes. The UI always displays effective
resolution, estimated memory, active backend, fallback reason and measured GPU
time. Manual edits change the profile to Custom. Grids exceeding 512 per axis automatically enforce sparse tile culling.

## Phase 0 — safety and budget foundation

Status: **in progress**

- [x] Retire the legacy gas runtime without coupling cleanup to production work.
- [x] Add domain quality profiles.
- [x] Add a serialized per-domain memory budget.
- [x] Enforce the budget before grid allocation.
- [x] Make managed profiles prefer Vulkan Compute.
- [ ] Query Vulkan memory budgets with `VK_EXT_memory_budget`.
- [ ] Reserve headroom for renderer/TLAS/textures; never consume the full heap.
- [ ] Overflow-safe byte calculations for every buffer/image allocation.
- [ ] Report requested versus effective dimensions and the exact clamp reason.
- [ ] Add dispatch, NaN/Inf and field-range validation counters.
- [ ] Add graceful domain pause on allocation/dispatch failure.

Acceptance:

- No profile can request an unbounded allocation.
- Allocation failure pauses only the affected domain and leaves the scene usable.
- Effective memory stays below the configured domain budget and available VRAM.

## Phase 1 — fully GPU-resident gas timestep

- Persistent ping-pong velocity, density, temperature, fuel and reaction fields.
- Persistent pressure, divergence and solver scratch.
- GPU source injection, buoyancy, force fields, turbulence and combustion.
- One command-buffer timestep with explicit compute barriers.
- Remove per-stage upload, synchronize and download operations.
- Read back only compact statistics and explicit cache/export requests.
- Publish simulation fields directly to Vulkan RT.

Acceptance:

- Normal playback performs no full-grid CPU readback.
- A timestep has a bounded number of submissions, targeting one submission.
- GPU and CPU reference fields pass impulse, plume and conservation tests.

## Phase 2 — collider and pressure production path

- GPU collider SDF and dirty-region updates.
- Fractional MAC-face weights and moving-solid face velocity.
- Solid-aware divergence, Poisson stencil and gradient subtraction.
- Thermal-expansion divergence target on GPU.
- Swept fast-collider coverage.
- Gas MGPCG/multigrid pressure solver with residual diagnostics.

Acceptance:

- Enabling a collider or fire expansion never causes CPU fallback.
- Moving-collider momentum transfer is stable and leak-free.
- Pressure residual reaches the profile threshold without fixed-iteration guessing.

## Phase 3 — high-order transport and upres

- RK2/RK3 characteristic tracing.
- MacCormack/BFECC scalar and velocity advection with monotonic limiter.
- Collider-aware backtrace clipping.
- Vorticity/curl diagnostic fields.
- Velocity-guided render upres and wavelet turbulence.
- Density erosion driven by age, vorticity and temperature.

Acceptance:

- Detail retention visibly exceeds first-order semi-Lagrangian at equal resolution.
- No new extrema, negative density or collider tunnelling.
- Upres remains temporally attached to simulated motion.

## Phase 4 — production pyro fields

- First-class density, Kelvin temperature, fuel, reaction/flame, soot, age and
  velocity fields.
- Stable combustion and cooling model.
- Soot generation and absorption coupling.
- Artist ramps that do not alter solver stability.
- Sparse cache/export preserving all field semantics.

Acceptance:

- Fire emission never falls back to density when a valid temperature field exists.
- Smoke extinction and flame emission can be authored independently.
- Missing fields produce deterministic, physically safe results.

## Phase 5 — Vulkan RT reference integrator

- Canonical full-interval ray coverage.
- Direction-aware voxel footprint under anisotropic transforms.
- Separate primary, shadow and hierarchy traversal budgets.
- Deterministic Beer-Lambert extinction/emission integration.
- Dual-lobe phase function with energy bounds.
- Energy-conserving multiple-scattering octave approximation.
- Geometry visibility and importance-sampled local lights.
- Deep sun-transmittance cache for large volumes.

Acceptance:

- Uniform-medium images match analytic transmittance.
- Camera direction and anisotropic scale do not change integrated opacity.
- Quality profiles change variance/cost without changing the expected radiance.

## Phase 6 — sparse traversal and temporal stability

- GPU active-brick mask and density min/max/majorant hierarchy.
- Empty-space skipping for primary and shadow rays.
- Blue-noise, frame-decorrelated march jitter.
- Velocity-based reprojection.
- Volume-aware history rejection and disocclusion handling.
- Stable shadow history.

Acceptance:

- Empty regions consume traversal work, not density-sample budget.
- Animated reference scenes show no structured flicker or persistent ghost trails.
- Sparse scenes scale with active bricks rather than full domain volume.

## Phase 7 — material authoring and bake

- Spatial volume graph VM: mapping, noise, ramp, math and field sampling.
- Instruction/register/noise-octave budgets.
- Graph-aware conservative occupancy.
- Bake graph fields to sparse GPU/render cache.
- Live/baked parity testing.
- Production presets: smoke, fire, explosion, cumulus, storm and ash.

Acceptance:

- Uniform graphs add effectively zero per-step cost.
- Spatial graph cost is visible and bounded.
- Presets behave consistently across domain scale and voxel resolution.

## Instrumentation

Every domain will expose:

- requested/effective resolution;
- estimated/allocated CPU and GPU bytes;
- active backend and fallback reason;
- timestep, advection, pressure and cache GPU milliseconds;
- pressure iterations and final residual;
- active bricks and skipped distance;
- primary/shadow samples and volume RT milliseconds;
- NaN, clamp, allocation and dispatch failure counts.

## Reference suite

- Rising smoke plume with wind and vortex force fields.
- Moving collider through smoke.
- Fuel ignition with thermal expansion.
- Dense explosion with soot and emissive core.
- Sparse cumulus cloud with sun and local light.
- Camera inside volume.
- Uniform and anisotropically transformed analytic media.
- Long animation for temporal stability and resource-lifetime testing.

Changes enter the Final profile only after passing safety, image-quality and
performance thresholds on this suite.
