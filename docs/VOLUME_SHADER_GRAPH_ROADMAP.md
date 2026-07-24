# RayTrophi Volume Shader Graph Roadmap

## Design contract

The volume graph is a second typed output of the unified Material Graph. It
extends `VolumeShader`; it does not replace the existing VDB,
gas, cloud, fire, or fluid render paths. A graph only overrides connected output
slots. Unconnected slots continue to use the owning `VolumeShader`, preserving
old projects and presets.

The compiler separates two execution domains:

- **Uniform closure:** values constant across a volume. These are folded into
  `GpuVolumeShaderData` and have zero per-step GPU cost.
- **Spatial field:** values depending on position, VDB grids, or procedural
  textures. These compile to a bounded Vulkan VM evaluated during ray marching.

One graph may drive both `Material Output.Surface` and `Material Output.Volume`.
The editor and serialization are shared, while each output compiles to a separate
runtime program. Vulkan RT is the reference backend. CPU and OptiX parity follow after the graph
and physical coefficient conventions stabilize.

Containers never become materials. VDB, Gas, Fluid, Cloud and SDF own field
data, transforms and march quality; the selected material asset owns optical
appearance. Density is composed as:

`sampled source field * container multiplier * material/graph multiplier`.

For an SDF volume, the same asset's Surface output drives the dielectric
boundary while its Volume output drives absorption/scattering/emission inside.

## Physical output model

The graph exposes Density, Scattering Color/Strength, Absorption Color/Strength,
Emission Color/Strength, and Anisotropy. The integrator interprets them as
non-negative per-distance coefficients. Anisotropy is clamped to `[-0.99, 0.99]`.

## Phases

### Phase 1 - foundation

- [x] Versioned, serializable compiled volume-output model.
- [x] Per-slot override mask and backward-compatible `VolumeShader` ownership.
- [x] Validation/clamping at the CPU/GPU boundary.
- [x] Zero-cost homogeneous fold into the existing GPU volume payload.
- [x] Typed `Material Output.Volume` socket in the existing Material Graph.
- [x] `Principled Volume` closure node sharing the common math/color node library.
- [x] Material Graph asset reference from VDB, gas, fluid and simulation shaders.
- [x] Standalone FluidObject uses the same VolumeShader/material reference as VDB/Gas/domains.
- [x] Legacy `Volumetric` materials materialize as `Principled Volume -> Material Output.Volume`.
- [x] VDB/Gas graph picker reads the material catalog, not only already-opened graph instances.

### Phase 2 - homogeneous Vulkan vertical slice

- [x] Compile constant/color/math chains into the homogeneous output.
- [x] Live compile, typed connection diagnostics, and project round-trip storage.
- [x] Publish graph changes to every bound VDB/Gas consumer without requiring its panel to be open.
- [x] Unify material creation, slot assignment, graph targeting, and backend invalidation.
- [x] Preserve source density when a material is bound (multiplicative density semantics).
- [x] Route both NanoVDB and legacy Gas through `VolumeShader::toGPU()`.
- [x] Resolve an SDF boundary's IOR/roughness from the same asset's Surface output.
- [ ] VDB, gas, cloud, and fluid regression matrix.

### Phase 3 - spatial field VM

- [ ] Position/object-position, Mapping, Noise, Wave, Gradient, Math, Mix, Ramp.
- [ ] Separate SSBO and versioned bytecode header.
- [ ] Constant folding, dead-code elimination, 32-register limit, instruction budget.
- [ ] Density and emission evaluation in primary and shadow marches.

### Phase 3b - mesh volume boundary

- [x] Treat a closed triangle mesh with only `Material Output.Volume` as a medium boundary
      (Vulkan front/back segment, OptiX pre-surface march, CPU fallback fixed).
- [ ] Robust entry/exit pairing for camera-inside, nested and overlapping closed meshes.
- [ ] Route the bounded interval through the Vulkan volume integrator instead of surface closest-hit.
- [ ] Reject or diagnose open/non-manifold boundaries rather than rendering a black surface.

Initial closure note: the ordinary closed-manifold case now passes through as a
medium on all three renderers. Nested/overlapping meshes, camera-inside entry
state and the full Vulkan spatial bytecode march remain production-hardening
work rather than being silently claimed by this first boundary slice.

### Phase 4 - volume data nodes

- [x] Typed `Volume Info` node contract: density, temperature, flame/fuel, velocity, position.
- [x] Direct `Volume Info.Density` identity path (zero VM cost).
- [x] Vulkan bounded lowering for direct `3D Noise -> Density` and explicit
      `Volume Info.Density * 3D Noise` (both preserve the source grid).
- [x] Fix live FBM Scale propagation by executing the compiled field program at
      every Vulkan march sample instead of copying one bounded noise descriptor.
- [x] General Vulkan spatial lowering for Noise/Wave/Gradient, Math, Vector Math,
      Mix, Ramp/Curve and all Principled Volume coefficient sockets.
- [ ] Named/custom NanoVDB grid node and runtime binding table.
- [x] Semantic `Volume Grid` node for density, temperature/heat, flame, fuel,
      velocity, color, emission and voxel size; unknown names diagnose and read zero.
- [ ] Arbitrary custom NanoVDB grid binding table beyond the semantic fields above.
- [ ] Missing-grid diagnostics and deterministic fallback values.
- [x] Persistent NanoVDB accessor reuse around the spatial VM march loop.

### Phase 5 - Vulkan production closure

- [x] Blackbody/temperature graph node interoperable with ramps and Volume Info.
- [x] Per-sample multiple-scattering control on Principled Volume.
- [x] Deterministic timeline `Time` node using the shared animation clock.
- [x] Optimized `Cloud Shape` opcode: animated wind advection, broad cumulus
      body, height profile, domain warp and high-frequency edge erosion.
- [ ] Dedicated curl-noise vector output for general-purpose field advection.
- [ ] Cloud presets: cumulus, stratocumulus, storm and volcanic ash.
- [ ] Optional 3D field bake/cache so finalized procedural clouds avoid VM cost.
- [ ] Phase-function model selection and explicit backward-lobe controls.
- [ ] Scripting API, presets, documentation, regression tests and performance counters.

### Phase 5b - legacy VDB zero-overhead pipeline

The first spatial-VM integration keeps old VDBs functionally compatible, but
including the VM in the shared volume closest-hit shader still reduces GPU
occupancy even when the runtime branch is inactive. Temperature sampling,
surface-only programs and direct Density passthrough have already been removed
from the legacy hot path, and the volume VM register budget is temporarily
limited to 12 compacted live registers. A measured residual slowdown remains.

- [x] Sample temperature only for temperature emission or an active volume program.
- [x] Do not bind surface-only material programs to a VDB march.
- [x] Fold direct `Volume Info/Grid Density -> Density` passthrough to zero VM work.
- [x] Use a volume-specific 12-register VM budget with safe folded fallback.
- [ ] Compile separate `Legacy VDB` and `Graph VDB` closest-hit shader variants.
- [ ] Add separate SBT hit groups and route each volume instance by program presence.
- [ ] Ensure the legacy variant contains no Material VM register file, interpreter,
      material texture descriptor access, or program-buffer access.
- [ ] Benchmark empty/sparse/dense VDB scenes against the pre-graph baseline.
- [ ] Acceptance: graph-disabled legacy VDB performance within 5% of baseline;
      graph-enabled cost reported by march steps, VM instructions and noise octaves.

### Phase 6 - CPU backend

- [ ] Interpret the same validated spatial bytecode in the CPU volume integrator.
- [ ] Match Vulkan coefficient, coordinate and missing-grid semantics.

### Phase 7 - OptiX backend

- [ ] CUDA interpreter generated from the same opcode contract.
- [ ] Vulkan/CPU/OptiX image parity and performance regression suite.

## Production volume target

The target is a production-quality, artist-directed sparse volume system in the
same problem class as dedicated real-time pyro/cloud tools. The goal is not to
hide quality problems by increasing `max_steps`. Step count remains a safety
ceiling; sampling rate derives from voxel size, optical depth and the selected
quality budget. Live graph authoring and final rendering use the same material
semantics, while finalized procedural fields may be baked for predictable cost.

### Phase 8 - voxel-aware integration

- [x] Derive primary step length from voxel size and a quality multiplier.
- [ ] Keep `max_steps` as a bounded safety limit rather than the quality control.
- [x] Optical-depth-aware adaptive stepping in dense regions.
- [x] Conservative transmittance early-out.
- [ ] Camera-inside and boundary-crossing correctness at every quality level.
- [x] Quality presets:
      Draft `1.5-2.0 voxels`, Medium `0.75-1.0`, High `0.35-0.5`,
      Ultra `0.2-0.3`.
- [ ] Report effective primary/shadow samples and early-out ratio.

#### Phase 8 continuation note (2026-07-23)

- Vulkan RT and CPU now convert native VDB voxel size through the object's
  world transform before deriving primary and shadow-march spacing. This fixes
  strongly down-scaled legacy VDBs becoming sparse or nearly empty after a
  quality preset is selected. OptiX keeps its previously correct path.
- CPU cover stepping now uses the actual ray interval (`tExit - tEnter`) rather
  than the world-AABB diagonal, and no longer clamps tiny scaled voxels to a
  `0.001` world-unit minimum.
- Closed-mesh Volume materials now carry `Shadow Steps` and `Shadow Strength`
  through the Vulkan material ABI and perform density-aware self-shadow
  marching, including spatial Volume Graph density evaluation.
- OptiX closed-mesh volume self-shadowing was already present.
- CPU closed-mesh volume lighting/shadow traversal exists, but exact
  `Shadow Steps` / `Shadow Strength` parity with Vulkan and OptiX remains a
  follow-up acceptance item. Add a regression scene with a closed sphere/cube,
  homogeneous and graph-driven density, one directional and one point light,
  plus camera-inside coverage.
- Atmosphere aerial/LUT compositing remains intentionally deferred until the
  voxel-aware integration work is complete.

### Phase 9 - sparse traversal and filtering

- [x] Vulkan NanoVDB hierarchy-based empty-space skipping for primary rays.
- [x] Vulkan NanoVDB hierarchy-based empty-space skipping for shadow rays.
- [ ] OptiX and CPU sparse-traversal parity.
- [ ] Active-tile/min-max density metadata for procedural and baked fields.
- [ ] Trilinear baseline with optional high-quality tricubic reconstruction.
- [ ] Distance/footprint-aware volume LOD or mip representation.
- [ ] Gradient/detail filtering to prevent high-frequency erosion aliasing.
- [ ] Acceptance: sparse legacy VDBs gain quality without proportional step-cost
      growth and never regress against the Phase 5b legacy baseline.

Sparse hierarchy traversal is deliberately restricted to legacy/baked NanoVDB
density. A Volume Graph can synthesize density inside inactive source tiles, so
graph-driven fields retain full sampling until graph-aware occupancy metadata is
available.

Vulkan RT is the reference backend for the parity reset. Its primary integration
now uses deterministic full-interval fixed segments, and self-shadow evaluation
frequency is separated from per-shadow-ray sample count through
`Shadow Update Stride`. OptiX and CPU will follow only after Vulkan acceptance.
VDB sequence uploads and UI-driven volume/program updates are serialized against
the Vulkan render thread; resized binding-9 buffers are retired only after GPU
completion.
Rapid manual VDB scrubbing additionally waits once per changed-grid upload batch
before rewriting persistent device-address buffers. Replace this safety stall
with a fenced ring/triple-buffer after the Vulkan reference path is stable.

Note: the later NVIDIA driver access violation during RT pipeline creation was
traced to `hair_shadow_anyhit` reading bindings 10/11 without ANY_HIT visibility
in the descriptor layout, not to the volume shader. The volume SPIR-V validation
guards remain useful defensive checks.

`Max Steps` is now a strict cost ceiling on Vulkan, OptiX, and CPU; the earlier
automatic 2x expansion made low authored budgets misleading and multiplied
nested self-shadow work. Custom `Voxels / Sample` spacing extends to 8 for
deliberately coarse previews of small or distant assets.

### Phase 10 - production cloud lighting

- [x] Primary volume raymarch with material-controlled step/max-step quality.
- [x] Volume shadow march and material shadow-step/strength controls.
- [x] Direct-light volume self-shadow transmittance.
- [ ] Fix atmosphere/volume compositing parity in empty and near-empty regions:
      an intersected volume AABB must preserve the same aerial-perspective
      transmittance/inscatter as the unobstructed background ray. Currently some
      empty volume pixels bypass or double-separate the aerial contribution,
      producing a background-contrast silhouette that is then amplified by the
      final display LUT.
- [ ] Regression cases for zero density, below-cutoff density, sparse edge voxels
      and a volume AABB extending across sky/horizon geometry.
- [ ] Phase-function model selection in the material closure.
- [ ] Dual-lobe Henyey-Greenstein forward/backward parameters.
- [ ] Energy-bounded silver-lining control.
- [ ] Density-aware powder effect.
- [ ] Energy-conserving multi-scatter octave approximation.
- [ ] Sky/atmosphere ambient contribution and optional ground-bounce tint.
- [ ] Upgrade the existing shadow march with adaptive stepping and stronger
      transmittance early-out.
- [ ] Directional-light deep-shadow/transmittance cache for large cloud fields.
- [ ] Multiple-light quality budget and deterministic fallback.

### Phase 11 - cloud and ash authoring

- [x] Optimized animated `Cloud Shape` field with body, height, warp and erosion.
- [x] Deterministic timeline Time and Wind advection.
- [ ] General Curl Noise vector node.
- [ ] Weather/Coverage map input.
- [ ] Separate base-shape, detail and erosion channels.
- [ ] Height-dependent erosion and density remapping.
- [ ] Cumulus anvil/top shaping, wind shear, turbulence and dissipation.
- [ ] Presets: cumulus, stratocumulus, storm cloud and volcanic ash.
- [ ] Parameter normalization so presets behave consistently across volume scale
      and voxel resolution.

### Phase 12 - production fire and pyro shading

- [ ] Temperature, flame, fuel, soot and velocity as first-class bound grids.
- [ ] Temperature-to-Kelvin remap and Blackbody emission.
- [ ] Independent flame mask and emission-intensity shaping.
- [ ] Soot-driven absorption and smoke/fire energy balance.
- [ ] Hot-core highlight compression/clipping controls.
- [ ] Missing-grid fallbacks that remain deterministic and physically safe.
- [ ] Fire, smoke, explosion and ember presets.

### Phase 13 - temporal stability

- [ ] Blue-noise march jitter instead of coherent per-ray hash planes.
- [ ] Frame-decorrelated sampling tied to the deterministic timeline.
- [ ] Volume-aware temporal accumulation and rejection.
- [ ] Disocclusion handling for animated density fields.
- [ ] Stable shadows during playback and offline animation rendering.
- [ ] Acceptance: no visible step banding, structured flicker or temporal trails
      in the reference cloud/fire animation suite.

### Phase 14 - graph field bake and cache

- [ ] Bake supported procedural graph outputs to a sparse 3D field.
- [ ] NanoVDB/portable cache output with graph and source-field content hashes.
- [ ] Dirty-region or frame-level invalidation for animated fields.
- [ ] Live mode executes the VM; Final mode may consume the equivalent baked field.
- [ ] Preserve density, temperature, flame, fuel, velocity, color and emission
      semantics across live and baked paths.
- [ ] Background bake progress/cancel and memory/disk budget reporting.
- [ ] Image-parity test between live graph and baked result.

### Phase 15 - reference quality and performance gates

- [ ] Reference scenes: sparse VDB cloud, dense storm, closed-mesh procedural
      cloud, fire/smoke explosion, volcanic ash and camera-inside volume.
- [ ] Vulkan RT reference images for Draft/Medium/High/Ultra.
- [ ] CPU and OptiX parity within documented stochastic tolerance.
- [ ] Counters: primary steps, shadow steps, skipped distance, VM instructions,
      noise octaves, cache hit rate and volume GPU time.
- [ ] Automated performance regression thresholds per reference scene.
- [ ] No graph: within 5% of the pre-graph legacy baseline.
- [ ] Uniform graph: effectively zero per-step graph overhead.
- [ ] Spatial graph: cost scales with reported instructions/octaves and stays
      inside its selected quality budget.

## Performance and safety budgets

- Graph disabled: no runtime branch beyond the CPU-side fold.
- Uniform-only graph: no additional shader instruction or descriptor.
- Spatial programs: immutable bytecode, validated offsets, bounded registers (12
  compacted live registers in the Vulkan volume VM; 32 in the surface VM) and
  instructions, no tracing nodes, no recursion, and deterministic missing-data fallback.
- Existing `VolumeShader` values remain the fallback for every unwritten slot.
