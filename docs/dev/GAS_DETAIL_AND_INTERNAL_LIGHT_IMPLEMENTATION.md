# Gas detail transport and emission-to-smoke lighting

Status: code implemented, compiler/GPU/live acceptance pending. No builds,
application launch, simulation stepping or cache regeneration by the agent.
Existing cached simulation data and live material settings were not edited.

## Source fidelity: scalar characteristic integration

The solver already has limited MacCormack advection, including GPU forward and
correction passes. It would have been incorrect to diagnose it as simply missing
MacCormack. Both passes previously traced trajectories with one Euler velocity
sample. The scalar path now uses the explicit midpoint characteristic:

`p_mid = p - 0.5*dt*v(p); p_back = p - dt*v(p_mid)`.

The reverse MacCormack trace uses the same method with negative dt. Its source
stencil limiter uses the matching midpoint backtrace, so corrections remain
bounded by the original donor cells. Original boundary fetches and solid masks
remain in place. The CPU scalar reference and Vulkan GPU scalar kernels use equivalent
helpers. Velocity advection is unchanged; this is not a claim to upgrade the
entire solver's order, conserve scalar mass exactly, provide vorticity refinement,
or make arbitrary high-CFL trajectories accurate.
The separate CUDA compute implementation is not upgraded by these GLSL edits.

This improves characteristic accuracy in rotating/shearing velocity fields;
it does not create absent small-scale turbulence. Extra velocity samples cost
work. Higher grid counts alone still do not establish fine-scale source detail.
Inventory shaders include the same scalar kernels and need rebuilding too.

The existing RAM scrub cache stores tile-sparse half floats. That is a deliberate
precision reduction, not a newly discovered float32 source path. This increment
does not rewrite its format, increase memory use, or destructively clear caches.
The previous live field arrived through this cache/NanoVDB route. New midpoint
advection affects only frames recomputed after the user builds; an old 100-frame
cache cannot gain source detail from a renderer change.

## Internal fire light: field integration, not emitter-list re-enabling

The disabled one-block emitter-list estimator remains disabled. Instead, the new
shared marcher integrates the existing camera emission source through density and
RGB extinction over a ray inside the same volume:

`L += T * emission_source * (1 - exp(-sigma_t * dt))`.

It then attenuates T over the segment. The receiving smoke applies its own RGB
scattering albedo in the camera integral. Light therefore depends on the actual
temperature/ramp, density, independent intensity response and intervening smoke;
it does not add an unoccluded glow based on distance to the volume centre.

Both dense gas and NanoVDB use their normal field samplers. No live-only block
address or imported-VDB emitter metadata is required. The RT primary thermal
emission evaluation is moved into `volume_internal_rt.glsl` and reused by light
queries, including existing flame boost and radiance limiting. The preview
adapter calls its existing `emissionAt` directly. This keeps camera and incident
source evaluation consistent within each renderer.

Vulkan RT samples a direction from the material's dual-HG phase; phase/pdf then
cancels. It clips the segment at scene solids using the existing solid probe.
Finite chord integration is still biased/discretized; a small hot core can be
missed at insufficient steps. Spatial reuse above stride 1 is an additional
approximation and the source-direction estimator needs adequate accumulated
samples. This is single scattering of volume emission, not full emissive
multiple scattering or a validated unbiased renderer.

RayFusion uses six fixed axial directions with normalized dual-HG weights. This
is stable but angularly approximate. It models attenuation within the emitting
volume; this version does not include solid geometry or other overlapping media
along those six internal rays and can leak light through embedded objects.
RT disables this gather for overlapping companions or spatial material programs
because their extinction/source evaluation would otherwise be inconsistent.
The implementation does not yet light a different volume from an emitter in a
separate volume. These boundaries must not be advertised as full Houdini parity.

## Controls, persistence and API

Two quality fields use previously unused integers at GPU offsets 280/284; the
Vulkan volume table remains 624 bytes. Host transfer structs carry both fields.

| Field | Default / bounds | Meaning |
| --- | --- | --- |
| `internal_light_steps` | 0, integer 0..32 | 0 disables; otherwise segments per incident ray |
| `internal_light_stride` | 8, integer 1..32 | Camera samples between incident-light updates |

Both controls are in the existing Physical volume / fire response panel and
`volume_material::configure`, so UI, Python and IPC share atomic validation.
`volume.get_material` reports them. Old project quality blocks default to 0,
preserving the previous lighting cost; new Blackbody Fire and Artistic Fire
recipes select 16 steps / stride 4. Physical Cloud disables this fire-only gather.
No work is performed for emission mode off. Zero receiving scattering produces
no internally scattered light. CPU/OptiX do not implement this Vulkan feature.

Example after building (exact target from `rt.volume.list_materials()`):

```python
rt.volume.set_material("gas:Ground Burst Gas",
                       internal_light_steps=16, internal_light_stride=4)
```

External IPC: `volume.set_material`,
`{target:"gas:Ground Burst Gas", settings:{internal_light_steps:16, internal_light_stride:4}}`.
No transport settings or intensity calibration were applied to the user's open
scene. The prior extreme density/absorption and response powers still require
calibration; internal light cannot penetrate arbitrarily opaque smoke.

The sampling help text now correctly describes 0.5 as two samples per voxel,
subject to step-budget stretching. A 333-step limit still cannot honour a 0.155
voxel request along every long interval.

## User build and verification checklist

1. Build C++ normally. Recompile `volume_closesthit.rchit` and
   `material_preview_volume.frag`, with the three new `volume_internal_*.glsl`
   includes available. No new pipeline or descriptor binding is required.
2. Recompile `sim_grid_advect_scalar.comp`, `sim_grid_maccormack_scalar.comp`,
   `sim_inventory_advect_scalar.comp` and `sim_inventory_maccormack_scalar.comp`;
   all consume the new `sim_grid_scalar_trace.glsl` through their shared includes.
3. Run standalone `scripts/test/grid_advection_trace_test.cpp` as C++17 with
   `source/include` on the include path and assertions enabled. It checks exact
   uniform translation, rotating-flow error against the analytical trajectory,
   reverse direction and zero dt. It does not validate the compiled GPU kernel.
4. Run `volume_material_authoring_test.cpp` and the external
   `rt_test_volume_material_ipc.py` after manually opening Studio. The probe
   defaults to read-only; --apply tests and restores controls.
5. Preserve the current cache before generating a separate comparison simulation.
   Compare the SAME cached frame with internal light off versus 16/stride 1,
   then 32/stride 1. Check dark/cold fields, zero emission, zero scattering and
   scale changes. Compare RT with RayFusion while accounting for its approximation.
6. For source fidelity, compute a separate new cache with identical emitter,
   time-step, substep and turbulence settings. Compare field gradients/peak loss,
   plume shape and runtime over several frames; do not compare only a single
   overexposed beauty image. Check open/closed/periodic boundaries and embedded
   solids. Existing cache frames intentionally retain their original solver data.
7. Profile both renderers. Steps 16 / stride 4 adds up to four source samples
   per primary sample amortized in RT and 24 in the six-ray preview, plus density
   lookups and RT solid probes. Skipping, termination and cache scheduling change
   actual cost. No realtime frame-rate guarantee is made.

Agent-run verification: non-build source contracts, Python syntax, targeted
whitespace checks. C++ numerical tests, shader compilation, field-comparison
experiments and rendered/visual acceptance are pending; no measured quality
improvement or elimination of the user's blocks is claimed yet.
