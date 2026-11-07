# Sparse SDF reconstruction and automatic Ice material

Source implemented; user compilation and live acceptance are pending.
User subsequently confirmed the automatic Ice material works on frozen matter;
the visual recipe is considered basic and further refinement was deferred.
No project/shader compilation was run by the agent. Per user preference, no new
C++ test files are retained: verification uses IPC and manual scenes.

## Sparse surface change

The old weighted-centroid reconstruction could fill an empty gap between a few
particles. The default positive surface offset further enlarged isolated drops.
Splat did not exhibit this because it draws the individual carriers.

CPU and GPU now compute the same effective support N = sum(w)^2 / sum(w^2).
At N <= 4, the reconstructed field is constrained by the nearest original
particle sphere, without positive surface dilation. Between 4 and 16 the
constraint fades smoothly; N >= 16 retains the existing dense reconstruction.
The bound uses original particle positions, so centre smoothing cannot pull a
sparse pair together and fill the space between them. Existing smoothing uses
the same confidence and stays disabled for unsupported sparse regions. This is
a reconstruction correction, not a change to solver mass, melting or density.

Per-particle support radii are calculated once rather than taking a cube root
for every particle-cell pair. GPU uses a new O(N) support-radius buffer: 400 kB
payload for 100k particles, with the same additional upload per rebuilt surface.
There is no new dense grid channel. The extra nearest-distance calculation still
has a runtime cost; it is not a claimed speedup.

`sim_surface_gather_supported` is a fresh 8-binding / 128-byte-PC pipeline name.
An old 6-binding gather SPIR-V cannot be mistaken for the new shader. Missing
new kernels refuse the GPU build and retain the existing CPU fallback.
Sub-voxel droplets can still be undersampled: this prevents artificial bridges
but does not invent geometric detail below the sampling resolution.

## Ice material

Frozen SDF carriers identified by the canonical `kParticleFlagFrozen` use an
editable **Ice (Automatic)** asset when no authored source/domain surface
material overrides them. Its recipe is the existing `substance:Ice` preset:
IOR 1.31, full transmission, roughness 0.18, pale tint and mild interior
absorption. The shared preset now includes that absorption, so explicit material
creation and automatic creation use the same recipe.

CPU and GPU composition use one `surfaceMaterialForParticle` resolver. Authored
source materials win; an authored domain surface material suppresses the
automatic fallback. Without either, a domain's explicit Ice binding supplies
the frozen fallback; otherwise the automatic asset is created once in the
material registry. Existing assets are reused, never overwritten. Melting
removes the frozen fallback and restores the liquid's normal material lookup.
Snow's granular solid phase is not automatically converted into clear ice.

This reuses the existing composition field and can require that field even in
a previously single-material domain containing ice/water. Its existing cost is
three floats per simulation-grid cell. The slot table now supports ten slots
(fallback, eight distinct bindings and frozen fallback); fresh
`sim_surface_composition_phase` uses the matching table stride. No separate
high-resolution ice interior grid is introduced.

Rendering may lazily allocate the automatic asset; `fluid.surface_parity` only
looks up existing assets and never creates one. Frozen flags participate in the
surface signature so a phase change without motion rebuilds material lookup.
Automatic material IDs are resolved from the current registry each time, not
cached across project loads.

## UI, scripting and IPC

The asset appears in the ordinary material library and uses the existing
material inspector/node editing. Explicit creation uses existing `material.create`
with type `substance:Ice`; assignment uses `fluid.set_substance_material` with
substance `Ice` and a material name. Python exposes the same material/domain
services. Normal validation for unknown substances/materials and missing domains
is unchanged. No UI-only operation or independent phase/solver setting was added.

## User build and live checklist

1. Build C++, including the registered `FrozenSurfaceMaterial.cpp`. Compile and
   deploy both new shaders: `sim_surface_gather_supported.comp` and
   `sim_surface_composition_phase.comp`, then restart. The main all-shader script
   discovers them; the simulation-only batch also includes them.
2. Small isolated-drop/melting scene: compare Splat and SDF with the same state.
   Empty gaps should remain open; sparse drops should not grow into a large pool.
   Check default smoothing, smoothing=0, and anisotropic CPU fallback. Dense water
   and a thin sheet need visual regression checks too.
3. Run existing external IPC `fluid.surface_parity` on the same paused state.
   Expect gpu_ok, matching signs and matching composition IDs/weights within
   the existing tolerances. Run the existing surface GPU fixture only in a
   separate empty scene; it creates/deletes domains and steps the simulation.
4. Freeze unassigned Water SDF: Ice (Automatic) should appear and shade the frozen
   region. Check mixed ice/water and melt it back. Particle mass/count must not
   change because of the rendering change. Compare RT and RayFusion.
5. Assign a custom domain or Water surface material and verify it remains in
   control. Separately bind Ice with existing UI/IPC and verify that fallback
   is used when source/domain materials are unset. Edit the automatic asset,
   pause/resume and save/reopen: the edited asset must not be regenerated.
6. Read `perf.get_gpu_memory` and surface build/upload stats after a real rebuilt
   surface. Measure additional nearest-support work; cache playback is not a
   reconstruction benchmark. The automatic material does not imply volumetric
   random-walk snow rendering or a new fog field.

## Previously verified by user

RayFusion SDF shadows now work under all tested light types. Matter Fog Spread
changes the footprint. Fine fog detail remains limited by voxel resolution;
the user accepted this limitation and moved the focus to SDF and Ice.
