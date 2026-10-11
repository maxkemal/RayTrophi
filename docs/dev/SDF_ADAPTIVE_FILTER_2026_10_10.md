# Adaptive SDF filter — 2026-10-10

The previous fixed six-neighbour filter had a shrinking world-space footprint as the surface resolution multiplier increased. The active reconstruction now samples at simulation-voxel spacing; the sparse halo is enlarged consistently. This does not make changing the simulation resolution itself invariant.

The gather pass records compact per-work-lane effective support (sum w)^2 / sum w^2. Four or fewer effective contributors receive no smoothing; the strength increases smoothly to full strength at sixteen. This evidence is frozen across filter iterations. Opposing exterior samples inhibit smoothing to protect thin structures/necks. Missing support and domain-boundary stencils are left unchanged. Total displacement from the original field is bounded to 0.2 simulation voxels. Scratch and confidence are transient compact arrays, not dense grids or cached state.

This is a geometric heuristic, not material-aware reconstruction or an exact volume/topology conservation algorithm. High-support small droplets can still be smoothed, and diagonal sheets are not guaranteed protected by the axial guard. Material-specific surface tension, viscosity and radius policies remain pending. Existing smoothing_iterations is the common UI/script/IPC control; zero bypasses filtering. Solver particles are unchanged.

User verification after C++ build: fixed cached frame/camera, compare smoothing 0/2/4, surface multiplier 1/2. Inspect stream lumps, thin floor sheets, isolated drops and gaps in Solid and RT. Record build/smooth/work-cell counters and memory; confirm increased refinement does not reduce the physical filtering footprint. Visual and native performance acceptance pending.

## Small fragments and traversal

Solid/RayFusion compositor, Vulkan RT surface query and RayFusion shadow query share surface_sdf_march_policy.glsl. Previously a 192/512 step cap enlarged sampling spacing over long volume intersections, so entry and exit could both lie between two samples. The new policy grows the budget first, keeping <=0.45 voxel spacing until a pathological 16384-step bound. This can increase GPU cost on long/dense views. It is not an empty-space acceleration implementation.

A sampled field still cannot preserve arbitrary sub-voxel droplets: if all lattice values are exterior, no marcher can reconstruct the missing component. Trilinear interpolation and grazing rays also preclude a universal visibility guarantee. Analytic droplet fallback/local refinement remains pending; do not enlarge all droplets to conceal this limit.

Build BOTH C++ and affected shaders (user only). Include dependency: surface_sdf_march_policy.glsl. Compare paused geometry from two cameras to isolate traversal loss, then successive cached frames to inspect reconstruction/temporal loss. No application mutation was needed during source changes.
