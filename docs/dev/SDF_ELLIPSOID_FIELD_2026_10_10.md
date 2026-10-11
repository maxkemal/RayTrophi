# Ellipsoid field correction — 2026-10-10

Live liquid SDF is traced directly; no tessellation/mesh generation in the main surface drawing path.

FluidLevelSet now applies the weighted anisotropy metric to the final implicit surface residual as well as the support kernel. Covariance eigenvalues are converted to standard-deviation axis lengths before aspect clamping and volume normalisation. The implicit residual is divided by its local metric gradient magnitude to retain approximate world-distance units. This is not an exact signed distance field; the varying kernel weights add spatial derivatives not included in that approximation.

Existing anisotropy_enabled controls this correction through the existing shared UI, scripting and IPC parameter path. Disabled mode remains spherical. Sparse support bounds remain conservative (maximum axis stretch^(2/3)). No absolute velocity stretching or solver particle mutation is introduced.

Pending: common splat/spray ellipsoid representation, isolated-drop deformation and capillary relaxation, density-aware body/sheet/drop radius policy. Foam and bubbles retain distinct physics/materials. Do not remove splat or claim the complete common shape system is delivered.

User build/test: compile C++; compare anisotropy off/on at the same cached frame, quality multiplier 1 and identical camera/material; inspect falling stream, isolated drops and thin floor sheets in Solid and RT. Record SDF gather/build, render_sync, work_cells and memory. No live performance or visual acceptance has been performed for this change.
