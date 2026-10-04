# C2 — Vulkan liquid active windows (2026-10-04)

## C2b source bundle — occupancy and pressure (2026-10-04)

New code, not covered by the earlier C2a live PASS. Build/live acceptance is
deferred to a combined batch at the user's request. No builds or application
launches were performed by Codex.

- The liquid path now uses the existing weighted GPU occupancy scatter and
  compact solid-cell stamp. The resulting mask is reused by viscosity and
  pressure; their duplicate mask uploads are skipped. If occupancy fails,
  the original host mask builder and upload path take over.
- `FluidGpuPressure.inl` extracts the existing MGPCG implementation from the
  large simulation driver. `FluidActivePressure.h` owns window routing and
  the appended 76-byte ABI; CUDA and full-grid kernels retain their 52-byte ABI.
- Seventeen window shader variants compile from the SAME source as full-grid
  kernels. Divergence, diagonal, CG vector operations, SpMV and fused/double
  reductions use the same rectangular window and global storage strides.
  The scalar reduction receives the reduced block count. Padded lanes contribute
  zero and still reach all workgroup barriers. Plain and variational-solid
  matrices are covered; GFM and periodic fallback policy is unchanged.
- Host-provided masks derive bounds from all fluid rows plus neighbour halo;
  the GPU-mask path uses authoritative liquid positions and the existing
  conservative support/CFL planner. No pressure row is deliberately discarded.
- Pressure/residual cold-start initialization, occupancy clear, P2G clear and
  final face-gradient/solid enforcement remain full-grid. This preserves empty
  cells, vacated support and moving-collider state. C2's cold-clear optimization
  remains open; C2 is not marked complete on the basis of shader routing alone.

Shared panel/Python/IPC diagnostics now include `occupancy_on_gpu`,
`pressure_window_used` and `pressure_window_cells`. An unused or failed pressure
window reports zero cells. These are runtime diagnostics, not serialized
authoring options. No physical timestep, pressure tolerance or iteration budget
was reduced.

### Deferred combined acceptance

1. User runs `RayTrophiStudio/compile_shaders.bat`, then the application build.
   The script also generates all 17 `_window.spv` files. No manual shader list
   or second build command is needed.
2. In an external terminal run
   `python scripts/test/rt_probe_fluid_active_window_matrix_ipc.py`.
   This creates/removes a hidden scratch domain, tests localized GPU occupancy
   and pressure, compares one step against CPU (centroid <1e-4 m, mean speed
   <1e-3 m/s), then tests periodic fallback. ALL existing liquids advance three
   times by 1/120 s; none are reset or reseeded. Use an empty scene if those
   small steps are unwanted. The report is `.tmp/fluid_active_window_live.json`.
3. Localized existing-scene probe:
   `python scripts/test/rt_test_fluid_active_window_ipc.py --expect-window --expect-pressure --expect-occupancy`.
   For a liquid occupying the whole domain omit the expect flags.
4. In the same acceptance batch verify open outflow, moving colliders, viscosity,
   two domains of different grid sizes and cache/resume. Measure total/pressure
   time at unchanged physical settings; dispatch shrinkage is not a speedup proof.

`python scripts/test/check_fluid_window_contracts.py` passed: 17 source/registry
pairs, ABI/binding parity, float64 gates, barrier-safe reduction tails and project
XML. The standalone C++ source test also gained mask-bounds/dispatch-policy
checks but was not compiled or run. Earlier C2a acceptance below remains valid
only for that earlier binary.

## C2a history — P2G normalize

C2a was built by the user and live routing was checked. It was the first C2
code slice; the C2b changes above extend occupancy and MGPCG after that binary.

2026-10-04 live update: user built the changes. External named-pipe probes
confirmed the active normalize kernel on a hidden 1,000-particle scratch liquid:
1,200 / 34,848 cells (3.44%), GPU P2G/pressure/G2P all active. Periodic scratch
liquid correctly reported window=false, cells=0 with GPU P2G retained. The scratch
domain was removed. The existing `Physics Domain 1` read-only probe passed with
window=false; that result alone does not distinguish full support from a large
CFL halo. No old-build numerical/performance A/B or standalone C++ test was run.
The live matrix report is `.tmp/fluid_active_window_live.json`.

The original probe assumed `Grid Domain 1` and indexed `measured` on the
application-level error result. It now reports `ok=false` clearly and selects
the sole liquid domain when no name is supplied. The Release script copy was
updated. Run `python scripts/test/rt_test_fluid_active_window_ipc.py` for the
existing scene. `rt_probe_fluid_active_window_matrix_ipc.py` tests a temporary
domain and advances all existing liquids twice by 1/120 s; it never resets them.

`FluidActiveWindow.h` computes half-open cell bounds from the canonical APIC
position/velocity SoA. Quadratic MAC support, one pressure-neighbour margin
and CFL travel are conservatively included. Normalize uses full-grid strides;
buffers, coordinates, P2G scatter and physical parameters are unchanged.
Full clear guarantees that faces outside the window remain zero, including
faces occupied in the previous step. No additional transfer or submit is added.

Only localized Vulkan liquid uses the new kernel. Granular resident positions,
CUDA, periodic boundaries, empty/invalid inputs and a full-size window keep
the existing normalize path. A dispatch failure uses the existing complete
solver fallback; the partially normalized field is not normalized a second time.

The existing panel diagnostics and `rt.fluid.step_stats(domain)` /
`fluid.step_stats` IPC share these read-only fields:

- `normalize_window_used`: the successful GPU P2G used the bounded kernel.
- `normalize_window_cells`: bounded cell-box volume; zero if unused. It is
  not occupied-cell count, total MAC face count or pressure dispatch count.
- `full_grid_cells`: the liquid solver's grid size, including in Matter domains.

## User build and acceptance

1. Run `RayTrophiStudio/compile_shaders.bat`, then build the application.
   The new shader is `sim_fluid_normalize_window.comp` (2 buffers, 40-byte PC).
2. Step a localized, non-periodic Vulkan liquid inside a large domain. Run
   from an external terminal:
   `python scripts/test/rt_test_fluid_active_window_ipc.py "Grid Domain 1" --expect-window`.
   The panel's normalize cell ratio must match the IPC values. Python's
   `rt.fluid.step_stats("Grid Domain 1")` must expose the same fields.
3. Compare against the previous build at identical dt, voxel, particles,
   pressure iterations and boundaries: positions, velocities, closed-tank mass
   and runout must agree within the existing solver tolerance. Record P2G and
   total time; smaller dispatch is not itself a measured speedup.
4. Move liquid between distant parts of the domain, including both boundary
   faces and open outflow. No residual velocity may remain in vacated cells.
   Test colliders/moving solids with unchanged pressure/boundary settings.
5. Periodic liquid, full-domain liquid, CPU/CUDA and resident granular cases
   must keep the old path; `normalize_window_used=false` and cells=0.

The standalone source test `scripts/test/fluid_active_window_test.cpp` checks
MAC face bounds, uniqueness, every quadratic scatter support index, CFL growth,
invalid inputs and stale-host fallback. Build it in a developer terminal with
the project include directory and `source/src/Math/Vec3.cpp`, then run it;
do not compile with `NDEBUG` because the checks use assertions.

Codex checked Python syntax, project XML and source ABI wiring only. C++/GLSL
tests and live IPC probes have not been run. Next C2 slice is occupancy/mask
and MGPCG dispatch cropping, with pressure neighbour and reduction semantics
validated independently; full-grid clear removal requires an explicit ownership
contract because pressure/viscosity can write beyond the P2G support.
