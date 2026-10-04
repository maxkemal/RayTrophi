# Granular GPU occupancy and bounded resident submissions

2026-10-02: user build and supplied-scene live measurement complete. Disposable
scratch-scene acceptance, fallback and collider/thermal regression checks remain pending.

This continues the measured 991666-particle, 128^3, 48-substep baseline in
`granular_residency_live_2026-10-02.json`: mean total 2717.82 ms, upload
590452232 bytes, download 763617216 bytes, 102 batch-end calls. These are CPU
stage/transfer timers; the 2006.10 ms advect row includes queued GPU work.

## Changes

- `FluidGpuOccupancy.inl` constructs occupancy from resident positions and mass
  fractions. It clears the existing float mask, stamps a compact solid-cell list,
  then scatters mass using the new `sim_fluid_occupancy.comp` kernel. The solids
  upload occurs once per frame; its persistent buffer grows only when needed and
  is released with the domain compute buffers. No extra dense grid is allocated.
- The existing eligible path remains closed, inviscid Vulkan granular domains,
  GPU forces, free-surface configuration, no solid/frozen parcels or substance
  viscosity. Liquid, open, periodic and other fallback paths keep host masks.
- Intermediate advect tails retain positions along with velocity/affine. They
  publish positions whenever either material-coordinate generation will refresh,
  retaining the original per-substep schedule (including period 1 and odd periods).
  Final tails publish positions before frame-level render/cache/API consumers.
- Missing/failed occupancy recovers resident particle state before building a
  host mask. Failed recovery prevents a stale host fallback. A tail dispatched
  before a failed readback still recovers its output instead of advecting twice.
- Vulkan submits at the 512-descriptor-set capacity before allocating another
  set, then resets its counter with the pool. Long subcycles therefore remain
  bounded and below the 1024 timestamp-pair capacity per submission.
- The unused GPU `material_flags` allocation/upload was removed. All shader
  consumers use `state_flags`, which stress update writes before settle/readback.
  This removes 4*N device bytes and 4*N upload bytes whenever state is uploaded.
- Granular cleanup now destroys `bond_scale`; validity also requires that buffer.

The mask remains float: it stores weighted mass, not merely three cell classes.
It preserves solid=-1, air=0, clamped mass in [0,1], and the mass>0.02 cutoff.
Float atomic summation order can differ from the host order, as already occurs
in P2G. No stress, deformation or affine precision/substep/stiffness was reduced.

At the recorded size, removing 48 mask uploads saves 402653184 bytes/frame.
Removing 47 intermediate position readbacks saves 559299624 bytes/frame when
no material refresh occurs. Each refresh adds one 11899992-byte position readback.
Removing the unused flag stream additionally saves 3966664 bytes per full state
upload and that amount of device storage. These are source-derived traffic
savings, not measured speedups. Compact collider storage is 4*solid_count bytes
(a four-byte dummy binding when there are no solids).

Shared CG/gas/combustion allocation policy, stress compression and geometric
particle-capacity growth are outside this bounded change. Measure this result
before expanding the optimization scope or returning to the unified domain plan.

## User build and acceptance

1. Run `RayTrophiStudio/compile_shaders.bat`, then the normal C++ build. Confirm
   `sim_fluid_occupancy.spv` is deployed to the executable's `shaders` directory.
   Codex did not compile shaders, build or launch the app.
2. Open an empty, paused scene. In a separate terminal run:

   ```powershell
   python scripts/test/rt_test_granular_stiffness_residency_ipc.py
   python scripts/test/rt_test_fluid_c1_residency_ipc.py
   ```

   The granular test checks legacy budgets, >64 physical substeps, closed/open
   motion parity, refresh period 1 versus 240, reduced position traffic, matching
   gather/stress/settle/advect timestamp counts, and presence of GPU occupancy.
   It restores the previous timing toggle. Missing occupancy must fail the new
   performance gate, even if the functional host fallback remains correct.
3. Run existing granular CPU/Vulkan parity and soft-stability tests in disposable
   scenes. Also check an odd material refresh period (e.g. 5), moving colliders,
   fractional mass from thermal/burn exchange, and a collider with particles near
   solid faces. Verify no new penetration, invalid flags or mass loss.
4. In a disposable runtime copy omit only `sim_fluid_occupancy.spv`: expect a
   warning and host position/mask traffic, with correct motion and material
   coordinates. Test missing advect and solid-face-clamp shaders separately;
   fallback must publish positions before a host tail or material reset.
5. In the supplied paused, single-domain scene, start from the same full initial
   state for each measurement. The following calls advance simulation globally:

   ```powershell
   python scripts/test/probe_granular_kernels.py "Grid Domain 1" --steps 5 --output docs/dev/granular_occupancy_untimed.json
   python scripts/test/probe_granular_kernels.py "Grid Domain 1" --steps 5 --gpu-timing --output docs/dev/granular_occupancy_timed.json
   ```

   Restart/reset to the same initial state before the second run. Timing is
   off for the first run and restored to its prior value after each probe.
   `gpu_stages` separates gather, stress, stress P2G, settle, advect and occupancy;
   `gpu_kernels` retains all rows, including shared clears. Timing is global to
   the compute context, so the probe rejects multiple simulation domains.
   Use IPC calls from the external terminal, never the embedded workspace.
6. Compare total_ms, upload/download bytes, batch_end_calls **and** synchronize
   calls/time. Some fence waits move to capacity-triggered submissions; a shorter
   advect row or fewer batch ends alone does not prove a faster frame. Confirm
   48/48 substeps, requested/effective Young=381300 Pa, matching particle counts,
   zero invalid flags and comparable damage/deformation. Verify cache scrub/resume
   separately; this change does not repair the earlier cache-resume issue.

Both `rt.fluid.step` and IPC `fluid.step` retain the same simulation core. GPU
timing uses the existing `rt.perf.set_gpu_kernel_timing/gpu_kernel_timings` and
IPC `perf.set_gpu_kernel_timing/perf.gpu_kernel_timings`; no new authoring API,
UI-only operation or binding-specific solver was introduced.

## Live result after user build (2026-10-02)

The user opened the same scene after compiling. External `scripts/test/rt_ipc.py`
access initially received Windows error 5 and succeeded with sandbox escalation.
The closed Vulkan domain started empty at timeline frame 0. Its existing source
filled it through 238 manual 1/24-second steps to 991666 particles. No scene
settings, reset, seed, visibility or renderer configuration were changed.

Five subsequent steps with timestamps disabled, then five with timestamps
enabled, produced the following means. The two sequences are adjacent physical
states, not an identical-state instrumentation A/B. The earlier baseline also
comes from a separate run: the percentage is an observed record comparison, not
proof of identical trajectories.

| Metric | Prior residency record | GPU occupancy, timestamps off |
| --- | ---: | ---: |
| total_ms | 2717.82 | 1815.48 |
| upload_bytes | 590452232 | 184061632 |
| download_bytes | 763617216 | 206697590.4 |
| batch_end_calls | 102 | 9.2 |

Observed reductions: total time 33.20%, uploads 68.83%, downloads 72.93%.
Four samples downloaded 204317592 bytes; one downloaded 216217584 bytes, exactly
one extra 11899992-byte material-refresh position stream. All samples retained
48 required/executed substeps, requested/effective Young=381300 Pa, no stiffness
cap, zero invalid particles and max damage=0. This does not establish full
deformation/trajectory parity or complete the separate regression checklist.

GPU timestamp means over five following steps (total host time 1791.62 ms):

| Kernel | GPU ms/frame | Calls/frame |
| --- | ---: | ---: |
| sim_fluid_p2g_scatter | 979.39 | 144 |
| sim_fluid_granular_stress_p2g | 360.40 | 144 |
| sim_fluid_g2p | 190.59 | 48 |
| sim_fluid_advect_tail | 44.31 | 48 |
| sim_fluid_granular_stress_update | 26.33 | 48 |
| sim_fluid_occupancy | 13.57 | 96 |
| sim_fluid_granular_settle | 6.49 | 48 |

GPU total across all kernels averaged 1650.20 ms. Gather/stress/settle/advect
counts agree with the 48-step subcycle, and occupancy ran twice per substep
(solid stamp plus mass scatter). The main remaining cost is the two grid scatter
paths, not the constitutive stress update. Investigating scatter locality/atomic
contention or reusing its neighborhood traversal is a future candidate; the
timestamps alone do not prove atomic contention is the cause.

Long resident submissions moved execution waits into the host P2G/dispatch row.
Do not interpret that row's increase as a measured increase in the P2G shader.
Internal backend capacity flushes are not all represented by the public context
`synchronize_calls` counter; total time and device timestamps remain the useful
comparison here.

At completion GPU timing was restored to disabled. Timeline stayed at frame 0,
cache reported 0 bytes/0 frames and valid=false, and the live digest retained
991666 particles (mean speed 0.1690614). These were fresh solver steps rather
than cached playback. The open scene was left at this advanced physical state.

Raw reports:
- [Fill](granular_occupancy_fill_2026-10-02.json)
- [Timestamps disabled](granular_occupancy_untimed_2026-10-02.json)
- [Timestamps enabled](granular_occupancy_timed_2026-10-02.json)
