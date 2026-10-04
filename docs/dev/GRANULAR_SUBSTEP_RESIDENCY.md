# Granular stiffness preservation and particle residency

2026-10-02: source implementation complete; build and live acceptance pending.

Follow-up source change: [GPU occupancy and bounded resident submissions](GRANULAR_GPU_OCCUPANCY.md)
removes intermediate position/mask traffic. The description and measurements
below document the preceding residency baseline, not the follow-up result.

## Baseline supplied by the user

Closed Vulkan domain, 991,666 particles, 128 cubed grid:

- Total 2332.52 ms; P2G 152.76 ms; G2P 1634.10 ms; advect 150.70 ms.
- Wave CFL requested 48 elastic substeps; legacy limit granted 32.
- Authored Young modulus 381300 Pa; delivered modulus 176400 Pa.
- The load diagnostic independently reported a 703367 Pa stiffness threshold.

This was a material change, not merely a performance limit. G2P timing includes
stress update, settle, GPU submission/execution waits and readbacks, summed over
the elastic subcycle. It does not isolate the gather shader.

## Core changes

`Fluid/GranularStepPolicy.h` grants the entire maximum of the wave and strain
requests for both CPU and GPU orchestration. No 32/64 budget truncates that
request. GPU stress receives a sufficiently small dt to retain authored Young;
CPU stress already uses authored Young and now gets the same subcycle count.

`granular_max_solver_substeps` remains a legacy serialized/IPC/Python field,
including its existing 1..64 storage clamp and get/set round trip. It has no
effect on physical substep count. UI shows adaptive subcycling rather than an
ineffective quality slider. IPC discovery and Python help document this change.
UI, `rt.fluid.step` and `fluid.step` share the existing simulation scheduler/core;
there is no binding-specific solver implementation or new operation.

GPU transfer stages now live in the focused
`source/src/Physics/Fluid/FluidGpuTransferStages.inl`, included in the simulation
driver's private namespace because its push-constant types remain private there.
No new compiled translation unit or shader is required.

On the closed, inviscid Vulkan granular path with GPU forces, free-surface
configuration and no solid/frozen parcels or substance viscosity:

1. P2G reuses the current device particle streams after each successful substep.
2. Intermediate G2P keeps velocity, affine and granular tensors on device.
3. Advect downloads positions for the next occupancy mask and the existing
   per-substep material-coordinate refresh schedule. It skips velocity downloads.
4. Final G2P downloads velocity/affine/tensors; final advect publishes current
   positions and velocity. Frame-level render/cache consumers retain host data.
5. Before CPU fallback or a lost particle reuse stamp, the residency helper
   recovers all particle state. Failed recovery refuses host fallback on stale
   velocity/affine/stress. Open, periodic, viscous and solid/frozen configurations
   retain their existing transfer paths.

For N particles and S substeps, the eligible path removes approximately
`60*N*(S-1)` download bytes (G2P velocity/affine plus duplicate tail velocity)
and `64*N*(S-1)` upload bytes (position/velocity/affine/mass fraction). At
991,666 particles and 48 substeps these are about 2.60 GiB down and 2.78 GiB up
per frame versus the previous path running the same 48 physical substeps.
These are source-derived savings, not measured speedups. Positions and the
occupancy mask still cross the CPU boundary each substep.

Deferring G2P submission moves GPU execution waits into the advect readback.
Judge speed by `total_ms`, upload/download bytes and physical state; a lower
`g2p_ms` alone does not establish a speedup. Additional physical substeps may
offset some of the transfer savings. The load warning remains independent.

## User build and acceptance checklist

Build the application using the normal project workflow. This change does not
modify shader source. Codex has not built or launched the application.

In an empty, paused scene, from a separate terminal:

```powershell
python scripts/test/rt_test_granular_stiffness_residency_ipc.py
```

The test creates/removes a scratch domain, requests more than 64 substeps,
checks legacy budgets 1/32/64, effective/requested stiffness, GPU execution,
invalid particles, repeated motion and closed versus open transfer traffic.
It also exercises the CPU planner. `fluid.reset/step` are global; the test
rejects existing domains before any mutation. It must not run on the user's
million-particle scene.

Run the existing CPU/Vulkan parity and soft-stability tests in a disposable
scene, and the liquid C1 residency test in an empty scene to cover unaffected
liquid G2P/advect behavior. For CPU fallback, test a build without an available
G2P or advect-tail kernel in a disposable scene; reject stale-state consumption.

For the supplied scene, clear/rebuild its old simulation cache and simulate
from a full initial/keyframe state. At the same settings expect needed/run
substeps 48/48, effective Young approximately 381300 Pa and no stiffness cap.
Do not compare playback of cached frames with a fresh solver step. Read traffic
from an external terminal while fresh simulation is advancing:

```powershell
python scripts/test/probe_granular_transfer.py "Grid Domain 1" 15
```

Record total, G2P + advect, transfer bytes/calls, particle count, invalid flags,
damage and deformation. Verify keyframe/intermediate-frame scrub and resume
separately: the earlier frame-cache handoff's resume issue is still open.


## Live measurement after user build (2026-10-02)

External IPC manual stepping filled the same source-driven scene to 991666
particles, then collected five solver samples. Timeline frame remained 0; cache
remained one 2174832-byte initial frame, so no playback cache was measured.

Mean total 2717.82 ms; P2G 12.02 ms; G2P 64.76 ms; Advect 2006.10 ms.
48 required/run substeps; requested/effective Young 381300 Pa; no cap or
invalid particles. Upload 590452232 bytes/frame; download 763617216 bytes/frame.

Absolute total is higher than the supplied 2332.52 ms / 32-substep sample,
while physical substeps increased by 50 percent. G2P waits moved into Advect.
This is not an identical-state A/B: the new trajectory also had max damage 0,
unlike the supplied old trajectory. CPU memcpy is not established as the main
remaining bottleneck; GPU execution and transfer fence waits are still combined.
Raw samples: [granular_residency_live_2026-10-02.json](granular_residency_live_2026-10-02.json).
