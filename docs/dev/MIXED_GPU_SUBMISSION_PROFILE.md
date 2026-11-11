# Mixed MPM/liquid GPU submission profile — 2026-11-09

Source implemented; C++ build and live comparison pending. No live test, scene
change, application launch or C++ build was performed in this implementation turn.
No shader source/ABI changed in this part; no shader compilation is needed for it.
The previous adhesion shader delivery remains required for its matching C++ code.

## Observed baseline (read-only, previous diagnostic turn)

User scene, paused at frame 80: 100000 parcels, 72^3 = 373248 cells, Vulkan,
compact canonical liquid MAC. Last step: 1950.6489 ms, 180 MPM wave/grid substeps,
62044 dispatch calls, 1928 transfer batch ends, 180 explicit synchronizations.
Recorded batch-end host time 911.4889 ms and synchronization time 545.8195 ms.
These wrapper durations are not independent GPU-stage timings and must not be
blindly summed. Physical substep count, material stiffness and solver tolerances
must not be weakened merely to hide this cost.

## Changes

1. `captureSparseMacFlip`: the canonical compact case records three page-copy
   kernels and publishes an ordered snapshot without a CPU fence. No host write
   targets the compact velocity pages in this route. Following GPU consumers
   remain in the same ordered command stream with existing dispatch barriers.
   Dense/publication mode still synchronizes before possible direct-mapped host
   boundary uploads. All failed-copy paths still synchronize before recovery.
   For the observed mixed scene, this removes the explicit wait after each of
   the 180 snapshots. It does not remove required topology/convergence readbacks.
2. Generic Vulkan pressure reductions combine pending computation, partials copy
   and fence in one `beginTransferBatch/download/endTransferBatch` sequence.
   Previously explicit synchronization preceded an immediate download submission.
   Non-Vulkan synchronization behavior is unchanged. Host partials are used only
   after successful batch completion. This generic-loop improvement is separate
   from the observed scene's sparse device-scalar pressure path.
3. Added disjoint host-stage attribution in `MixedGpuStageStats.h` and a shared
   JSON serializer. Probe switching only reads a CPU clock and changes a counter
   sink; it never dispatches, downloads or synchronizes. Its counters merge back
   into the outer frame probe exactly once, including error unwinding. Explicit
   finalization precedes publishing the successful physics statistics.

Stages: setup, forces, occupancy, liquid_transfer, mpm_transfer, flip_snapshot,
viscosity, pressure, contact, gather, advection, publication. Setup covers mixed
state preparation/allocations; publication includes final readbacks and host
commit. Liquid/MPM transfer stages include their first boundary pass. Gather
includes the post-contact boundary re-enforcement. These stage names are profiler
regions, not additional numerical operators.

No physical common-clock, transfer order, pressure iteration cap, residual
threshold, boundary condition, conservation law or material parameter changed.
No asynchronous un-fenced buffer retirement was introduced. Upload-only batches
and mandatory host-result completion continue to follow the existing backend
contract. A snapshot marked valid means its copy is ordered for subsequent GPU
consumers; it does not promise that CPU-visible memory has already completed.

## Publication batching closure — same build, 2026-11-09

The final canonical position/velocity/affine readback and fourteen MPM constitutive
sidecar readbacks now share one transfer batch. `GranularFrameReadback.h` holds
only the downloaded columns and drives both enqueue and commit from one column
list. Destinations stay alive/unmoved through `endTransferBatch()`, including a
failed enqueue. All copies complete before the first state commit. Undersized
destinations reject publication after draining queued reads safely.

Only granular/elastic owner slots receive the MPM sidecars. Shared canonical
transport, stable IDs, substance, rest mass, fractions, pore water and liquid
sidecars cannot be overwritten by this helper because those columns are absent.
This removes the second full FluidParticles clone and its separate sidecar fence
without duplicating canonical position/velocity downloads. The compact MAC host
publication still has its own required batch and decode; pressure topology counts
and convergence checks retain their required host completion.

Added `granular_frame_readback_test.cpp` for a user test target: fourteen distinct
CPU-buffer patterns, one caller-owned batch, correct owner commit, unchanged
transport/identity/mass/liquid state, and undersized destination refusal. The
existing probe regression also covers exception unwinding and probe restoration.
These C++ tests are source-delivered; not compiled or executed by Codex.

This closes the agreed physics-preserving source part. It does not claim that
all pressure readbacks or all submission overhead are gone, and does not include
multirate liquid/MPM integration. No shader changed in this closure.

## Inspection surfaces and timing semantics

The existing domain performance panel has **Mixed GPU stage counters**.
Both `rt.fluid.step_stats(domain)` and IPC `fluid.step_stats` expose the same
`mixed_stages` schema, serialized by `mixedGpuStagesJson`:

- `measured`, `timing_kind = host_wall_including_waits`.
- `common_substeps` and `grid_steps`.
- `stages[name]`: `host_wall_ms`, `visits`, transfer bytes/calls, dispatch calls,
  batch-end calls/time and explicit synchronization calls/time.

These host intervals include queued work awaited at a later readback. They are
not true GPU execution times and must not be labelled as such. Stage counter
sums are subsets of existing outer frame counters; adding both would double-count.
Batch-end and synchronization timer values can overlap/nest; do not add them to
host_wall_ms. `visits` counts instrumented regions, not pressure iterations.
Nonmixed/unmeasured/asleep paths report `measured=false`, not an invented zero.
For device kernel cost use existing `perf.set_gpu_kernel_timing` and
`perf.gpu_kernel_timings`; this change does not automatically enable timestamps
or silently add their overhead.

## Verification and next user build

Source checks run without compiling C++: Matter GPU ABI/contact/order checks,
sparse MAC canonical/storage checks, grain contracts, domain-wall contracts,
Python inspector AST, explicit Visual Studio project XML and diff whitespace.
Descriptor generation/check includes the new schema documentation.

Added C++ regressions, not compiled/run by Codex:

- `mixed_gpu_stage_profile_test.cpp`: real CPU buffer transfers under nested
  probes; correct region attribution, unchanged transfer result, preservation of
  existing frame counters, restoration and idempotent finish without new waits.
- `sparse_flip_queue_test.cpp`: recorder backend checks three queued compact
  copies with no wait; legacy host-overwrite fence and failure-recovery fence
  remain; no upload/readback is introduced by snapshot capture.

User should rebuild C++ (APICSolverStats/FluidStepStats layouts changed).
Do not run the following inspection until the user chooses to drive a new step:

```text
python scripts/test/rt_inspect_mixed_stage_profile.py "Physics Domain 1"
```

This inspector itself is read-only: no play/step/reset/scene write. It requires
pause after the user's chosen step, checks finite phase timings, frame/phase
counter consistency, zero explicit canonical FLIP synchronization, and unchanged
particle count/position/velocity digests across the inspection. User-controlled
same-scene acceptance should compare the same initial state and dt, unchanged
180-step requirement, finite state and mass/momentum/energy behavior. Inspect
`flip_snapshot.synchronize_calls` (canonical successful path should be zero),
outer submission counters, total step wall time and pressure-stage readbacks.
Build/source tests alone establish neither visual parity nor a speedup factor.
Actual GPU waits may move into a necessary later submission; the wall-time
improvement must be measured rather than equated with removed timer values.

Further opportunities intentionally not folded into this safe part: eliminate
CPU tile-list packing of compact collider weights, reduce mandatory pressure
readbacks through device-side scheduling, and later evaluate multirate liquid/
MPM integration with conservative contact exchange. Their data-lifetime,
allocation and numerical contracts need their own focused changes and tests.

## Single-build acceptance sequence

1. One user C++ rebuild covering changed headers and translation units. No new
   shader build/deployment is needed for this optimization part.
2. User opens the same snow/water scene and chooses one or more simulation steps,
   then pauses. Do not force step count, change materials, lower stiffness or
   change pressure tolerance as part of the performance comparison.
3. Run the read-only inspector command above once. Keep its report with the scene
   state/dt: physical common/grid substeps, total_ms, per-stage dispatch/readback/
   wait counts, particles and GPU route. Compare with the documented old baseline;
   avoid treating unrelated scene states as a controlled timing comparison.
4. In the same acceptance session inspect snow/water contact, wall hold/release,
   finite state, expected inventory, and cached rewind/continuation. C++ CPU-buffer
   regressions can be included in the user's normal test build if available.
   No second C++ build is required merely to read the new diagnostics.

The user deferred live acceptance. Source completion is not numerical/performance
acceptance; no live command from this sequence was run by Codex.

## User build: read-only acceptance observation — 2026-11-09

The user rebuilt C++ and left the same scene open. The inspector completed with
exit 0 without stepping, playing, clearing or changing the scene. Frame **31**,
paused; **84934** parcels; grid **72^3**, common/grid substeps **180/180**.
New stats schema, finite host phase times, phase/frame counter accounting, zero
explicit compact FLIP fences, and unchanged particle/position/velocity digests
across the read-only inspection passed. Publication used **2** batches with
**22** readback calls, matching transport+14 sidecars together plus compact MAC
publication. This confirms the new path executed; it is not full physical parity.

Observed total **1894.7583 ms**, **48364** dispatch calls, **1642** batch ends,
**0** explicit synchronizations, upload **268535280 bytes**, download
**20890440 bytes**. Batch-end host time **1231.3574 ms**.

Dominant host regions (including queued work awaited there):

| Region | Host ms | Batch ends | Dispatches | Upload bytes |
| --- | ---: | ---: | ---: | ---: |
| pressure | 1287.8837 | 1441 | 36588 | 206260992 |
| liquid_transfer | 501.5090 | 180 | 2700 | 0 |
| setup | 24.5956 | 2 | 1 | 33883516 |
| publication | 20.9664 | 2 | 0 | 0 |

The old baseline was frame 80 with 100000 parcels, whereas this observation is
frame 31 with 84934 parcels. Do not quote the totals as a controlled speedup.
The removed explicit wait migrated in part into later required readback waits;
zero synchronize_calls does not mean zero GPU/CPU waiting. Pressure and P2G
capacity decisions still cause many submission boundaries. Further focused work
should address repeated compact solid-weight packing/upload and GPU topology/
convergence readbacks before claiming the approximately two-second cost resolved.
No new C++ or shader change was made during this inspection. Native C++ test
binaries and full scene physics acceptance remain separate from this read-only
counter check.
