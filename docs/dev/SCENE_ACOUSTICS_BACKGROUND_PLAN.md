# Scene acoustics: GPU-first bounded solver plan

Implementation checkpoint: [GPU direct-path source delivery](SCENE_ACOUSTICS_GPU.md)
now implements shared AS queries, GPU reduction and frame-fence integration.
The planning statements below predate this source delivery. Build/runtime gates
remain open; hidden-wall AS expansion and later propagation stages remain future work.

## Revised direction: reuse GPU scene hits

2026-10-16 user correction: CPU BVH synchronization is deliberately lazy. The
production acoustic query path must first reuse the existing GPU acceleration
and hit infrastructure. The CPU-worker proposal below is historical design
context, not the selected next implementation. Keep CPU tracing as an explicit
reference/diagnostic path; automatic acoustic updates must not force CPU vertex
readback or a lazy CPU BVH build/refit.

Source evidence:

- `Viewport/RayFusionSceneAS.cpp` owns scene BLAS/TLAS preparation and
  `getRayFusionHitInstances`, with GPU positions, indices, triangle counts and
  material-ID buffers in TLAS custom-index order.
- `rayfusion_probe_trace_beta.comp` already reads instance custom index,
  primitive index, barycentrics, distance and front-face information using
  `rayQueryEXT`.
- `Viewport/RayFusionProbeTrace.cpp` demonstrates compute dispatch against the
  TLAS, but currently drains in-flight viewport work and downloads results in a
  synchronous call. Reusing that entire call would retain a CPU/GPU wait.
- RayFusion scene assembly excludes `rayFusionExcluded` and zero-mask instances.
  This differs from the current acoustic default `include_hidden=true`.

### Selected implementation sequence

### Required submission and data-transfer contract

User requirement, 2026-10-16: acoustic queries follow the renderer's submission
and fence lifecycle. They must not introduce host waits between devices, stall
rendering to obtain sound results, or download full hit/scene data in normal use.

- Prefer recording the acoustic compute pass after the owning backend's AS
  updates and appropriate AS-build-to-compute barriers in its scheduled GPU work.
  Fence completion is a host observation mechanism, not a GPU dependency barrier.
  Separate-queue operation requires GPU semaphore dependencies and queue-family
  ownership transfers where applicable; do not assume a compute queue gives free
  overlap or that separate devices can share Vulkan resources directly.
- Maintain a fixed small ring of request/result slots. Poll completion without
  blocking (`vkGetFenceStatus` or completed timeline value). If every slot is
  busy, coalesce the newest pending request or skip this refresh; never wait for
  a slot merely to improve acoustic freshness. Do not reset a fence or overwrite
  a slot until its owning submission is complete. A borrowed render fence is
  observed through the renderer's ownership/generation, never reset by audio.
- Current render code is an integration reference, not an unconditional nonblocking
  API: `submitTraceTonemapAsync` waits when reusing an occupied frame slot, and
  single-copy AS updates have in-flight hazard guards. Audio must not add another
  reader that forces these guards to wait longer. Prefer the same ordered submit;
  defer an independent query if safe AS resource lifetime cannot be guaranteed.
  Cancelled requests still retain GPU resources until their submission completes.
- Compute bounded crossing reduction, object-barrier deduplication, six-band
  transmission products and path distance/delay on GPU where practical. Download
  only a compact per-source summary: request/revision identity, validity/coverage/
  overflow status, six energy values and distance/delay. Fix the C++/shader ABI
  and total byte budget during implementation; do not transport JSON or per-face
  metadata through the GPU hot path. Detailed bounded hits are explicit diagnostic
  output only. Sorting/grouping must retain existing boundary semantics.
- If a part cannot run on GPU, transfer only the bounded inputs that CPU processing
  actually needs. No geometry/texture readback and no synchronous result download.
  On non-coherent host-visible memory, invalidate completed mapped ranges before
  reading them. GPU completion, host visibility and request currency are separate
  checks; all must pass before main-thread application.
- Keep final device playback/DSP on the existing audio engine unless a later
  measured design justifies otherwise. GPU geometry/reduction does not require
  GPU PCM generation. The audio callback consumes already available coefficients
  and never polls fences, waits for a renderer or initializes a GPU device.
- GPU resources/queues still contend for hardware time and bandwidth. Bound ray,
  crossing, dispatch and update budgets and measure added frame cost, result age,
  transfer bytes and skipped/coalesced work. No claim of zero render impact or
  guaranteed concurrent execution is implied by asynchronous submission.

### Backend resource selection

Backend/resource clarification from the same user discussion:

- Vulkan RT: borrow the active renderer's compatible scene AS through a backend
  query provider. Do not awaken or retain the separate raster RayFusion AS.
  `ensureRayFusionSceneAS` explicitly refuses non-interactive rendering, and
  `yieldRayFusionSceneAS` releases its AS to give VRAM to the rendered backend.
  Acoustic work must respect this ownership gate.
- Raster with a current RayFusion AS: borrow that AS without enabling GI,
  reflection, lighting, probe or texture/material bounce resources.
- Raster without an AS: raster buffers alone cannot perform hardware ray queries.
  Enabling acoustic GPU queries has a real BLAS/TLAS build/refit and memory cost;
  expose that capability/cost and allocate lazily only while needed. Reuse GPU
  geometry buffers where compatible, rather than copying the scene.
- A different Vulkan device/context cannot consume an AS handle borrowed from
  another device. Backend switching revokes requests and retires old resources
  through the owning device. Unsupported providers report bypass/unavailable;
  they do not silently initialize another rendering device or force CPU BVH work.

Use a minimal acoustic hit path. Direct-path transmission needs bounded crossings,
distance and stable object identity; primitive/front-face/geometric normal are
added where diagnostics, boundary handling or reflection geometry requires them.
Do not require UVs, textures, PBR shading, radiance, lights, environment maps or
full closest-hit rendering payloads. Acoustic energy coefficients live in a small
object/substance response table. Query/result buffers can be small; acceleration
structures can still be large, so a minimal payload is not a negligible-memory
guarantee. Surface current/shared versus acoustic-only AS bytes, scratch/result
bytes and update costs separately when implementing diagnostics.

1. Audit the active backend's acceleration structure ownership, readiness,
   generation and hit-identity mappings. RayFusion provides a Vulkan starting
   point; do not assume its TLAS is valid in every rendering/backend mode.
   Report unsupported/unavailable/stale coverage explicitly.
2. Introduce a focused shared GPU segment-query contract: bounded batch of
   source/listener endpoints, t bounds and request/revision tokens; results carry
   distance, instance/object identity, primitive ID, barycentrics, geometric
   normal/front-face and coverage/overflow state. Resolve physical Substance
   bindings by stable object identity, not by treating PBR material IDs as acoustic
   profiles. GPU buffers remain derived from canonical flat mesh/DNA data.
3. Reuse current BLAS and eligible TLAS resources. Audit hidden-object and instance
   coverage first. If the render TLAS excludes acoustic geometry, evaluate a
   separate acoustic TLAS sharing the same compatible BLAS. Do not change render
   visibility or claim complete acoustic coverage from a visible-only TLAS.
4. Record a bounded compute ray-query pass in the backend's submission lifecycle.
   Use bounded per-flight result/staging slots and fences or timeline completion
   polling. Main-thread collection must not wait for GPU completion, drain the
   viewport queue, or reuse a descriptor/buffer/AS allocation still in flight.
   Resource retirement must follow submission completion, including cancelled work.
5. Implement bounded ordered crossings, rather than only closest-hit or any-hit.
   Preserve coplanar deduplication, ambiguity reporting, per-object barrier
   grouping and thin-surface semantics. Candidate traversal order is not distance
   order. Overflow makes the response unknown; do not silently truncate loss.
   Do not inherit visual alpha/transparency rejection as acoustic transmission.
6. Feed completed geometry reports into the existing shared acoustic response
   and preview service. Preserve UI/Python/IPC configuration parity, latest-result
   ownership, disable/clear cancellation and bypass rules. The audio callback
   receives only prepared gain/filter state; it never dispatches or waits on GPU.

Benchmark end-to-end latency, queue contention, AS update cost and readback cost,
as well as query time. Reusing a current AS is the expected advantage; a few rays
alone do not establish a universal GPU speedup. Reflections and larger ray batches
can use the same contract after direct-path parity is established.

Final acceptance adds GPU/reference crossing and material parity; render-hidden
walls; stale/deforming geometry; overflow; unavailable ray-query capability;
backend switching; in-flight scene replacement; and evidence that acoustic updates
do not trigger lazy CPU BVH construction. Any new compute shader is compiled by
the user with the Studio build, never by Codex. No GPU implementation or measured
speedup is claimed by this plan revision.

## Earlier CPU-worker proposal (superseded)

2026-10-16 planning checkpoint. This document defines the next implementation
slice after the accepted direct-path preview. It does not claim worker code,
compilation, listening acceptance or completion of physical propagation.

## Starting point

The external direct-path/native PCM acceptance passed on 2026-10-16; see
[the listening checkpoint](SCENE_ACOUSTICS_PREVIEW.md). The accepted model applies
one broadband amplitude multiplier derived from six transmission-energy bands.
It does not apply travel delay, spectral filtering or reflections.

Source inspection confirms that `tickSceneAcoustics` calls `querySceneAcoustics`
for each enabled source approximately every 250 ms. The query captures flat
geometry, resolves materials, constructs a BVH and traces the segment on the
main thread. Four enabled sources can repeat scene capture four times per tick.
The current immutable snapshot is a suitable worker boundary; scene capture
itself must continue to read scene-owned data on the main thread.

## Next delivery: scheduling without changing the acoustic model

Keep `acoustics_trace` synchronous and read-only for explicit diagnostics. The
existing `acoustics_preview` operation remains the shared configuration entry for
UI, Python and IPC; its automatic refresh uses the background solver. There is
no new UI-only solver, independent material table or hidden simulation consumer.

1. Split query preparation from evaluation. Preparation validates options,
   resolves authored source/listener poses, copies canonical flat TriangleMesh/DNA
   geometry and material data, and records request ownership. Evaluation accepts
   only owned immutable values and calls the existing trace core.
2. Capture one shared scene snapshot per preview refresh, then use it for the
   maximum four enabled source segments. Reuse one BVH across those segments.
   Do not introduce a persistent geometry cache until invalidation is reliable.
3. Use one lazy service-owned worker, one active batch and one replaceable pending
   batch. A batch has at most four paths and the existing 100000-triangle,
   2048-object and 128-crossing limits. Status storage must also remain bounded.
4. Collect completed work without waiting in the viewport tick. Apply results and
   update native source gain on the main thread through the existing service.
   No scene, UI, renderer or native voice pointers enter the worker or callback.
5. Reject obsolete results before applying any attenuation. Stop/disable, source
   removal, scene clear/restore and service destruction revoke publication rights.
   Cancellation checks belong in BVH construction and traversal; destruction
   joins the worker before releasing its storage.

The first delivery removes BVH/tracing work from the main thread. Snapshot copying
and material resolution still run there and must be reported and measured.
Moving live scene reads to a worker is not an acceptable shortcut.

## Ownership and invalidation

Each request needs a monotonically increasing session/configuration token plus
an input stamp for scene identity, geometry revision, material fingerprint,
timeline/deformation state, source pose, listener pose, units and boundary mode.
Matching only the source ID is insufficient: an ID can be removed, recreated or
reconfigured while an earlier request is running.

The same token/stamp check gates success and error publication. An old failure
must not replace a newer successful report. An old completion must not reactivate
a disabled source or overwrite new units/material settings. Validate malformed
configuration before cancelling or replacing an existing request.

Audit geometry invalidation before trusting `g_scene_geometry_generation` alone:
object transforms, skinned/deformed positions, pending deletion and material edits
all affect the snapshot. If an authoritative revision is unavailable, recapture
and compare bounded immutable inputs before publication, or bypass that result;
do not label an incomplete revision check as current geometry authority.

For an unchanged input, keep the last accepted result during a refresh. When a
known input change invalidates that result, restore bypass gain 1 while the new
solve is pending. Preserve the existing bypass behavior for missing camera,
unsupported geometry, unknown material and resource failures. Continuous motion
can consequently remain bypassed; smooth dynamic propagation is a later delivery.

## Shared status and validation

Preserve existing preview status fields (`id`, `enabled`, settings, `status`,
`amplitude_gain`, optional `report`). Add a separate `solver` object so existing
consumers can continue reading transmission status:

- `state`: pending, running, ready, stale or failed.
- `request_id`: publication token for the latest submitted request.
- `applied_request_id`: token of the accepted report, or null before acceptance.
- `snapshot_ms` and `solve_ms`: measured CPU wall times for the completed request;
  omit unavailable timings rather than reporting zero.
- `error`: current request failure only, cleared by a newer successful result.

These are proposed fields, not currently implemented API. UI reads the same status
that `rt.audio.acoustics_preview_status` and `audio.acoustics_preview_status` return.
Configuration errors retain current Python/IPC error semantics and preserve state.
Runtime worker failures are reported as status with bypass gain, not thrown from
the viewport loop. Preview remains transient and absent from project serialization.

## Module and integration boundaries

- `AcousticScene`: validated request preparation and canonical snapshot capture.
- `AcousticTrace`: reusable immutable acceleration/tracing core, preserving
  current intersection and boundary-model semantics.
- New focused `AcousticSolver.h/.cpp`: bounded worker, batch ownership,
  cancellation, completed-result transfer and timing.
- `AcousticPreview`: configuration, current request/report state and gain policy.
- `AudioService`: minimal lifecycle and main-thread apply wiring.
- `AudioAcousticUI`: compact pending/error feedback in the existing source panel.

Check line counts before edits. Files over 2000 lines receive integration wiring
only. Register new modules in the existing project files. Keep synchronous trace
and automatic preview on the same numerical core; do not duplicate BVH or DSP logic
inside bindings.

## Acceptance gates

During implementation, perform static source/registration, lifecycle and API/IPC
contract audits. Do not build, launch Studio or add/run live probes in this slice.
The user requested test scripts at the final structural acceptance stage.

After the user builds Studio, the final external IPC/manual pass must check:

1. Unchanged-scene synchronous/background report equality and the previously
   accepted native PCM ratios for one/two barriers, clear path and bypass.
2. Source/listener motion, object transform/deformation, material edit and removal
   while work is running: no obsolete gain or report is applied.
3. Disable/re-enable with the same ID, units/mode edits, clear/restore/undo/redo,
   failed requests and shutdown: no late publication or orphan worker.
4. Four sources share one snapshot/BVH per refresh. Pending work remains bounded
   under rapid edits; invalid configuration preserves current valid ownership.
5. Viewport responsiveness and separate snapshot/solve timings at representative
   scene sizes. Report measured capacity; worker placement alone is not proof of
   acceptable frame time or a hard realtime guarantee.

Run live probes externally through `scripts/test/rt_ipc.py` while Studio is open.
The source build checklist is Studio C++ with the new solver module registered;
this planned slice requires no new shaders.

## Following slices

After scheduling: six-band audible direct-path filtering with smooth coefficient
changes and shared offline processing; then explicit closed-wall/thickness
semantics; then validated bounded early-reflection paths and delayed processing.
Do not postpone boundary-model correctness until reflections are presented as
physical propagation. Freehand spectral editing, event modulation, file workers
and portable export remain open items in the broader audio plan.
