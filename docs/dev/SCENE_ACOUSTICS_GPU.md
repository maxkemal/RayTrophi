# GPU direct-path acoustic preview

## Distant noise and foreground cost measurement, 2026-10-16

At the user's request, the four authored test sources now use one deterministic
four-second filtered-noise asset at camera-relative distances 50/100/150/250 m
(meters_per_unit=1). Sources share gain 0.5, min_distance=5, max_distance=500 and
rolloff=1. Geometry/material bindings were not changed. All four direct paths were
clear; these measurements therefore demonstrate spatial/distance gain, not
building transmission or diffraction. Travel delay remains reported, not applied.
Native Master measurements are engine PCM, not microphone SPL:

| Distance | RMS dBFS | Peak | Overload blocks |
|---|---:|---:|---:|
| 50 m | -52.44 | 0.0101031 | 0 |
| 100 m | -58.49 | 0.0046301 | 0 |
| 150 m | -60.69 | 0.0042176 | 0 |
| 250 m | -66.75 | 0.0018520 | 0 |

Combined four-source RMS was -49.72 dBFS with no overload. Random-noise sampling
windows and fades limit exact inverse-distance ratio comparisons. Evidence:
[noise/distance and first cost run](acoustics_neighborhood_cost_20261016.json).
GPU memory counters were identical before/after that run: tracked device-local
1330298800 bytes, driver usage 3871952896 bytes. This demonstrates no growth during
the run, not zero initial audio allocation; the slots already existed.

The first moving/configuration runs became background-window measurements. Four
configuration-to-ready wall times were 3.31/3.31/3.31/3.62 s, including serialized
IPC and scheduling. A direct foreground-PID check confirmed Studio was not
foreground, and the main-loop policy allows a 500 ms background wait. These are
not acoustic GPU execution times. See
[refresh wall latency](acoustics_neighborhood_refresh_latency_20261016.json).

The user then explicitly held Studio foreground. All 85 sampled focus checks
matched the Studio PID. Four approximately four-second matched-workload windows
gave the following whole-process CPU figures (100% means one CPU core):

| Workload | Audio/preview off | Four voices + preview on | Observed added core fraction |
|---|---:|---:|---:|
| Static sources | 18.64% | 24.22% | 0.056 |
| Four sources updated +/-0.3 units per cycle | 17.80% | 34.69% | 0.169 |

Moving control-cycle median was 123.86 ms off versus 147.86 ms on. A cycle includes
four source-update IPC calls, status/telemetry queries and application dispatch;
this is not solver time. Moving display-loop medians were 19.194/19.235 ms and
p95 20.084/20.314 ms. No sustained FPS or pure-audio cost is inferred from these
short windows. RT sample count remained 32 in every sample: these are converged
RT/idle-query measurements, not continuous path-tracing overhead measurements.
There are no dedicated acoustic GPU timestamps in this revision. Evidence:
[foreground matched-workload run](acoustics_neighborhood_foreground_cost_20261016.json).

All restoration operations succeeded. Four distant sources/noise asset remain
as requested; playback was stopped and the original timeline mode restored.
Project save was left to the user. Further acceptance needs active-render GPU
timing, repeated matched windows and defined material responses on blocked paths.

## Neighborhood post-build live check, 2026-10-16

The user reopened the large project after rebuilding. Four requested persistent
sources were recreated from the shared test WAV; see
[positions and initial GPU status](acoustics_neighborhood_capacity_check_20261016.json).
All four returned ready/clear. Importantly, the runtime AcousticGPU first-record
log explicitly reported `instances=2251 sources=1`; this establishes the acoustic
TLAS input-table entry count for this run. It is not a logical-object or flat
triangle count inferred from scene.list_objects.

A temporary downward 50-unit path from the current camera encountered one GPU
crossing and correctly reported unknown_material with bypass gain 1. The source
was restored, and all four paths returned clear again. A short audition of the
front-left source produced native Master RMS 0.00091303594 and peak 0.00193430902,
with overload_blocks=0. Playback was stopped and the original timeline mode was
restored. See [live evidence](acoustics_neighborhood_live_check_20261016.json).
Sources/asset remain as requested; geometry and material bindings were not edited,
and Codex did not save the project or build/launch Studio.

This confirms successful four-source GPU query use and nonzero PCM on the loaded
scene, including invalidation after a position change. It is not an FPS/GPU-time,
transfer-bus or capacity benchmark, nor evidence of calibrated building-wall
attenuation: the intersected material response was undefined. Material authoring
and measured performance/lifecycle stress remain further acceptance work.

## Neighborhood preparation and capacity revision, 2026-10-16

The user loaded a 2251-object project in RT and requested sources at different
positions. Four persistent spatial manual sources were added relative to the
current camera (front-left, front-right, distant-left, rear-right), sharing one
generated mono asset. They were not played, project geometry/materials were not
changed and the project was not saved by Codex. See
[setup/positions](acoustics_neighborhood_setup_20261016.json). Preview configuration
is session-only; the sources/asset are authored scene data.

The current binary reported GPU-unavailable coverage/budget status for all four.
2251 objects exceed the previous 2048-instance budget, although object count does
not by itself prove the actual TLAS count or exclude other coverage failures.
The source now shares one 4096-instance bound across RT/raster validation and
slot allocation. Over-budget RT errors include actual instance count and limit.
Three slot payloads reserve 787584 bytes (plus allocation/driver overhead), with
unchanged 64-byte-per-source/256-byte-batch results. No new AS or scene-geometry
copy is added. The shader ABI/binary is unchanged; a user C++ build is required.
This is not large-scene acceptance; further coverage/material/latency measurements
remain pending on that build. Earlier capacity/memory figures below describe the
previous revision.

## Post-fix Vulkan RT acceptance, 2026-10-16

The user rebuilt the mask-cast fix and opened RT. Preflight confirmed Rendered
mode, ready Vulkan pipeline (cached creation 0.1327951 s), one Default_Cube and
empty audio state. The external probe completed all 13 cases without the previous
crash; see [RT acceptance report](acoustics_gpu_rt_fixed_acceptance_20261016.json).
Native PCM RMS/peak ratios were 1 for clear/bypass/disabled, 0.5 for one object
barrier, 0.25 for two barriers or two thin cube boundaries, and 0 for zero
transmission. Unequal-band response matched the expected broadband coefficient.
Reports identified `active_vulkan_as`, `cpu_bvh=false`, ready/failed status as
appropriate and 64 result bytes per source. Invalid configuration and source
removal checks passed. Only Default_Cube remained and test audio was removed;
the history-pinned derived material could not be removed. Playback mode was
restored, while ordinary next_id/explicit-mode/project-history effects remain.

This closes the reproduced UTF-8 RT startup-query crash and the small-scene
Vulkan RT/native PCM acceptance gate. RayFusion had passed separately. It does
not establish large-scene capacity, transfer/GPU timing measurements, full
in-flight lifecycle stress, hidden-wall expansion or later propagation stages.
No build or application launch was performed by Codex.

## RT crash root cause and source fix, 2026-10-16 10:48

The stack-enabled retry reproduced Signal 22 and saved
[the external attempt](acoustics_gpu_rt_stack_retry_20261016.json), plus ignored
runtime StartupCrashStack.txt/StartupCrashSignal.dmp. Offline DbgHelp resolution
against the matching EXE/PDB identified JSON serializer `dump_escaped` -> JSON
`dump` -> `audio::tickSceneAcoustics` at the `identity.dump()` stamp creation.
This is before AcousticGpuPass initialization/dispatch, explaining absent pass logs.

The RT revision builder streamed `TLASInstance::mask` directly. Its type is
`uint8_t`, so ostream writes a character, not decimal digits. Default mask 0xFF
therefore inserted invalid UTF-8 into the revision embedded in JSON. The source
now explicitly casts the mask to `std::uint32_t` before insertion. RayFusion's
revision used decimal strings and did not enter this RT-specific path.

Source inspection and a non-build byte/UTF-8 model check confirm the defect and
the numeric representation for all 256 mask values. They do not establish a
post-fix C++/GPU pass: another user C++ build and external probe are still required.
No shader change is required for this fix. The process exited during the attempt;
cleanup was unavailable and must be checked after the user reopens. This evidence
supersedes earlier uncertainty about whether the driver/dispatch was responsible
for these recorded crashes; it is not a claim that all RT lifecycle cases pass.

## RT diagnostic retry, 2026-10-16 10:40

User rebuilt and reopened RT. Preflight confirmed a ready cached pipeline
(0.1453969 s), one cube and empty audio state. The external retry again measured
baseline PCM then disconnected at the first enabled clear-path status query.
[Retry report](acoustics_gpu_rt_retry_20261016.json) was successfully saved by the
repaired failure path; cleanup failed because the application exited. Signal 22
was logged at 10:40:50. No AcousticGPU pipeline/record breadcrumb was present in
the scene log; last entries concerned test-wall geometry/AS updates. This does
not establish whether the fault precedes acoustic dispatch or logging was lost.

IPC had already successfully read state, created objects and played/measured
audio. Errors 109/232 describe the disconnected pipe after process exit, unlike
the initial sandbox access error 5. No evidence currently attributes the crash
to external IPC permissions. The user reports VS did not catch the error.

The existing signal handler exited without a stack. A focused FatalSignalTrace
module now records current-thread addresses, module paths/RVAs and attempts a
current-context minidump before that exit. Raw evidence is flushed first, because
in-process dump writing is best-effort. Expected files beside the executable:
StartupCrashStack.txt and StartupCrashSignal.dmp. Preserve matching EXE/PDB files
for symbolization. This is further source instrumentation, not a verified crash
fix; a user C++ build and reopen are required, with no new shader change.

## RT attempt after pipeline readiness, 2026-10-16

The user confirmed RT readiness. External preflight verified Rendered mode and
Vulkan `rt_pipeline_state=ready` (driver pipeline creation had taken 143.615 s).
The probe measured baseline tone PCM, then lost its pipe while waiting for the
first enabled clear-path result: ReadFile error 109, then WriteFile error 232.
The user confirmed the application closed by itself. No process remained;
StartupCrash.log recorded Signal 22 at 10:32:27. No stack/module diagnosis is
available, so neither a specific driver fault nor an acoustic root cause is proven.

The RT acceptance failed. Cleanup could not be confirmed and the original probe
also failed before writing its report because its final scene query used `call`.
[Attempt evidence](acoustics_gpu_rt_acceptance_20261016.json) is reconstructed from
the command output/log, not a successfully completed probe report. The probe now
uses `try_call` for final scene/audio snapshots so future disconnects still save
the primary error and unavailable-cleanup state. Bounded first-use pipeline,
record/submit and first-completion diagnostics were added to AcousticGPU source.
These diagnostics require a new user build; they are not a confirmed crash fix.
No application restart/build was performed by Codex. After the user reopens,
inspect possible recovered test walls/source before another probe.

## RayFusion external acceptance, 2026-10-16

After the user authorized RayFusion testing, the already active Material/scene
viewport had a ready AS. The existing external probe passed all 13 cases on the
user-built application; see
[GPU RayFusion report](acoustics_gpu_rayfusion_acceptance_20261016.json).
Accepted reports identify `active_vulkan_as`, `cpu_bvh=false` and a 64-byte result
per source. Native engine PCM RMS/peak ratios were 0.5 for one object barrier,
0.25 for two, 0.25 for two thin cube boundaries, 1 for unknown-material bypass,
0 for zero transmission, and 1 after disabling. Unequal-band RMS ratio was
0.48131764821 versus expected 0.48131763594.

This is actual pre-OS/device native PCM evidence, not microphone/SPL or a measured
GPU transfer/performance trace. It does not establish Vulkan RT acceptance,
busy-slot/AS-lifetime stress behavior, large-scene capacity or all stale-result
cases. The initial hidden-instance count did not prevent this probe after its
scene rebuild; that observation alone does not establish a hidden-wall bug/fix.
The user's active Material mode was preserved. Only Default_Cube remained and
test emitters/assets/clips/previews were removed. A derived material remained
history-pinned; ordinary next_id/project modification effects were preserved.
No build, shader compilation or application launch was performed by Codex.

## First external attempt, 2026-10-16

The user built Studio and offered the one-cube scene for testing. External IPC
reached the new GPU status path. Native baseline tone PCM was present, but the
acceptance stopped at the clear-path case with `gpu_unavailable`: the active
Solid viewport had never built a RayFusion AS (`builds=0`, `ready=false`). This
confirms explicit bypass reporting, not successful acoustic GPU dispatch.
See [attempt report](acoustics_gpu_acceptance_20261016.json).

A temporary Rendered switch succeeded, but `viewport.render_frames(count=2)`
timed out before main-thread dispatch; the RT pipeline still reported `none`.
No acoustic RT/PCM pass or performance result is claimed. Solid was restored;
only Default_Cube remained, no audio emitters/assets/clips or preview entries
remained. The test leaves normal undo/project modification effects and can leave
its derived material pinned by history. No build or application launch was run.
Before a large-scene benchmark, resolve AS availability in the intended viewport
mode and repeat the small-scene acceptance.

2026-10-16 source delivery. No C++ build, shader compilation, Studio launch or
live GPU/PCM acceptance was performed. Previous CPU-preview acceptance does not
establish correctness or performance of this revision.

## Shared operation and resources

The existing UI checkbox, `rt.audio.acoustics_preview(...)` and external IPC
`audio.acoustics_preview` configure the same session service. Parameters and
validation bounds are unchanged. Status remains `acoustics_preview_status`;
enabling/reconfiguring adds a monotonic `preview_revision`. No new authoring data,
file access, telemetry consumer or UI-only operation is added.

- Vulkan RT borrows its current render BLAS/TLAS and derived geometry buffers.
- Raster borrows an already available RayFusion AS on that viewport device.
- Missing/currently incompatible AS, missing shader, unsupported backend or
  incomplete coverage produces bypass gain 1 with status/error. Audio does not
  build/reclaim RayFusion AS, initialize another device or trigger CPU BVH work.
- `acoustics_trace` remains the explicit synchronous CPU reference diagnostic.

RT custom indices are not unique object IDs: normal meshes can share customIndex
0 and volumes reserve slots. GPU queries therefore use actual TLAS instance order
(`rayQueryGetIntersectionInstanceIdEXT`). Physical responses resolve by canonical
object name through ObjectSubstance. Render material IDs, texture alpha and PBR
transparency are not acoustic coefficients. Legacy RT representatives supply
identity only; geometry comes from current GPU buffers with flat-object coverage
checks, never from per-face facade collections.

## Submission, completion and idle viewports

`AcousticGpuPass` records into the owning RT/raster frame command buffer after
rendering and AS updates, with AS/compute/host memory barriers. Submission
notification associates work with the frame fence. Three fixed acoustic slots
own separate descriptor sets and mapped buffers. Fence polling never waits;
results are collected before renderer fence reset/destruction. Audio never resets
render fences. Busy slots defer the newest pending request; pending replacements
are bounded. Cancellation revokes publication but retains in-flight GPU storage.
Device teardown drains existing render work before releasing the pass.

An idle viewport submits only acoustic compute through an available existing
RT/raster frame slot and its usual fence. It checks readiness without waiting;
no tonemap, image copy/download or sample increment is needed. These submissions
participate in existing AS mutation/resource drain guards. Raster does not publish
old image staging as a new frame. New audio hot paths add no queue/device-idle or
host fence waits. Existing renderer waits and hardware contention remain possible;
zero render impact and hard realtime behavior are not claimed.
Host backend access also uses try-locks: a busy render thread defers acoustic
work instead of making the UI wait for its backend lock. At RT convergence the
enqueue path can schedule the small idle query directly, because the main loop
no longer requests progressive samples.

## GPU processing and budgets

The shader enumerates triangle candidates without closest-hit confirmation or
pruning, deduplicates same-object coplanar crossings, detects coincident
noncoplanar ambiguity and computes six-band transmission products for thin
surfaces or named-object barriers. It binds no textures, lights, GI/probe or
radiance data. Path distance/delay is reported; delay is not applied. Existing
playback gain remains sqrt(mean(six energy fractions)).

Limits: four sources, 2048 AS instance entries, 128 retained crossings and 1024
raw candidates per ray. Overflow, unknown material, AABB candidates, invalid
geometry and ambiguity bypass instead of publishing truncated attenuation.
Traversal order is not presented as ordered detailed hits: only summaries return.

Each result is 64 bytes: at most 256 result bytes copied to CPU per batch.
Coherent host-visible storage needs no staging submission/invalidation. Each
of three slots reserves 131072 instance-table bytes, 128 ray bytes and 256 result
bytes: 394368 requested mapped-buffer bytes total, plus allocation alignment,
descriptor/pipeline allocations and bounded CPU metadata. No second geometry
upload or audio-specific AS is introduced. Slots remain allocated until device
teardown. Small result transfer does not imply a small shared AS residency.

## Currency, coverage and status

Approximately every 250 ms, preview checks backend identity, AS/geometry revision,
material fingerprint, timeline frame, configuration revision and source/listener
pose. Changed inputs bypass the previous result at that update; only matching
completion stamps apply. Backend geometry is revalidated before dispatch. Continuous
edits can remain bypassed. Restore/undo/redo clears configuration as before.

Status retains `status`, `amplitude_gain` and `report`. Report and entry `solver`
fields expose backend, pending/ready/failed state, `cpu_bvh=false` and
`result_bytes_per_source=64`. GPU reports contain `crossings`/`raw_candidates`
instead of detailed `hits`. CPU diagnostic trace retains detailed hits. The
existing external acceptance probe accepts either count and checks GPU metadata.

Missing canonical flat objects (including hidden walls), skipped AS geometry,
non-flat scene objects, hair/procedural AS entries, limits and unresolved identity
cause conservative bypass. An acoustic-only hidden-wall TLAS is not delivered.
Named-object triangle totals must also match canonical flat indices; partially
represented objects or mismatched render tessellation are not silently accepted.
Raster checks geometry/skin revisions and placement signatures; RT checks actual
TLAS count against its mapping. These checks do not claim all render representations
are covered. GPU float geometry/distance-scaled tolerance differs from CPU doubles;
large coordinates, very thin surfaces and grazing paths need numerical acceptance.

Spectral filtering, coefficient smoothing, wall thickness, reflections and GPU
PCM processing remain separate stages. First-use pipeline creation, metadata and
material resolution are host work; no measured total performance is claimed.

## User build and final acceptance

1. Compile `acoustic_segment.comp` with the normal shader script and build Studio
   C++. Deploy `acoustic_segment.spv` beside the other shaders. Four new backend
   modules and the focused scene module are registered in VS; CMake uses its
   existing recursive source glob.
2. With Studio open, use a small audio-empty flat scene and a ready Vulkan RT AS,
   then raster with a ready RayFusion AS. From an external terminal run
   `python scripts/test/rt_test_acoustics_preview_ipc.py --report <report.json>`.
   The existing probe was updated but not run; no new test script was introduced.
3. Compare one/two barriers, clear path, unknown material, thin surfaces, zero and
   unequal transmission, and native PCM ratios against the CPU diagnostic. Include
   coplanar edges/ambiguous boundaries and documented float tolerances.
4. Check idle/converged updates without extra samples/image readback; busy-slot
   deferral; raster presentation and RT resumption; resize/ring recreation.
5. Change source/listener/walls/material/configuration during work; disable,
   remove/recreate, undo/redo, switch backend/scene. No obsolete gain or invalid
   fence/descriptor/AS access may occur.
6. Hidden walls, unsupported geometry/backend, missing SPV, stale/unavailable AS
   and crossing/candidate overflow must report bypass with no CPU fallback.
7. Measure frame/GPU latency, residency, transfer bytes, busy-slot behavior and
   responsiveness. Static structure is not runtime or audio-quality acceptance.

Static checks cover project XML/registration, Python AST, C++/shader ABI layout,
fence-hook ordering, forbidden waits/downloads in new modules and diff whitespace.
They do not compile or execute shaders.
