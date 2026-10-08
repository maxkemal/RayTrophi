# Matter disk cache playback

Cache restore now publishes a version newer than both the previous runtime
snapshot and the restored snapshot. The render bridge and particle label/view
caches therefore observe every restored frame, including disk frames whose
serialized format does not contain a runtime version.

Restore also clears the uploaded particle counts for primary and Matter liquid
compute buffers. Existing device handles must not expose positions from the
previous live solve while playback uses a host cache snapshot.

Render-only disk caches omit the secondary Matter liquid grid. Restore rebuilds
that missing layout using the baked logical bounds and the canonical phase layout
resolver. This prevents the empty grid's default 1-metre voxel from overriding
the authored fluid visual radius. Physical grain radii retain their existing
solver-owned behavior. RAM snapshots with a liquid grid retain that grid.

## Playback allocation and I/O follow-up

The first radius fix allocated a complete secondary liquid solver grid for every
disk frame, although playback only consumes its layout. Restore now sets the
layout without velocity, pressure, divergence, density or solid work arrays.
The canonical phase-storage synchronizer detects missing pressure storage and
allocates the solver grid before a subsequent live step. Storage-match queries
also reject a layout without solver storage.

Disk-loaded state vectors are moved into the runtime instead of deep-copied.
RAM callers retain value semantics; temporary decompressed snapshots also move.
Primary grid allocation skips unused gas channels for specialized fluid domains.

Positions, both material-coordinate generations and foam positions use bulk
reads when Vec3 matches the existing three-float format. Mass and pore fields
use bounded 4096-particle block reads with the original interleaved field order.
The format version and writer remain unchanged; read failures still reject a
truncated frame. Particle identity validation remains unchanged.

Profiling adds `sim.cache.disk.read_system`,
`sim.cache.disk.allocate_primary_grid` and `sim.cache.restore.prepare` to the
existing performance recorder. Primary solver-grid allocation, granular
sidecars, per-frame material routing and CPU instance updates remain costs of
this playback path. These changes do not guarantee a particular frame rate.

The shared runtime restore applies to UI, scripting and IPC playback; no new API
operation or cache format is introduced. Existing caches can be replayed.

## User build and verification

1. Build RayTrophi Studio using the normal project configuration.
2. Open an existing baked Matter scene with fluid and granular carriers.
3. Scrub between frames with visibly different positions, then play the baked
   range. Confirm positions change even when particle count stays constant.
4. Compare the fluid sphere radius before baking and during cached playback.
   Change Voxel Radius Factor / Visual Size Multiplier and confirm the cached
   display updates. Granular carriers should retain their physical grain radius.
5. Test a Matter domain with a liquid phase voxel override, and a specialized
   fluid domain. Confirm both use their intended voxel size during playback.
6. Repeat after a live solve and after reopening the project, then rewind and
   resume live simulation outside the baked range. Check for stale positions.
7. Compare the same 10-second baked range in solid mode before and after this
   build: observed playback FPS, CPU utilization, peak process working set and
   frame particle count. Use the same viewport, timeline FPS and render settings.
8. Check a mixed wet-grain cache for unchanged mass, pore values, materials and
   foam appearance. Include more than 4096 particles to cross I/O block boundaries.
9. Disable the disk cache, reset and step live simulation. Confirm both Matter
   phase grids receive solver storage and no stale cache positions appear.

Builds and live application verification were not run by Codex.

## Live cache inspection, 2026-10-08

An external `scripts/test/rt_ipc.py` probe inspected the user's open scene;
the application was already running. No build, bake or project save was run.
The timeline was paused at frame 250. Frames 30 through 34 were read from the
existing disk cache, then the timeline was returned to frame 250.

The Matter domain contained 99,166 fluid carriers, a 128 cubed grid, and eight
visual children per carrier (793,328 virtual grains). Five restored frames took
233.875 ms in `sim.timeline.restore_frame_disk` (46.775 ms/frame), including
220.620 ms in `sim.cache.disk.read_system` (44.124 ms/frame) and 56.379 ms in
primary grid allocation (11.276 ms/frame). Render synchronization took
204.510 ms (40.902 ms/frame). These nested timings must not be added together.
Paused event waits are included in `loop.frame`; that scope is not a playback
FPS measurement. No solver-step scope advanced during the probe.

The follow-up reader change bulk-reads substance tags, constitutive model,
identity and persistent flag columns. Identity validation uses one contiguous
sorted scratch copy instead of a node allocation per particle. Zero, duplicate,
and out-of-range identities still reject the frame; the original particle
order and file format are preserved. Its timing is exposed as
`sim.cache.disk.particle_identities`. This source change has not been built or
measured live. Primary grid allocation and render publication remain costs.

After building, repeat the same cached range and compare the above scopes.
Also replay a mixed-substance cache and confirm particle labels and identities
remain intact. A lower CPU percentage alone is not evidence of faster playback.

## Second live inspection after the user build

The same five frames measured 46.864 ms/frame for disk restore, 43.406 ms/frame
for the reader, 11.471 ms/frame for primary allocation, and 37.598 ms/frame for
render synchronization. Identity read/validation was only 0.235 ms/frame. The
previous change did not resolve the overall bottleneck. The paused timeline was
returned to its original frame 0. No project settings were edited or saved.
Frame 30's file is 34,517,129 bytes; five separate Python raw-file reads took
10.2 to 11.7 ms each. This is a different reader and is only a baseline, not a
measurement of the C++ decoder or continuous playback FPS.

The next source change makes the primary Matter grid render-only on disk restore:
layout and saved scalar fields are restored, but velocity, pressure, divergence,
solid and tile scratch arrays are not allocated. Specialized fluid/gas paths
retain their existing allocation. Before a subsequent live solve, the shared
phase synchronizer allocates missing scratch and preserves saved scalar fields
and gas mass/energy when the layout/type/channels are unchanged. Layout changes
retain the existing reset behavior. The reader also configures a 1 MiB file buffer
before opening the frame. The disk format and writer are unchanged.

Added timing scopes: `sim.cache.disk.scalar_fields` and `sim.render.fog_volume`.
These changes have not been built or measured in the application. After the user
build, compare the same five frames and check Matter gas/fire playback as well as
liquid playback. Resume a live step outside the baked range and verify solver
storage is allocated and saved scalar/mass fields are not inadvertently cleared.

## Compact zero scalar fields (cache v12)

The next live probe confirmed disk restores in the open scene despite the
parallel domain-roadmap change to configuration hashing. Frames 30 through 34
took 28.783 ms/frame for restore and 31.272 ms/frame for render synchronization.
Primary grid allocation was negligible. Scalar field reads accounted for
26.263 ms/frame. The original timeline frame 60 was restored.

Frame 30 had four dense 128 cubed scalar columns, all positive zero (density,
temperature, fuel, interaction). They occupied 33,554,432 payload bytes.
Cache v12 marks an all-positive-zero float column with the high bit of its
u64 count and omits the payload. Loading reconstructs the same full-length zero
vector. Empty fields remain empty. Nonzero values, negative zero and NaNs use
the original raw payload; no tolerance or lossy quantization is used. This
reduces disk traffic but still allocates restored scalar vectors in RAM.

The manifest, system, soft-body and rigid-body readers accept v11 and v12.
Older versions remain rejected; an old binary does not understand v12.
Configuration-signature validation is unchanged. A valid old v11 bake can
still play but gains no compact disk encoding without rebaking.

An offline format roundtrip on the existing frame-30 file reduced its predicted
size from 34,517,129 to 962,697 bytes; expansion reproduced every original byte.
This was a Python format check, not execution of the C++ writer/reader. No
existing cache file was changed. C++ live playback and speed are unverified.

User verification after building:
1. Replay the existing v11 cache to confirm compatibility.
2. Bake the same scene to a separate cache directory with v12; keep the old
   cache for comparison. Compare frames 30 through 34, file size, and timings.
3. Check gas/fire with nonzero scalar fields, mixed pore/mass fields, and rigid
   and soft-body cached playback. Check particle motion, radius and materials.
4. Check a truncated frame is rejected, and switching back to live simulation
   hydrates scratch while retaining restored scalar fields where appropriate.

## Live v12 result and Save As cache ownership

The user's rebuilt/rebaked scene used `matter_domain_test1.simcache`, manifest
v12. Frame 30 was 962,697 bytes. Frames 30 through 34 averaged 8.278 ms for disk
restore, 6.542 ms for the reader, 4.973 ms for scalar fields and 34.437 ms for
render synchronization. The original paused frame 0 was restored. No cache was
deleted by the probe. Continuous playback FPS is not inferred from scrub timing.

The user reported deleting the old cache after Save As. Inspection found the
project path changes while the scene's old disk-cache binding survives. The
shared ProjectManager save path now uses ProjectSimulationCache.h: after a
successful Save As changing the canonical cache directory, detach the old binding
and bind only a matching cache at the new project's directory if present. Never
delete/copy/move the old cache. Save Copy and ordinary Save retain bindings.
Save As changing cache ownership during an active bake is refused to prevent
the bake finishing against the previous project's directory. The implementation
is shared by UI and existing API save operations. The fix is not built/live-tested.

User checks: save project A with a bake, Save As B, verify B has no binding to
A's cache; bake/clear B and verify A's files are unchanged. Reopen A and confirm
its cache still loads. Save Copy and same-path Save must retain the current
binding. During a bake, same-path Save may continue, but Save As B must report
that the bake needs to finish or cancel.

## Skip surface publication when all live parcels resolve elsewhere

In the open v12 scene's frame 30, all 12,500 carriers resolved to Splat; Surface
had no live parcels, but its authored default kept the surface resource enabled.
Surface statistics still reported a 22.195 ms build with zero active cells.
The SceneData publication gate incorrectly used the domain's total particle
count rather than the canonical resolved view's live presence.

The gate now calls `view_plan.anyLiveIn(FluidView::Surface)`. Authored surface
resource intent remains stable. The existing empty-content path deactivates
publication and invalidates the upload signature without retiring the resource.
When surface parcels return, the existing reupload gate reactivates it. Fog
publication runs before this gate and keeps its own live-parcel/gas decisions;
whitewater on a surface remains a channel of a live carrier surface, not a
standalone level-set source. Splat drawing remains on its existing bridge.

The shared path covers UI, script and IPC timelines, live steps and disk replay.
This avoids building empty surface SDF/UVW/composition and volume payloads; it
does not claim to accelerate reconstruction of a genuinely populated surface.
Added nested scopes `sim.render.surface.level_set`,
`sim.render.surface.material_coordinates`, `sim.render.surface.composition`
and `sim.render.volume.publish` for the next live comparison. Their times must
not be added to their enclosing render-sync scope.

No build or post-change live measurement was performed. User checks after build:
1. Replay the same v12 cache (no rebake needed) and compare frames 30 through 34.
   In this splat-only scene the surface builder/publisher counters should not
   advance, while splat positions and radii still update.
2. Route body to Surface and verify the liquid appears, then return it to Splat
   and verify no old liquid surface remains. Repeat paused and during playback.
3. Check a mixed Surface/Splat/Fog scene, and gas/fire-only playback, to ensure
   occupied routes still publish. Check rewind to frame 0 then play again.

## Confirmed live result after the surface-gate build

The same v12 cache, frames 30 through 34, measured 0.055 ms/frame in render
synchronization (previously 34.437 ms/frame) and 7.630 ms/frame in disk restore
(previously 8.278 ms/frame). Combined measured scopes fell from 42.715 to
7.685 ms/frame, about 5.6 times less time. No surface reconstruction/publication
scope advanced. Frame 34 had 14,166 carriers, all resolving to Splat, with zero
live Surface/Fog parcels; surface build telemetry was 0 ms. The existing cache
remained valid and the original paused frame 0 was restored. These are five
scrubbed-frame timings, not a continuous playback FPS or a populated-surface
benchmark. Mixed Surface/Fog reactivation still needs user visual verification.
