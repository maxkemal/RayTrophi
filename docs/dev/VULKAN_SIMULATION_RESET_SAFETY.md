# Vulkan Simulation Reset and Rewind Safety

> **Durum:** REFERANS — Yururlukteki muhendislik kurali (2026-08-09).

Status: active engineering rule, recorded 2026-08-09.

## Observed failure class

A high-polygon melt source/collider can be stable by itself, as can APIC
SurfaceSDF fluid and gas. Windows TDR becomes likely when timeline rewind,
cache restore, pause, or a viewport transition concentrates their expensive
resource changes in one render frame:

- source-mesh vertex/normal write-back and BLAS refit/rebuild,
- fluid SurfaceSDF destruction, reconstruction, and RT binding replacement,
- gas buffer/volume publication,
- particle instance-pool changes and TLAS update.

This is a cross-system resource-lifetime problem, not a reason to disable one
simulation feature.

## Mandatory rules

1. Timeline Start/End is playback-range metadata. Editing it must not clear the
   simulation cache, change the resident frame, or request frame-zero rewind.
   Extending 250 to 500 preserves cached frames and computes only new frames.
2. Geometry, fluid SurfaceSDF, and gas destroy/recreate work must not be grouped
   into one render frame. A transition scheduler must spread work over frames.
3. Resource replacement order is: stop new submissions, fence every owning
   queue, detach old bindings, retire old resources, publish replacements, then
   resume rendering. Render and simulation-compute queues are separate owners.
4. Per-frame device-wide idle is forbidden as a workaround. Fences belong only
   at ownership changes, rewind/cache restore, capacity changes, and structural
   rebuild boundaries.
5. Heavy deforming-mesh BLAS refits are budgeted. Chemistry and APIC continue at
   full rate; only visual mesh write-back/refit cadence may be reduced.
6. Melt must not continuously recook ObjectMeshSDF. Rebuild is an explicit user
   or command action and must run through the safe transition scheduler.
7. Reset is idempotent. Geometry, volume, instance, and simulation buffers must
   have one clear owner and may be retired only once.
8. Required telemetry: queued/completed geometry refits, SurfaceSDF rebuilds,
   gas uploads, retirement-queue depth, fence duration, and transition cost per
   frame. A large operation must be attributable before changing code.

## Required stress gate

Use one scene containing a high-polygon mesh emitter/collider, melt deformation,
APIC SurfaceSDF fluid, and gas. Repeat:

- Play and Pause,
- frame-zero rewind and cache restore,
- End Frame expansion from 250 to 500 while playing and while paused,
- Solid to Rendered and Rendered to Solid transitions.

Playback-range edits must produce no reset. Structural resets must complete
without a one-frame workload spike, invalid resource access, or Windows TDR.

