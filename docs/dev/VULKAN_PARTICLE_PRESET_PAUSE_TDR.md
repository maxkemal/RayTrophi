# Vulkan RT Particle Preset Pause TDR

> **Durum:** ARSIV — Cozuldu; TDR kok neden postmortem'i.

## Symptom

Particle-heavy presets such as Ground Burst, Explosion, and Fireball could
trigger a Windows TDR when playback was paused and resumed repeatedly. The
first pause could succeed; the second pause or the following Play operation
often terminated the application in the Vulkan/NVIDIA driver.

Gas-only scenes and the Campfire preset were normally stable. This made the
volume system look suspicious, but gas was an amplifier rather than the root
cause: the failing presets combine a dense volume with hundreds of RT particle
instances and their source BLASes.

## Root causes

### Gas solid probes treated every particle as an embedded solid

The decisive isolation test was changing the same preset domain from Gas to
Fluid: the particle scene then survived repeated pauses. Vulkan's volume
closest-hit shader probes for solid geometry inside a volume AABB so it can stop
the march before an embedded mesh. Simulation particles previously used the
ordinary `0xFF` TLAS mask, so every spark, ember, or debris instance inside the
gas counted as a solid. A detected solid launches additional binary-search
`traceRayEXT` probes. Hundreds of particles therefore multiplied the ray work
inside every affected pixel and bounce until a long dispatch exceeded Windows'
GPU watchdog. Gas-only and low-particle scenes stayed below that threshold.

Transient simulation particles now use mask bit `0x04`. Primary rays use
`0xFF`, so the particles remain visible, while volume solid probes use `0xF9`
and exclude both volume AABBs (`0x02`) and transient particles (`0x04`). Real
scene solids remain eligible for the embedded-solid handoff.

Ground-aligned Gas domains still encounter a real solid: the ground plane. The
old probe returned only hit/miss, then launched another five or six
`traceRayEXT` calls as a binary search for the hit distance. A dedicated
volume-probe payload mode now returns `gl_HitTEXT` from any-hit directly. Both
fog and isosurface branches locate the solid with exactly one probe, preserving
the handoff while bounding the shader workload.

### Vulkan simulation commands escaped their frame

The particle/gas Vulkan compute backend recorded commands between simulation
operations, but its `endFrame()` implementation was empty. Upload/dispatch-only
tails could therefore remain unsubmitted across a timeline Pause/Play boundary.
The pause drain waited for the render backend but not the independent simulation
compute context. A later cache restore or buffer resize could replace resources
still referenced by the recorded tail; a subsequent synchronization then
submitted stale work and could produce a delayed second/third-pause TDR.

The Vulkan simulation backend now submits and fences any outstanding work at
every simulation-frame boundary. Timeline Pause also explicitly synchronizes
the simulation compute context before draining render/RT work.

### Dead pool slots used a singular Vulkan RT transform

The stable particle pool represents inactive slots with scale `(0, 0, 0)`.
Vulkan previously copied those matrices into `VkAccelerationStructureInstanceKHR`
with mask `0xFF`. A zero-scale transform is singular and therefore invalid for
an acceleration-structure instance. Large bursts created many such slots, and a
pause/cache restore could collapse many of them at once. NVIDIA could accept an
initial build but later hang during a TLAS update, producing the delayed
first/second-pause TDR.

Vulkan now keeps inactive slots index-stable using an identity transform and
instance mask `0`. The TLAS upload boundary also validates every transform and
applies the same identity-plus-mask-zero fallback, preventing any other scatter
or simulation path from submitting NaN, infinite, or singular matrices.

### Pause incorrectly requested a full Vulkan rebuild

`Main.cpp` treated every Play-to-Pause edge as a reason to set
`g_vulkan_rebuild_pending`. A playback-state change is not a structural scene
change. For burst presets this unnecessarily concentrated particle instance
recreation, acceleration-structure teardown/rebuild, and volume work into the
same frame, which could exceed the Windows GPU timeout.

OptiX retains its historical pause rebuild. Vulkan now only resets
accumulation on this edge. Actual structural mutations continue to request
their own rebuild through the existing structural-change paths.

### The particle instance pool was shrinking despite its contract

`ParticleRenderBridge.cpp` documented a monotonic instance pool but used:

```cpp
const bool pool_grew = cap != inst.size();
```

Consequently, restoring a cached/paused frame with fewer particles shrank the
pool and marked the scene structural. Replaying then grew it again and caused
another structural rebuild. Repeated Pause/Play cycles therefore produced
exactly the expensive rebuild pattern that the monotonic pool was intended to
avoid.

The pool now grows only when `cap > inst.size()`. Slots beyond the current
particle SoA remain allocated and are collapsed with scale zero. This preserves
stable per-slot source/material bindings and keeps ordinary pause/resume
updates on the transform/refit path.

### RT visibility was implemented as a structural deletion

Disabling a particle system's **Render in Ray Tracing** option cleared its
instance array and requested a full Vulkan/OptiX rebuild. Re-enabling it then
recreated the pool. Visibility is not a topology mutation: the bridge now keeps
the pool and source bindings alive, collapses its transforms, and requests only
a transform/TLAS refit. The same rule applies to the particle RT debug-disable
path.

### Particle-to-gas state validation

The particle-to-gas deposit correctly multiplied density, temperature, and fuel
rates by `dt`; velocity is an authored impulse used by the explosion presets.
Temporarily changing that impulse to a rate changed the visible explosion but
did not remove the TDR, so the original impulse response was restored.
Incoming and accumulated face velocities remain finite-checked and bounded.
Particle SoA lengths, world/grid coordinates, and channel indices are validated
before a grid write, so stale cache tails or NaN particle state cannot turn into
an invalid gas-grid address.

### Live Gas buffers were written while Vulkan RT still read them

The Vulkan volume path consumes device addresses for the simulation's live
density, temperature, fuel, and flame buffers. Interactive Vulkan rendering can
keep two asynchronous frame slots in flight. The next simulation frame reused
and overwrote those same buffers after synchronizing only the simulation queue;
that does not wait for reads on the RT queue. Dense particle-driven explosions
made the race reproducible after several frames even without Pause.

The interactive simulation driver now drains Vulkan RT before advancing a live
or playing simulation. This deliberately serializes this shared-buffer boundary
until the fields use per-frame buffers or an explicit cross-queue semaphore.

### Accumulation reset retained stale volume temporal history

Every changed Gas frame reset ordinary path-tracing accumulation, but
`resetAccumulation()` left the separate volume temporal ping-pong history valid.
Old volume samples could therefore be accepted after the grid moved and appear
as stationary trails. An accumulation reset now invalidates volume temporal
history as well.

## Why Campfire and gas-only scenes did not reproduce it

- Gas-only scenes have no large RT particle instance pool to shrink and grow.
- Campfire emits relatively few continuous particles, so the forced work often
  remained below the TDR threshold.
- Burst and mushroom-style presets create hundreds of particles at once and
  combine them with dense gas, making the unnecessary rebuild substantially
  more expensive.

## Regression checklist

1. Run Ground Burst for several seconds and repeat Play/Pause at least ten
   times.
2. Repeat with Explosion and Fireball/mushroom-style presets.
3. Verify Campfire and gas-only scenes still pause and resume normally.
4. Confirm normal playback changes particle transforms through refit and that
   a genuine source/material/topology change still requests a structural
   rebuild.
5. If a similar TDR returns, first inspect any path that sets
   `g_vulkan_rebuild_pending` on a non-structural state transition and any
   supposedly stable instance pool that shrinks or reorders its slots.
6. Never hide a retained Vulkan RT instance with zero scale. Preserve its
   transform with an invertible matrix and set its instance mask to zero.
7. A timeline mutation boundary must drain both the simulation-compute context
   and the render backend; waiting for one does not flush the other.
8. Keep transient simulation particles on their dedicated RT mask. Adding them
   back to the volume embedded-solid probe recreates multiplicative ray work.
9. Never pass non-finite particle state to `worldToGrid`, and bound accumulated
   MAC-face values without silently changing an authored preset impulse.
10. A live simulation buffer cannot be overwritten while an asynchronous RT
    frame still reads its device address. Use per-frame buffers, a cross-queue
    semaphore, or drain the RT consumer at the ownership boundary.
11. Invalidating ordinary accumulation after a dynamic volume update must also
    invalidate the separate volume-temporal history.

## OptiX playback boundary follow-up

The simulation may still use the Vulkan compute backend while OptiX renders.
Live dense field addresses are Vulkan device addresses and must never be copied
into an OptiX/CUDA volume packet; OptiX now remains on its CUDA/NanoVDB route.

Compute-backend recreation exposed another stale-handle problem. Vulkan buffer
IDs previously restarted at `1` for every backend instance. A cached handle from
the previous instance could collide with an unrelated buffer in the new
instance, pass the size-based stale check, and make `ensureComputeBuffer()`
resize or destroy the wrong `VkBuffer`. Vulkan simulation buffer IDs are now
process-unique. Structural destroy/resize also waits for the full Vulkan device,
because a compute-queue fence alone does not cover a live field read from
another queue.

The Vulkan simulation backend also borrows the Vulkan render backend's
`VkDevice`; it does not own an independent device. Backend switching previously
destroyed that render device while `SimulationWorld` retained the compute
backend. The next Play then failed inside `vkCreateBuffer` on the dead device.
The simulation backend is now released before Vulkan renderer teardown.

Finally, GPU-Compute auto selection was sticky: once the context was Vulkan,
choosing CUDA kept Vulkan instead of retrying CUDA. CUDA-preferred selection now
actually replaces Vulkan, and handles from a previous API/backend are discarded
without passing their opaque IDs to the new backend's `destroyBuffer`.

## Gas authoring and resize follow-up

`FluidGrid::allocate()` previously used `vector::resize(newSize, 0)`. When a
domain resolution changed, C++ preserved the old vector prefix even though the
flattened X/Y/Z strides had changed. The solver then interpreted old rows and
slices using the new dimensions, producing repeated or mirrored small blocks.
Layout changes now initialize every staggered/scalar field from scratch and
invalidate all lazy collider, face-weight, level-set, and tile caches.

Live dense Vulkan Gas does not need NanoVDB's sparse padding threshold. Applying
the default `0.04` cutoff discarded valid low-rate smoke deposits before
extinction, so explosion presets could show heat/fire with almost no smoke.
Dense fields now use only a small numerical-noise cutoff.

Object-bound particle emitters now resolve their origin from current world
bounds rather than a cached import-pose centroid. Point emitters also expose
their actual world position in the UI; `local_offset` remains the optional
source-relative adjustment.

Explicit per-domain Vulkan selection now takes precedence over the global
auto-GPU preference. Previously a resize/resync could leave the combo displaying
Vulkan while the shared compute context selected another backend; choosing
Vulkan again manually rebuilt the expected context. The UI now distinguishes
the requested backend from the active runtime backend and Gas domains report
whether their last step was full core-GPU, partial/hybrid, or CPU fallback.
Vulkan allocation failures include buffer name, byte count, and `VkResult` in
the Console instead of failing silently.

## Distinction from the hair pause crash

The earlier hair failure was a buffer-ownership error: a borrowed external
geometry buffer was destroyed during a full rebuild and later destroyed again.
This particle case shared the Pause/Play trigger but was rebuild amplification
leading to TDR, not the same double-free. Keeping that distinction prevents a
future driver-side stack from sending the investigation down the wrong path.
## Volume shader data-contract audit

The live Gas render path has a stricter channel contract than a general VDB:
Vulkan RT reads the simulation's dense `density` and `temperature` buffer
addresses directly. The old UI offered `fuel` and `interaction` selections, but
changing the string did not change those GPU addresses. Gas now displays the
fixed channels as read-only; general VDB assets retain channel selection.

Temperature is converted from simulation heat to Kelvin before rendering.
`temperature_min` / `temperature_max` are now published for standard VDBs as
well as Gas and are used consistently by Vulkan RT, OptiX/CUDA, and the CPU
marcher for both blackbody and color-ramp evaluation. Normalized fallbacks map
into the authored interval instead of using a hidden 1000–4000 K range.

The UI also prevents zero/inverted density-remap and temperature ranges. Vulkan
live Gas deliberately uses a very small numerical density cutoff instead of the
authored sparse-VDB cutoff; otherwise rate-deposited smoke disappears. This is
now explained in the UI. `edge_falloff` remains CPU-only until the fixed GPU
volume-instance ABI carries that field, so its label now states that limitation.

## Late-frame Gas boundary slowdown and velocity burst

The Vulkan MAC-grid sampling helpers used to clamp integer indices after an
advection backtrace left the domain, but retained the original floating sample
coordinate. The resulting interpolation fractions could be below zero or above
one. When a buoyant mushroom cap reached the top boundary, the shader therefore
extrapolated velocity instead of interpolating it. That produced a delayed
velocity/vorticity burst, apparent reheating and a sharp late-frame workload
increase.

`sim_grid_advect_velocity`, `sim_grid_advect_scalar`, and
`sim_grid_maccormack_scalar` now clamp floating MAC sample coordinates before
calculating indices and interpolation fractions. Open-boundary scalar fetches
still return their configured background, so smoke, fuel, and heat can leave
the domain; only invalid velocity extrapolation is removed.

The velocity push constants now carry the boundary mode as well. Open-domain
velocity samples return zero outside the MAC grid, and the MacCormack correction
is suppressed when its reverse trace exits the domain. This prevents the upper
wall from behaving like a clamped/closed velocity reservoir.

A separate late-frame cost cliff occurred when density reached zero. Dense
Vulkan volumes have no NanoVDB topology to skip empty space, so an empty AABB
made every ray consume the full march budget (opaque volumes often terminate
earlier). The live-volume manager now keeps addresses and TLAS slots stable but
marks whether renderable density exists. Empty domains are omitted from the
volume packet, leaving their existing SSBO slot inactive until density returns.

Near-empty domains were still expensive because a single residual cell kept the
whole allocation render-active. Gas telemetry now records the inclusive active
density-cell bounds. The live Vulkan packet transforms a one-cell-padded version
of those bounds into the volume's local AABB, giving dense Gas the occupied-box
equivalent of NanoVDB topology skipping.

Finally, an open pressure boundary could generate a normal velocity pointing
back into the domain after gradient subtraction. At the top face this appeared
as hot gas being pumped downward. Vulkan's grid gradient kernel now preserves
outward velocity on open faces but clamps projection-created inflow to zero.

## 2026-07-29: frame-linear Vulkan VRAM growth

The Mushroom preset was observed to grow process VRAM by roughly 1 GB/s even
in Solid view and during a disk-cache bake. The same scene used about 1.8 GB
VRAM with CUDA compute, so this growth is not the required dense-grid working
set.

Two Vulkan allocation hazards were corrected. Adaptive resolution changes no
longer destroy and recreate the complete grid buffer set; individual buffers
retain capacity and grow only when required. `resizeBuffer` now also preserves
the original usage/device-address contract and reacquires `unordered_map`
iterators after `createBuffer`, which may rehash the map.

During diagnosis, `SimulationComputeVulkan.cpp` temporarily reported live
buffer and staging totals every 24 frames. The instrumentation was removed
after the owner was identified so normal sessions do not accumulate console
noise.

The diagnostic produced the decisive signature:

```text
frame=0  live_buffers=60
frame=24 live_buffers=1500
frame=48 live_buffers=2940
```

This was exactly 60 leaked buffers per simulation frame. Vulkan
`createBuffer()` assigned the numeric handle ID but did not assign
`handle.backend`; `ComputeBufferHandle` therefore retained its default `CPU`
tag. On the next frame `ensureComputeBuffer()` treated every Vulkan handle as a
foreign-backend handle, allocated a replacement, and intentionally did not
destroy the supposedly foreign allocation. CUDA and CPU already set this tag
correctly. Vulkan now sets `h.backend = type()` before returning the handle.

## Timeline cache replay volume parity

Live Vulkan Gas samples the simulation's dense buffers directly, while a RAM
timeline-cache restore temporarily rebuilds a host NanoVDB snapshot. The cached
arrays were complete; the visual mismatch came from different sampling
contracts:

- the dense sampler treats `(i,j,k)` as a cell-centred value at
  `(ijk + 0.5) * voxel_size`;
- the live NanoVDB conversion previously placed that value at the voxel corner;
- sparse conversion removed density below `1e-4`, then the volume shader often
  applied the authored VDB cutoff of `0.04`;
- cached temperature was discarded below 300 K after the bridge's Kelvin
  scaling.

Together these differences exposed bright grey/white voxel shells during cache
playback even though the first live simulation pass looked smooth. Live
NanoVDB conversion now translates its transform by half a voxel, retains
density down to `1e-6`, retains scaled temperature down to `1e-3`, and marks
live-simulation NanoVDB packets with the dense path's `1e-5` shader cutoff.
Live and cached playback were visually verified to match.

## Follow-up: external EmberGen VDB sequence parity

An EmberGen campfire VDB sequence currently renders correctly on the CPU
reference path but differs on both GPU backends:

- Vulkan appears to cut density/topology too aggressively, producing a harder
  result.
- OptiX appears substantially less dense.

Keep this separate from live Gas timeline caching. The external sequence uses
authored sparse-file transforms, value ranges and background/topology, so it
must not inherit the low numerical cutoff intended specifically for converted
live-simulation grids. Future investigation should compare the same frame's
density min/max, active bbox, voxel transform, shader density cutoff/remap and
temperature scaling at the CPU, Vulkan and OptiX upload boundaries.

## Partial timeline cache restart at its end frame

Reproduction:

1. Add an explosion/particle Gas preset.
2. Play from frame 0 and stop at an intermediate frame, for example frame 50,
   so only frames 0-50 exist in the timeline cache.
3. Rewind to frame 0 and play the cached interval again.
4. When playback reaches frame 50, the particle/Gas simulation does not
   continue from the restored frame-50 state. It restarts with behavior
   resembling frame 0.
5. Repeating pause, rewind and play reproduces the same reset at the partial
   cache boundary.

The first uninterrupted play is correct if it is allowed to reach the intended
final frame before pausing. This confirmed a cache-boundary/resume-state bug,
not a solver or renderer failure.

Root cause: the RAM frame cache restored the Gas/Fluid grid and discrete
particle SoA, but not the particle system's runtime emission state. Rewinding
cleared `burst_consumed`, fractional emitter/flow accumulators and deterministic
spawn serials. Cached frames still looked correct because their particle/grid
data was restored directly; the first step beyond the cache then saw a freshly
armed one-shot emitter and fired the explosion again as if it were frame zero.

The particle frame snapshot now also captures and restores emitter
accumulators, burst-consumed flags, flow emission counters, deterministic spawn
serial and moving-collider history. A true rewind resets the spawn serial and
collider history, while resuming at a cached boundary restores their
frame-matching values. This preserves the valid cached interval and continues
the first uncached frame from its real predecessor instead of restarting.
