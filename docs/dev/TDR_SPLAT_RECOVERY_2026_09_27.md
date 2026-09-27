# Splat sphere device-loss investigation — 2026-09-27

Live IPC evidence: 12,500 water particles, `virtual_particles`, granular disabled,
and `granular_virtual_measured=false`. Raster telemetry reports device loss;
recovery reports zero losses, zero attempts, no pending rebuild. The last log
reports raster submission failure after an RT-to-Solid transition. This identifies
an observation point, not the GPU operation that originally caused the reset.

The backend-switch block could take its same-backend early return before reaching
device-loss recovery. Recovery was also conditional on a backend-change request,
and normal initialization ran before the loss handler. The patch makes a pending
loss enter this block independently, bypasses same-backend and OptiX shortcuts,
and skips normal teardown/initialization so the existing autosave, teardown and
backoff handler runs first. Both requested GPU modes are cleared for CPU fallback.

Frame-slot waits, trace-slot waits, geometry one-shot submit/wait and device-idle
wait now report device loss at the observed operation. Geometry update stops when
its drain detects loss. This does not establish or fix the initial TDR cause.

User build and live verification remain required:

1. Start with a small splat scene in Solid; record recovery counters.
2. Leave the app unfocused until idle, focus it, then select Vulkan RT.
3. Repeat with unchanged particle count, then with a pool-growth frame.
4. If loss recurs, preserve SceneLog before restarting. Check that recovery counts
   the loss and schedules backoff without requiring another backend selection.
5. After recovery, confirm a new frame appears. Compare explicit spheres and
   virtual particles with granular disabled before testing granular bulk output.

Do not interpret stale raster triangle counts after device loss as current
geometry counts. A healthy-device test is needed for geometry lifetime analysis.

## Persistent HUD and bounded recovery

The recovery HUD reads the existing loss/pending/given-up state independently
of timed notifications. It names CPU fallback, pending viewport recovery, stopped
recovery, and requested RT return. A sample count above zero is reported separately
from selecting Vulkan; initializing a device alone is not proof of a rendered frame.
RT return uses the existing backend selector. Viewport recovery does not request RT.

Both device initialization entry points respect the driver-reset time gate.
Viewport initialization also respects the stopped state. Three consecutive failed
viewport initializations stop retries; a successful initialization resets this
attempt budget. The existing Python `rt.viewport.retry_device_recovery()` and IPC
`viewport.retry_device_recovery` reset the budget without clearing a pending reset
delay. Existing recovery-status fields report pending/given_up/attempts.

Structural RT operations emit `[VulkanTransition] begin/end` records with CPU wall
duration and pending loss state. These are not GPU timestamps. A missing end record
locates an unfinished CPU call, not necessarily the original faulty GPU command.
Device initialization and failed recovery attempts also have explicit records.

After user build, verify CPU fallback text remains visible through a long stall;
verify retry cannot bypass the reset delay; verify failed initialization stops at
three attempts; retry manually and check that CPU remains selected until an explicit
Vulkan selection. Preserve the transition log from focus gain to the first failure.
