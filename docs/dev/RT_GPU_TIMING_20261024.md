# Opt-in RT GPU timing capture

The first timing layer uses Vulkan timestamp queries around the camera RT
dispatch, rather than CPU wall time from viewport.render_frames (which currently
calls the CPU renderer) or the Vulkan adapter's constant-zero ms_per_sample.
It is intended to diagnose expensive initial samples and long GPU work that
coincides with desktop/mouse stalls. It does not by itself fix GPU saturation.

## Normal rendering cost boundary

- Default off, transient; not saved in projects or enabled on startup.
- No query pool allocation until explicitly armed.
- Disabled frame paths issue no timestamp commands, shader clock operations,
  atomics, result reads, extra submissions or extra fence/queue/device waits.
  Only inert CPU guards remain at integration points.
- GPU atomics in the older Volume Counters feature are independent. Keep those
  OFF when taking uncontaminated timing captures; collect counts separately.
- Explicit arming creates/resets the query pool in one setup submission and
  waits for that setup once. This happens only on the user's capture request,
  outside measured samples. No new synchronization is added to captured frames.
- A capture has 64 unique query pairs and never overwrites in-flight queries.
  After 64 camera dispatches no more timing commands are recorded.
- Refresh only reads available results; it never uses QUERY_RESULT_WAIT_BIT
  or forces GPU idle. Disabled captures can still retrieve pending results on
  an explicit refresh. Pools are retained until device teardown, which already
  drains in-flight work, rather than introducing a wait when switched off.

## What is measured

Each row reports capture_index (0..63), the zero-based progressive sample_index,
render width/height and camera RT dispatch elapsed GPU milliseconds. The four
camera submission paths are instrumented: standalone trace, fused HDR copy,
fused tonemap/readback and async frame slots. Photon tracing, aerial preparation,
tonemap, copy, presentation and UI work are outside the timestamp bracket.
Timestamp scheduling/dependency effects can affect dispatch elapsed time; this
is not exclusive per-instruction shader time or a full frame time.

The host metrics separately measure existing synchronous submit fence waits
and existing async frame-slot waits. They are CPU wall milliseconds, NOT extra
GPU times and must not be added to GPU dispatch ms. They cover those specific
wait sites only, not all renderer/resource/denoiser waits. A measured flag and
wait count distinguish an unobserved wait path from a cheap observed wait.

No solid-probe / SDF / camera-density / shadow-march shader-clock attribution is
claimed by this layer. Whole-dispatch GPU times plus separate existing counters
first distinguish long GPU work from host waiting; shader-component attribution
requires a separately gated profiling shader path, not ordinary Vulkan timestamps.

## Canonical controls

Volume Performance panel: "Capture next 64 RT dispatches", manually reset
accumulation, then "Refresh RT GPU timings". The panel never polls GPU results
automatically. It shows mean/max GPU ms, host waits, a dispatch plot and a copyable
JSON report. The checkbox and APIs share the same service; there is no scene or
simulation mutation and no automatic accumulation reset on toggle.

Python:

```python
rt.render.set_gpu_timing(True)
rt.reset_accumulation()
# Return to the normal render loop. Query later, never wait inside the script UI.
result = rt.render.gpu_timing_stats()
rt.render.set_gpu_timing(False)
```

External named-pipe IPC (scripts/test/rt_ipc.py):

```python
c.call("render.set_gpu_timing", enabled=True)
c.call("reset_accumulation")
# Let the natural Vulkan loop render; poll later from this external process.
report = c.call("render.gpu_timing_stats")
c.call("render.set_gpu_timing", enabled=False)
```

Do not call viewport.render_frames for Vulkan timing. Keep the camera,
resolution, frame, materials, lights and adaptive settings fixed between trials.
Arm while converged, then reset accumulation so capture_index 0 corresponds to
sample_index 0. Check rows rather than assuming they match. Sum rows 0..15 and
0..63 for equal-dispatch comparisons; GPU time excludes presentation. Adaptive
sampling may change active pixel counts between different scene configurations.

## Validation and failures

Setter accepts only a boolean enabled field; unknown/missing/type-invalid fields
reject before mutation. Stats accepts no fields. Setter requires Render access;
stats is Read. Both fail clearly without an initialized Vulkan render device.
Unsupported queue timestamps or query allocation failure reject arming. Rearming
while queries are pending rejects without overwriting the current capture.
The counter's timestampValidBits and timestampPeriod are used for wrap-safe
conversion; no GPU clock frequency is guessed. Query read failures/submission
failures are reported, not silently converted to 0 ms. unavailable/unmeasured
results have no numeric mean/max. Device destruction clears capture resources.

Non-build checks: python scripts/test/check_rt_gpu_timing_contracts.py, existing
RT optical contracts, phase contracts, IPC capability audit and descriptor check.
The new Python tests are executed; no unused C++ test file is added.

User build/live checklist: build application, leave capture OFF and compare normal
FPS; arm/reset/refresh and verify 64 rows with positive GPU times, stable dimensions
and sample indices; disable and verify unchanged scene/normal performance; compare
the same cube-visible/hidden and two-Fog configurations. On an async render path,
check async_slot_host_wait_* independently of sync_submit_host_wait_*. Missing
live GPU execution remains unverified until the user builds this change.

## Live equal-sample GPU ablation after user build

Verified with external IPC at frame 28, fixed camera, 1680x945, solid world and
one point light. Cube visibility/geometry was held fixed. Every capture was
validated as exactly progressive sample indices 0..63; volume atomic counters
were disabled during all timing captures.

| Fog configuration | Authored shadow steps, mean GPU ms | Shadow steps zero, mean GPU ms |
| --- | ---: | ---: |
| Snow + vapor | 77.59 (repeat 74.09) | 35.10 |
| Snow only | 38.42 | 29.92 |
| Vapor only | 39.15 | 27.08 |
| Both Fog hidden, SDF retained | 12.94 | — |

Both-Fog shadow ablation reduced mean dispatch time by 39.0..42.5 ms, compared
with 8.50 ms for Snow alone and 12.07 ms for vapor alone. This confirms the
user's manual diagnosis of a large joint-medium/shadow interaction cost. The
shadow-off ablation changes illumination and adaptive/secondary path work; the
differences are not exclusive per-stage GPU times and are not additive material
costs. Baseline repeat varied about 4.5%; it does not explain the large ablation
delta. No renderer implementation or quality settings were changed permanently.

All authored materials/visibility, camera, frame, world, lights and physical Fog
statistics were verified restored. GPU timing and atomic counters were then
explicitly left OFF for normal use; the original reports and all raw timestamp
captures are preserved in `rt_gpu_ablation_20261024.json`. This runtime check
supersedes the pending-live status for the timestamp capture layer above.
