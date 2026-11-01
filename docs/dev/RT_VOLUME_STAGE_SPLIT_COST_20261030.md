# First live split-stage cost check, 2026-10-30

External `scripts/test/rt_ipc.py` verification found Vulkan RT ready with
`volume_fog_closesthit`, `volume_surface_closesthit` and `volume_intersection`
installed. Driver pipeline creation reported 205.364 s, with no per-stage
creation feedback. This confirms split driver acceptance, not image parity or
a compiler-time improvement against a cache-matched legacy run.

The existing scene was paused at frame 99. Both gas fields were visible.
Two independent GPU captures used sample indices 8..23, 1680 x 945, with volume
counters disabled. A separate full-viewport counter pass followed eight samples.

| Capture | GPU dispatch median |
|---|---:|
| Current split, run 1 | 139.675536 ms |
| Current split, run 2 | 136.844592 ms |
| Historical legacy A+B | 123.110928 ms |
| Historical legacy A+B repeat | 122.921888 ms |

## Same-build legacy restart control

The user reopened legacy mode. External IPC verified installed
`volume_closesthit`/`volume_intersection`, with neither split stage present, RT
ready. Two independently rearmed captures at the same dimensions and sample
indices measured **139.463824 / 144.265696 ms**. Driver creation reported
220.797 s, again without per-stage feedback; no cache-matched compile-time
claim is made. Raw artifact:
`docs/dev/.captures/rt-volume-legacy-control-20261030.json` (ignored).

Camera, frame 99, gas material values and render settings match the split
capture, but gas fields changed again. Vapor density sum is 600.626307 versus
611.127454 in split; pilot sum is 179.780967 versus 186.455951. Density and
temperature hashes differ, and active-cell counts differ as well. This is not
merely a publication-version mismatch. Current field identity was stable within
each capture, but was not preserved across reopening. The mechanism causing the
different snapshots is not established by these measurements.

The two modes currently occupy the same approximate GPU-time band, with no
demonstrated large split-only penalty and no demonstrated speedup. A causal
percentage comparison remains invalid. Repeated startup A/B runs without
preserving voxel identity should stop; any next rigorous A/B must preserve a
frozen field snapshot or compare pipeline variants inside one process.

Legacy counters per traced path: camera density 13.71089, shadow density
19.66587, majorants 9.62521, light-cache hits 0.001540, step-budget exhaustion
0.11356. These are similar workload indicators to split, not component timings.
Timing/counters were restored disabled, region [0,0,1,1], with Pause/frame 99
confirmed afterward. No camera, material, visibility or physical-step edits
were made. No build or application launch was performed by the agent.

**The historical rows are not a valid controlled split/legacy A/B.** Camera,
frame, listed render settings and all historical gas material parameters match,
but both density hashes and the temperature hash differ. Relative to that
report, vapor density sum rose 6.387% and pilot density sum rose 7.845%; active
cell counts also changed. Geometry, SDF and light identity across runs is not
established. The apparent higher GPU time cannot be attributed to stage splitting
from this comparison. No speedup is established either.

Current counter indicators per traced path: camera density samples 13.63385,
shadow density samples 19.20386, majorant queries 9.62636, light-cache hits
0.001785, step-budget exhaustion 0.11335. They are work indicators, not component
timings; they do not isolate the contribution of shadows or driver resources.

The gas hashes, camera, frame, materials and render settings remained identical
through the current captures. No visibility, camera or simulation control was
edited and no physics step/build/app launch was performed by the agent. Timing
and counters were restored to disabled, full counter region [0,0,1,1], frame 99
and Pause verified over IPC afterward. Accumulation was reset for measurement.
Named-pipe sandbox error 5 was handled by rerunning externally with escalation.

The collector now explicitly drains/rearms GPU timestamps for each repetition.
An initial attempt reused capture 1 for capture 2; that duplicate-repeat report
is marked rejected and excluded from the table. Three mock collector tests cover
independent rearming, fixed sample selection, context rejection and control
restoration. No shader/source implementation was changed for this measurement.

Raw accepted artifact: ignored
`docs/dev/.captures/rt-volume-split-cost-20261030.json`.
Rejected duplicate-repeat attempt:
`docs/dev/.captures/rt-volume-split-cost-20261030-rejected-repeat.json`.

The planned restart control below has now been performed and rejected as a
causal A/B because voxel identity changed; retain it as the historical procedure:
restart the **same executable and shader files** with
`RT_VK_SPLIT_VOLUME=0`, reopen the same paused scene without regenerating fields,
and compare to the accepted current context. Field hashes must match before a
timing delta is accepted. Keep dimensions, sample range, camera and all settings
fixed; require installed legacy `volume_closesthit` without the two split stages.
Do not rebuild between captures. The collector accepts the current report as
its next `--baseline`; it reports context differences rather than silently
treating changed fields as equal. Startup selection prevents switching this
policy via the current IPC API; no hot-reload mechanism was invented.
