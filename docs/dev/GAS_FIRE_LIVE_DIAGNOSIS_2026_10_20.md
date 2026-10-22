# Ground Burst live diagnosis — detail and internal fire lighting

User supplied Houdini reference and RayFusion screenshot. Inspected the already
open Studio through external `scripts/test/rt_ipc.py`; no material, solver,
timeline, renderer or quality settings were changed. The plume measurement
requested a readback, not a simulation step. No screenshots, build or app launch.
Pipe access initially failed with Windows error 5; the same external probe
succeeded with sandbox escalation.

## Measured state

- Domain `Ground Burst Gas`, grid 283 x 213 x 283 = 17,058,957 total cells;
  voxel 0.02826855 world units. Domain extent approximately 8 x 6 x 8.
- 1,466,689 active density cells from domain telemetry. Plume measurement at
  threshold 0.01 counted 335,626 cells (1.97% of total). These are different
  thresholds/counters, not contradictory resolution measurements.
- Actual RT and viewport packets both use NanoVDB volume id 337, volume type 2,
  with density and temperature present. The render slot is published. The
  viewport packet-order diagnostic reports `published=false`; that flag alone
  does not prove a missing packet because this backend has no TLAS identity order.
- Both `gas:Ground Burst Gas` and `vdb:337` report matching authored material.
- Density multiplier 52.139, scatter coefficient 6.48, scatter colour
  [0.5,0.45,0.4], absorption coefficient 4.96, colour [0.15,0.12,0.1].
- Response enabled: temperature power 6.843, density power 4, EV +5.333,
  saturation 2. Temperature interval 100..5000, scale 1.2, five-stop ramp enabled.
- Random walk false, multi scatter 0. Event settings 138/322 are inactive.
- Primary voxel multiplier 0.155, max steps 333; shadow steps 10, stride 6.
  Viewport quality reports auto. Runtime volume instrumentation is disabled;
  its zero counters cannot establish an effective per-ray march count.
- Solver plume heat mean 0.131479, peak 0.533203. The bridge's legacy x3000
  conversion produces a shader-temperature peak approximately 1599.61.
  This is not a measurement of physically calibrated Kelvin.

## Confirmed mechanisms

1. **Emission does not currently illuminate its own smoke through volume NEE.**
   `VulkanBackend_Volumes.cpp` explicitly publishes `emissive_list_address=0`
   and capacity 0. The RT block emitter sampler is therefore unreachable for
   these packets. Its accompanying comment says it was disabled to avoid
   live-only one-block Monte Carlo spikes. Do not simply re-enable that unstable
   estimator. The RT walk also excludes emissive media. RayFusion's inspected
   volume pass adds `emissionAt` to the camera integral but does not transport it
   as incident light into surrounding smoke. Shared emission response equations
   do not imply shared complete internal illumination. This is a renderer
   capability gap, not a preset or resolution problem.
2. **Current settings turn the medium into a strongly absorbing opaque shell.**
   At the measured density peak 0.215299, mapped density is about 11.2255.
   With the red-channel extinction coefficients, one voxel transmits only about
   2.65% (homogeneous-voxel estimate). Red scattering albedo including the tint
   is about 0.283; other channels are similarly low. Interior details are hidden
   even if the source grid contains them. These estimates are local, not a full
   rendered-ray measurement.
3. **Emission response produces a narrow, highly amplified selection.**
   The measured peak maps to u=0.30604 and ramp coordinate 0.36725, so most of
   the yellow/white ramp is unused. u^6.843 is only 0.0003028 at that peak.
   EV +5.333 then adds gain ~40.31. Density power 4 raises differences in
   already-scaled density to the fourth power, amplifying dense lumps while
   suppressing sparse cells. The existing radiance limiter can then flatten
   bright peaks. This explains why increased exposure can produce bright
   red/white spots without a broad warm internal light field.
4. **Requested subvoxel spacing is not guaranteed by Max Steps.**
   Requested step is 0.0043816 world units. An 8-unit chord would require 1826
   samples to honour it, whereas a 333-step ceiling gives ~0.0240. Exact chord
   length depends on camera direction and interval. RT distributes the complete
   interval across the capped budget; RayFusion also applies a viewport ceiling
   of 96/256/512 by quality. Instrumentation was not enabled, so no precise live
   effective count is claimed. The current UI help text inverts the interpretation
   of 0.5: `primaryStep()` actually multiplies voxel size, so 0.5 means two samples
   per voxel before budget stretching, not every other voxel.
5. **Resolution does not establish equal simulation detail.**
   The supplied Houdini image is a different dataset/solver/look and frame; there
   is no same-field reference comparison. Fine wrinkles must exist in the source
   density/temperature fields; the shader cannot reconstruct absent turbulence.
   NanoVDB bridge inspection uses float grids and trilinear reads. This audit
   does not establish that cache precision, solver advection or source noise are
   lossless; those require field statistics/gradient comparison.

## Required next implementation

- Implement an emission-driven incident-light estimator for both live dense gas
  and cached/imported NanoVDB. Integrate with scattering and self-transmittance;
  use matching source evaluation for camera and light samples. A block light
  approximation must be labelled and stable; physically accurate RT needs a
  properly weighted volume emitter sampler and/or emissive multiple scattering.
- Carry sufficient emitter metadata across the NanoVDB/cache path; do not rely
  only on source_type 5 buffers that this scene is not using.
- Implement a bounded RayFusion approximation from the same emitter authority,
  with explicit quality/performance controls rather than exposure compensation.
- Calibrate a gentler response and realistic extinction on the unchanged cached
  frame, then compare step budgets and source gradients. Preserve and restore
  every edited setting during acceptance. No calibration changes were made here.
- Fix the reversed sampling tooltip and expose requested versus effective spacing
  diagnostics so the user can see when a step budget defeats the requested value.

No code or visual fix is claimed by this diagnostic snapshot. The previous
material increment is insufficient to deliver Houdini-like internal fire lighting.
