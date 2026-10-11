# Fog and cloud reference audit

External named-pipe reads of the open Vulkan Rendered scene found a fluid fog
volume, Sphere_1, and completed accumulation at 128 samples. No scene/material/
cloud settings were changed. Volume counters were disabled, so their zero values
are not measurements. No fresh benchmark or screenshot was acquired; no build or
shader compilation was run.

## Fog workload

`fluid.get_fog_shader(domain="Physics Domain 1")` returned density multiplier 100,
scatter coefficient 2.42, white scattering, zero absorption, random walk enabled,
51 maximum events and 13 exact events. Forward anisotropy was 0.872, back lobe
-0.513, and lobe mix zero. The actual mean g is therefore -0.513 and the forward
similarity reduction in `volumeRandomWalk` is disabled. Exact-events changes do
not accelerate that configuration. Every active sample still pays for free-flight
density queries, per-event lighting/shadows and collision probes. Visually clean
or denoised output does not itself reduce this per-sample workload.

Raygen applies adaptive skipping to every pixel before tracing, including fog and
cloud paths. Its criterion is relative standard error, not raw sample variance.
A larger threshold skips sooner; a smaller threshold is stricter. The source
default is 0.01 (1% relative error), but current Vulkan adaptive settings are not
reported by `render.get_settings`; this audit did not establish the active value.
The current guard also refuses to stop exactly zero-M2 pixels. No adaptive policy
change was made in this audit.

## Cloud workload and boundary defect

`world.get_clouds` reported reference path tracing enabled, 8 camera bounces,
secondary full march disabled, 16 RT production steps, one 1200 m-thick layer at
1200 m altitude and extinction 0.05/m. Production step count does not govern
reference tracing. Reference mode performs a camera transmittance query plus
delta-tracked flights and sun transmittance queries per event. Each tracking
query permits up to 2048 candidate collisions/segments. Fog paths escaping to
the sky can additionally invoke secondary cloud tracing (capped at 3 events).
These are source budgets, not measured event counts or GPU times.

Previously, tracking-cap exhaustion could return `false` from free flight with
the same output as real sky escape. The lighting caller then added sky radiance
after scattering. Ratio tracking could also return its partially evaluated
transmittance without accounting for the remaining cloud. Both can leak light
into capped paths; whether caps caused the reported whitening needs verification.

`cloudFreeFlight` now marks a capped flight with negative hit distance and the
caller only adds escape sky for a completed flight. `cloudTransmittance` returns
zero if work ends before traversal completes. The budget is unchanged, so this
does not solve reference-mode performance. The bounded fallback is intentionally
dark-biased when a cap is hit; it must not be described as an unbiased reference
result on capped paths.

## User verification

1. Recompile the cloud-consuming shaders through the normal shader workflow.
2. Compare cloud interiors and horizon at the same camera, frame, exposure and
   sample count. Check that cap-hit bright patches reduce without unacceptable
   dark areas. If whitening persists, examine transport/phase/exposure separately.
3. To attribute cost, compare reference clouds enabled/disabled while retaining
   fog path tracing, then compare fog walk/march with cloud settings unchanged.
   Use fresh accumulation and the same resolution/sample count for each pair;
   restore original settings afterwards. Do not compare a completed idle frame
   or stale raster telemetry against an active RT sample.
4. For fog budget experiments, compare 51 vs 32 maximum events at equal samples.
   This can darken truncated paths; inspect lighting as well as speed. Changing
   lobe mix is a material appearance change, not a free optimization.
5. Check the UI's current adaptive threshold/minimum samples. A 0.01 to 0.02
   comparison tests a more permissive stopping criterion, not faster individual
   random walks. Keep thin details and bright outliers in the comparison.
