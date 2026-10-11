# Cloud shape first — 2026-10-11

Status: source implementation, not compiled or visually accepted. User priority:
formation, layering and detail first; RT transport cost and cinematic lighting next.

## First source increment

`cloud_common.glsl` remains the single density implementation for Vulkan RT,
RayFusion, cloud shadows and the GPU query sampler.

- Each volumetric layer has a different horizontal noise orientation and phase.
  Weather coverage coordinates remain unchanged, including precipitation and the
  tracking majorant. Existing layer altitude, thickness, scale and type controls
  continue to define the arrangement; enabling layers still adds rendering work.
- The existing large-scale base lookup now precedes the small-scale lookup. Its
  unused channels warp the smaller formations, reducing regular billow alignment.
  The two scales use a less commensurate frequency ratio. This reduces visible
  repetition; the underlying textures are still periodic.
- Stratiform layers retain more continuous bodies, while convective types retain
  stronger separation between large formations.
- Convective base erosion is reduced and inverted crown detail develops gradually
  with height. Stratiform erosion stays softer. Detail still only removes density.

The density path retains two base-noise reads, one curl read and one detail read
after weather/profile rejection. Weather and optional shear reads are unchanged.
Additional work is coordinate arithmetic, not new textures, octaves or march
steps. This is a lookup-budget statement, not a measured GPU performance claim:
changed occupancy and shadowing can still change total rendering time.

No new authoring controls, packet layout or serialization were introduced.
UI and `rt.world.set_clouds` / IPC `world.set_clouds` retain the shared CloudState
validation and commit path. `rt.world.sample_clouds` / IPC `world.sample_clouds`
consume the same updated GPU density shader. Existing saved settings remain valid,
but their rendered formations change after the shaders are rebuilt.

## Static verification

- Layer rotations preserve horizontal length (coefficients 0.8/0.6).
- Weather lookup and RT majorant coordinates were not modified.
- Shape and coverage factors remain clamped to [0, 1]. For erosion threshold
  `e` in [0, 0.45], `clamp((s-e)/(1-e), 0, 1) <= s` for `s` in [0, 1].
  Therefore detailed density <= cheap density <= coverage * extinction.
- No build, shader compilation, application launch or live scene mutation.

## User build and acceptance

1. Recompile the consumers of `cloud_common.glsl` through the normal workflow:
   `cloud_sample.comp`, `cloud_raster.comp`, `cloud_shadow.comp`, `miss.rmiss`
   and `closesthit.rchit`, including the sphere/volume closest-hit variants
   generated from that source. A C++ build is not required by this increment.
2. Load the rebuilt shaders through the normal application restart/reload workflow.
   The already-open application still uses its previous shader binaries.
3. Compare the same camera, timeline frame, seed, exposure, resolution and sample
   count before/after. Inspect wide views and closer crowns: repetition, flat base
   continuity, distinct layer patterns, silhouette and edge breakup.
4. Separately check stratus, stratocumulus, cumulus and cumulonimbus using the
   existing type control. A near-stratus type cannot become a towering cumulus
   merely by raising render quality. Check all three layers individually and
   together, then cirrus, horizon and wind playback for seams or sudden changes.
5. After compilation, external `scripts/test/rt_ipc.py` queries can compare
   `world.sample_clouds` for both devices at identical points and detail modes;
   check finite, nonnegative values and the density bounds above. Never run the
   pipe probe inside the application's embedded script workspace.
6. Compare fresh completed-sample progression for cost, not converged idle FPS.
   Do not change reference/production transport during the shape comparison.

This increment does not change vertical type profiles, weather field generation,
cirrus or light transport. Visual acceptance should decide whether the next shape
increment needs stronger type-specific silhouettes or weather-scale organization.
Maximum quality and a faster RT result have not yet been demonstrated.
