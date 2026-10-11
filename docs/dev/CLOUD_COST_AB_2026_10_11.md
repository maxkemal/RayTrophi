# Live storm cloud cost, 2026-10-11

The existing application was inspected through the external IPC client. No build,
shader compilation, application launch, or new test script was performed.
Raw settings and measurements: `cloud_cost_live_2026_10_11.json`.

Scene: 1680 x 945 RayFusion viewport, three linked storm regions spanning
800–12500 m. The middle region is 7100 m thick. Cirrus is enabled; lightning and
cloud precipitation are disabled. Both cloud marches use 128 steps; RayFusion
starts at resolution divisor 1. RT reference and full secondary march are off.

| Temporary configuration | RayFusion GPU ms/frame | RT wall ms/completed sample |
|---|---:|---:|
| Original storm | 1151.50 | 417.75 |
| RayFusion resolution divisor 2 | 322.87 | — |
| Liquid layers disabled, cirrus retained | 22.33 | 24.03 |
| Middle region disabled | 734.30 | 287.96 |
| All layers retained, formation role/group set to zero | 227.69 | 88.61 |

RayFusion windows contain 12–14 recorded frames after a three-second settling
period and counter reset. GPU timestamps are available. Telemetry warns that wall
time exceeds the recorded GPU durations, so these are GPU frame costs, not a
sustained FPS estimate. RT uses natural completed sample progression after two
warmup samples; windows contain 6–10 additional samples. RT's reported
`ms_per_sample` is zero, so RT figures use wall time and include scheduling/IPC
overhead. The two columns are not directly comparable. These are short A/B
windows, not a statistical benchmark.

The unlinked case changes cloud morphology and occupied sample counts as well
as removing linked storm calculations. Its improvement cannot be attributed
entirely to arithmetic: it identifies the linked storm density path as the main
optimization target, not a measured percentage for a single function.

Source findings:

- `cloud_common.glsl::cloudStormEnvelope` checks nine neighboring cells and up
  to four lobes per cell for every relevant linked density evaluation. Region
  interpolation adds further work. Both render paths share this density code.
- Production `cloud_rt.glsl` evaluates cheap and detailed view density. Every
  occupied view step then evaluates six sun samples, one far sun sample, and
  two ambient samples. Sun/linked ambient samples query the shared density
  field. Thus expensive formation evaluation is multiplied by lighting.
- This illumination also produces natural shafts and remains active for
  occupied cloud samples. It is not a separate god-ray pass controlled by the
  old checkbox. Its isolated cost was not measured because that would require
  a shader change and a user build.
- `raygen.rgen` skips the legacy primary god-ray pass when liquid cloud layers
  are active (`!cloudHasLiquidLayers()` guard).
- RayFusion also recomputes a 512 x 512 cloud shadow map with up to 32 optical
  depth samples per crossed interval. It shares the expensive density shape.
- `VulkanViewportBackend.cpp` records the cloud raster and cloud shadow dispatch
  before ending the `ShadowAtlas` timing interval. A large `shadow_atlas`
  reading therefore includes cloud computation; it is not evidence that mesh
  shadows alone are expensive.

All cloud settings and the original material viewport mode were restored.
`towering_cumulus` was removed from the canonical preset registry and application
branch, and the IPC descriptor was updated. UI, scripting and IPC already use
that common registry. Existing scenes retain their authored layer data. Removal
becomes visible after the user's next C++ build/restart; the running binary still
reports the old preset.

Recommended next source change: bound/reuse linked envelope and region work
before reducing visual light quality; then decouple shadow-map updates from
unchanged frames. Divisor 2 is a measured temporary performance option, not a
persisted scene change. User validation after any optimization should compare
storm silhouettes, shading and natural shafts in both paths at the original
settings. Cirrus and lightning should retain their accepted appearance.

## Follow-up source optimization (user build pending)

- Production view march now resolves base and detailed density in one call.
  The returned base density preserves the existing empty-step decision; an
  erosion-only hole still uses the original fine step. This removes the second
  complete storm shape evaluation at the same view position.
- Lobe solving rejects vertical and squared-distance bounds before square root
  and smoothstep. The bound includes the maximum 0.26 billow displacement, so
  it only rejects zero-contribution lobes. A fully saturated envelope union
  returns immediately; additional lobes cannot raise its value above one.
- Storm fine erosion fades out for normalized shape density 0.75–0.95 in the
  interior. Base/crown bands keep erosion active. This avoids curl/detail
  texture reads in full cores; unlike the pruning above, this intentionally
  changes interior microdensity and needs visual acceptance. The cheap density
  remains an upper bound for the detailed density.
- Existing view termination at transmittance below 0.005 is retained. Dense
  material is still integrated for extinction and lighting; it is not treated
  as empty space. No larger uncertified ray jumps were introduced.

Both Vulkan paths share these shader functions. No new authoring parameter,
ABI change, or new test script was added. No shader compilation was run.
The timings above are the pre-optimization baseline, not evidence of a measured
speedup from the edited source. After compiling affected cloud consumers, inspect
storm silhouettes, underside/crown detail, holes, interior brightness, natural
shafts and lightning from outside and inside the cloud, then repeat the original
cost measurement at 128 steps and RayFusion divisor 1.
