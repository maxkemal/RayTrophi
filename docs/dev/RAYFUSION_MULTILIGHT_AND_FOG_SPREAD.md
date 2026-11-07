# RayFusion SDF light coverage and Matter Fog spread

User live acceptance: SDF shadows confirmed under all tested RayFusion light
types; Matter Fog Spread changes the footprint. Sub-voxel detail remains limited
by grid sampling. Implementation below records the delivered source change.

## Source changes (awaiting user build)

RayFusion's mesh screen-shadow pass now traces each covered visible scene light,
up to 32, rather than reusing one directional visibility. Each pixel stores a
32-bit blocked-light mask plus camera depth bits: still 8 bytes per pixel.
Directional, point, spot and area lights use their own direction/light segment.
Finite rays stop before the light, so an object behind it cannot cast a shadow.
Area lights use the same centre sample as current RayFusion direct lighting;
this change does not add area penumbra sampling.

The shared native NanoVDB SDF query runs for every uncovered light segment. SDF
receivers also call it directly, using the same library with their existing raw
volume buffer and NanoVDB setup. Mesh screen-mask consumers keep their existing
depth match. Non-directional atlas maps remain available for receivers absent
from the mesh prepass; fog transmittance still multiplies surface visibility.
Native opaque SDF membership follows the existing opacity/transmission cutoffs.

Compute binding 7 contains a small 32-entry light table (1024 bytes). Push size
remains 128 bytes. Pixel encoding has a new version magic 0x52545349, published
by the new compute shader rather than blindly advertised by the host. Host,
producer and consumer shader variants must be rebuilt together. Geometry remains
on the existing flat scene/TLAS path. Existing UI, Python and IPC RT shadow
operations continue using the same backend service.

Matter Fog now reads the existing `fluid.set_fog(spread_voxels=...)` value while
building carrier fields. Density uses the existing separable Gaussian helper;
temperature uses matching mass-weighted spread of temperature numerator and
weight. Gas aggregate/source fields are unchanged. Invalid/nonfinite input is
rejected before publication; existing scratch accounting bounds the blur peak.
The temporary density blur storage ends before temperature gathering.

Matter domains expose **Fog View / Fog Spread (voxels)** whenever their view plan
contains Fog. The UI calls `rtapi::setFluidFog`, the same service as scripting and
IPC. Existing serialization stores the setting. No new authoring API is needed.

This is Gaussian sigma in simulation voxels, **not a hard physical diameter**.
Zero gives the original trilinear footprint; larger values widen/soften carrier
fog. A Gaussian has no fixed outer boundary; the helper drops tiny tails. The
simulation grain radius and solver state are unchanged. Full-grid CPU blur adds
render-publication work on large domains; no speedup is claimed. Matter erosion
and resolution scaling are outside this slice and are not exposed by this new
focused Matter control. Legacy non-Matter controls retain their existing path.

## User build and acceptance

1. Build C++ and compile/deploy all shaders, then restart. Required changed shader
   entrypoints: `rayfusion_rt_shadow.comp`, `material_preview_frag.frag`, its
   `material_preview_covered` variant, and `material_preview_sdf_surface.frag`.
   The shared `surface_sdf_shadow_query.glsl` is included by compute and SDF.
2. Enable RayFusion RT shadows with existing UI or `viewport.set_rt_shadow` and
   Scene lighting. Test directional, point, spot and area lights individually,
   including a scene with **no directional light**. Check SDF shadows on mesh
   and SDF receivers, two lights with different shadow directions, moved lights,
   and an SDF blocker behind a point light (must not block that light).
3. Check fog transmittance, transparent/cutout mesh fallback, pause and scene
   reload. Area-shadow sharpness should match the centre-sample limitation.
4. In a paused Fog carrier scene compare spread 0, 1.5 and 3 through the panel,
   `rt.fluid.set_fog(domain, spread_voxels=...)` and IPC `fluid.set_fog`. Footprint
   must widen, particle radius/position/count must not change. Check a constant
   hot carrier's temperature remains constant throughout its spread footprint.
   Save/reopen to verify the existing setting persists. Check gas stays unchanged.
5. On a **small, paused** Matter fixture with isolated populated Fog carriers,
   `python scripts/test/rt_test_matter_fog_spread_ipc.py <domain>` checks actual
   field hashes/footprint, gas identity and unchanged simulation version. It
   restores spread afterward. Do not use the 222-cubed scene: the inspection
   ceiling remains 128 MiB. Not run against the user's scene by agent.

Non-build checks passed: `check_rayfusion_multilight_shadow_contracts.py`,
`check_matter_fog_snapshot_contracts.py`, Python syntax, project XML and scoped
whitespace checks. These are not shader execution or visual acceptance.
