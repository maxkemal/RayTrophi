# Cloud transport before scene surfaces — 2026-10-11

Status: source implemented; non-build checks pass. User compilation and visual/live
acceptance are pending. Supersedes the earlier intentional miss-only/sky-only limit.

## Problem and finite transport

RT previously evaluated cloud radiance only in `miss.rmiss`; an intersected mountain
therefore received surface shading with no cloud between camera and surface.
RayFusion composited its cloud output only in the sky shader, behind geometry.

`cloudMarchLimited` and `cloudRenderLimited` now take a finite world ray length. The
same field, weather bounds, roles and materials are evaluated up to that endpoint.
Camera transmittance and the reference walk's first collision stop there; cirrus and
precipitation also honor the endpoint. Their existing unbounded wrappers remain the
sky path, so cloud behind a mountain cannot contribute directly along the finite
camera segment while a miss still sees distant cloud.

A convex-altitude endpoint check skips finite segments whose endpoints both lie
below every cloud shell. Rain disables that skip because it extends below the shell.
No new density model, geometry facade, authoring toggle or material copy is introduced.
Cloud settings remain CloudState, edited through the existing UI/Python/IPC service.

## Vulkan RT integration

Focused `cloud_hit_segment.glsl` adds the segment source to hit radiance and multiplies
both hit radiance and continuation attenuation by segment transmittance. The mesh
and sphere closest-hit wrapper uses the actual `gl_HitTEXT`; hair uses the same
operation. This applies to primary and later scene intersections, including a
camera inside cloud. Every early return in existing shading still passes through
the small wrapper. Large existing hit files receive only include/call/entry wiring.

The wrappers preserve camera/photon metadata flags that AOV packing previously
overwrote. Photon traces with `PL_NO_SKY_RADIANCE` do not receive sky/cloud radiance.
Straight transparent passes with a cloud source are classified as medium passes so
added in-scatter does not consume a GI bounce. Transparent classification is shared
with raygen through `plIsTransparentPass` rather than copied into another shader.
The payload layout is unchanged.

Volume-hit continuation uses its published projected endpoint, deferring a same-origin
solid retrace to the following real hit. This retains depth ordering for straight
volume exits; it is a composition approximation for overlapping participating media.
An internal volume random walk and cloud walk are not a single jointly sampled
medium. Reference cloud scatter continuation still uses the existing cloud/sky
lighting model, not arbitrary scene intersections for every internal scattering
leg. This work fixes the view-segment placement and occlusion, not all cloud/scene GI.

Existing aerial/fog processing and geometric denoiser AOVs remain. Mixed-air treatment
and representative cloud AOV/DoF depth can need further calibration; no claim of
fully unbiased joint atmosphere/volume transport is made.

## RayFusion opaque geometry

New `cloud_scene.comp` runs after geometry/HDR work, when the current D32 depth is
available, and before raster post/DoF/tonemapping. It leaves sky pixels untouched:
the existing reduced-resolution temporal sky pass already handles them.

For each opaque pixel it converts projection depth to view depth, divides by the
camera-direction cosine to obtain ray length, and integrates only that finite
segment. It composites into the existing scene-linear RGBA16F target once. The
shared `cloud_raster_frame.glsl` carries the same 176-byte camera/world frame as
the sky pass. No new full-size image or density texture is allocated.

The geometry pass uses full resolution even if the sky divisor is greater than 1:
a coarse sky pixel cannot safely supply the endpoint at a mountain silhouette.
It uses realtime step quality and a stable pixel seed. It has no separate temporal
history; motion and thin cloud noise therefore require visual acceptance. Geometry
depth is the opaque depth buffer, not a deep transparency or blended-volume buffer.
Transparent raster layers and cloud-weighted DoF depth are not solved exactly by it.
Existing orthographic RayFusion cloud exclusion remains.

Raster cloud layout adds depth binding 12 and HDR storage binding 13 and a 24-byte
push range. The new pass has its own descriptor set, separate from the sky/history
sets already bound earlier in the command buffer. Shared LUT writes include that
set after their existing idle/rebuild boundary. Target descriptors are cached and
invalidated with viewport target destruction; ordinary frames do not rewrite
in-flight sets. HDR storage support already exists. Barriers order color/HDR writes,
the cloud compute update and subsequent post reads; depth uses its existing
read-only transition and nearest sampler.

`VulkanCloudScenePass.cpp` provides the focused adapter wiring and is registered in
the MSVC project and filters; CMake already glob-collects Backend sources. The new
compute pipeline loads `cloud_scene.spv` with the cloud resources. Missing shader
reports an explicit warning and skips geometry composition; sky cloud remains.

## Script/IPC and diagnostics

Existing canonical operations (`rt.world.set_clouds` / `world.set_clouds`, presets
and bounded `world.sample_clouds` segments) continue to drive the shared field.
There is no UI-only operation for placement or a second cloud material authority.

`rt.world.cloud_stats()` / IPC `world.cloud_stats` now report
`backends[].scene_depth_pipeline_available`. This is RayFusion pipeline availability,
not a measured cost, proof of shader compatibility or a rendered acceptance result.
Read after initializing a RayFusion viewport with clouds enabled.

External read-only diagnostic after building:

```python
from rt_ipc import RtIpc
ipc = RtIpc()
try:
    print(ipc.call("world.cloud_stats"))
finally:
    ipc.close()
```

## User build and acceptance

Build C++ and shaders together. Required changed consumers are `raygen`, `miss`,
`closesthit`, `sphere_closesthit`, `hair_closesthit`, `volume_closesthit`,
`cloud_raster`, `cloud_shadow`, `cloud_sample`, plus **new `cloud_scene.comp`**.
The normal full shader workflow includes the new `.comp`; no payload or cloud packet
field-layout change is made in this increment. Restart with matching outputs.

1. Keep the same mountain/camera/frame/exposure. Put the cloud base below the peak.
   A ridge nearer than the cloud must occlude it; cloud before the ridge must
   attenuate the ridge and contribute its own radiance. Compare sky beside the ridge
   to ensure the miss path was not composited twice.
2. Move the camera inside/below/above cloud. Check near foreground meshes and the
   mountain background, then circle the ridge to expose silhouette changes.
3. Check RT production and reference separately; then RayFusion. Verify the new
   pipeline flag for the viewport role and absence of the missing-pipeline warning.
4. Check alpha cutouts, translucent passes, reflected surfaces, hair, straight fog
   volume exits and a solid inside a volume. These detect double-counted ray prefixes
   and camera/photon metadata regressions. Mixed-media/reference lighting remains
   approximate as described above.
5. Check resize/mode transitions and shader/resource reload under Vulkan validation
   if available. Inspect aerial/fog and DoF separately; geometry depth does not
   represent foreground cloud's weighted DoF depth.
6. `python scripts/test/check_cloud_shape_contracts.py` covers finite intervals,
   cirrus/rain endpoint clipping, hit wrappers/metadata, RayFusion ray-depth
   conversion/post order, descriptor separation/lifetime and straight-pass budgets.
   Source structure, project registration and shader-plan inventory were also
   checked without invoking a compiler. This does not confirm GLSL compilation,
   runtime synchronization, visual correctness or GPU performance.

Codex ran no project build, shader compilation or application launch and did not
modify the open scene. Surface cloud work adds cost where a finite segment enters
cloud; the endpoint guard avoids full marches for most below-cloud terrain.
