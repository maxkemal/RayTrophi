# RT shadow 2a review (2026-09-08)

The displayed shadows still come from the cascade atlas. The RT producer runs
after the HDR/transmission passes and its mask has no consumer. Compiling the
existing producer cannot switch the displayed shadows to ray queries.

## Corrections in this review

- `ready` now requires an enabled material preview pass, valid resources/TLAS,
  and a recorded dispatch. Allocation alone, a failed dispatch prerequisite, or
  disabling the switch no longer reports success. This is command recording
  status, not proof of GPU completion or valid image contents.
- `rays` is an upper bound (width times height), not a measured ray counter:
  sky pixels return before traversal. An inactive pass reports zero.
- Depth uses `texelFetch` at mip zero. A single hard-shadow ray uses the exact
  sun direction instead of a randomly perturbed direction.
- Depth barriers cover early and late depth tests and subsequent depth reads.
  Mask writes have a write-to-write dependency across dispatches.
- The single descriptor set is drained before rewriting it. This is necessary
  while it is shared by pending frame submissions; per-frame descriptor sets
  should remove this serialization when integrating the consumer. Frame A/B
  measurements include this wait and are not isolated GPU traversal timings.

Both Python and IPC retain the existing `set_rt_shadow` / `rt_shadow` service.
No new command or parameter was introduced.

## Remaining work before 2b is valid

1. Split depth production and shading into separate compatible render passes,
   preserving attachments with LOAD and generating the mask between them.
2. Evaluate candidate alpha using the flat scene instance/UV/material data.
   The current opaque query still blocks the entire foliage card.
3. Match each mask to its actual light. The producer currently uses only the
   world sun direction; it does not represent independent directional lights.
4. Restrict screen-space consumption to the surface represented by its depth.
   `rtPreviewShadow` also serves volumes and SDF surfaces at other positions;
   substituting a pixel mask unconditionally is incorrect.
5. Define transparent prepass behavior, preserve volume transmittance, and
   retain cascade fallback for unsupported lights/surfaces or producer failure.
6. Measure the completed path. The two probe timings do not establish an upper
   bound on full-screen shadow cost: origins, directions, hit rate and alpha
   traversal differ. The quoted approximately 5 ms is an extrapolation.

## User build and runtime checklist

No builds, shader compilation, or application launches were performed here.

1. Run `compile_shaders.bat` from the `RayTrophiStudio` directory, then build the
   existing project in the IDE. The existing project entry covers the C++ file.
2. In Material Preview, enable `viewport.set_rt_shadow {"enabled": true}`,
   render a frame, then query `viewport.rt_shadow`. Expect `ready: true` only
   after dispatch recording with a valid TLAS. Check the reported reason otherwise.
3. Disable it and query again: expect `ready: false`, `rays: 0`, `reason: disabled`.
4. Enable/disable with a fixed camera: displayed pixels must remain unchanged.
   Move the camera, resize the viewport, and switch modes with Vulkan validation
   enabled; check for descriptor-in-use and image synchronization errors.
5. Compare sustained foreground frame timings with geometry in view and with
   only sky in view. Do not interpret `rays` as an actual traversal count.
6. Camera-attached RT shadows and foliage alpha correctness remain 2b tests;
   they cannot pass visually until a consumer is implemented.

Static checks passed: `audit_rt_shadow_pass.py`, `audit_raster_depth_prepass.py`,
`audit_shader_struct_layout.py`, and `audit_ipc_capabilities.py`.
