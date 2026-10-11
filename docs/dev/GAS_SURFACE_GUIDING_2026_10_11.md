# Vulkan RT surface sampling toward emissive gas

## Status: rolled back after scene verification

The user compiled this experiment and reported increased noise and cost in the
open Nuclear Detonation scene. The shared diffuse scatter has been restored to
cosine-only sampling; the helper and its numerical test have been removed.
The text below records the abandoned experiment, not current render behavior.
The synthetic bright-box variance improvement did not predict this sparse,
large-volume scene. Recompile closesthit, sphere_closesthit and volume_closesthit
to activate the rollback. The earlier live-only smoke NEE fix remains in place.

## Behavior

Diffuse surface continuation previously used only a cosine hemisphere proposal.
Small bright gas regions were reached rarely, causing noisy fire illumination
on surrounding solid surfaces. The shared diffuse scatter now mixes that proposal
with a direction toward a uniformly sampled point inside an emissive volume's
world-space bounds. One continuation ray is still traced.

`gas_surface_guiding.glsl` owns selection, transformed bounds, the box directional
PDF and the throughput correction. `bsdf_scatter.glsl` includes and calls it,
covering triangle/sphere surfaces and diffuse fluid isosurfaces. Metal, dielectric
specular and transmission lobes retain their existing sampling.

This is continuation importance sampling, rather than an additional volume NEE
lighting term. No second fire-light contribution is added, no BSDF-hit emission
is suppressed, and no new user-authored setting or API operation is introduced.
The previous live-only smoke NEE remains disabled.

## Probability and cost

For a uniform point in a box of volume V, the direction PDF is
`(far^3 - near^3) / (3 V)`, where near/far are positive ray-box intersection
distances measured from the actual continuation origin. The proposal is
`0.5 cosinePDF + 0.5 boxPDF`; diffuse throughput is multiplied by
`cosinePDF / mixturePDF`. Full hemisphere support is preserved and this multiplier
cannot exceed two. A box sample below the hemisphere is a zero-contribution event,
not a resampled direction. Selection is deterministic for a given surface and
published volume table, so there is no missing emitter-selection probability.

Selection scans volume metadata without reading density/temperature cells. It
chooses a nearby grid-backed blackbody/ramp volume using brightness and angular
size, excluding fluid surfaces, degenerate boxes, missing grids, wholly
behind-surface boxes and boxes farther than sixteen bounding radii. There is
no added shadow ray or emissive-list readback. More continuation rays reaching gas
can still increase gas-march work; GPU timing has not been measured.

The proposal uses whole volume bounds, not a hot-cell distribution. Sparse fire
inside a large smoke domain, very glossy surfaces and many differently bright
emitters can therefore retain noise. Live and cached volumes use the same method,
but their published bounds can differ and need not produce identical variance.

## Static validation

- `python scripts/test/check_gas_surface_guiding_math.py`: passed constant incident
  radiance checks for external, internal, horizon-crossing and behind-surface
  boxes. A controlled bright-box experiment preserved mean illumination within
  sampling error and reduced variance to 0.2198 of cosine-only sampling.
  This is a numerical transport check, not measured renderer performance.
- `python scripts/audit_shader_struct_layout.py`: all shared layouts passed.
- No payload, descriptor or GPU structure ABI changed.
- No build, shader compilation, app launch or live IPC probe was run by Codex.

## User build and scene checks

1. Recompile shaders using the normal shader workflow (for example,
   `powershell -ExecutionPolicy Bypass -File .\RayTrophiStudio\compile_shaders.ps1`
   from the repository root). `closesthit`, `sphere_closesthit` and
   `volume_closesthit` all consume the changed shared module.
2. In Vulkan RT, compare the campfire's diffuse ground/container surfaces at the
   same camera, frame, exposure and sample budget. Compare convergence and GPU
   frame time, then allow accumulation to check stable mean brightness.
3. Repeat uncached simulation and cache playback. Check that the previous
   live-only smoke sparkle does not return, including scatter coefficient zero.
4. Check an opaque object between the fire and ground: the guided ray uses normal
   scene traversal, so solid occlusion and gas attenuation should remain intact.
5. Move/rotate/scale the gas domain; test a surface inside the bounds and one
   beside a box crossing the horizon. Check for missing illumination or NaNs.
6. Check a scene without emissive gas and metallic/glass surfaces. The diffuse
   helper returns the original weight when no eligible target exists; the other
   lobes do not use this proposal. OptiX/Solid/RayFusion are unchanged.
