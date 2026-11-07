# SurfaceSDF direct shadows in Vulkan RT

## Live result and remaining issue (2026-11-06)

The user compiled the fix and confirmed SurfaceSDF casts shadows under all
tested light types in RT. RayFusion was separately confirmed to cast SDF shadows
only under directional lights. Source inspection agrees: `rtShadowCoveredLight`
selects directional lights and `rayfusion_rt_shadow.comp` traces a constant
`sunDirection`. Point and area light SDF shadows remain open work in RayFusion.
Do not describe RayFusion's current SDF shadow support as covering all lights.

Follow-up source implementation is now prepared in
[RayFusion multilight and Fog spread](RAYFUSION_MULTILIGHT_AND_FOG_SPREAD.md).
The directional-only statement above records the tested binary; the new
multilight implementation was subsequently compiled and confirmed by the user
under all tested RayFusion light types.

The direct-shadow TLAS mask is 0x05 (solids and splats), excluding SurfaceSDF
instances (0x08). The participating-medium shadow march also deliberately skips
source_type 4. Together these made reconstructed opaque surfaces visible to
camera rays but invisible to direct lighting. RayFusion already has an explicit
surface-proxy shadow query.

`surface_sdf_rt_shadow.glsl` adds a bounded raw NanoVDB surface query for Vulkan
RT. It clips the light segment in domain space, uses the existing fine surface
march policy and the 0.5 proxy threshold, and rejects missing/inactive fields.
Like RayFusion, material opacity below 0.1 or transmission above 0.5 skips opaque
occlusion. This is binary opaque visibility, not dielectric coloured shadows.
Fog density remapping and fog shadow controls do not modify the surface proxy.

Mesh/splat direct lights and sun use the query through the existing volume-shadow
entry point. Flash lighting now calls that entry point too. SurfaceSDF BSDF
lighting uses the same surface query, with its existing NanoVDB sampler. The
TLAS mask stays unchanged: accepting an entire volume AABB as a shadow hit would
cast a box shadow rather than a reconstructed surface shadow.

## User build and visual verification

- Compile and deploy `closesthit.rchit` and `volume_closesthit.rchit`, including
  the new GLSL dependency, then restart the application. No C++ build is needed
  for this fix. The independent particle-label memory change still needs C++.
- In the snow scene, place an opaque SDF surface above a receiving mesh. Compare
  RT and RayFusion under a point/area light and the sun; test flash separately.
- Check the silhouette of the shadow, not just its presence. Empty parts of the
  domain must remain lit. Move/rotate the domain and check the shadow follows.
- Check a surface receiving another SDF's shadow, and check for self-shadow acne.
- Change fog shadow strength/density: this must not remove the opaque SDF shadow.
- Check transmissive water/glass remains exempt from binary opaque occlusion.

Source inspection and whitespace checks only; shaders have not been compiled
or executed by the agent. Hair-specific lighting is outside this change.
