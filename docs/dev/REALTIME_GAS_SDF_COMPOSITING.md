# Realtime gas in front of SDF surfaces

The HDR transmission replay now draws surfaces first, copies their completed
depth to the sampled depth snapshot, then composites participating media.
Gas marching therefore ends at the opaque/SDF surface and foreground gas is
no longer overwritten by the SDF draw. The colour snapshot used for refraction
remains the opaque scene. No public operations or shader ABI changes.

Scope: the normal HDR transmission replay. The resource-failure fallback and
legacy direct viewport path are unchanged. Refracted gas behind transparent
fluid remains unsupported by this correction.

Static validation: `python scripts/audit_realtime_sdf_surface.py`.
Build and application verification are performed by the user:

- Build the application normally; this change requires no shader recompilation.
- Place dense gas between the camera and a fluid SDF: gas should cover the fluid.
- Move gas completely behind opaque fluid: it should not bleed over the surface.
- Move the camera through an intersecting gas/fluid scene: foreground gas should
  persist up to the surface, without a hard disappearance over the fluid.
- Check opaque mesh occlusion and a gas-only scene for regressions.
- Check Vulkan validation output for image layout and synchronization errors.

Cost: one additional depth copy and render-pass restart per transmission replay.
