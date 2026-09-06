# Realtime SurfaceSDF

Material Preview now consumes fluid `SurfaceSDF` data from the same producer
packet and NanoVDB grid used by Vulkan RT. It does not polygonize or create a
second geometry authority. The raster pass reads the 624-byte
`VkVolumeInstance` table, intersects the volume bounds, marches the shared
field at `iso=0.5`, reconstructs a gradient normal, and writes the resulting
surface depth into the main viewport depth attachment.

The pass is deliberately isolated in
`Viewport/MaterialPreviewSdfSurface.cpp` and
`material_preview_sdf_surface.frag`. The large backend files contain only
descriptor, lifetime, and draw-order wiring. Binding 20 is a read-only alias of
the viewport device's volume table; Vulkan RT keeps its existing binding 9 and
shader pipeline unchanged.

## Current parity contract

`viewport.quality().sdf_surface` and IPC `viewport.quality` report
`shared_nanovdb_depth_pbr`.

- Shared with Vulkan RT: field bytes, volume transform, active bounds,
  `iso=0.5` crossing, voxel-sized gradient normal, surface material index, IOR,
  roughness, Base Color/albedo texture, roughness/metallic/transmission/emission
  texture channels, scene lights, Physical Sky sun, shadow receiving, IBL, and
  authoritative raster depth.
- Bounded realtime behavior: at most 16 volume records are inspected per
  pixel; march limits are 72/144/256 for Performance/Balanced/Quality-Full.
  Reflection/refraction uses the environment continuation and is not recursive
  scene traversal.
- Not yet at RT parity: SDF shadow casting into the raster atlas, screen-space
  scene refraction, material graph VM, advected UVW/composition blending,
  opacity/porosity field modifiers, and SurfaceSDF selection-mask ownership.

## Static verification

Run without building:

```text
python scripts/audit_realtime_sdf_surface.py
python scripts/audit_shader_struct_layout.py
python scripts/audit_ipc_capabilities.py
```

## User build and visual test

1. Run `RayTrophiStudio/compile_shaders.bat`, then build the application.
2. Open a fluid domain in `SurfaceSDF` mode and switch to Material Preview with
   Scene lighting.
3. Confirm the surface appears without switching to Rendered mode, updates
   while simulation runs, and correctly occludes the grid and opaque meshes.
4. Sweep Realtime Quality from Performance through Full. Silhouette position
   must remain stable; only fine detail and cost may change.
5. Compare Material Preview and Vulkan RT at a fixed camera for surface shape,
   transform, IOR/Fresnel, roughness, texture anchoring, and depth ordering.
6. Place an opaque object first behind and then in front of the liquid. The
   front object must win depth; the liquid must win when it is nearer.
7. Query scripting and IPC `viewport.quality`; both must return
   `sdf_surface: shared_nanovdb_depth_pbr`.

