# Cloud material, tower variation and reference cirrus — 2026-10-11

Status: source complete, non-build checks pass; C++/shader build and visual/live
acceptance are pending. Supersedes the first increment's build-only-shaders advice.

## Shape and missing cirrus

User accepted improved small formations, but tall cumulus columns remained too
uniform. The existing large-formation read now supplies horizontal cell development:
weather A organizes fields, while finer channels vary neighbouring tower heights,
vertical proportions and crown lean. The vertical base-noise coordinate is anchored
to the layer base and has a continuous oblique component. No additional shape-noise
lookup was added. The formation read now occurs before local-height rejection, so
some previously rejected samples pay one read earlier; unchanged lookup count does
not guarantee unchanged total cost.

User clarified that the horizontal blocking disappears when increasing the existing
UI step setting. No marcher step policy, step control or budget was changed here.
The earlier hypothesis attributing those blocks to shape generation is unconfirmed.

Cirrus now has broad packets and finer bent, tapered fibres rather than similarly
stretched noise at both scales. It retains one base and one detail texture read.
Cirrus coverage remains a threshold control, not a calibrated sky-area percentage.

`cloudRender` previously returned from production through `cloudMarch`, which called
`cloudCirrus`, but never called cirrus in its reference continuation. Reference mode
now composites the same authored thin sheet behind the volumetric clouds. This fixes
the missing call; it does not turn the ice sheet into a volumetric random walk. The
sheet is still a bounded single-scattering approximation, not an unbiased reference
for cirrus scattering. Its existing layer-order assumption (cirrus behind the liquid
layers) remains; an opaque lower cloud can legitimately hide it.

Read-only external IPC confirmed the open scene used reference tracing and had cirrus
disabled at inspection. No live settings were changed. New code is not yet running.

## One material authority

The VDB-listed internal sky-cloud object is a derived legacy proxy, not the Vulkan
cloud field. Its VolumeShader is overwritten by `ensureInternalSkyCloudVolume`.
Editing it therefore failed to author the Vulkan material.

The list now labels it `[World] Clouds`. Selecting that row routes to a focused
CloudState editor instead of imported-VDB transforms, density remap and graph controls
that do not apply to this field. The same material controls are available in World >
Clouds. The VDB editor commits through `rtapi::setCloudsJson`; the World panel retains
its validated `World::setClouds` path. Python and IPC use the same API service and
CloudState validation. Existing imported VDB and simulation-owned volume panels keep
their existing behavior.

`CloudState.material` is shared by all volumetric layers and cirrus:

| Field | Default | Accepted range / meaning |
| --- | --- | --- |
| `density_multiplier` | 1 | 0..100, density scale |
| `scattering_coefficient` | 1 | 0..100, multiplier of nominal layer extinction |
| `scattering_color` | [1,1,1] | linear RGB 0..1, single-scattering albedo tint |
| `absorption_coefficient` | 0 | 0..100, achromatic absorption multiplier |
| `override_anisotropy` | false | bool; otherwise retain droplet/ice phase |
| `anisotropy` | 0.75 | -0.99..0.99, matched HG evaluation and sampling |
| `emission_color` | [1,1,1] | linear RGB 0..1 |
| `emission_strength` | 0 | 0..100000, source radiance per optical depth |

Total extinction = nominal extinction * density multiplier * (scattering +
absorption). RGB scattering albedo = scattering colour * scattering / total.
The colour is an albedo tint, not independently coloured absorption: darkening a
component removes its scattered energy. Zero density or zero coefficient sum disables
the field, including emission. Defaults retain the old white, non-absorbing transport.

Both Vulkan paths consume the material. Reference events add emission then attenuate
scattering throughput by the RGB albedo. Production uses the albedo for its source
and reduces approximate higher-order scattering with mean albedo. Cirrus uses the
same material with its bounded sheet source. Cloud density queries and shadows use
the same total extinction scale; the existing weather majorant remains valid because
the GPU layer extinction is scaled identically. Density bounds refer to this effective
extinction, not the unscaled `layers[].extinction_per_m` alone.

The GPU packet grows from 240 to 272 bytes. Buffer allocations, descriptor ranges and
history keys use `sizeof(CloudParamsGPU)` and therefore include the new fields. C++
and all cloud shader consumers must be rebuilt together. No extra noise texture,
descriptor binding or per-sample noise lookup is added for material controls.

Cloud JSON, project save/load and world keyframes use `cloudsToJson/cloudsFromJson`;
material fields are included and interpolate with the cloud keyframe block. Missing
material fields in older files use neutral defaults. Existing cloud shape presets
preserve the authored material, consistently in UI and IPC. Invalid ranges, malformed
RGB arrays and non-finite values are rejected before commit; authored state stays
unchanged. Existing overlap and render-job validation still applies.

OptiX/CPU keep their documented frozen legacy cloud approximation. The new optical
material transport is implemented for Vulkan RT and RayFusion; it is not advertised
as material parity with those legacy renderers. Volume Graph/remap/temperature shader
controls are not silently mapped to unsupported atmosphere operations.

## Script and IPC

Embedded Python (the API call itself is short; external IPC remains required for
live frame-loop tests):

```python
rt.world.set_clouds(material={
    "density_multiplier": 1.2,
    "scattering_coefficient": 1.0,
    "scattering_color": [1.0, 0.95, 0.9],
    "absorption_coefficient": 0.05,
    "override_anisotropy": True,
    "anisotropy": 0.7,
})
material = rt.world.get_clouds()["material"]
```

External equivalent: `ipc.call("world.set_clouds", material={...})`, then
`ipc.call("world.get_clouds")["material"]`. Partial material objects merge with
current values; they do not reset omitted fields. Changing settings resets accumulation
and marks the project modified through the existing commit service.

## Validation and user checklist

- Run `python scripts/test/check_cloud_shape_contracts.py`: non-build consumer
  wiring and noise-budget checks, including reference cirrus and both material paths.
- Build C++ normally, including `CloudMaterialUI.cpp`, and recompile the cloud
  consumers using `RayTrophiStudio/compile_shaders.ps1` or the normal shader workflow:
  `cloud_sample`, `cloud_raster`, `cloud_shadow`, `miss`, `closesthit` and
  `sphere_closesthit`. Reload the application with matching binaries. The shader
  plan generates sphere closest-hit from `closesthit.rchit` with `SPHERE_HIT=1`.
- Same camera/frame/seed/exposure: compare tall cumulus silhouette and variation;
  use the existing step setting for sampling quality, rather than changing transport
  while judging the shape.
- Enable cirrus in the UI; compare production and reference with volumetric layers
  off, then on. Check visible high packets, curved fibres and lower-cloud occlusion.
- Select `[World] Clouds` in VDB and change density, scatter colour, absorption,
  phase override and emission individually. Read back via script/IPC; check World >
  Clouds shows the same state and that edits survive multiple frames and save/open.
- `python scripts/test/rt_test_cloud_material_ipc.py` is a read-only schema check.
  After building, `python scripts/test/rt_test_cloud_material_ipc.py --apply` runs
  deliberate live edits and GPU query checks, then restores original cloud settings
  in `finally` and verifies restoration. It restarts accumulation. Run from a
  separate process, never from Studio's embedded workspace. Use `--backend render`
  to repeat on the other Vulkan role when available.
- Compare active completed-sample progression for cost. Shape occupancy and cirrus
  becoming visible in reference can add work; no speedup is claimed here.

Codex ran no project build, shader compilation or application launch. Live acceptance
of the new source and visual quality remain pending user compilation.
