# Cloud direction, contours and tall formations — 2026-10-11

Status: source implemented; non-build checks pass. C++/shader compilation and
visual/live acceptance are pending. Previous material and reference-cirrus work
was built by the user: material edits work in the VDB panel and cirrus appears in RT.

## Observed baseline

External named-pipe reads found the storm settings active: one 9 km-thick layer,
8 km cell period, 400 m detail period, erosion 0.45, type 0.95, plus cirrus. Production
RT was selected with 1024 steps; realtime steps were 512. Material edits were retained
(density 0.98, scatter 1.33, overridden anisotropy 0.935).

The viewport capture below shows the nested contour pattern reported by the user.
No scene or camera settings were changed. Capture was temporarily enabled to obtain
the frame and then disabled. This image is a baseline, not a rendering of the new code.

Compiled baseline (temporary capture removed)

The source uses several cellular distance channels both for formations and erosion,
and folded absolute-value gradient noise for the base shape. These are plausible
contributors to the regular contours; the screenshot does not prove a unique cause.
The new implementation targets those correlations. It does not claim that every
visible contour is a texture boundary or that the issue is already visually fixed.

## Implemented changes

- Base and detail noise generation bake in a smooth periodic vector warp. Base
  gradient noise is no longer folded through `abs`, and the alpha channel uses
  independent signed gradient structure for lean/taper instead of another nested
  cellular distance octave. Texture sizes/formats and runtime lookup counts are
  unchanged. This adds generation work, not per-sample texture reads; no measured
  performance improvement is claimed.
- Tall types have a less vertically stretched body, plus intermediate-scale flank
  erosion from an existing higher-frequency base channel. Fine erosion is stronger
  on tall types while retaining base protection. Both erosion stages only remove
  density, preserving detailed density <= cheap density <= effective local majorant.
- `storm` now uses cell period 6500 m, detail period 700 m and erosion 0.65. The detail
  period is the texture repeat length, not an individual billow's diameter. The
  broader detail scale reduces the gap between large bodies and details that the
  marcher can sample. No marcher budgets or UI step limits changed.
- A new canonical `towering_cumulus` preset has a 6.5 km-deep layer, type 0.75,
  cell period 4200 m and detail period 500 m, without requiring a cirrus anvil.
  Preset registry, UI list, Python and IPC all use `cloudPresetNames/cloudPreset`.

The already-open scene retains its authored values. Reapplying `storm` after building
uses the revised defaults; existing saved storm-like scenes do not change their
stored scales automatically. Existing preset application preserves the material.

## Direction authoring

`layers[i].rotation_degrees` and `cirrus.rotation_degrees` default to 0 and accept
finite values from -360 to 360. They are exposed in World > Clouds and the VDB-listed
`[World] Clouds` editor. The layer control remains editable with climate derivation
enabled, because the climate does not derive this field.

Layer angles rotate the field about world Y, relative to its advected origin.
Positive angles turn the pattern's +X direction toward +Z. The inverse rotation is
precomputed in C++ and used for shape, detail and weather coordinates. Weather and
the majorant share `cloudWeatherUV`, including offsets; rain coverage also uses this
same weather lookup. An orthogonal rotation preserves segment travel length, so the
existing tracking neighbourhood has the same travel bound. Wind advection/shear
still come from climate and are not themselves authored by this angle.

Cirrus uses the actual horizontal climate wind velocity for its frame, rather than
normalizing accumulated displacement (which was zero at time zero). Its angle is
an offset from that direction; with no horizontal wind the reference direction is +X.

JSON save/load and keyframes use the same CloudState serialization/validation.
Missing angles in older files use zero. Angle keyframes interpolate linearly, like
the existing cloud block; e.g. 0 -> 360 explicitly represents a full turn. Invalid
values are rejected before commit and leave authored state unchanged.

Python:

```python
rt.world.set_clouds(layers=[{"rotation_degrees": 35.0}],
                    cirrus={"rotation_degrees": -20.0})
rt.world.apply_cloud_preset("towering_cumulus")
```

External equivalents:

```python
ipc.call("world.set_clouds", layers=[{"rotation_degrees": 35.0}],
         cirrus={"rotation_degrees": -20.0})
ipc.call("world.apply_cloud_preset", preset="towering_cumulus")
```

Angles belong to the preset's shape, so applying a preset resets them to its default
zero. The material and quality are preserved. Vulkan RT and RayFusion consume the
new orientation; the documented frozen OptiX/CPU legacy approximation does not
implement these field rotations.

## Build and acceptance

The GPU layer grows to 64 bytes and the complete cloud packet to 336 bytes.
**Build C++ and all cloud shader consumers together.** Include `cloud_noise.comp`
this time, as well as `cloud_sample`, `cloud_raster`, `cloud_shadow`, `miss`,
`closesthit` and `sphere_closesthit` through the normal workflow. Restart with matching
binaries so the baked noise textures regenerate; rebuilding only the consumer shaders
cannot update existing noise textures.

1. Compare the captured camera/frame and current authored settings first. Inspect
   nested contours and detail on both volumetric clouds and cirrus. A fresh compiled
   comparison is required to establish whether the proposed cause was correct.
2. Reapply `storm` separately, then try `towering_cumulus`. Inspect bases, towering
   silhouette, sidewall cuts, small billows and the storm anvil. Keep exposure,
   resolution and transport fixed while comparing shape.
3. Rotate each enabled layer 0/37/90/360 degrees; check weather coverage, silhouette,
   shadows and rainfall coverage rotate together. Rotate cirrus independently and
   check its direction at time zero and during wind playback.
4. Confirm World and VDB angle controls read the same state. Check Python/IPC
   readback, invalid-angle rejection, keyframe motion and save/open persistence.
5. `python scripts/test/check_cloud_shape_contracts.py`: 8 non-build checks passed,
   including shared weather/tracking rotation, packet field order and the unchanged
   runtime noise lookup budget. Python syntax, rotation-length invariance and
   erosion-bound checks also passed. These do not compile GLSL or prove visual quality.
6. After building, external `python scripts/test/rt_test_cloud_orientation_ipc.py`
   checks the schema without edits. Add `--apply` for angle round-trip, GPU full-turn
   equivalence, changed field and rejection checks. It temporarily edits angles and
   restores all cloud settings in `finally`, restarting accumulation. Run from a
   separate process; use `--backend render` for the other Vulkan role when available.

Codex ran no build, shader compilation or application launch. New visual quality,
live angle acceptance and actual cost remain unverified until the user builds.
