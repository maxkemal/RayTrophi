# Explicit cutout coverage and visibility-first material shading

Status: user built the application; **live foliage A/B measured on 2026-09-10**.
Codex drove the already-open application through IPC, without building or
launching it. Four scattered asset needle materials now have cutout enabled.

## Live scattered-asset test, 2026-09-10

The material panel does not expose the user's directly scattered asset
materials. They are nevertheless registered and reachable through the existing
material IPC service. The probe discovered four foliage materials with opacity
textures (Pin pond need 1/2 and Pin jef need 1/2), verified no explicit
transmission/map, and changed only their `alpha_cutout` parameter.

`scripts/ipc/Probe-FoliageCutout.ps1` records off/on/off/on windows with warmup,
22 measurement camera updates per window, and captures outside timed windows.
Resolution was 1680x945; camera and geometry were consistent within each pair.

| Applied settings | Cutout | GPU frame ms | Main ms | Transmission ms |
| --- | --- | --- | --- | --- |
| Performance, RT shadow off, no depth prepass | Off | 274.14 / 272.74 | 145.87 / 145.06 | 91.61 / 90.77 |
| Performance, RT shadow off, no depth prepass | On | 180.93 / 180.38 | 143.28 / 142.03 | 1.48 / 1.47 |
| Balanced, RT shadow ready, depth prepass on | Off | 325.07 / 323.69 | 181.71 / 180.84 | 121.79 / 121.41 |
| Balanced, RT shadow ready, depth prepass on | On | 82.41 / 81.12 | 59.75 / 58.04 | 1.48 / 1.44 |

Balanced frame cost fell about 75%; main shading remains the largest measured
cost. Depth prepass stayed around 18.4–19.0 ms. Geometry was unchanged between
cutout states: 34 draws, 29,038,508 triangles in Balanced; 21,151,903 triangles
in Performance. These are GPU timing windows, not sustained interactive FPS.

Records and before/after JPEGs: `tmp/cutout-live-20260910-094824*` (Performance)
and `tmp/cutout-live-20260910-095016*` (Balanced). Visual inspection found no
obvious missing trees; binary coverage changes fine needle edges. Water/glass,
cross-renderer parity and the remaining acceptance matrix are not established
by these captures. A pipeline-ready log was not located during this test.

The IPC roundtrip/rejection/restoration test passed on Pin pond need 1.
Camera, capture setting, original Performance quality and RT-shadow-off setting
were restored. All four needle materials were intentionally left with cutout
enabled. The project was not saved by the probe. Asset/scatter UI access and
automatic imported alpha-mode mapping remain unresolved authoring work.

## Authoring contract

### Closing update: defaults and remaining work

#### Reopen investigation, 2026-09-10 (supersedes automatic-foliage expectation)

The user reopened `E:/rt_k/lanspace.rtp` and lost the speedup. Read-only IPC
inspection confirmed all four needle materials had `alpha_cutout=0`; RT shadows
were enabled and ready. The saved `.rtp.shared` material records contain no
`alpha_cutout` field for these four materials, so the documented legacy default
correctly loads false. The earlier live edits had not been saved.

Direct inspection of the JSON chunks in the actual shipped `Pinus ponderosa.glb`
and `pine_jeffreyi.glb` found all four needle materials authored as **BLEND**, not
MASK. Therefore the new MASK import default does not cover these assets even on
a fresh import. Automatic conversion of this library's BLEND foliage remains
unimplemented; do not describe the MASK change as solving this scene's default.
A future explicit asset coverage policy must distinguish these tested foliage
materials from genuine BLEND surfaces and preserve deliberate saved overrides.

The four live values were set back to 1 and read back successfully, retaining
the user's quality/shadow settings. Current Performance + RT timing windows:
61.03 / 61.88 ms GPU, 43.19 / 43.92 ms main, 1.42 / 1.48 ms transmission.
Submitted geometry differed between windows (19.58M / 22.76M triangles), so
these are recovery observations, not a controlled speedup against the earlier
Balanced A/B. Record: `tmp/raster-postbuild-20260910-101959.json`.

The original project was not overwritten, and no reload was driven. The live
cutout edits need a project save to persist; save/reload is not verified here.
No renderer changes or build were made in this investigation.

The user additionally tested transmission and foliage scenes and reported no
visible problems. The measured coverage optimization is accepted for this
session; this is not exhaustive cross-backend certification.

Two final source changes await the user's next build:

- Vulkan material-preview RT shadows are requested by default. Existing hardware,
  shader, scene and depth-prepass eligibility checks and atlas fallback remain.
  An explicit off setting still wins; Performance without a depth prepass does
  not become RT-ready merely because the request defaults to on.
- The shared glTF importer automatically enables coverage for `MASK` materials
  with the standard 0.5 cutoff, including foliage asset/scatter imports. `BLEND`
  and explicit optical transmission retain their semantics. Nonstandard MASK
  cutoffs keep legacy handling until a configurable cutoff is supported.
  Existing serialized materials retain their stored setting; this is not a
  blanket migration of old projects or a name-based foliage heuristic.

Cutout is a material coverage semantic, not a universal quality switch. Its
eligible rendering optimization runs automatically; the checkbox/API remain
available for authoring exceptions. Imported BLEND foliage cannot safely be
reclassified solely from the presence of an opacity texture.

Remaining measured bottlenecks in Balanced + RT + cutout:

1. Main pass: 58–60 ms. Profile material/sky lighting and remaining pixel
   overdraw separately before choosing the next optimization. These stage
   timings alone do not identify the expensive shader instructions.
2. Depth prepass: 18–19 ms, with about 29 million submitted triangles. Assess
   foliage LOD, projected-size rejection and alpha coverage sampling next.
   Preserve flat mesh data and existing GPU visibility/indirect drawing.
3. Transmission: 1.44–1.48 ms in the measured forest, no longer the dominant
   cost. This does not predict a screen-filling glass/water scene's cost.
4. Fine cutout edges still need temporal/coverage antialiasing work. Graph
   coverage, mixed materials and custom import cutoffs remain conservative.

The total 81–82 ms is a substantial optimization, not yet a realtime frame
budget. Do not attribute all gains to RT shadows alone: the measured fast
configuration combines cutout, the depth prepass and covered material shading.

Final source checks passed: material coverage audit, raster material visibility
audit and diff whitespace check for the importer/document. No build was run.
Next build acceptance: fresh default RT request with readiness/fallback check;
fresh scatter import of standard MASK versus BLEND; old-project saved cutout
roundtrip. Recheck custom cutoff handling before extending automatic coverage.

Principled materials now expose `alpha_cutout`: 0 = legacy opacity behavior,
1 = binary surface coverage with threshold 0.5. Default is 0, including old
scene files. Values other than exactly 0 or 1, nonfinite values, wrong types,
and missing materials are errors through the existing material service.
Rejected values do not mutate the material. Transmission remains independently
authored; alpha cutout never invents transmission from intermediate alpha.

Material panel: **Alpha Cutout**, beside Alpha. The widget calls the same
`rtapi::setMaterialParamByName` service as Python and IPC; it does not write
another renderer-specific flag. The field is serialized as `alpha_cutout` and
copied by both Principled copy operations. GPU flag bit 25 carries the setting
through existing snapshots/material uploads with no buffer stride change.

Python:
```python
rt.materials.set_param(material_name, "alpha_cutout", 1)
assert rt.materials.get_param(material_name, "alpha_cutout") == 1
```

IPC:
```powershell
Invoke-RtIpc material.set_param @{material_name=$name;param='alpha_cutout';value=1}
Invoke-RtIpc material.get_param @{material_name=$name;param='alpha_cutout'}
```

`scripts/ipc/Set-FoliageCutout.ps1 -MaterialNames $names -Enabled 1` changes only
the explicitly supplied materials, reads all originals before writing, verifies
each write and attempts rollback on failure. It leaves successful edits authored;
it does not save the project. Enabling coverage can change fractional leaf edges
to binary edges. It is opt-in and is not an automatic filename heuristic.

For the previously measured scene, the saved inventory contains four needle
materials with opacity textures. After rebuilding and opening that same scene:
```powershell
$snapshot = Get-Content .\tmp\raster-material-snapshot.json -Raw | ConvertFrom-Json
$names = @($snapshot | Where-Object { $_.textures.slot -contains 'opacity' } | ForEach-Object { $_.name })
.\scripts\ipc\Set-FoliageCutout.ps1 -MaterialNames $names -Enabled 1
.\scripts\ipc\Probe-RasterMaterialCost.ps1 -BaselineOnly -Frames 26
```
This is an explicit migration of those four known materials, not an importer
policy. A different scene needs its own material list. Setting `-Enabled 0`
selects legacy behavior again. Rebuilding alone does not opt old leaves in.

## Visibility and shading

The main and camera-depth shaders share `previewSurfaceOpacity`; cutout turns
sampled coverage into 0/1 before the main shader's expensive material work.
Graph opacity overrides are still evaluated before final coverage in the normal
material shader. Replay classification shares the same cutout flag and excludes
opacity maps as a transmission reason when cutout is enabled. Actual authored
transmission, transmission textures, bubbles and graphs remain conservative.

A second compilation of the same material fragment source defines
`PREVIEW_COVERED_SHADING`, enabling `early_fragment_tests`. Its pipeline uses
EQUAL depth comparison with depth writes disabled. Only the main shading draw
selects this pipeline, only after an actual depth prepass, and only for meshes
whose complete material membership has exact coverage parity. The classifier
rejects graphs, tile breaking, explicit transmission/maps, water/bubble/volume
flags, unknown or external CPU material provenance, and legacy opacity maps or
partial opacity. Mixed meshes remain on the normal pipeline if any member fails.
Opaque impostor coverage is preserved. Transmission replay always uses the
normal pipeline, preserving its back-depth and refraction behavior.

Missing/failed covered pipeline falls back to the normal material pipeline and
logs a warning. Successful creation logs:
`[Coverage] EQUAL / early-tests material pipeline ready.`
Lifecycle follows normal pipelines, including resize retention and destruction.

CPU get_opacity, Vulkan any-hit/legacy transmission, OptiX alpha/transmission
consumers, RT shadow candidates and RayFusion bounce candidates use the shared
binary-coverage rule. Existing renderer differences in graph evaluation, mip
selection, tile-break UVs and erosion remain outside the exact-prepass fast path;
cross-renderer visual parity for these complex cases is not claimed by a static
audit. In particular graph-driven coverage still needs its own full visibility
evaluation design before it can enter the fast path.

## Validation and user build

Passed source-only checks: `audit_material_coverage.py`,
`audit_raster_material_visibility.py`, `audit_raster_depth_prepass.py`,
`audit_raster_gpu_instancing.py`, `audit_shader_struct_layout.py`,
`audit_rayfusion_bounce.py`; new PowerShell/Python scripts parse successfully.

Delivered but not executed: standalone C++ coverage/classification test at
`scripts/test/raster_material_visibility_test.cpp`, Python API test at
`scripts/test/rt_test_material_coverage.py`. IPC roundtrip/rejection test at
`scripts/test/rt_test_material_coverage_ipc.ps1` passed in the live scene.

User build:

1. Build C++/CUDA normally. New `MaterialPreviewCoverage.cpp` is registered in
   the vcxproj. Rebuild affected CUDA/PTX consumers as well as the host binary.
2. Run `RayTrophiStudio/compile_shaders.bat`. It now also produces
   `material_preview_covered.spv` from `material_preview_frag.frag` with the
   required define. Rebuild normal preview, shadow prepass, closest-hit/any-hit,
   RayFusion shadow and bounce shaders together. Do not mix old SPVs with the
   new material flag behavior.
3. Optional CPU test in a developer shell:
   `cl /nologo /std:c++17 /EHsc scripts\test\raster_material_visibility_test.cpp /Fe:tmp\raster_material_visibility_test.exe /Fo:tmp\raster_material_visibility_test.obj`
   then `tmp\raster_material_visibility_test.exe`.

Acceptance:

- Old scene without cutout enabled retains legacy appearance. Toggle cutout on
  the four known needle materials, verify silhouettes and absence of refracted
  leaf edges. Water/glass keep their explicitly authored transmission.
- Keep the same camera, resolution, balanced quality and RT shadows. Record
  baseline/repeat windows and applied geometry. Check pipeline-ready log.
  No numeric speed target is promised before this build is measured.
- With only cutout/opaque materials and no volumes/graphs/transmission maps,
  replay must be SKIPPED. Adding actual transmission must enable it immediately.
- Disable both RT shadow and depth prepass: coverage still works with the normal
  pipeline. Restore them: silhouettes must match and covered shading may resume.
- Test mixed-material meshes, graph opacity, tile breaking, proxies, GPU culling
  on/off, opaque-to-glass edits, reassignment, clone, save/reload, and viewport
  resize. Unsupported fast-path cases must render through the normal pipeline.
- Compare Vulkan RT, OptiX and CPU cutout silhouettes/shadows on a simple textured
  card, including alpha 0, 0.49, 0.5, 1 and cutout with explicit transmission.

Deferred: alpha-to-coverage/MSAA or temporal edge treatment, automatic imported
alpha-mode mapping, graph visibility evaluation, and finer material subdraws.
