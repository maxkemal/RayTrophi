# Raster opacity and transmission replay filtering

> Follow-up implementation: [explicit coverage / EQUAL shading](MATERIAL_COVERAGE_PIPELINE_2026_09_10.md).
> This page records the earlier replay-filter patch and its measured build.

2026-09-09: implemented in source; user subsequently built and tested the open
scene. Post-build timing results are below; full visual regression checklist
remains outstanding. Baseline:
[live A/B measurements](RASTER_FRAME_COST_AB_2026_09_09.md).

## Behavior

The material fragment shader now evaluates authored opacity immediately after
UV transformation/tile breaking. For materials without a graph, zero-coverage
pixels leave before terrain layers, albedo, normal, roughness, and lighting.
During transmission replay, sampled opaque pixels also leave there when no
transmission map, graph, or bubble behavior can make them transmit. Opacity is
sampled once and reused; graph opacity overrides still run before final discard.
The existing 0.1 alpha floor, partial-opacity legacy refraction, and opaque
impostor semantics are preserved. This is not a new cutout mode.

Replay no longer submits provably opaque meshes. Both the scene replay gate and
individual draws use the same mesh classifier. The scalar possibility predicate
is shared literally between C++ and GLSL in `raster_material_policy.h`.
Opacity maps, transmission maps, material programs, partial scalar opacity, and
bubbles conservatively retain a mesh. A mesh with mixed materials is retained
as a whole; unknown/incomplete CPU data and external material buffers retain
draws. Graphs are classified per material using the canonical uploaded program
offset table. Volumes and SDF retain their existing replay scheduling.

Only the unique material-ID membership is cached, not a may-transmit Boolean.
All 11 existing material-ID invalidation sites invalidate this cache alongside
the RayFusion content hash. Reassignment, sculpt rebuilds, proxy generation,
cloning, and ID compaction therefore rebuild membership lazily. Scalar, texture,
and graph edits use live material/program state without rescanning vertices.
The cache reads the flat raster ID stream; no Triangle facade traversal is added.

Both passes retain the existing compacted instance buffer and GPU indirect
commands. This change does not disable frustum culling or submit a second list
of unculled instances. The live observation was a cost spike when a few trees
enter the frame, not evidence that all offscreen fragments execute.

This internal optimization adds no UI authoring operation or setting. Existing
script/IPC material operations continue through the same renderer upload path;
there is no UI-only classifier or duplicated binding implementation.

## Validation performed

- `python scripts/audit_raster_material_visibility.py`: passed; checks all 11
  invalidation sites, shared policy, early alpha ordering and graph guards,
  replay filtering, volume scheduling and indirect draw wiring.
- Existing `audit_raster_depth_prepass.py` and `audit_raster_gpu_instancing.py`:
  passed. These are static checks, not compilation or GPU behavior tests.
- Standalone C++ regression test delivered at
  `scripts/test/raster_material_visibility_test.cpp`; **not compiled/run**.
  Covers opaque-to-glass edits, map edits, program updates/removal, same-size ID
  reassignment, mixed/impostor IDs, ID clamping, missing/external data and bounds.

## User build and acceptance checklist

1. Rebuild C++ and `material_preview_frag.frag` to
   `material_preview_frag.spv` using the normal shader build
   (`RayTrophiStudio/compile_shaders.bat` includes this shader). Both new shader
   includes must be available at compilation. No new .cpp registration is needed.
2. Optional standalone CPU regression in an MSVC developer shell, workspace root:
   `cl /nologo /std:c++17 /EHsc scripts\test\raster_material_visibility_test.cpp /Fe:tmp\raster_material_visibility_test.exe /Fo:tmp\raster_material_visibility_test.obj`
   then `tmp\raster_material_visibility_test.exe` (exit 0 means pass).
3. Open the original foliage/water scene at the saved camera, balanced preset,
   1680x945. Compare main/transmission timings to the recorded baseline with
   matching applied geometry/settings. Drive the camera through sparse and dense
   foliage. Check needle silhouettes, near/far proxies, water and glass refraction.
4. In an opaque scene with no volumes/programs/maps/partial opacity, replay must
   be SKIPPED. Adding transmission by UI, Python material API or
   `material.set_param` IPC must enable it without reloading the scene; restoring
   opacity must remove the unnecessary replay. Repeat with opacity/transmission
   textures, graph-only transmission, and mixed-material meshes.
5. Reassign a mesh from opaque to glass and back; delete/compact material IDs;
   sculpt/rebuild/clone a mesh. Glass must never disappear due to cached membership.
6. Check volume-only and SDF scenes still composite correctly. Toggle RT shadows
   and depth prepass; this patch does not require either and must work with both
   disabled. Check GPU instancing/culling enabled and disabled.

## Deliberately unresolved

This does not introduce EQUAL or early_fragment_tests: camera prepass coverage
differs from the main shader for partial alpha/graphs/tile breaking, and the
opaque phase has a back-depth atomic side effect. Making early depth unconditional
would risk losing glass thickness or coverage. Foliage with intermediate alpha
still may require replay and keeps the existing appearance. No speedup magnitude
is claimed before the rebuilt shader is measured. Remaining sky sampling cost
and per-mesh subdraw splitting are separate follow-ups if this remains expensive.

## Post-build live measurement, 2026-09-09 21:52

User reported a modest improvement after building. Two unchanged-settings
windows (27 GPU-marked frames each) recorded 318.884 / 320.028 ms GPU frame,
181.591 / 182.250 ms main, and 116.281 / 116.760 ms transmission. Both have
29,366,548 visible triangles, 34 draws, balanced, scene lighting, 1680x945,
RT shadows ready, six cascades replaced, depth prepass and GPU culling enabled.
Second-window CPU recording: 0.732 ms; prepass 19.105 ms; RT shadow 1.525 ms.

Earlier balanced reference windows were approximately 372–381 ms frame,
204–209 ms main, 146–150 ms transmission, at 29,327,572 visible triangles.
The post-build result is consistent with an approximately 14–16% lower frame
cost, but the camera was not recorded with the old baseline and geometry counts
differ; **this is not a controlled exact-camera old/new binary A/B**. No sustained
FPS claim is made: telemetry marks individually driven dirty frames.

Camera restored; no material, quality, world, or shadow settings changed.
Raw data: `tmp/raster-postbuild-20260909-215224.json`. Reproduce without material
edits using `Probe-RasterMaterialCost.ps1 -BaselineOnly -Frames 26`; this mode
does not load the scene-specific material snapshot and writes timestamped data.
Main plus transmission still account for approximately 93% of GPU frame cost.
