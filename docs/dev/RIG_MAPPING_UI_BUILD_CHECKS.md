# 2D rig mapping and synchronized pose preview

Status: user confirmed build and runtime of the 2D mapping/pose-preview UI.
The prior hierarchy rest-basis workflow also builds and animates correctly.
Dedicated Python/IPC and CPU regression execution remains pending. Foot
contact correction remains a later IK/contact milestone.

## User workflow

1. Build the project, import target character and source rig/clip.
2. Open the existing bottom Node Editor > Animation editor, then **Retarget**.
   **Graph** keeps the existing animation graph editor. Dock/float behavior is
   inherited from the existing Animation surface; no new permanent shelf.
3. Choose target, source rig and clip; enable rest-basis correction if needed.
4. Source appears on the left, target on the right. Pick a source joint then a
   target joint to create an override. Existing channel matches link across the
   panels; choosing a source highlights its current mapped target when available.
5. Use Front XY / Side ZY / Top ZX, Fit and Zoom. Shared scale preserves relative
   rig sizes; each rig is centered on fit and that center stays fixed during
   preview so root motion stays visible. Fit after a large transform change.
6. List selection handles overlapping helper joints; **Find node** filters the
   lists. **Unresolved source only** filters source channel diagnostics. Use
   **Use auto for selected source** or **Clear all overrides** to reset mapping.
7. Play/Pause and the seconds slider sample source and target simultaneously.
   Successful conversion shows the bound target clip. Incomplete compatible
   preflight shows animated source and target bind pose. Invalid parameter/map/
   conversion errors show bind skeletons with an explicit error, keeping edits
   accessible. Preview does not change actual character playback or graph.
8. Check mapping and bake; choose the resulting clip on the target Graph.
   Hierarchy and visual editor share the same per-target mapping/options state.

## Shared service and agent contract

Python/IPC `anim.sample_clip_binding` takes source_character, source_clip,
target_character plus optional time_seconds=0, node_map={}, mode="same_rig",
translation_scale=1. It shares mapping/conversion with preview/bind. Seconds
loop at duration using AnimationData's existing sampler, including its current
interpolation behavior. This is not a new animation runtime.

Returned object: binding report, requested time_seconds, duration_seconds,
target_pose_source (`bound_clip` / `bind`), source[] and target[]. Each row contains
unique `name`, parent unique name (empty for root), `world_transform` as row-major
16 floats. Position indices are 3/7/11. Transforms are accumulated in model
space; scene placement, skin offsets and scene/controller/ozz state are excluded.
All hierarchy nodes are returned, including helpers. UI selection uses existing
rig.select_bone service for actual bones; non-bone helpers remain UI selectable.

```python
p = rt.anim.sample_clip_binding(source, clip, target, time_seconds=0.5,
                               mode="rest_basis", node_map=mapping)
assert p["binding"]["ready"]
print(p["target_pose_source"], p["target"][0]["world_transform"])
```

Invalid time: invalid_preview_time (finite/nonnegative). Invalid parent indices,
cycles: invalid_preview_hierarchy. Nonfinite sampled matrices: invalid_preview_pose.
Existing binding/retarget errors and type semantics remain. No pose arrays are
returned by API failure. UI bind fallback is sampled by the same canonical core.
IPC is Read; sampling does not modify clips, scene selection, history or playback.

## Verification after user build

- Check both docked and floating Animation editor Graph/Retarget tabs, target
  switching, hierarchy/visual state sharing and viewport shortcut suppression.
- Identity and A/T rest cases: slider poses should match baked target playback
  at equal times. Check root/helper axes and unit scale scenarios from
  RETARGET_BUILD_CHECKS.md, including nonzero root motion.
- Pick source then target on canvas and in lists; verify override, mapped pair
  highlight, named errors for duplicate targets and recovery after reset.
- Invalid/incomplete mappings keep skeletons editable and prevent baking.
- Confirm preview doesn't start, stop or seek real characters; graph topology,
  clip inventory and history stay unchanged until explicit bind.
- Check visual responsiveness on a long Mixamo clip. First implementation stages
  conversion per sample; persistent preview caching/performance tuning remains
  a follow-up if large clips need it.
- Shared `rig_mapping_preview_contract.check_sample` is called by embedded
  rt_test_clip_binding.run and external test_retarget_ipc.py. It checks finite
  matrices, parent/name integrity, loop boundary and unchanged target inventory.
  Those larger tests eventually bind a clip; run on a test project.
- User compiles/runs rig_pose_preview_test.cpp with assertions enabled, C++17/EHsc;
  link Animation/RigPosePreview.cpp, Math/Vec3.cpp, Math/Matrix4x4.cpp, include
  source/include. Core coverage: global composition, bind fallback, seconds loop,
  reordered parent array, invalid hierarchy/time and nonfinite poses.
- Existing ClipBinding core tests don't call the new sample API and need no
  additional source. Builds/CPU tests/application checks were not run by Codex.

Remaining: mapping preset persistence/revisions, rest alignment controls, zoom
around pointer/pan polish, topology/role inference, editable rig/template creation,
and contact IK. This is the first interactive mapping surface, not auto-rig.


## Deferred UI polish requested by the user

Planned, not implemented: mouse-wheel zoom around the cursor, middle-mouse pan,
and a subtle hover-revealed draggable splitter for left-panel/canvas sizing.
The splitter is the proposed "ghost scrollbar" affordance; it resizes panels.
Keep Fit/reset, minimum panel widths and existing dock/float behavior. Verify
that local canvas gestures do not affect viewport navigation or bone mappings.
These refinements can follow editable rig and template infrastructure.
