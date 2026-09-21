# Rest-basis retarget: Phase 0B

Status: user confirmed hierarchy rest-basis build, workflow and target playback.
New 2D mapping/sample API checks are in RIG_MAPPING_UI_BUILD_CHECKS.md. Codex
has not compiled tests, built the project or launched the application.
Manual mapping panel build/visibility is user-confirmed; manual map playback
was deferred to later IPC testing. Automatic same-rig playback was confirmed.

## Workflow

1. User builds the project. Import target character and source skeleton/clip.
2. Target hierarchy: **Add clip from imported rig**, choose source rig/clip.
3. Enable **Correct rest-pose / bone basis**. Default motion scale is 1.
4. Map renamed nodes and ancestors/helpers if needed, then **Check mapping**.
5. Successful check validates the same converted clone used by commit; it is
   a numerical preflight, not a live side-by-side motion preview.
6. **Bake retargeted clip to this character**; select the new target clip on
   its AnimGraph. Verify target mesh/overlay and unchanged source playback.
7. Verify undo/redo and save/reopen preserve the generated target clip.

Both existing operations accept the same optional parameters in Python and IPC:

```python
options = dict(node_map={}, mode="rest_basis", translation_scale=1.0)
preview = rt.anim.preview_clip_binding(source, clip, target, **options)
if preview["ready"]:
    bound = rt.anim.bind_clip(source, clip, target, **options)
```

Defaults remain `mode="same_rig"`, `translation_scale=1.0`, `node_map={}`.
The same-rig path continues to copy absolute local channels. Reports now include
`mode` and `translation_scale`. No new IPC method/capability is introduced;
preview remains read-only, bind remains scene-write with undo.

## Transform contract

Quaternions are (w,x,y,z), unit rotations, column-vector matrices T*R*S.
Let source/target local rest rotations be Rs/Rt and rest-global rotations Gs/Gt.
The node basis change is C = inverse(Gs)*Gt. Each source local rotation key Q
becomes Rt * inverse(C) * inverse(Rs) * Q * C. Thus a source rest key becomes
target rest; equal rest frames preserve the key. This transports local angular
motion in rest frames; it does not enforce identical animated global rotations
when parent motion or limb geometry differ.

Translation keys become targetRestPosition + inverse(targetParentGlobalRotation)
* sourceParentGlobalRotation * (key-sourceRestPosition) * translation_scale
* sourceParentGlobalUniformScale / targetParentGlobalUniformScale.
Root parent frames are identity. Motion scale applies to all authored translation
deltas, including root/helpers; it is not inferred from character height.
Relative uniform scale keys become targetRestScale * (keyScale/sourceRestScale).
Missing channels stay missing and use target bind fallback in existing samplers.
Times/duration/rate are preserved. Affine vector transfer and constant unit
quaternion multiplication preserve the existing linear/slerp interpolation.

Hierarchy globals are computed from parent indices independent of node ordering.
All nodes in each imported hierarchy must have finite, reconstructable TRS,
positive uniform rest scale and valid acyclic parent indices. Reflections,
shear and nonuniform rest/animated scale are deliberately unsupported. This
conservative first solver does not silently discard deformation components.

Errors: `invalid_retarget_mode`, `invalid_translation_scale` (finite, >0, <=10000),
`translation_scale_requires_retarget` for nondefault scale with same_rig,
`invalid_retarget_rest`, `unsupported_retarget_rest`, `invalid_retarget_hierarchy`,
`invalid_retarget_keys`, `unsupported_retarget_scale_keys`. Existing node-map and
hierarchy errors remain. Key times must be finite, strictly increasing and
within [0,duration]; vector values finite, quaternion norms nonzero.
Python type errors use TypeError; service failures use ValueError. IPC invalid
parameter types use `invalid_parameter`; service failures use named codes.
Conversion failures discard the staged clip and set ready=false before commit.

## User test checklist

- Identity rest rotation, target limb translations of different lengths: keys
  keep motion, target rest positions remain intact.
- Source A/T rest key maps exactly to target rest; animated rotation follows
  the documented rest-frame angular direction.
- Rotated root/helper and different uniform units: parent-space motion and
  explicit motion scale work; no source clip or target skeleton mutation.
- Missing channels continue to use target defaults. Reject duplicate target,
  omitted parent override, invalid quaternions/times, shear and nonuniform scale.
- Run the embedded `rt_test_clip_binding.run(source, clip, target,
  mode="rest_basis", translation_scale=1.0, node_map=mapping)` on production rigs.
- Run external `scripts/test/test_retarget_ipc.py` with the same rig/map inputs;
  compare output clip and channel inventory with Python/UI. This test adds a clip.
- Build/run `scripts/test/rest_basis_retarget_test.cpp` with assertions enabled,
  linking Animation/ClipBinding.cpp, Animation/Retarget.cpp, Math/Vec3.cpp and
  Math/Matrix4x4.cpp from source/src; include source/include, use C++17/EHsc.
  The same-rig regression now also links Retarget.cpp and those math sources.

Synchronized skeleton pose preview is now delivered in source; user checks pending.
Remaining: rest-pose alignment controls, topology/role
retarget, limb proportion policies, IK/contact correction and foot-slide metrics,
versioned mapping presets. No claim of full humanoid/nonhuman retarget is made.
The next UI milestone is a contextual 2D mapping editor; no permanent shelf is
introduced by this delivery. Fast auto-rig remains a separate template/edit/bind
milestone in RIG_AUTHORING_ROADMAP.md.
