# Rig serialization audit and hierarchy persistence

Status: hierarchy persistence/fallback delivered in source; user save/reopen
verification pending. Builds and application/CPU tests were not run by Codex.

## What is currently saved

| Data | Project persistence |
| --- | --- |
| Bone indices, default local transforms, parents, offsets, weighted-joint set, per-model inverses | Existing BoneData save/load |
| Imported and generated target clips, model ownership, TRS key values/times, duration/rate, frame range | Existing AnimationDataList save/load |
| AnimGraph topology/settings and character context playback flags | Existing graph/context save/load |
| NodeHierarchy authored names, unique keys, parent indices, local matrices | Added in this delivery |
| Derived children, skeletonNodes, controller/ozz runtime caches | Rebuilt after load |
| Manual node_map, mode/motion-scale authoring options and mapping presets | Not persisted yet; baked target clip already stores resulting channels |
| Preview time/plane/zoom, editor panel state, transient bone selection and overlay-visible setting | Not persisted by this rig delivery |

Do not confuse saved baked results with a saved retarget authoring session.
Saving a target-bound clip is enough to play that clip after reopening; resuming
its original editable mapping recipe still needs planned preset/session storage.

## Implementation and compatibility

The audit found that model contexts saved metadata but omitted NodeHierarchy.
The loader clears/recreates contexts, so the original imported node tree was
lost. BoneData and clip presence alone do not provide exact authored node names
and every helper/animated scene node required by retarget/2D preview.

New projects store `importedModelContexts[].nodeHierarchy` with version=1,
nodes[] containing name, uniqueName, parent and localBind (16 row-major floats).
Node array order and matrices are preserved, including nonuniform/sheared rest
matrices: serialization does not apply the retarget solver's narrower limits.
Children are derived from parent indices. Duplicate authored names are allowed;
unique keys must be nonempty/unique, transforms finite and parent links valid
and acyclic. Version/type failures and malformed graphs are rejected.
The pure deserializer stages output, leaving the supplied hierarchy unchanged
on failure. Existing project load itself is not an atomic scene transaction.

Older files without this field rebuild the available prefixed bone/helper tree
from BoneData; authored names are inferred by removing the character prefix.
Parents are placed before children. If the old data has multiple roots, an
identity `__LegacyRoot` helper joins them for existing node-zero tree walkers.
Missing parent data/cycles are errors. Missing local transforms for old indexed
bones use identity compatibility defaults. Never-stored scene nodes or their
original names/transforms cannot be recovered by this fallback; reimport may
be required for exact legacy retarget matching.

UI, Python `rt.project.save/open` and IPC `project.save/open` already use the
same ProjectManager path, which now calls the focused RigSerialization module.
No separate UI implementation or new persistence endpoint is introduced.
Project operations keep their existing failure semantics; hierarchy details
appear in project errors/logs. Pure helper validation codes include
unsupported_rig_hierarchy_version, invalid_rig_hierarchy_format,
invalid_rig_node_key, invalid_rig_parent, cyclic_rig_hierarchy,
invalid_rig_transform and missing_legacy_rig_parent.

## User validation after build

1. Use a test scene with skinned target, meshless source and a previously baked
   same-rig/rest-basis target clip. Save to a new disposable .rtp path.
2. Reopen without reimporting. Verify hierarchy/overlay, both clip lists and
   playback of the original and generated target clip on its graph.
3. Reopen Animation > Retarget. Rechoose rigs/clip/options (session options are
   not persisted). Compare pose at equal times; bind a further clip after reopen.
4. Run embedded `rt_test_rig_roundtrip.run(output_path, source_character,
   source_clip, target_character, node_map=mapping, mode="rest_basis")` from
   scripts/test. It saves/reopens the current scene, resets current history and
   makes the supplied new output path active. It compares hierarchy JSON,
   canonical bone/clip data and model-space sampled poses.
5. Run the existing unskinned FBX import fixture test; it now also checks saved
   hierarchy joint/ancestor keys.
6. On a disposable copy, remove the nodeHierarchy fields to test legacy fallback.
   Verify root-first hierarchy, finite poses and clear errors for missing parents.
7. Compile/run scripts/test/rig_serialization_test.cpp with assertions enabled,
   C++17/EHsc, include source/include, link Animation/RigSerialization.cpp,
   Math/Vec3.cpp and Math/Matrix4x4.cpp. Cases cover exact round-trip, derived
   children, duplicate authored names, rejected corrupt data with unchanged
   output, nonfinite save data and legacy single-root/forest/cycle handling.

Next editable-rig work must ship topology/rest-transform persistence together
with stable identities, four derived representations, undo and script/IPC parity.
