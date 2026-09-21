# Same-rig clip binding — Phase 0B first delivery

Status: user confirmed automatic same-rig binding and target AnimGraph playback.
Manual node overrides are implemented; user build/runtime verification is pending.
General rest-pose/anatomical retargeting remains a later step.

## Why the second Mixamo clip did not move the first character

Each import has a separate character identity and prefixed channel names.
The source clip samples its source joints; the target animator only owns its
target clips. Merely choosing another import's raw clip does not transfer it.

The shared ClipBinding core matches original authored node names, verifies all
channel targets and ancestor chains, clones/rekeys the clip, and sets its
modelName to the target character. Source rig/clip, target bones, weights and
bind offsets are preserved. Only target controller/ozz runtime is staged and
replaced. The graph reads the refreshed target clip list on its next evaluation.
No graph topology or graph clip selection is changed automatically.

## User workflow after build

1. Import the skin-bound Mixamo character and its original animation.
2. Import the same character's meshless animation FBX as a source rig.
3. Under Characters / Models, expand the **skin-bound target character**.
4. Open **Add clip from imported rig**. Choose the meshless source rig and clip.
5. Click **Check mapping**. Inspect mapped/missing/ambiguous/hierarchy counts.
6. For the same exported rig, click **Add same-rig clip to this character**.
7. In the target AnimGraph Clip node, select the reported **Added** clip name.
   This is a new target-bound clip; selecting the original source name still
   refers to the source import.

Default transform differences are reported. `ready` means all channel and parent
mapping is complete. It does not prove identical rest bases, units, anatomical
roles or body proportions. This operation intentionally copies absolute local
TRS for a caller-selected same exported rig. Different rest/basis/scale needs
the later retarget solver and explicit correction policy.

## Canonical script and IPC surfaces

Python:

```python
report = rt.anim.preview_clip_binding(source_character, source_clip, target_character)
result = rt.anim.bind_clip(source_character, source_clip, target_character, output_name="")
```

IPC: `anim.preview_clip_binding` and `anim.bind_clip`, with source_character,
source_clip, target_character and optional output_name (bind only). Preview is
Read; bind is SceneWrite and undoable. Empty output_name produces a unique name;
an explicit conflicting name is rejected. The operation rebuilds preflight on
commit on the main thread; it does not trust an earlier cached report.

Report: ready, mode=same_rig, source/target/clip identity, output_clip, matches
(source/target/authored_name), unmapped, ambiguous, hierarchy_mismatches and
rest_difference_count. Incompatible mapping is a valid ready=false preview;
commit rejects it with incompatible_clip_binding. Missing source/target/clip,
duplicate source clip name, invalid timing and name conflicts reject without
changing scene data. Normal parameter type errors are Python TypeError / IPC
invalid_parameter; validated service errors are Python ValueError / IPC named
error code. Unexpected staging errors return clip_binding_failed.

## Validation after user build

- Perform the two-import Mixamo workflow above. The new target clip must animate
  the target mesh and its skeleton overlay; original source playback still works.
- Check identical rigs with different import prefixes and bone index ordering.
- Check unmatched names, duplicate authored names and different parent chains:
  preview must report problems and bind must not modify clip inventory.
- Bind while the target is crossfading. Both controller clipA/clipB references
  must remain valid. The source character's playback must be preserved.
- Undo removes only the generated target clip and restores prior target runtime;
  redo restores it. Save/reopen preserves the bound canonical clip and mappings.
- After undo while a Clip node names the removed clip, expect that missing clip
  to stop producing a valid pose; graph selection is not rewritten by undo.
- Run embedded production script coverage on the imported rigs:

  ```python
  import sys
  sys.path.insert(0, "E:/RayTrophi_projesi/raytracing_Proje_Moduler/scripts/test")
  import rt_test_clip_binding
  result = rt_test_clip_binding.run(source_character, source_clip, target_character)
  ```

- Repeat preview/bind through IPC; verify returned report and target `anim.clips`
  / `anim.source_channels` inventory. Selecting the returned clip in the target
  graph must yield the same visible animation as the UI-generated clip.
- CPU core test (user runs in Developer PowerShell, assertions enabled):

  ```powershell
  cl /nologo /std:c++17 /EHsc /I RayTrophiStudio\source\include scripts\test\same_rig_clip_binding_test.cpp RayTrophiStudio\source\src\Animation\ClipBinding.cpp RayTrophiStudio\source\src\Animation\Retarget.cpp RayTrophiStudio\source\src\Math\Vec3.cpp RayTrophiStudio\source\src\Math\Matrix4x4.cpp /Fe:tmp\same_rig_clip_binding_test.exe
  .\tmp\same_rig_clip_binding_test.exe
  ```

Static checks: focused module/source registration, project XML, Python syntax,
descriptor generation and capability/mirror audit. No builds, CPU test executable
or application verification were run by Codex.

Rest-basis conversion is now delivered in source; see RETARGET_BUILD_CHECKS.md.
Remaining Phase 0B work: visual preview, role mapping, rest alignment and presets. The older file-asset prefix rewrite path
is not this imported-rig binding service and needs consolidation in that work.

## Manual mapping delivery

In **Manual node mapping**, choose target unique node names for differently named
source nodes. Include root/parent/helper overrides when their exported names differ.
**Auto by name** removes an override. Run **Check mapping** after each change;
then bind and choose the resulting target clip in AnimGraph.

Both Python and IPC accept optional `node_map` on `anim.preview_clip_binding` and
`anim.bind_clip`. It is an object/dict of source uniqueName -> target uniqueName.
Omitted/empty mapping preserves automatic matching. Python example:

```python
mapping = {"2_Spine": "1_Spine", "2_Armature": "1_Armature"}
report = rt.anim.preview_clip_binding(source, clip, target, node_map=mapping)
if report["ready"]:
    bound = rt.anim.bind_clip(source, clip, target, node_map=mapping)
```

Unknown keys/values return `unknown_source_node` / `unknown_target_node`;
two mapped source nodes targeting one target return `duplicate_target_mapping`.
IPC malformed mapping types return `invalid_parameter`; Python uses typed dict
conversion. Hierarchy mismatch remains a valid preview with `ready=false` and
binding returns `incompatible_clip_binding`. Failures create no target clip.
Overrides replace automatic matches; collision checking includes automatic nodes.

User verification after build: renamed joint + renamed helper map successfully;
omitted helper produces a hierarchy diagnostic; duplicate targets/unknown nodes
fail without adding a clip. Verify Python and IPC with identical mapping payloads,
undo/redo and save/reopen of the baked target clip. The C++ regression fixture
covers these core cases but has not been compiled/run by Codex. Mapping presets
are not persisted yet; copied target clip channels are the saved result.
The current hierarchy editor will move into the contextual animation UI later.

Manual mapping panel build/visibility is user-confirmed; actual manual mapping
and playback tests were deferred by the user for later IPC verification.

Visual mapping and synchronized sample service are now delivered in source;
see RIG_MAPPING_UI_BUILD_CHECKS.md. Rest-basis hierarchy workflow is user-confirmed.
