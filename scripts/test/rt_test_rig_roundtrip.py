"""Embedded production save/reopen test; user runs on a disposable project.

Supply a new .rtp output path. Reopening resets the current scene/undo history;
the output becomes the active project. Mapping options are supplied explicitly
because mapping presets/editor state are not persisted yet.
"""
import json
import math
from pathlib import Path


def run(output_path, source_character, source_clip, target_character,
        node_map=None, mode="rest_basis", translation_scale=1.0):
    import rt
    output = Path(output_path).resolve()
    assert output.suffix.lower() == ".rtp" and not output.exists()
    options = dict(node_map={} if node_map is None else node_map,
                   mode=mode, translation_scale=translation_scale)
    initial = rt.anim.sample_clip_binding(source_character, source_clip, target_character, **options)
    assert initial["binding"]["ready"], initial
    time = initial["duration_seconds"] * 0.25
    before = rt.anim.sample_clip_binding(source_character, source_clip, target_character,
                                        time_seconds=time, **options)
    output.parent.mkdir(parents=True, exist_ok=True)
    rt.project.save(str(output))
    saved = json.loads(output.read_text(encoding="utf-8"))
    hierarchies = {c["importName"]: c["nodeHierarchy"] for c in saved["importedModelContexts"]}
    assert all(hierarchies[c]["version"] == 1 and hierarchies[c]["nodes"]
               for c in (source_character, target_character))
    rt.project.open(str(output))
    after = rt.anim.sample_clip_binding(source_character, source_clip, target_character,
                                       time_seconds=time, **options)
    assert after["binding"]["ready"] and after["binding"]["matches"] == before["binding"]["matches"]
    for side in ("source", "target"):
        assert len(before[side]) == len(after[side])
        for a, b in zip(before[side], after[side]):
            assert a["name"] == b["name"] and a["parent"] == b["parent"]
            assert all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5)
                       for x, y in zip(a["world_transform"], b["world_transform"]))
    rt.project.save(str(output))
    reopened = json.loads(output.read_text(encoding="utf-8"))
    assert {c["importName"]: c["nodeHierarchy"] for c in reopened["importedModelContexts"]} == hierarchies
    assert reopened["animationDataList"] == saved["animationDataList"], "Canonical clip/key data changed"
    assert reopened["boneData"] == saved["boneData"], "Canonical bone data changed"
    print("PASS: hierarchy/authored names, pose preview, bone data and all baked clips survive reopen")


def run_binding(output_path, character):
    """Save/reopen an already bound disposable scene to a NEW .rtp path."""
    import rt
    output = Path(output_path).resolve()
    assert output.suffix.lower() == ".rtp" and not output.exists()
    before = rt.rig.get_binding(character)
    assert before["bound"] and all(part["present"] for part in before["parts"])
    names = [part["mesh"] for part in before["parts"]]
    stats = {name: rt.rig.weight_stats(name) for name in names}
    rows = {name: rt.rig.get_weights(name, 0) for name in names}
    bones = rt.rig.list_bones(character)
    output.parent.mkdir(parents=True, exist_ok=True)
    rt.project.save(str(output))
    saved = json.loads(output.read_text(encoding="utf-8"))
    model = next(c for c in saved["importedModelContexts"] if c["importName"] == character)
    assert model["rigBoundMeshes"] == names
    rt.project.open(str(output))
    assert rt.rig.get_binding(character) == before
    for name in names:
        assert rt.rig.weight_stats(name) == stats[name] and stats[name]["weights_valid"]
        assert rt.rig.get_weights(name, 0) == rows[name]
    after = rt.rig.list_bones(character)
    for a, b in zip(bones, after):
        assert a["name"] == b["name"] and a["bone_index"] == b["bone_index"]
        assert all(b[key] for key in ("in_bonedata", "in_skeleton_nodes", "in_node_hierarchy", "in_ozz_skeleton"))
        assert all(math.isclose(x, y, rel_tol=1e-5, abs_tol=1e-5) for x, y in zip(a["world_transform"], b["world_transform"]))
    rt.project.save(str(output))
    assert json.loads(output.read_text(encoding="utf-8"))["boneData"] == saved["boneData"]
    print("PASS: bound part registry, owned skeleton, influences and ownership survive native reopen")


def run_rest(output_path, character):
    """Actual native save/reopen after batch rest editing; disposable fixture."""
    import rt
    output = Path(output_path).resolve()
    assert output.suffix.lower() == ".rtp" and not output.exists()
    before = rt.rig.list_bones(character)
    assert before and all(b["authoring_owned"] and not b["weighted"] for b in before)
    output.parent.mkdir(parents=True, exist_ok=True)
    rt.project.save(str(output))
    saved = json.loads(output.read_text(encoding="utf-8"))
    model = next(c for c in saved["importedModelContexts"] if c["importName"] == character)
    rt.project.open(str(output))
    assert rt.rig.get_selection()["bones"] == []
    after = rt.rig.list_bones(character)
    assert len(after) == len(before)
    for a, b in zip(before, after):
        assert a["name"] == b["name"] and a["parent"] == b["parent"] and a["bone_index"] == b["bone_index"]
        assert all(b[k] for k in ("in_bonedata", "in_skeleton_nodes", "in_node_hierarchy", "in_ozz_skeleton"))
        for key in ("world_transform", "local_rest_transform", "scene_transform"):
            assert all(math.isclose(x, y, abs_tol=1e-5, rel_tol=1e-5) for x, y in zip(a[key], b[key]))
    rt.project.save(str(output))
    reopened = json.loads(output.read_text(encoding="utf-8"))
    assert reopened["boneData"] == saved["boneData"]
    assert next(c for c in reopened["importedModelContexts"] if c["importName"] == character)["nodeHierarchy"] == model["nodeHierarchy"]
    print("PASS: batch rest hierarchy, stable indices, actor placement and four representations survive native reopen")


def run_pose(output_path, character):
    """Save/reopen authored bound rig clips to a new .rtp; explicitly replaces scene."""
    import rt
    output = Path(output_path).resolve()
    assert output.suffix.lower() == ".rtp" and not output.exists()
    rt.rig.set_mode("pose", character)
    state = rt.rig.get_pose_state(character)
    assert state["clips"] and state["clip"]
    clip = state["clip"]
    frame = state["frame"]
    before = state["local_transforms"]
    coverage = rt.rig.get_pose_coverage(character)
    joint_profile = rt.rig.get_joint_profile(character)
    rt.rig.set_mode("scene")
    output.parent.mkdir(parents=True, exist_ok=True)
    rt.project.save(str(output))
    saved = json.loads(output.read_text(encoding="utf-8"))
    authored = [c for c in saved["animationDataList"] if c.get("rigAuthoring") and c["modelName"] == character]
    assert authored and any(c["name"] == clip for c in authored)
    rt.project.open(str(output))
    rt.rig.set_mode("pose", character)
    rt.rig.select_pose_clip(character, clip)
    rt.rig.set_pose_frame(frame)
    after = rt.rig.get_pose_state(character)
    assert set(before) == set(after["local_transforms"])
    for bone, values in before.items():
        assert all(math.isclose(a,b,abs_tol=1e-5,rel_tol=1e-5) for a,b in zip(values,after["local_transforms"][bone]))
    assert rt.rig.get_pose_coverage(character) == coverage
    assert rt.rig.get_joint_profile(character) == joint_profile
    rt.rig.set_mode("scene")
    rt.project.save(str(output))
    reopened = json.loads(output.read_text(encoding="utf-8"))
    assert reopened["animationDataList"] == saved["animationDataList"]
    assert reopened["boneData"] == saved["boneData"]
    model = lambda project: next(c for c in project["importedModelContexts"] if c["importName"] == character)
    assert model(reopened)["rigAnatomy"] == model(saved)["rigAnatomy"]
    print("PASS: native authored bone keys, evaluated FK and flat weights survive reopen")
