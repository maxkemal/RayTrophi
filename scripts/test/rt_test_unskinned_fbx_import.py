"""Run inside RayTrophi Python on a disposable scene after the user builds.

Uses the production import/save/open API, exercising the reader's final
acceptance gate and persistence, beyond the CPU joint collector unit test.
The supplied output path becomes the active project's path.
"""
import json
from pathlib import Path


def verify_snapshot(path, prefix=None):
    scene = json.loads(Path(path).read_text(encoding="utf-8"))
    bones = scene["boneData"]
    indices = bones["boneNameToIndex"]
    if prefix is None:
        candidates = []
        for ctx in scene["importedModelContexts"]:
            candidate = ctx["importName"]
            if all(candidate + "_" + joint in indices for joint in ("Hip", "Knee", "Ankle")):
                candidates.append(candidate)
        assert len(candidates) == 1, "Use a scene with exactly one three-joint fixture import"
        prefix = candidates[0]
    names = {role: prefix + "_" + role for role in ("Armature", "Hip", "Knee", "Ankle")}
    assert all(names[role] in indices for role in ("Hip", "Knee", "Ankle")), "Missing explicit joints"
    assert len({indices[names[role]] for role in ("Hip", "Knee", "Ankle")}) == 3
    assert names["Armature"] in bones["boneDefaultTransforms"], "Missing ancestor transform"
    parents = bones["boneParents"]
    assert parents[names["Hip"]] == names["Armature"]
    assert parents[names["Knee"]] == names["Hip"]
    assert parents[names["Ankle"]] == names["Knee"]
    assert not any(name.startswith(prefix + "_") for name in bones["weightedBoneNames"])
    contexts = [ctx for ctx in scene["importedModelContexts"] if ctx["importName"] == prefix]
    assert len(contexts) == 1 and not contexts[0]["hasAnimation"]
    hierarchy = contexts[0]["nodeHierarchy"]
    assert hierarchy["version"] == 1
    node_keys = {node["uniqueName"] for node in hierarchy["nodes"]}
    assert set(names.values()) <= node_keys, "Missing persisted hierarchy joints/ancestor"
    return prefix, {name: indices[name] for name in names.values() if name in indices}


def run(output_path):
    import rt

    fixture = Path(__file__).resolve().parent / "fixtures/rig_authoring/unskinned_three_joint.fbx"
    output = Path(output_path).resolve()
    assert output.suffix.lower() == ".rtp", "Supply a disposable .rtp output path"
    assert not output.exists(), "Supply a new path so the test does not overwrite a project"
    output.parent.mkdir(parents=True, exist_ok=True)
    clips_before = rt.anim.source_clips()
    rt.scene.import_model(str(fixture))
    assert rt.anim.source_clips() == clips_before, "Skeleton-only fixture must not add clips"
    rt.project.save(str(output))
    prefix, original = verify_snapshot(output)
    rt.project.open(str(output))
    rt.project.save(str(output))
    restored_prefix, restored = verify_snapshot(output, prefix)
    assert restored_prefix == prefix and restored == original, "Joint indices changed on round-trip"
    print("PASS: meshless FBX import, explicit joints, ancestor hierarchy, save/reopen")
