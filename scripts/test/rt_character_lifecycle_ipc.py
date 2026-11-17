"""Live, external Studio acceptance after the user builds.

Run while nobody edits the scene. Saves a fresh backup, exercises deletion,
save-time mesh compaction, undo/redo and reopen, then restores the backup.
The output directory must be new. Does not launch or build either application.
"""
import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc, RtIpcError


def read_project(path):
    return json.loads(path.read_text(encoding="utf-8"))


def bone_slots(project, character):
    prefix = character + "_"
    return {name: slot for name, slot in project["boneData"]["boneNameToIndex"].items()
            if name.startswith(prefix)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delete", required=True, dest="target")
    parser.add_argument("--keep", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert args.target != args.keep
    args.output.mkdir(parents=True, exist_ok=False)
    client = RtIpc()
    backup = args.output.resolve() / "backup.rtp"
    report = {"status": "FAIL", "checks": []}
    backed_up = False

    def save(name):
        path = args.output.resolve() / (name + ".rtp")
        assert client.call("project.save", path=str(path))
        return path, read_project(path)

    def names():
        return {row["name"] for row in client.call("anim.characters")}

    try:
        original_names = names()
        assert {args.target, args.keep} <= original_names
        assert client.call("project.save", path=str(backup))
        backed_up = True
        original = read_project(backup)
        original_objects = set(client.call("scene.list_objects"))
        keep_slots = bone_slots(original, args.keep)
        target_slots = bone_slots(original, args.target)
        assert keep_slots and target_slots
        target_clips = {clip["name"] for clip in original["animationDataList"]
                        if clip.get("modelName") == args.target}

        def verify_deleted(project):
            assert names() == original_names - {args.target}
            assert bone_slots(project, args.keep) == keep_slots
            assert not bone_slots(project, args.target)
            assert not any(ctx["importName"] == args.target
                           for ctx in project["importedModelContexts"])
            assert not any(clip["name"] in target_clips
                           for clip in project["animationDataList"])
            for key in ("boneDefaultTransforms", "boneOffsetMatrices", "boneParents"):
                assert not target_slots.keys() & project["boneData"][key].keys()
            sets = project.get("kinematic_colliders", {}).get("sets", [])
            assert not any(row["target_character"] == args.target for row in sets)
            objects = set(client.call("scene.list_objects"))
            expected = {name for name in original_objects
                        if not name.startswith(args.target + "_")}
            assert objects == expected, (objects, expected)

        for params in ({}, {"character": True}, {"character": ""},
                       {"character": "__missing_character__"}):
            try:
                client.call("anim.delete_character", **params)
            except RtIpcError:
                pass
            else:
                raise AssertionError("invalid deletion accepted: " + repr(params))
            assert names() == original_names
        report["checks"].append("invalid requests preserve scene")

        client.call("anim.delete_character", character=args.target)
        deleted_path, deleted = save("deleted")
        verify_deleted(deleted)
        report["checks"].append("delete, compact, preserve other character bone slots")
        client.call("undo")
        _, restored = save("restored")
        assert names() == original_names
        assert bone_slots(restored, args.target) == target_slots
        assert bone_slots(restored, args.keep) == keep_slots
        assert set(client.call("scene.list_objects")) == original_objects
        report["checks"].append("undo after save-time compaction")
        client.call("redo")
        _, redone = save("redone")
        verify_deleted(redone)
        report["checks"].append("redo")
        client.call("project.open", path=str(deleted_path))
        verify_deleted(read_project(deleted_path))
        report["checks"].append("save/reopen does not resurrect character or clips")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        try:
            if backed_up:
                client.call("project.open", path=str(backup))
                report["backup_restored"] = True
        finally:
            client.close()
            (args.output / "report.json").write_text(
                json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
