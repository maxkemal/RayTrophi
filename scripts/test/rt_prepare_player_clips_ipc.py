"""Prepare a separate two-clip project in an already open Studio via external IPC.

Replaces Studio's current scene; save any work before invoking. Never launches
an app, builds C++, overwrites the input project, or uses Player for authoring.
"""
import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--fbx", type=Path, required=True)
    parser.add_argument("--character", default="1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, fbx, output = (p.resolve() for p in
                            (args.project, args.fbx, args.output))
    assert project.is_file() and fbx.is_file(), "input project/FBX missing"
    assert output.suffix.lower() == ".rtp", "output must be .rtp"
    assert not output.exists(), "refusing to overwrite an existing output"
    assert output != project, "output must be a separate project"
    output.parent.mkdir(parents=True, exist_ok=True)
    client = RtIpc()
    try:
        client.call("project.open", path=str(project))
        before = client.call("anim.clips", character=args.character)
        characters = {c["name"] for c in client.call("anim.characters")}
        client.call("scene.import_model", path=str(fbx))
        sources = [c for c in client.call("anim.characters")
                   if c["name"] not in characters]
        assert len(sources) == 1, f"expected one imported clip character: {sources}"
        source = sources[0]["name"]
        clips = client.call("anim.clips", character=source)
        assert len(clips) == 1, f"expected one FBX clip: {clips}"
        params = dict(source_character=source, source_clip=clips[0]["name"],
                      target_character=args.character, mode="same_rig")
        preview = client.call("anim.preview_clip_binding", **params)
        assert preview["ready"], preview
        assert client.call("anim.clips", character=args.character) == before
        result = client.call("anim.bind_clip", **params)
        assert result["ready"], result
        after = client.call("anim.clips", character=args.character)
        assert len(after) == len(before) + 1, after
        # The original import is no longer needed after the cloned target clip.
        client.call("anim.delete_character", character=source)
        assert any(c["name"] == result["output_clip"] for c in
                   client.call("anim.clips", character=args.character))
        client.call("project.save", path=str(output))
        client.call("project.open", path=str(output))
        reopened = client.call("anim.clips", character=args.character)
        assert {c["name"] for c in reopened} == {c["name"] for c in after}
        print(json.dumps(dict(status="PASS", project=str(output),
                              character=args.character, clips=reopened), indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
