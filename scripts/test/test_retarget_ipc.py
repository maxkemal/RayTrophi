"""External production contract. User starts app; this test adds a target clip."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import probe_import_export_parity as ipc
from rig_mapping_preview_contract import check_sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--clip", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--node-map", default="{}", help="JSON source uniqueName -> target uniqueName")
    parser.add_argument("--translation-scale", type=float, default=1.0)
    args = parser.parse_args()
    params = dict(source_character=args.source, source_clip=args.clip,
                  target_character=args.target, node_map=json.loads(args.node_map),
                  mode="rest_basis", translation_scale=args.translation_scale)
    pipe = ipc.connect()
    try:
        def call(method, values):
            response = ipc.call(pipe, method, values)
            if "error" in response:
                raise RuntimeError(response["error"])
            result = response["result"]
            if isinstance(result, dict) and "__error" in result:
                raise RuntimeError(result)
            return result
        before = call("anim.clips", {"character": args.target})
        source = call("anim.source_channels", {"clip": args.clip})
        check_sample(call, params)
        preview = call("anim.preview_clip_binding", params)
        assert preview["ready"], preview
        assert call("anim.clips", {"character": args.target}) == before
        bound = call("anim.bind_clip", params)
        assert bound["ready"] and bound["mode"] == "rest_basis"
        assert bound["translation_scale"] == args.translation_scale
        assert bound["matches"] == preview["matches"]
        after = call("anim.clips", {"character": args.target})
        assert len(after) == len(before) + 1
        assert any(c["name"] == bound["output_clip"] for c in after)
        assert call("anim.source_channels", {"clip": args.clip}) == source
        channels = call("anim.source_channels", {"clip": bound["output_clip"]})
        assert {c["node_name"] for c in channels} <= {m["target"] for m in bound["matches"]}
        print("PASS: read-only preflight, target registration, source preservation", bound["output_clip"])
    finally:
        ipc._kernel32.CloseHandle(pipe)


if __name__ == "__main__":
    main()
