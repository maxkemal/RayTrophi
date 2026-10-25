"""External-terminal source-geometry acceptance after the user's build.

Run on a paused disposable scene with two existing liquid-phase flow sources
and at least one scene mesh. Restores both geometry settings in finally.
Does not start the application, advance simulation, build, or save the scene.
"""

import argparse
import time

from rt_ipc import RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sphere-source", required=True)
    parser.add_argument("--mesh-source", required=True)
    parser.add_argument("--objects", nargs="+", required=True)
    args = parser.parse_args()
    assert args.sphere_source != args.mesh_source
    ipc = RtIpc()
    names = [args.sphere_source, args.mesh_source]
    before = {name: ipc.call("flow_source.get", name=name)["splat_geometry"] for name in names}

    def get(name):
        return ipc.call("flow_source.get", name=name)["splat_geometry"]

    generated_objects = []
    try:
        ipc.call("flow_source.set_splat_geometry", name=names[0], mode="sphere")
        assert get(names[0])["mode"] == "sphere"
        assert get(names[1]) == before[names[1]], "Sphere edit leaked into other emitter"
        ipc.call("flow_source.set_splat_geometry", name=names[1], mode="mesh_grains",
                 objects=args.objects, near_distance=2.5, mesh_budget=7)
        mesh = get(names[1])
        assert mesh["mode"] == "mesh_grains" and mesh["objects"] == args.objects
        assert mesh["mesh_budget"] == 7 and mesh["near_distance"] == 2.5
        assert get(names[0])["mode"] == "sphere"
        ipc.call("flow_source.set_splat_geometry", name=names[1], mesh_budget=0)
        zero = get(names[1])
        assert zero["objects"] == mesh["objects"] and zero["mesh_budget"] == 0
        ipc.call("undo")
        assert get(names[1]) == mesh
        ipc.call("redo")
        assert get(names[1]) == zero

        for invalid in ({"mode": "unknown"}, {"near_distance": 0},
                        {"near_distance": -1}, {"mesh_budget": -1},
                        {"mesh_budget": True}, {"mesh_budget": 4294967296},
                        {"objects": []}, {"objects": ["__MISSING_EXEMPLAR__"]},
                        {"objects": [args.objects[0], args.objects[0]]},
                        {"mode": "scene_object", "objects": args.objects[:1] * 2},
                        {"unknown_field": 1}):
            try:
                ipc.call("flow_source.set_splat_geometry", name=names[1], **invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Accepted invalid geometry: %r" % invalid)
            assert get(names[1]) == zero, "Rejected geometry changed live authoring"
            assert get(names[0])["mode"] == "sphere", "Invalid request changed other emitter"

        ipc.call("flow_source.set_splat_geometry", name=names[1], mode="scene_object",
                 objects=args.objects[:1])
        assert get(names[1])["mode"] == "scene_object"
        assert get(names[0])["mode"] == "sphere"
        print("PASS: legacy source authoring, validation and undo/redo")
    finally:
        try:
            for name in reversed(names):
                ipc.call("flow_source.set_splat_geometry", name=name, **before[name])
            for node in generated_objects:
                if ipc.call("scene.object_exists", name=node):
                    ipc.call("scene.delete", name=node)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
