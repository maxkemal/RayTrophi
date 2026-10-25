"""External-terminal acceptance after the user build, on a paused disposable scene.
Checks per-substance output geometry and independent procedural generation.
"""
import argparse
import time
from rt_ipc import RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--substance", default="Sand")
    parser.add_argument("--other-substance", default="Water")
    args = parser.parse_args()
    ipc = RtIpc()
    created = []

    def get(substance):
        return ipc.call("fluid.get_substance_splat_geometry", domain=args.domain,
                        substance=substance)

    def set_geometry(**fields):
        return ipc.call("fluid.set_substance_splat_geometry", domain=args.domain,
                        substance=args.substance, **fields)

    before = get(args.substance)
    other = get(args.other_substance)
    try:
        result = ipc.call("scene.generate_grains", name="OutputGeometry_%s" % time.time_ns(),
                          type="rock", count=2, origin=[0, 60, 0])
        created.extend(result["objects"])
        assert get(args.substance) == before, "Generator unexpectedly assigned output geometry"
        assert get(args.other_substance) == other
        set_geometry(mode="scene_object", objects=created)
        assigned = get(args.substance)
        assert assigned["objects"] == created and assigned["mode"] == "scene_object"
        assert get(args.other_substance) == other
        ipc.call("undo")
        assert get(args.substance) == before
        ipc.call("redo")
        assert get(args.substance) == assigned
        for fields in ({"objects": []}, {"objects": [created[0], created[0]]},
                       {"objects": ["__MISSING__"]}, {"mode": "domain"},
                       {"mesh_budget": -1}, {"near_distance": 0}):
            try:
                set_geometry(**fields)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Accepted invalid geometry: %r" % fields)
            assert get(args.substance) == assigned
        set_geometry(mode="mesh_grains", near_distance=3.0, mesh_budget=7)
        limited = get(args.substance)
        assert limited["objects"] == created and limited["near_distance"] == 3.0
        assert limited["mesh_budget"] == 7
        assert get(args.other_substance) == other
        print("PASS: independent generator, multi-object substance output, undo/redo, validation")
    finally:
        try:
            set_geometry(**before)
            for node in created:
                if ipc.call("scene.object_exists", name=node):
                    ipc.call("scene.delete", name=node)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
