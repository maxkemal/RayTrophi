"""Run from an external terminal with Studio open, after rebuilding.

Checks shared grain library authoring, atomic rejection, and IPC survival after
wrong JSON types. Restores all touched settings; does not advance the timeline.
Use --render on a paused, populated, splat-routed DEM, MPM or fluid domain.
The active camera must be within Near Distance of at least one particle.
"""

import argparse
import json
import time

from rt_ipc import RtIpc, RtIpcError


def settings(ipc, domain):
    return ipc.call("fluid.get", domain=domain)["grain_mesh_near"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc()
    original = settings(ipc, args.domain)
    required = {"shape", "source_materials"}
    if not required.issubset(original):
        ipc.close()
        raise RuntimeError("Rebuild required: running executable lacks grain library options")
    frame = ipc.call("timeline.get_frame")
    restore = {
        "near_mesh_enabled": original["enabled"],
        "near_mesh_shape": original["shape"],
        "near_mesh_source_materials": original["source_materials"],
    }
    try:
        ipc.call("fluid.set_splat_geometry", domain=args.domain,
                 near_mesh_shape="voronoi", near_mesh_source_materials=True,
                 near_mesh_enabled=args.render)
        current = settings(ipc, args.domain)
        assert current["shape"] == "voronoi"
        assert current["source_materials"] is True

        # Reject the whole patch: its valid field must not be applied either.
        for bad in ("unknown_shape", [[], [], []], 42):
            try:
                ipc.call("fluid.set_splat_geometry", domain=args.domain,
                         near_mesh_shape=bad, near_mesh_source_materials=False)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Invalid shape was accepted: %r" % (bad,))
            after = settings(ipc, args.domain)
            assert after["shape"] == "voronoi"
            assert after["source_materials"] is True

        for shape in ("rock", "voronoi"):
            # Start from a hidden pool so an earlier active split cannot satisfy
            # this shape's asynchronous wait before its library has rebuilt.
            ipc.call("fluid.set_splat_geometry", domain=args.domain, near_mesh_enabled=False)
            if args.render:
                deadline = time.monotonic() + 30
                while settings(ipc, args.domain)["status"] == "active":
                    if time.monotonic() >= deadline:
                        raise RuntimeError("Previous mesh pool did not hide")
                    time.sleep(0.05)
            ipc.call("fluid.set_splat_geometry", domain=args.domain,
                     near_mesh_shape=shape, near_mesh_enabled=args.render)
            current = settings(ipc, args.domain)
            assert current["shape"] == shape
            if args.render:
                deadline = time.monotonic() + 30
                while True:
                    info = ipc.call("fluid.get", domain=args.domain)
                    current = info["grain_mesh_near"]
                    assert ipc.call("timeline.get_frame") == frame, "Pause the timeline first"
                    assert info["primary_visual_children"] == 1
                    if current["status"] == "active" and current["near_instances"] > 0:
                        assert current["exemplars"] > 0
                        assert current["exemplar_triangles"] > 0
                        break
                    if time.monotonic() >= deadline:
                        raise RuntimeError("No visible mesh split: %s" % json.dumps(current))
                    time.sleep(0.05)
                print("PASS: %s visible meshes=%s" % (shape, current["near_instances"]))
        print("PASS: library settings, atomic validation, IPC exception survival")
    finally:
        try:
            ipc.call("fluid.set_splat_geometry", domain=args.domain, **restore)
            restored = settings(ipc, args.domain)
            assert restored["shape"] == original["shape"]
            assert restored["source_materials"] == original["source_materials"]
            assert restored["enabled"] == original["enabled"]
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
