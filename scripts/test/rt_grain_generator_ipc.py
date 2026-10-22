"""External-terminal acceptance for scene.generate_grains after a user build.

Creates tiny temporary libraries, checks flat geometry and one-step undo/redo,
and deletes the created objects. Does not save/open a project or run simulation.
Use --keep to leave the libraries for material and project round-trip inspection.
"""

import argparse
import time

from rt_ipc import RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc()
    created = []
    try:
        for invalid in ({"count": 65}, {"arms": 2}, {"count": True},
                        {"diameter_min": -1}, {"material": "__MISSING_GRAIN_MATERIAL__"},
                        {"type": [[], [], []]}, {"domain": "__MISSING_GRAIN_DOMAIN__"}):
            before = ipc.call("scene.list_objects")
            try:
                ipc.call("scene.generate_grains", **invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Invalid generator input accepted: %r" % invalid)
            assert ipc.call("scene.list_objects") == before, "Rejected call changed scene objects"
        for kind in ("rock", "voronoi", "snow", "crystal"):
            name = "GrainProbe_%s_%s" % (kind, time.time_ns())
            result = ipc.call("scene.generate_grains", type=kind, name=name, count=3,
                              seed=8127, diameter_min=0.02, diameter_max=0.04,
                              size_distribution="log_uniform", origin=[0, 60, 0])
            created.extend(result["objects"])
            assert 0 < result["count"] <= 3
            for row in result["examples"]:
                info = ipc.call("scene.object_info", name=row["object"])
                assert info["meshes"] == 1 and info["triangles"] > 0
                assert info["has_bounds"]
                assert 0.02 <= row["diameter"] <= 0.04
                assert abs(max(info["world_size"]) - row["diameter"]) < 0.0001
                assert abs(info["world_center"][1] - 60) < 0.001
            ipc.call("undo")
            for name in result["objects"]:
                assert not ipc.call("scene.object_exists", name=name)
            ipc.call("redo")
            for name in result["objects"]:
                assert ipc.call("scene.object_exists", name=name)
            print("PASS:", kind, "flat meshes", result["count"], "undo/redo")
        densities = []
        for rounding, detail in ((0.0, 4), (0.6, 4), (0.6, 8)):
            result = ipc.call("scene.generate_grains", type="voronoi", count=1,
                              name="RoundedCell_%s" % time.time_ns(), seed=8127,
                              roundness=rounding, resolution=detail, origin=[0, 60, 0])
            created.extend(result["objects"])
            assert result["count"] == 1
            densities.append(result["examples"][0]["triangles"])
        assert densities[0] < densities[1] < densities[2], densities
        print("PASS: single Voronoi example and real rounded-surface detail", densities)
        print("PASS: atomic validation and IPC survival")
    finally:
        try:
            if not args.keep:
                for name in created:
                    if ipc.call("scene.object_exists", name=name):
                        ipc.call("scene.delete", name=name)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
