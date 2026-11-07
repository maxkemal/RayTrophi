"""External IPC regression. Run after the user's build with Studio open and paused.

Run once in Solid/Material Preview and once in Rendered mode. Never starts Studio.
Creates one temporary plane, changes global perf counters and scene history.
"""
import time
import uuid

from rt_ipc import RtIpc


def main():
    client = RtIpc()
    call = client.call
    name = None
    forbidden = (
        "accel.cpu.embree_build", "accel.vulkan_rt.rebuild",
        "accel.vulkan_rt.update_geometry", "accel.vulkan_rt.raster_geometry",
        "accel.vulkan_solid.raster_geometry", "accel.vulkan_rt.mesh_refit.blas",
        "accel.vulkan_rt.mesh_refit.tlas", "uv.attribute_rebuild_fallback",
    )
    try:
        assert call("uv.capabilities").get("attribute_updates"), "Build UV attribute updates first"
        name = call("scene.add_primitive", type="plane",
                    name="UV_Attribute_" + uuid.uuid4().hex[:10], size=1.0)
        target = dict(object=name, mesh=0)
        # Exclude the fixture's legitimate initial upload from the edit measurement.
        time.sleep(1.0)
        before = call("uv.layout", **target, limit=10)
        call("perf.reset")
        result = call("uv.transform", **target, revision=before["revision"],
                      vertices=before["faces"][0]["vertex_indices"][:2],
                      offset=[0.125, -0.25], allow_texture_change=True)
        assert result["changed"] and not result["topology_changed"]
        after = call("uv.layout", **target, limit=10)
        time.sleep(0.25)
        call("undo")
        assert call("uv.layout", **target, limit=10) == before
        time.sleep(0.25)
        call("redo")
        assert call("uv.layout", **target, limit=10) == after
        time.sleep(0.25)
        rows = {row["name"]: row for row in call("perf.list")}
        assert rows.get("uv.attribute_publish", {}).get("count", 0) == 3, rows
        for section in forbidden:
            assert rows.get(section, {}).get("count", 0) == 0, (section, rows.get(section))
        uploads = {key: value for key, value in rows.items() if key.startswith("uv.")}
        print("PASS: UV-only transform/undo/redo without AS/raster rebuild", uploads)
    finally:
        try:
            if name is not None:
                call("scene.delete", name=name)
        finally:
            client.close()


if __name__ == "__main__":
    main()
