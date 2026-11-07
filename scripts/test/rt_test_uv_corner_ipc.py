"""Run externally after the user's build, with Studio open. Never launches Studio.

Changes scene history using temporary planes. Includes an in-process Python API check
via script.run_file; the child script never opens IPC from inside the application.
"""
from pathlib import Path
import uuid

from rt_ipc import RtIpc, RtIpcError


def check_atlas_split(call, shape):
    name = call("scene.add_primitive", type=shape,
                name="UV_Atlas_" + shape + "_" + uuid.uuid4().hex[:10], size=1.0)
    try:
        target = dict(object=name, mesh=0)
        before = call("uv.layout", **target, limit=10000)
        result = call("uv.unwrap", **target, revision=before["revision"],
                      resolution=256, padding=2, allow_texture_change=True)
        after = call("uv.layout", **target, limit=10000)
        assert result["vertex_splits"] > 0 and result["topology_changed"]
        assert after["vertices"] > before["vertices"]
        assert after["triangles"] == before["triangles"]
        if shape == "sphere":
            assert result["atlas_preserved_faces"] > 0
            assert after["faces"][0]["uv"] == before["faces"][0]["uv"]
            assert after["faces"][-1]["uv"] == before["faces"][-1]["uv"]
        else:
            assert result["atlas_preserved_faces"] == 0
        call("undo")
        assert call("uv.info", **target)["revision"] == before["revision"]
        call("redo")
        assert call("uv.info", **target)["revision"] == after["revision"]
        print(f"PASS: {shape} atlas vertex split, preserved faces and undo/redo")
    finally:
        call("scene.delete", name=name)


def main():
    client = RtIpc()
    call = client.call
    name = "UV_Corner_IPC_" + uuid.uuid4().hex[:10]
    created = False
    try:
        caps = call("uv.capabilities")
        assert caps["corner_remap"] and caps["set_corners"], "Build the corner remap sources first"
        assert caps.get("vertex_selection"), "Build the UV vertex selection sources first"
        name = call("scene.add_primitive", type="plane", name=name, size=1.0)
        created = True
        target = dict(object=name, mesh=0)
        before = call("uv.layout", **target, limit=10)
        assert before["vertices"] == 4 and before["triangles"] == 2
        corners = [uv[:] for face in before["faces"] for uv in face["uv"]]
        corners[3] = [corners[3][0] + 0.125, corners[3][1] + 0.25]
        edited = call("uv.set_corners", **target, revision=before["revision"],
                      corners=corners, allow_texture_change=True)
        assert edited["changed"] and edited["topology_changed"]
        assert edited["vertices"] == 5 and edited["vertex_splits"] == 1
        after = call("uv.layout", **target, limit=10)
        assert after["triangles"] == 2
        assert [uv for face in after["faces"] for uv in face["uv"]] == corners
        assert before["faces"][0]["vertex_indices"] == after["faces"][0]["vertex_indices"]
        try:
            call("uv.set_corners", **target, revision=before["revision"],
                 corners=corners, allow_texture_change=True)
        except RtIpcError as error:
            assert "uv_stale_revision" in str(error)
        else:
            raise AssertionError("Stale corner write accepted")
        call("undo")
        assert call("uv.layout", **target, limit=10) == before
        call("redo")
        assert call("uv.layout", **target, limit=10) == after
        selected = after["faces"][0]["vertex_indices"][:2]
        moved = call("uv.transform", **target, revision=after["revision"],
                     vertices=selected, offset=[0.25, -0.5], allow_texture_change=True)
        assert moved["changed"] and not moved["topology_changed"]
        moved_layout = call("uv.layout", **target, limit=10)
        for old_face, new_face in zip(after["faces"], moved_layout["faces"]):
            for vertex, old_uv, new_uv in zip(old_face["vertex_indices"],
                                             old_face["uv"], new_face["uv"]):
                expected = [old_uv[0] + 0.25, old_uv[1] - 0.5] if vertex in selected else old_uv
                assert new_uv == expected
        for invalid in ([], [-1], [99999], [0.5], [True]):
            try:
                call("uv.transform", **target, revision=moved_layout["revision"],
                     vertices=invalid, offset=[1, 1], allow_texture_change=True)
            except RtIpcError as error:
                assert "uv_invalid_" in str(error)
            else:
                raise AssertionError("Invalid vertex selection accepted")
            assert call("uv.info", **target)["revision"] == moved_layout["revision"]
        try:
            call("uv.transform", **target, revision=moved_layout["revision"],
                 vertices=selected, islands=[0], offset=[1, 1], allow_texture_change=True)
        except RtIpcError as error:
            assert "uv_invalid_selection" in str(error)
        else:
            raise AssertionError("Mixed island/vertex selection accepted")
        call("undo")
        assert call("uv.info", **target)["revision"] == after["revision"]
        call("redo")
        assert call("uv.info", **target)["revision"] == moved_layout["revision"]
        child = Path(__file__).with_name("rt_test_uv_corner_python.py").resolve()
        call("script.run_file", path=str(child))
        check_atlas_split(call, "torus")
        check_atlas_split(call, "sphere")
        print("PASS: external corner IPC and embedded Python API remap/undo tests")
    finally:
        try:
            if created:
                call("scene.delete", name=name)
        finally:
            client.close()


if __name__ == "__main__":
    main()
