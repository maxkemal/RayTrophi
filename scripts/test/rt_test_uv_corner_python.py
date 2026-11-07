"""In-process rt.uv test. Invoked by the external IPC test; never imports RtIpc.

Creates/deletes one temporary plane. All NumPy views are copied before topology edits.
"""
import uuid

import numpy as np
import rt


def main():
    assert rt.uv.capabilities().get("vertex_selection"), "Build the UV vertex sources first"
    name = "UV_Corner_Python_" + uuid.uuid4().hex[:10]
    created = False
    try:
        name = rt.scene.add_primitive("plane", name, 1.0)
        created = True
        args = dict(object=name, mesh=0)
        before = rt.uv.layout(**args, limit=10)
        positions = rt.mesh.positions(name).copy()
        normals = rt.mesh.normals(name).copy()
        assert before["vertices"] == 4 and before["triangles"] == 2
        original_indices = [v for face in before["faces"] for v in face["vertex_indices"]]
        corners = [uv[:] for face in before["faces"] for uv in face["uv"]]
        corners[3] = [corners[3][0] + 0.125, corners[3][1] + 0.25]
        result = rt.uv.set_corners(**args, revision=before["revision"],
                                   corners=corners, allow_texture_change=True)
        assert result["vertices"] == 5 and result["vertex_splits"] == 1
        assert result["topology_changed"]
        after = rt.uv.layout(**args, limit=10)
        new_positions = rt.mesh.positions(name).copy()
        new_normals = rt.mesh.normals(name).copy()
        for face, original in zip(after["faces"], before["faces"]):
            for new_index, old_index in zip(face["vertex_indices"], original["vertex_indices"]):
                assert np.array_equal(new_positions[new_index], positions[old_index])
                assert np.array_equal(new_normals[new_index], normals[old_index])
        assert [uv for face in after["faces"] for uv in face["uv"]] == corners
        try:
            rt.uv.set_corners(**args, revision=after["revision"], corners=corners[:-1],
                              allow_texture_change=True)
        except RuntimeError as error:
            assert "uv_corner_count_mismatch" in str(error)
        else:
            raise AssertionError("Invalid corner count accepted")
        no_op = rt.uv.set_corners(**args, revision=after["revision"], corners=corners,
                                  allow_texture_change=True)
        assert not no_op["changed"] and no_op["vertices"] == 5
        rt.undo()
        restored = rt.uv.layout(**args, limit=10)
        assert restored["revision"] == before["revision"]
        assert [v for face in restored["faces"] for v in face["vertex_indices"]] == original_indices
        assert np.array_equal(rt.mesh.positions(name), positions)
        rt.redo()
        assert rt.uv.info(**args)["revision"] == after["revision"]
        selected = after["faces"][0]["vertex_indices"][:2]
        moved = rt.uv.transform(**args, revision=after["revision"], vertices=selected,
                                offset=[0.25, -0.5], allow_texture_change=True)
        assert moved["changed"] and not moved["topology_changed"]
        transformed = rt.uv.layout(**args, limit=10)
        for old_face, new_face in zip(after["faces"], transformed["faces"]):
            for vertex, old_uv, new_uv in zip(old_face["vertex_indices"],
                                             old_face["uv"], new_face["uv"]):
                expected = [old_uv[0] + 0.25, old_uv[1] - 0.5] if vertex in selected else old_uv
                assert new_uv == expected
        assert np.array_equal(rt.mesh.positions(name), new_positions)
        rt.undo()
        assert rt.uv.info(**args)["revision"] == after["revision"]
        print("PASS: corner Python API, surface/normals, no-op, invalid count and topology undo/redo")
    finally:
        if created:
            rt.scene.delete(name)


if __name__ == "__main__":
    main()
