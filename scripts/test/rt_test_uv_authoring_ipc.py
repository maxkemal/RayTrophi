"""UV authoring acceptance probe. Run externally after the user's build.

Creates a uniquely named cube and deletes it afterwards. Scene/history become modified.
Never run from the application's embedded interpreter. Does not launch the application.
"""
import math
import uuid

from rt_ipc import RtIpc, RtIpcError


def expect_error(call, method, fragment, **params):
    try:
        call(method, **params)
    except RtIpcError as error:
        assert fragment in str(error), str(error)
    else:
        raise AssertionError(f"{method} should reject {fragment}")


def coordinates(layout):
    return [face["uv"] for face in layout["faces"]]


def main():
    client = RtIpc()
    call = client.call
    name = "UV_Acceptance_" + uuid.uuid4().hex[:10]
    created = False
    restore_target = None
    try:
        caps = call("uv.capabilities")
        assert caps["uv_set"] == 0 and not caps["seam_authoring"]
        assert not caps["corner_remap"] and not caps["texture_preservation"]
        restore_target = next((t for t in call("uv.targets")
                               if not t["object"].startswith(("UV_Acceptance_", "UV_Extended_"))), None)
        call("scene.add_primitive", type="cube", name=name, size=1.0)
        created = True
        assert any(t["object"] == name for t in call("uv.targets"))
        target = dict(object=name, mesh=0)
        before = call("uv.layout", **target, limit=100)
        assert before["triangles"] == 12 and before["next_face"] is None
        page = call("uv.layout", **target, first_face=2, limit=2)
        assert [f["face"] for f in page["faces"]] == [2, 3]
        assert page["next_face"] == 4
        edit = dict(**target, revision=before["revision"], islands=[0], offset=[0.125, 0.25])
        expect_error(call, "uv.transform", "uv_texture_change_ack_required", **edit)
        assert call("uv.info", **target)["revision"] == before["revision"]
        edited = call("uv.transform", **edit, allow_texture_change=True)
        assert edited["changed"]
        after = call("uv.layout", **target, limit=100)
        for old_face, new_face in zip(before["faces"], after["faces"]):
            moved = old_face["island"] == 0
            for old_uv, new_uv in zip(old_face["uv"], new_face["uv"]):
                for axis, delta in enumerate((0.125, 0.25)):
                    expected = old_uv[axis] + delta if moved else old_uv[axis]
                    assert math.isclose(new_uv[axis], expected, abs_tol=1e-6)
        expect_error(call, "uv.transform", "uv_stale_revision", **edit,
                     allow_texture_change=True)
        call("undo")
        assert coordinates(call("uv.layout", **target, limit=100)) == coordinates(before)
        call("redo")
        assert coordinates(call("uv.layout", **target, limit=100)) == coordinates(after)
        info = call("uv.info", **target)
        expect_error(call, "uv.transform", "uv_invalid_island", **target,
                     revision=info["revision"], islands=[999999], allow_texture_change=True)
        expect_error(call, "uv.layout", "uv_invalid_page", **target, limit=10001)
        for operation in ("flip_u", "flip_v", "swap_uv"):
            current = call("uv.info", **target)
            call("uv." + operation, **target, revision=current["revision"],
                 islands=[0], allow_texture_change=True)
            repaired = call("uv.layout", **target, limit=100)
            for old_face, new_face in zip(after["faces"], repaired["faces"]):
                for old_uv, new_uv in zip(old_face["uv"], new_face["uv"]):
                    expected = old_uv
                    if old_face["island"] == 0:
                        if operation == "flip_u":
                            expected = [1 - old_uv[0], old_uv[1]]
                        elif operation == "flip_v":
                            expected = [old_uv[0], 1 - old_uv[1]]
                        else:
                            expected = [old_uv[1], old_uv[0]]
                    assert all(math.isclose(a, b, abs_tol=1e-6)
                               for a, b in zip(new_uv, expected))
            call("undo")
            assert coordinates(call("uv.layout", **target, limit=100)) == coordinates(after)
        info = call("uv.info", **target)
        all_uvs = [uv for f in after["faces"] for uv in f["uv"]]
        low = [min(uv[axis] for uv in all_uvs) for axis in range(2)]
        span = max(max(uv[axis] for uv in all_uvs) - low[axis] for axis in range(2))
        call("uv.normalize", **target, revision=info["revision"], allow_texture_change=True)
        fitted = call("uv.layout", **target, limit=100)
        for old_face, new_face in zip(after["faces"], fitted["faces"]):
            for old_uv, new_uv in zip(old_face["uv"], new_face["uv"]):
                for axis in range(2):
                    assert math.isclose(new_uv[axis], (old_uv[axis] - low[axis]) / span, abs_tol=1e-6)
        call("undo")
        assert coordinates(call("uv.layout", **target, limit=100)) == coordinates(after)
        info = call("uv.info", **target)
        if not caps["xatlas"]:
            expect_error(call, "uv.pack", "uv_backend_unavailable", **target,
                         revision=info["revision"], allow_texture_change=True)
        else:
            try:
                call("uv.pack", **target, revision=info["revision"],
                     resolution=512, padding=4, allow_texture_change=True)
            except RtIpcError as error:
                assert "uv_vertex_split_required" in str(error) or "uv_atlas_remap_required" in str(error)
                assert call("uv.info", **target)["revision"] == info["revision"]
            else:
                packed = call("uv.layout", **target, limit=100)
                assert all(math.isfinite(c) for f in packed["faces"] for uv in f["uv"] for c in uv)
                print("PASS: xatlas pack")
            info = call("uv.info", **target)
            try:
                call("uv.unwrap", **target, revision=info["revision"],
                     resolution=512, padding=4, allow_texture_change=True)
            except RtIpcError as error:
                assert "uv_vertex_split_required" in str(error) or "uv_atlas_remap_required" in str(error)
                assert call("uv.info", **target)["revision"] == info["revision"]
            else:
                unwrapped = call("uv.layout", **target, limit=100)
                assert unwrapped["vertices"] == before["vertices"]
                assert unwrapped["triangles"] == before["triangles"]
                assert all(math.isfinite(c) for f in unwrapped["faces"] for uv in f["uv"] for c in uv)
                print("PASS: xatlas unwrap")
        current = call("uv.info", **target)
        try:
            call("uv.project_view", **target, revision=current["revision"],
                 allow_texture_change=True)
        except RtIpcError as error:
            assert any(code in str(error) for code in
                       ("uv_projection_behind_camera", "uv_projection_no_camera", "uv_projection_degenerate"))
            assert call("uv.info", **target)["revision"] == current["revision"]
        else:
            projected = call("uv.layout", **target, limit=100)
            assert projected["vertices"] == before["vertices"]
            assert all(math.isfinite(v) and -1e-6 <= v <= 1.000001
                       for f in projected["faces"] for uv in f["uv"] for v in uv)
        call("scene.delete", name=name)
        created = False
        assert not any(t["object"] == name for t in call("uv.targets")), "Deleted UV target is still listed"
        expect_error(call, "uv.info", "uv_target_not_found", **target)
        print("PASS: UV isolation, normalization, paging, stale writes, undo/redo and deletion")
    finally:
        try:
            if created:
                call("scene.delete", name=name)
            if restore_target is not None:
                call("uv.open", object=restore_target["object"])
        finally:
            client.close()


if __name__ == "__main__":
    main()
