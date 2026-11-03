"""In-process Python binding probe: call via script.run_file, never use RtIpc here.

Requires one existing flat mesh with UVs. Queries and a verified no-op only; no history edits.
"""
import rt


def main():
    capabilities = rt.uv.capabilities()
    assert capabilities["uv_set"] == 0
    targets = rt.uv.targets()
    target = next(t for t in targets if not t["object"].startswith(
        ("UV_Acceptance_", "UV_Extended_")))
    args = dict(object=target["object"], mesh=target["mesh"])
    before = rt.uv.info(**args)
    page = rt.uv.layout(**args, limit=1)
    assert len(page["faces"]) == 1
    assert page["revision"] == before["revision"]
    result = rt.uv.transform(**args, revision=before["revision"],
                             offset=[0, 0], scale=[1, 1], pivot=[0, 0],
                             allow_texture_change=True)
    assert not result["changed"]
    assert result["revision"] == before["revision"]
    try:
        rt.uv.transform(**args, revision="obsolete", allow_texture_change=True)
    except RuntimeError as error:
        assert "uv_stale_revision" in str(error)
    else:
        raise AssertionError("Python binding accepted a stale write")
    try:
        rt.uv.transform(**args, revision=before["revision"],
                         offset=[float("nan"), 0], allow_texture_change=True)
    except ValueError:
        pass
    else:
        raise AssertionError("Python binding accepted non-finite JSON input")
    assert rt.uv.info(**args)["revision"] == before["revision"]
    print("PASS: rt.uv Python binding queries, no-op, stale write and non-finite input")


if __name__ == "__main__":
    main()
