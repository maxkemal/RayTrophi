#!/usr/bin/env python3
"""
Probe: two Asset Library foliage assets must not share a decoded texture.

WHAT IT MEASURES
    Loads two different library models into one scatter group, then asks the
    engine which texture each of their materials actually holds. Two conditions:

      * every texture identity belongs to exactly ONE asset. A texture reported
        by materials of BOTH assets means the two imports collided on a cache
        key and one tree is wearing the other tree's pixels.
      * each asset reports textures AT ALL. A material only lists a slot once a
        loaded texture was bound to it, so an empty result means the maps never
        decoded and the material fell back to its bare default.

    Run it TWICE in one process — once on a fresh scene and again after New
    Project — because the second load is a different code path from the first.

WHY IT IS NEEDED
    The collision is invisible from every other angle. Nothing fails, nothing is
    logged, both trees render — they just render with each other's maps, and the
    multi-material one (a pine) is the one you notice. It is also load-order and
    heap-state dependent: import one asset and it is clean; import an unrelated
    mesh first and it can go clean again. That is not something a human retests.

    Backends are irrelevant here — the damage is in the CPU-side import, so
    Vulkan, OptiX and CPU all show it. Do not go looking in a backend.

USAGE
    1. .\scripts\ipc\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/probe_foliage_asset_material_identity.py

    Optionally pass two asset relative paths to test a specific pair:
    python scripts/probe_foliage_asset_material_identity.py <pathA> <pathB>

EXIT CODE
    0 = PASS, 1 = FAIL (shared texture found), 2 = could not run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\.\pipe\RayTrophiStudio'
GROUP_NAME = "__probe_foliage_material_identity"

# Two assets that between them reproduced the original report: a many-material
# conifer plus a many-material broadleaf. Overridable on the command line.
DEFAULT_A = "vegetation/trees/Coniferous/Pinus ponderosa.glb"
DEFAULT_B = "vegetation/trees/Broadleaf Trees/Bauhinia_blakeana.glb"

_kernel32 = ctypes.windll.kernel32


def connect():
    handle = _kernel32.CreateFileW(PIPE_NAME, 0xC0000000, 0, None, 3, 0, None)
    if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
        print("FAIL(setup): RayTrophi Studio is not running "
              "(no \\.\pipe\RayTrophiStudio). Start it first.")
        sys.exit(2)
    return handle


def call(pipe, method, params=None, request_id=[0]):
    request_id[0] += 1
    msg = {"id": request_id[0], "method": method}
    if params:
        msg["params"] = params
    data = json.dumps(msg).encode("utf-8")
    written = wintypes.DWORD(0)
    if not _kernel32.WriteFile(pipe, data, len(data), ctypes.byref(written), None):
        raise OSError("WriteFile failed (%d)" % _kernel32.GetLastError())
    chunks = []
    while True:
        buf = ctypes.create_string_buffer(65536)
        read = wintypes.DWORD(0)
        ok = _kernel32.ReadFile(pipe, buf, 65536, ctypes.byref(read), None)
        chunks.append(buf.raw[:read.value])
        if ok:
            break
        if _kernel32.GetLastError() != 234:  # ERROR_MORE_DATA
            raise OSError("ReadFile failed (%d)" % _kernel32.GetLastError())
    return json.loads(b"".join(chunks).decode("utf-8"))


def result_of(response, what):
    if not isinstance(response, dict) or "result" not in response:
        print("FAIL(setup): %s -> %s" % (what, response))
        sys.exit(2)
    return response["result"]


def material_names(pipe):
    return {m["name"] for m in result_of(call(pipe, "material.list"), "material.list")}


def textures_of(pipe, material):
    bindings = result_of(call(pipe, "material.textures",
                              {"material_name": material}), "material.textures")
    # A pre-fix build returns bare slot-name strings and cannot answer this
    # question at all. Say so instead of silently reporting "no sharing".
    for entry in bindings:
        if not isinstance(entry, dict) or "texture" not in entry:
            print("FAIL(setup): this build's material.textures reports slots without "
                  "texture identity, so the measurement cannot be made. Rebuild.")
            sys.exit(2)
    return {e["texture"] for e in bindings if e["texture"]}


def main():
    asset_a = sys.argv[1] if len(sys.argv) > 2 else DEFAULT_A
    asset_b = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_B

    pipe = connect()
    try:
        call(pipe, "scatter.delete_group", {"group": GROUP_NAME})
        result_of(call(pipe, "scatter.create_group", {"name": GROUP_NAME}),
                  "scatter.create_group")

        before = material_names(pipe)
        result_of(call(pipe, "scatter.add_library_source",
                       {"group": GROUP_NAME, "relative_path": asset_a}),
                  "add_library_source(A)")
        after_a = material_names(pipe)
        result_of(call(pipe, "scatter.add_library_source",
                       {"group": GROUP_NAME, "relative_path": asset_b}),
                  "add_library_source(B)")
        after_b = material_names(pipe)

        mats_a = sorted(after_a - before)
        mats_b = sorted(after_b - after_a)
        if not mats_a or not mats_b:
            # Both assets already loaded this session, so neither registered new
            # materials. An empty comparison is not a pass.
            print("FAIL(setup): no new materials appeared (A=%d, B=%d). These assets "
                  "are already loaded — restart the app for a clean measurement."
                  % (len(mats_a), len(mats_b)))
            sys.exit(2)

        tex_a, tex_b = {}, {}
        for m in mats_a:
            for t in textures_of(pipe, m):
                tex_a.setdefault(t, []).append(m)
        for m in mats_b:
            for t in textures_of(pipe, m):
                tex_b.setdefault(t, []).append(m)

        if not tex_a or not tex_b:
            # Not a setup problem: a material only reports a slot once a LOADED
            # texture was bound to it. Zero textures across a whole asset means
            # the import produced textures that never decoded, and the material
            # fell back to its bare default look. That is the failure mode the
            # pixel-less embedded "cache" used to cause on a repeat load.
            print("FAIL: an asset ended up with no textures at all "
                  "(A=%d, B=%d). Its maps did not decode, so its materials are "
                  "bare — check the embedded-texture path in Texture.h."
                  % (len(tex_a), len(tex_b)))
            sys.exit(1)

        shared = sorted(set(tex_a) & set(tex_b))
        print("asset A: %-55s %2d materials, %2d textures" % (asset_a, len(mats_a), len(tex_a)))
        print("asset B: %-55s %2d materials, %2d textures" % (asset_b, len(mats_b), len(tex_b)))

        if shared:
            print("\nFAIL: %d texture(s) claimed by BOTH assets:" % len(shared))
            for t in shared:
                print("  %s" % t)
                print("     A: %s" % ", ".join(tex_a[t]))
                print("     B: %s" % ", ".join(tex_b[t]))
            print("\nThe two imports collided on a texture cache key. One asset is "
                  "rendering with the other's maps on every backend.")
            sys.exit(1)

        print("\nPASS: no texture is shared between the two assets.")
        sys.exit(0)
    finally:
        try:
            call(pipe, "scatter.delete_group", {"group": GROUP_NAME})
        except Exception:
            pass
        _kernel32.CloseHandle(pipe)


if __name__ == "__main__":
    main()
