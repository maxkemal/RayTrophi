#!/usr/bin/env python3
"""
RayTrophi Studio - "terrain-only scene, added object, wrong thing moves" check.

★★★ WHAT THIS SEPARATES

Reported symptom: in a scene that contains only a terrain, add a cube, select
the cube in the hierarchy, drag - and the TERRAIN moves. Two very different
faults produce that one picture:

  A. IDENTITY plumbing. Selecting "Cube" does not actually address the cube:
     world.objects is reordered whenever the terrain re-registers its mesh, and
     every cached SLOT INDEX (mesh_cache, tri_to_index, SelectableItem::
     object_index) is a guess about a moment. SceneSelection::isSelected used to
     treat a matching index as identity, so one object could answer for another.

  B. The UI CLICK / gizmo path only. Identity is fine at the value level and the
     fault is in the hierarchy row -> selection -> gizmo chain, which no script
     can drive.

This file measures A. If every check passes, the fault is B and the next look
belongs in scene_ui_hierarchy.cpp / scene_ui_gizmos.cpp - NOT in the selection
or transform API, which this file has just cleared.

★ Transforms are read back from the ENGINE, not from what we sent: a set that
reports ok while writing somewhere else is exactly the failure being hunted.

Usage:
    1. Start RayTrophi Studio (.\\scripts\\ipc\\Start-RayTrophi.ps1).
    2. python scripts/terrain_scene_selection_check.py
"""
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ipc_test_client import send_command, PIPE_NAME  # noqa: E402

TERRAIN = "SelCheck"
RES, SIZE, HSCALE = 256, 1000.0, 200.0

failures = []


def call(pipe, method, params=None):
    response = send_command(pipe, method, params, request_id=call.counter)
    call.counter += 1
    return response


call.counter = 1


def result_of(response):
    if not isinstance(response, dict) or response.get("error"):
        return None
    return response.get("result", response)


def check(label, condition, detail=""):
    if condition:
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  [FAIL] {label}" + (f"\n         {detail}" if detail else ""))
        failures.append(label)


def open_pipe():
    import ctypes
    import ctypes.wintypes as wintypes
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
    if handle is None or handle == (2 ** 64 - 1):
        print(f"[selection-check] FAIL: cannot connect to {PIPE_NAME}. Is RayTrophi Studio running?")
        sys.exit(1)
    handle = ctypes.c_void_p(handle)
    kernel32.SetNamedPipeHandleState(handle, ctypes.byref(wintypes.DWORD(2)), None, None)
    return handle


def translation_of(transform):
    """scene.get_transform returns a 4x4 (row-major) or a dict; accept both."""
    if isinstance(transform, dict):
        for key in ("position", "translation", "location"):
            if key in transform:
                return [float(v) for v in transform[key]]
        if "matrix" in transform:
            transform = transform["matrix"]
    if isinstance(transform, list) and len(transform) == 4 and isinstance(transform[0], list):
        return [float(transform[0][3]), float(transform[1][3]), float(transform[2][3])]
    if isinstance(transform, list) and len(transform) == 16:
        return [float(transform[3]), float(transform[7]), float(transform[11])]
    return None


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    pipe = open_pipe()
    print("=== terrain-only scene selection/transform identity check ===")

    call(pipe, "terrain.remove", {"name": TERRAIN})
    created = result_of(call(pipe, "terrain.create", {
        "name": TERRAIN, "resolution": RES, "size": SIZE, "height_scale": HSCALE}))
    check("terrain created", created is not None)
    if created is None:
        return 1
    terrain_object = TERRAIN + "_Chunk"

    cube = result_of(call(pipe, "scene.add_primitive", {"type": "cube", "name": "SelCheckCube",
                                                        "size": 1.0}))
    check("cube added", isinstance(cube, str) and cube, json.dumps(cube))
    if not isinstance(cube, str) or not cube:
        return 1
    print(f"  cube node name: {cube}")

    objects = result_of(call(pipe, "scene.list_objects")) or []
    names = [o.get("name") if isinstance(o, dict) else o for o in objects]
    check("both objects are in the scene", cube in names and terrain_object in names,
          f"listed: {names}")

    # ── 1. Does selecting the cube actually select the cube? ─────────────────
    call(pipe, "select.clear")
    call(pipe, "select.object", {"name": cube})
    selection = result_of(call(pipe, "select.list")) or []
    primary = next((s for s in selection if isinstance(s, dict) and s.get("primary")), None)
    check("select.object(cube) makes the CUBE the primary selection",
          primary is not None and primary.get("name") == cube,
          f"selection: {json.dumps(selection)}")

    # ── 2. Does a terrain mesh refresh steal that selection? ─────────────────
    # Re-registering the terrain mesh used to erase+append it in world.objects,
    # renumbering every slot the UI had cached.
    call(pipe, "terrain.set_mesh_resolution", {"name": TERRAIN, "mesh_resolution": 128})
    time.sleep(0.3)
    selection = result_of(call(pipe, "select.list")) or []
    primary = next((s for s in selection if isinstance(s, dict) and s.get("primary")), None)
    check("cube is STILL the selection after a terrain mesh rebuild",
          primary is not None and primary.get("name") == cube,
          f"selection: {json.dumps(selection)}")

    # ── 3. Does moving the cube move the cube - and ONLY the cube? ───────────
    terrain_before = translation_of(result_of(call(pipe, "scene.get_transform",
                                                   {"name": terrain_object})))
    cube_before = translation_of(result_of(call(pipe, "scene.get_transform", {"name": cube})))
    check("both transforms readable", terrain_before is not None and cube_before is not None,
          f"terrain {terrain_before}, cube {cube_before}")
    if terrain_before is None or cube_before is None:
        return 1

    moved = result_of(call(pipe, "scene.set_transform",
                           {"name": cube, "translation": [10.0, 3.0, -4.0]}))
    check("scene.set_transform(cube) accepted", moved is not None)

    terrain_after = translation_of(result_of(call(pipe, "scene.get_transform",
                                                  {"name": terrain_object})))
    cube_after = translation_of(result_of(call(pipe, "scene.get_transform", {"name": cube})))

    def moved_by(before, after):
        if before is None or after is None:
            return None
        return max(abs(a - b) for a, b in zip(after, before))

    cube_delta = moved_by(cube_before, cube_after)
    terrain_delta = moved_by(terrain_before, terrain_after)
    check("the CUBE moved", cube_delta is not None and cube_delta > 0.5,
          f"cube {cube_before} -> {cube_after}")
    # ★ The whole point of the file. A terrain that moves here means the name /
    # index / transform-handle plumbing addresses the wrong object.
    check("the TERRAIN did NOT move", terrain_delta is not None and terrain_delta < 1e-4,
          f"terrain {terrain_before} -> {terrain_after}")

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s) - the fault is in the identity plumbing (case A).")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("All value-level checks passed: selection and transform address the right object.")
    print("If the gizmo still moves the terrain, the fault is in the hierarchy-click /")
    print("gizmo path (case B) - scene_ui_hierarchy.cpp and scene_ui_gizmos.cpp.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
