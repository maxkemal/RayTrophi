#!/usr/bin/env python3
"""
RayTrophi Studio — IPC Named Pipe Test Client (Faz 4c)

Usage:
    1. Start RayTrophi Studio (it creates \\.\pipe\RayTrophiStudio).
    2. Run this script from a separate terminal:
       python scripts/ipc_test_client.py

The script sends several JSON commands and prints the responses.
It uses only the Python standard library (no win32pipe dependency).
"""
import json
import struct
import sys
import os

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'

def send_command(pipe, method, params=None, request_id=1):
    """Send a JSON command to the pipe and read the response."""
    msg = {"id": request_id, "method": method}
    if params:
        msg["params"] = params
    data = json.dumps(msg).encode("utf-8")

    # Write to pipe
    import ctypes
    import ctypes.wintypes as wintypes

    kernel32 = ctypes.windll.kernel32

    # WriteFile
    bytes_written = wintypes.DWORD(0)
    success = kernel32.WriteFile(
        pipe, data, len(data), ctypes.byref(bytes_written), None
    )
    if not success:
        raise OSError(f"WriteFile failed (error {kernel32.GetLastError()})")

    # Read the complete message; responses may be larger than one 64 KiB chunk.
    chunks = []
    while True:
        buf = ctypes.create_string_buffer(65536)
        bytes_read = wintypes.DWORD(0)
        success = kernel32.ReadFile(pipe, buf, 65536, ctypes.byref(bytes_read), None)
        chunks.append(buf.raw[:bytes_read.value])
        if success:
            break
        error = kernel32.GetLastError()
        if error != 234:  # ERROR_MORE_DATA
            raise OSError(f"ReadFile failed (error {error})")
    response_str = b"".join(chunks).decode("utf-8")
    return json.loads(response_str)


def main():
    import ctypes
    import ctypes.wintypes as wintypes

    kernel32 = ctypes.windll.kernel32

    GENERIC_READ = 0x80000000
    GENERIC_WRITE = 0x40000000
    OPEN_EXISTING = 3
    PIPE_READMODE_MESSAGE = 0x00000002
    INVALID_HANDLE_VALUE = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF

    print(f"[ipc-test] Connecting to {PIPE_NAME} ...")

    # Open the pipe
    handle = kernel32.CreateFileW(
        PIPE_NAME,
        GENERIC_READ | GENERIC_WRITE,
        0,       # no sharing
        None,    # default security
        OPEN_EXISTING,
        0,       # default attributes
        None     # no template
    )

    # Handle comparison (handle is a signed int on 64-bit)
    if handle == -1 or (handle & 0xFFFFFFFFFFFFFFFF) == INVALID_HANDLE_VALUE:
        err = kernel32.GetLastError()
        print(f"[ipc-test] FAIL: Cannot connect to pipe (error {err}).")
        print("           Is RayTrophi Studio running?")
        sys.exit(1)

    # Set read mode to message
    mode = wintypes.DWORD(PIPE_READMODE_MESSAGE)
    kernel32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None)

    print("[ipc-test] Connected!\n")

    test_id = 0
    tests_passed = 0
    tests_failed = 0

    def run_test(method, params=None, label=None, expect_error=False):
        nonlocal test_id, tests_passed, tests_failed
        test_id += 1
        tag = label or method
        try:
            resp = send_command(handle, method, params, request_id=test_id)
            has_error = "error" in resp

            if has_error and not expect_error:
                print(f"  [{test_id}] {tag}: ERROR (Unexpected) — {resp['error']}")
                tests_failed += 1
            elif not has_error and expect_error:
                print(f"  [{test_id}] {tag}: FAIL (Expected error but succeeded) — {resp.get('result')}")
                tests_failed += 1
            elif has_error and expect_error:
                print(f"  [{test_id}] {tag}: OK (Expected error caught) — {resp['error']}")
                tests_passed += 1
            else:
                result = resp.get("result", None)
                # Truncate long results for display
                result_str = json.dumps(result, ensure_ascii=False)
                if len(result_str) > 120:
                    result_str = result_str[:117] + "..."
                print(f"  [{test_id}] {tag}: OK — {result_str}")
                tests_passed += 1
        except Exception as e:
            print(f"  [{test_id}] {tag}: EXCEPTION — {e}")
            tests_failed += 1

    print("─── IPC Smoke Tests ───────────────────────────────────────────")

    # Basic queries
    run_test("version")
    run_test("scene.list_objects")
    run_test("timeline.get_frame")
    run_test("project.path")
    run_test("lights.list")
    run_test("batch", {"calls": [
        {"method": "version", "params": {}},
        {"method": "timeline.get_frame", "params": {}},
        {"method": "scene.object_exists", "params": {"name": "default_Cube"}},
    ]}, "batch(single main-thread hop)")

    # Object existence check
    run_test("scene.object_exists", {"name": "default_Cube"}, "exists(default_Cube)")
    run_test("scene.object_exists", {"name": "NONEXISTENT_OBJ"}, "exists(nonexistent)")

    # Object info
    run_test("scene.object_info", {"name": "default_Cube"}, "info(default_Cube)")

    # Procedural primitive creation (Faz 5.2a)
    run_test("scene.add_primitive", {"type": "cube", "name": "IpcTestCube", "size": 1.0}, "add_primitive(cube)")

    # Mesh Modifiers (Faz 5.2b)
    run_test("modifiers.get_stack", {"object": "IpcTestCube"}, "modifiers.get_stack")
    run_test("modifiers.add", {"object": "IpcTestCube", "type": "catmull_clark", "name": "IpcSubdiv", "levels": 2}, "modifiers.add(subdiv)")
    run_test("modifiers.remove", {"object": "IpcTestCube", "index": 0}, "modifiers.remove")

    # Scatter & Foliage System (Faz 5.2c)
    run_test("scatter.list_groups", {}, "scatter.list_groups")
    run_test("scatter.create_group", {"name": "IpcScatterGrp", "target_node": "IpcTestCube", "target_type": "mesh"}, "scatter.create_group")
    run_test("scatter.delete_group", {"group": "IpcScatterGrp"}, "scatter.delete_group")

    # Physics Engine (Faz 5.3a)
    run_test("physics.add_body", {"object": "IpcTestCube", "kind": "rigid", "motion_type": "dynamic", "shape": "box", "mass": 2.0}, "physics.add_body")
    run_test("physics.get_body", {"object": "IpcTestCube"}, "physics.get_body")
    run_test("physics.step", {"dt": 0.0166}, "physics.step")
    run_test("physics.reset", {}, "physics.reset")
    run_test("physics.remove_body", {"object": "IpcTestCube"}, "physics.remove_body")

    # Fluid Simulation Engine (Faz 5.3b)
    run_test("fluid.create_domain", {"name": "IpcFluidDomain", "voxel_size": 0.1}, "fluid.create_domain")
    run_test("fluid.seed", {"domain": "IpcFluidDomain", "particles_per_cell": 4}, "fluid.seed")
    run_test("fluid.get", {"domain": "IpcFluidDomain"}, "fluid.get")
    run_test("fluid.step", {"dt": 0.0166}, "fluid.step")
    run_test("fluid.remove_domain", {"domain": "IpcFluidDomain"}, "fluid.remove_domain")

    run_test("terrain.list", {}, "terrain.list")
    run_test("terrain.create", {"name": "IpcTerrain", "resolution": 64, "size": 128.0,
                                "height_scale": 24.0}, "terrain.create")
    run_test("terrain.get", {"name": "IpcTerrain"}, "terrain.get")
    run_test("terrain.evaluation_status", {"name": "IpcTerrain"}, "terrain.evaluation_status(idle)")
    run_test("terrain.sample_height", {"name": "IpcTerrain", "world_x": 0.0, "world_z": 0.0},
             "terrain.sample_height")
    run_test("terrain.calculate_flow", {"name": "IpcTerrain"}, "terrain.calculate_flow")
    run_test("terrain.apply_preset", {"name": "IpcTerrain", "preset": "default"},
             "terrain.apply_preset(default)")
    run_test("terrain.list_rivers", {}, "terrain.list_rivers")
    run_test("nodes.list", {"graph_type": "terrain", "graph_name": "IpcTerrain"},
             "nodes.list(terrain)")
    run_test("nodes.list_properties", {"graph_type": "terrain", "graph_name": "IpcTerrain", "node_id": 1},
             "nodes.list_properties(terrain noise)")
    run_test("terrain.erode", {"name": "IpcTerrain", "type": "invalid"},
             "terrain.erode(invalid) -> error", expect_error=True)
    run_test("terrain.carve_river", {"name": "IpcTerrain", "river": "__missing__"},
             "terrain.carve_river(missing) -> error", expect_error=True)
    run_test("terrain.cancel_evaluation", {"name": "IpcTerrain"},
             "terrain.cancel_evaluation(idle) -> error", expect_error=True)
    run_test("terrain.remove", {"name": "IpcTerrain"}, "terrain.remove")

    # Hair & Groom System (Faz 5.4a)
    run_test("hair.list", {}, "hair.list")
    run_test("hair.create", {"mesh": "IpcTestCube", "name": "IpcHairGroom",
                              "guide_count": 24, "children_per_guide": 1,
                              "points_per_strand": 6, "length": 0.2}, "hair.create")
    run_test("hair.get", {"name": "IpcHairGroom"}, "hair.get")
    run_test("hair.update", {"name": "IpcHairGroom", "length": 0.3,
                              "clumpiness": 0.7, "curl_frequency": 1.5}, "hair.update")
    run_test("hair.restyle", {"name": "IpcHairGroom"}, "hair.restyle")
    run_test("hair.list_presets", {}, "hair.list_presets")
    run_test("hair.apply_preset", {"name": "IpcHairGroom", "preset": "wavy"},
             "hair.apply_preset")
    run_test("hair.trim", {"name": "IpcHairGroom", "length_factor": 0.9}, "hair.trim")
    run_test("hair.grow", {"name": "IpcHairGroom", "length_factor": 1.1}, "hair.grow")
    run_test("hair.comb", {"name": "IpcHairGroom", "direction": [1.0, 0.2, 0.0],
                            "strength": 0.6, "root_stiffness": 0.8}, "hair.comb")
    run_test("hair.smooth", {"name": "IpcHairGroom", "strength": 0.35, "iterations": 2},
             "hair.smooth")
    run_test("hair.reset_simulation", {"name": "IpcHairGroom"}, "hair.reset_simulation")
    run_test("hair.bake", {"name": "IpcHairGroom"}, "hair.bake")
    run_test("hair.rename", {"name": "IpcHairGroom", "new_name": "IpcHairRenamed"},
             "hair.rename")
    run_test("hair.remove", {"name": "IpcHairRenamed"}, "hair.remove")

    # Mesh Paint Automation (Faz 5.4b)
    run_test("paint.ensure", {"object": "IpcTestCube", "resolution": 64}, "paint.ensure")
    run_test("paint.add_layer", {"object": "IpcTestCube", "name": "IpcPaintFill"},
             "paint.add_layer")
    run_test("paint.update_layer", {"object": "IpcTestCube", "layer_index": 1,
                                     "opacity": 0.8, "blend_mode": "overlay"},
             "paint.update_layer")
    run_test("paint.fill", {"object": "IpcTestCube", "layer_index": 1,
                             "channel": "base_color", "color": [0.2, 0.5, 0.8]},
             "paint.fill")
    run_test("paint.get", {"object": "IpcTestCube"}, "paint.get")
    run_test("paint.list_mask_presets", {}, "paint.list_mask_presets")
    run_test("paint.apply_mask", {"object": "IpcTestCube", "layer_index": 1,
                                   "preset": "edge_wear", "strength": 0.8, "seed": 42},
             "paint.apply_mask")
    run_test("paint.bake_height_to_normal", {"object": "IpcTestCube", "strength": 4.0},
             "paint.bake_height_to_normal")
    run_test("paint.duplicate_layer", {"object": "IpcTestCube", "layer_index": 1},
             "paint.duplicate_layer")
    run_test("paint.move_layer", {"object": "IpcTestCube", "from_index": 2, "to_index": 1},
             "paint.move_layer")
    run_test("paint.merge_down", {"object": "IpcTestCube", "layer_index": 1},
             "paint.merge_down")
    ipc_paint_png = os.path.join(os.environ.get("TEMP", "."), "raytrophi_ipc_paint_test.png")
    run_test("paint.export_channel", {"object": "IpcTestCube", "channel": "normal",
                                       "filepath": ipc_paint_png}, "paint.export_channel")
    run_test("paint.import_channel", {"object": "IpcTestCube", "layer_index": 1,
                                       "channel": "normal", "filepath": ipc_paint_png},
             "paint.import_channel")
    if os.path.isfile(ipc_paint_png):
        os.remove(ipc_paint_png)
    run_test("paint.flatten", {"object": "IpcTestCube"}, "paint.flatten")

    # Deterministic Sculpt Automation (Faz 5.4c)
    run_test("sculpt.get", {"object": "IpcTestCube"}, "sculpt.get")
    run_test("sculpt.mask_operation", {"object": "IpcTestCube", "operation": "fill",
                                         "undo": False}, "sculpt.mask_operation(fill)")
    run_test("sculpt.mask_operation", {"object": "IpcTestCube", "operation": "invert",
                                         "undo": False}, "sculpt.mask_operation(invert)")
    run_test("sculpt.paint_mask", {"object": "IpcTestCube", "points": [[0.0, 0.0, 0.0]],
                                     "radius": 5.0, "value": 1.0, "undo": False},
             "sculpt.paint_mask")
    run_test("sculpt.mask_operation", {"object": "IpcTestCube", "operation": "clear",
                                         "undo": False}, "sculpt.mask_operation(clear)")
    for sculpt_tool in ("draw", "inflate", "smooth", "flatten", "stamp", "noise"):
        run_test("sculpt.stroke", {"object": "IpcTestCube", "tool": sculpt_tool,
                                     "points": [[0.0, 0.0, 0.0]], "radius": 5.0,
                                     "strength": 0.01, "direction": [0.0, 1.0, 0.0],
                                     "seed": 42, "use_mask": False, "undo": False},
                 f"sculpt.stroke({sculpt_tool})")

    # Undo/redo descriptions
    run_test("undo_description")
    run_test("redo_description")

    # Render status (should be idle)
    run_test("render.status")

    # Node types
    run_test("nodes.types")

    # Camera (Faz 5.1a) — read, then a non-destructive fov round-trip
    run_test("camera.get")
    run_test("camera.set_fov", {"fov": 50.0}, "camera.set_fov(50)")

    # World (Faz 5.1c) — read + a sun elevation set
    run_test("world.get")
    run_test("world.set_sun_elevation", {"sun_elevation": 30.0}, "world.set_sun_elevation(30)")

    # Post-processing (Faz 5.1d) — read + exposure set
    run_test("post.get")
    run_test("post.set_exposure", {"exposure": 1.2}, "post.set_exposure(1.2)")

    # Node parameters (Faz 5.1b) — no graph in the default scene, so a missing
    # graph must error (surface check that the methods are wired).
    run_test("nodes.get_param",
             {"graph_type": "material", "graph_name": "__missing__", "node_id": 1, "pin_index": 0},
             "nodes.get_param(missing) → error", expect_error=True)
    run_test("nodes.set_param",
             {"graph_type": "material", "graph_name": "__missing__", "node_id": 1, "pin_index": 0, "value": 0.5},
             "nodes.set_param(missing) → error", expect_error=True)

    # Error cases
    run_test("scene.object_info", {"name": "DOES_NOT_EXIST"}, "info(missing) → error", expect_error=True)
    run_test("nonexistent_method", {}, "unknown method → error", expect_error=True)

    print(f"\n─── Results: {tests_passed} PASS, {tests_failed} FAIL "
          f"(total {tests_passed + tests_failed}) ─────────────────")

    # Close handle
    kernel32.CloseHandle(handle)

    if tests_failed > 0:
        print("[ipc-test] FAIL")
        sys.exit(1)
    else:
        print("[ipc-test] PASS")
        sys.exit(0)


if __name__ == "__main__":
    main()
