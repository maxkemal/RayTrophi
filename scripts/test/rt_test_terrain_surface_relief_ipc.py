#!/usr/bin/env python3
"""External IPC smoke test for Structural Hardness and Surface Relief."""

import ctypes
import ctypes.wintypes as wintypes
import sys

sys.path.insert(0, str(__file__).replace("\\", "/").rsplit("/", 2)[0])
from ipc_test_client import PIPE_NAME, send_command  # noqa: E402


NAME = "__IPC_TerrainSurfaceRelief"


def open_pipe():
    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateFileW(
        PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
    invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
    if handle == -1 or (handle & 0xFFFFFFFFFFFFFFFF) == invalid:
        raise RuntimeError("cannot connect to RayTrophi Studio IPC pipe")
    mode = wintypes.DWORD(2)
    kernel32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None)
    return handle


def main():
    pipe = open_pipe()
    request_id = 1

    def call(method, params=None):
        nonlocal request_id
        response = send_command(pipe, method, params, request_id=request_id)
        request_id += 1
        if response.get("error"):
            raise AssertionError(f"{method}: {response['error']}")
        return response.get("result", response)

    try:
        try:
            call("terrain.remove", {"name": NAME})
        except AssertionError:
            pass

        type_ids = {item["type_id"] for item in call("nodes.types")}
        assert {"TerrainV2.StructuralHardness", "TerrainV2.SurfaceRelief"} <= type_ids

        call("terrain.create", {
            "name": NAME, "resolution": 64, "mesh_resolution": 64,
            "size": 256.0, "height_scale": 64.0,
        })
        preset = call("terrain.apply_preset", {
            "name": NAME, "preset": "snowy_mountain_valley",
            "replace_graph": True,
        })
        assert preset["wiring_fault_count"] == 0, preset

        nodes = call("nodes.list", {
            "graph_type": "terrain", "graph_name": NAME,
        })
        by_type = {node["type_id"]: node for node in nodes}
        hardness = by_type["TerrainV2.StructuralHardness"]
        relief = by_type["TerrainV2.SurfaceRelief"]
        assert hardness["inputs"] == 3 and hardness["outputs"] == 3, hardness
        assert relief["inputs"] == 7 and relief["outputs"] == 4, relief

        call("nodes.set_property", {
            "graph_type": "terrain", "graph_name": NAME,
            "node_id": relief["id"], "property": "rillDepthMeters",
            "value": 0.45,
        })
        value = call("nodes.get_property", {
            "graph_type": "terrain", "graph_name": NAME,
            "node_id": relief["id"], "property": "rillDepthMeters",
        })
        assert abs(float(value) - 0.45) < 1.0e-6, value
        print("[terrain surface relief IPC] OK")
    finally:
        try:
            call("terrain.remove", {"name": NAME})
        except Exception:
            pass
        ctypes.windll.kernel32.CloseHandle(pipe)


if __name__ == "__main__":
    main()
