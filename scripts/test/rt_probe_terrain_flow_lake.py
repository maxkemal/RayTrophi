#!/usr/bin/env python3
"""Create one non-destructive RTAPI terrain probe and report hydrology state.

The probe never deletes or reuses an existing terrain. It intentionally keeps
the generated terrain in the open application so the reported fields can be
inspected visually after the script exits.
"""

from __future__ import annotations

import ctypes
import ctypes.wintypes as wintypes
import argparse
import json
from pathlib import Path
import sys
import time


SCRIPTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SCRIPTS))
from ipc_test_client import PIPE_NAME, send_command  # noqa: E402


def result_of(response):
    if not isinstance(response, dict):
        raise RuntimeError(f"invalid IPC response: {response!r}")
    if response.get("error"):
        raise RuntimeError(str(response["error"]))
    return response.get("result", response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--existing",
        help="Inspect an existing probe terrain instead of creating another one",
    )
    parser.add_argument(
        "--disable-pit-fill",
        action="store_true",
        help="Disable Hydraulic Erosion's legacy terminal pit-fill and re-evaluate",
    )
    parser.add_argument(
        "--lem-only",
        action="store_true",
        help="Disable droplet/multi-pass work and evaluate only the LEM cycle",
    )
    parser.add_argument(
        "--restore-defaults",
        action="store_true",
        help="Restore this probe's authored Alpine hydraulic settings and re-evaluate",
    )
    args = parser.parse_args()
    kernel32 = ctypes.windll.kernel32
    pipe = kernel32.CreateFileW(
        PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None
    )
    invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
    if pipe == -1 or (pipe & 0xFFFFFFFFFFFFFFFF) == invalid:
        raise RuntimeError(f"cannot connect to {PIPE_NAME}; win32={kernel32.GetLastError()}")
    mode = wintypes.DWORD(2)
    kernel32.SetNamedPipeHandleState(pipe, ctypes.byref(mode), None, None)

    request_id = 1

    def call(method, params=None):
        nonlocal request_id
        response = send_command(pipe, method, params or {}, request_id=request_id)
        request_id += 1
        return result_of(response)

    def wait_for_evaluation(name, evaluation):
        deadline = time.monotonic() + 900.0
        last_progress = -1.0
        while evaluation.get("state") == "running":
            progress = float(evaluation.get("progress", 0.0))
            if progress >= last_progress + 0.02 or last_progress < 0.0:
                print(
                    f"[probe] evaluation {progress * 100.0:5.1f}% "
                    f"node={evaluation.get('current_node_id', 0)}",
                    flush=True,
                )
                last_progress = progress
            if time.monotonic() >= deadline:
                raise RuntimeError("terrain evaluation did not finish within 15 minutes")
            time.sleep(0.5)
            evaluation = call("terrain.evaluation_status", {"name": name})
        if evaluation.get("state") != "completed":
            raise RuntimeError(f"terrain evaluation ended as {evaluation}")
        return evaluation

    try:
        terrains = call("terrain.list") or []
        existing = {str(item.get("name", "")) for item in terrains}
        if args.existing:
            name = args.existing
            if name not in existing:
                raise RuntimeError(f"terrain not found: {name}")
            created = None
            preset = None
            if args.disable_pit_fill or args.lem_only or args.restore_defaults:
                nodes = call(
                    "nodes.list", {"graph_type": "terrain", "graph_name": name}
                )
                hydraulic = next(
                    (
                        node
                        for node in nodes
                        if node.get("type_id") == "TerrainV2.HydraulicErosion"
                    ),
                    None,
                )
                if hydraulic is None:
                    raise RuntimeError("Hydraulic Erosion node not found")
                node_id = int(hydraulic["id"])
                if args.disable_pit_fill:
                    call(
                        "nodes.set_property",
                        {
                            "graph_type": "terrain",
                            "graph_name": name,
                            "node_id": node_id,
                            "property": "params.fillPits",
                            "value": False,
                        },
                    )
                    print("[probe] disabled legacy terminal pit-fill", flush=True)
                if args.lem_only:
                    for property_name, value in (
                        ("multiPass", False),
                        ("params.iterations", 0),
                        ("params.fillPits", False),
                        ("params.removeSpikes", False),
                    ):
                        call(
                            "nodes.set_property",
                            {
                                "graph_type": "terrain",
                                "graph_name": name,
                                "node_id": node_id,
                                "property": property_name,
                                "value": value,
                            },
                        )
                    print("[probe] evaluating LEM cycle without droplets", flush=True)
                if args.restore_defaults:
                    for property_name, value in (
                        ("multiPass", True),
                        ("params.iterations", 75000),
                        ("params.fillPits", True),
                        ("params.removeSpikes", True),
                    ):
                        call(
                            "nodes.set_property",
                            {
                                "graph_type": "terrain",
                                "graph_name": name,
                                "node_id": node_id,
                                "property": property_name,
                                "value": value,
                            },
                        )
                    print("[probe] restored authored Alpine hydraulic settings", flush=True)
                evaluation = call("terrain.evaluate", {"name": name})
            else:
                evaluation = call("terrain.evaluation_status", {"name": name})
        else:
            base = "CodexFlowLakeProbe"
            name = base
            suffix = 2
            while name in existing:
                name = f"{base}_{suffix}"
                suffix += 1

            print(f"[probe] creating {name} (512x512, 2 km, 320 m relief)", flush=True)
            created = call(
                "terrain.create",
                {
                    "name": name,
                    "resolution": 512,
                    "size": 2000.0,
                    "height_scale": 320.0,
                    "mesh_resolution": 512,
                },
            )
            print("[probe] applying relief preset", flush=True)
            relief_preset = call(
                "terrain.apply_preset", {"name": name, "preset": "snowy_mountain_valley"}
            )
            print("[probe] evaluating relief graph", flush=True)
            wait_for_evaluation(name, call("terrain.evaluate", {"name": name}))
            print("[probe] applying river_network preset", flush=True)
            river_preset = call(
                "terrain.apply_preset", {"name": name, "preset": "river_network"}
            )
            preset = {"relief": relief_preset, "river_network": river_preset}
            print("[probe] evaluating GPU terrain graph", flush=True)
            evaluation = call("terrain.evaluate", {"name": name})

        evaluation = wait_for_evaluation(name, evaluation)

        report = {
            "terrain": name,
            "created": created,
            "preset": preset,
            "evaluation": evaluation,
            "terrain_state": call("terrain.get", {"name": name}),
            "flow_authority": call("terrain.flow_authority", {"name": name}),
            "erosion_stats": call("terrain.erosion_stats"),
            "flow_stats": call("terrain.calculate_flow", {"name": name}),
            "slope_area_fit": call("terrain.slope_area_fit", {"name": name}),
            "rivers": call("terrain.list_rivers"),
            "nodes": call("nodes.list", {"graph_type": "terrain", "graph_name": name}),
        }
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        print(f"[probe] kept {name} in the open app for visual inspection", flush=True)
        return 0
    finally:
        kernel32.CloseHandle(pipe)


if __name__ == "__main__":
    raise SystemExit(main())
