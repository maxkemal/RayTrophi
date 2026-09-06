#!/usr/bin/env python3
"""Evaluate an already-authored terrain road graph without rewiring it."""

import argparse
import sys
import time

from rt_ipc import RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("terrain")
    args = parser.parse_args()
    client = RtIpc()
    try:
        nodes = client.call("nodes.list", graph_type="terrain", graph_name=args.terrain)
        road = next((node for node in nodes if node["type_id"] == "TerrainV2.RoadCarve"), None)
        if road is None:
            raise AssertionError("terrain graph has no Road Carve node")
        ports = client.call("nodes.list_ports", graph_type="terrain",
                            graph_name=args.terrain, node_id=road["id"])
        required = {("input", "height"), ("input", "curve"), ("output", "height")}
        connected = {(p["direction"], p["key"]) for p in ports if p["connected"]}
        if not required <= connected:
            raise AssertionError(f"missing connected ports: {required - connected}")

        client.call("terrain.evaluate", name=args.terrain)
        deadline = time.monotonic() + 90.0
        while True:
            status = client.call("terrain.evaluation_status", name=args.terrain)
            state = status.get("state")
            if state in ("completed", "idle"):
                break
            if state in ("failed", "cancelled"):
                raise AssertionError(status)
            if time.monotonic() >= deadline:
                raise AssertionError(f"evaluation timed out: {status}")
            time.sleep(0.1)
        print("[existing terrain road graph] PASS")
        print(f"  terrain={args.terrain!r} road_node={road['id']} status={status}")
        return 0
    except (AssertionError, RtIpcError) as exc:
        print(f"[existing terrain road graph] FAIL: {exc!r}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
