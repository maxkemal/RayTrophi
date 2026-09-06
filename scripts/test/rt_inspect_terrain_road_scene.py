#!/usr/bin/env python3
"""Read-only scene inventory for terrain curve/road diagnostics."""

import json
import sys

from rt_ipc import RtIpc, RtIpcError


def main():
    client = RtIpc()
    try:
        splines = client.call("spline.list")
        terrains = client.call("terrain.list")
        result = {"splines": [], "terrains": []}
        for item in splines:
            payload = client.call("spline.get", name=item["name"])
            result["splines"].append({"summary": item, "payload": payload})
        for terrain in terrains:
            name = terrain["name"] if isinstance(terrain, dict) else terrain
            nodes = client.call("nodes.list", graph_type="terrain", graph_name=name)
            road_nodes = []
            for node in nodes:
                if node["type_id"] not in (
                        "TerrainV2.CurveInput", "TerrainV2.CurveToMask",
                        "TerrainV2.RoadCarve", "TerrainV2.RoadFieldsOutput",
                        "TerrainV2.PublishField"):
                    continue
                road_nodes.append({
                    "node": node,
                    "ports": client.call(
                        "nodes.list_ports", graph_type="terrain", graph_name=name,
                        node_id=node["id"]),
                    "properties": client.call(
                        "nodes.list_properties", graph_type="terrain", graph_name=name,
                        node_id=node["id"]),
                })
            result["terrains"].append({"summary": terrain, "road_nodes": road_nodes})
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except RtIpcError as exc:
        print(f"[terrain road scene inspect] FAIL: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
