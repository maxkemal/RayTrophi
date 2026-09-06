#!/usr/bin/env python3
"""Live IPC smoke test for Curve Input -> Road Carve.

The test is observational with respect to spline authoring: it uses the first
existing SplineObject (or --spline NAME), never edits or removes it. Only a
reserved temporary terrain is created and cleaned up.
"""

import argparse
import math
import os
import sys
import time

from rt_ipc import RtIpc, RtIpcError


TERRAIN = f"__TerrainRoadCarveProbe_{os.getpid()}_{time.time_ns()}"
FIELDS = {
    "road_core": "infrastructure.road_core",
    "shoulder": "infrastructure.shoulder",
    "cut": "infrastructure.cut",
    "fill": "infrastructure.fill",
    "foliage_exclusion": "infrastructure.foliage_exclusion",
    # Six, not five. The ditch is what keeps the drainage network intact once
    # road_core is excluded from channel classification: the road pushes water
    # off its crown, the ditch carries it. Publishing the exclusion WITHOUT the
    # ditch is the plausible-looking failure - no river on the road, and the
    # water that should run beside it goes nowhere.
    "ditch": "infrastructure.ditch",
}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def wait_for_evaluation(client):
    deadline = time.monotonic() + 30.0
    while True:
        status = client.call("terrain.evaluation_status", name=TERRAIN)
        state = status.get("state")
        if state in ("completed", "idle"):
            return
        if state in ("failed", "cancelled"):
            raise AssertionError(f"terrain evaluation ended as {status}")
        if time.monotonic() >= deadline:
            raise AssertionError(f"terrain evaluation timed out: {status}")
        time.sleep(0.1)


def spline_fixture_extent(payload):
    matrix = payload.get("transform")
    points = payload.get("points", [])
    require(isinstance(matrix, list) and len(matrix) == 4, "spline transform is missing")
    require(len(points) >= 2, "road spline needs at least two control points")
    world = []
    for point in points:
        x, y, z = point["position"]
        world.append((
            matrix[0][0] * x + matrix[0][1] * y + matrix[0][2] * z + matrix[0][3],
            matrix[2][0] * x + matrix[2][1] * y + matrix[2][2] * z + matrix[2][3],
        ))
    # TerrainManager centers a [0,size] local heightfield in world space by
    # translating the TerrainObject to (-size/2, 0, -size/2).
    half_extent = max(max(abs(x), abs(z)) for x, z in world) + 16.0
    return max(32.0, math.ceil(half_extent * 2.0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--spline", help="existing SplineObject name")
    args = parser.parse_args()
    client = RtIpc()
    created = False
    stage = "scene inventory"
    try:
        splines = client.call("spline.list")
        spline_name = args.spline or (splines[0]["name"] if splines else None)
        require(spline_name, "create or draw one SplineObject before running this test")
        require(any(item["name"] == spline_name for item in splines),
                f"spline not found: {spline_name}")
        spline_payload = client.call("spline.get", name=spline_name)
        fixture_size = spline_fixture_extent(spline_payload)

        stage = "temporary terrain creation"
        client.call("terrain.create", name=TERRAIN, resolution=256,
                    mesh_resolution=64, size=fixture_size, height_scale=32.0)
        created = True
        nodes = client.call("nodes.list", graph_type="terrain", graph_name=TERRAIN)
        height_source = next((item for item in nodes
                              if item["type_id"] == "TerrainV2.HeightmapInput"), None)
        if height_source is None:
            for item in nodes:
                if item["type_id"].endswith("Output"):
                    continue
                ports = client.call(
                    "nodes.list_ports", graph_type="terrain", graph_name=TERRAIN,
                    node_id=item["id"])
                if any(port["direction"] == "output" and port["key"] == "height"
                       for port in ports):
                    height_source = item
                    break
        require(height_source is not None,
                f"temporary graph has no Height source: {[n['type_id'] for n in nodes]}")

        stage = "road graph construction"
        curve_input = client.call(
            "nodes.add", graph_type="terrain", graph_name=TERRAIN,
            type_id="TerrainV2.CurveInput")
        road = client.call(
            "nodes.add", graph_type="terrain", graph_name=TERRAIN,
            type_id="TerrainV2.RoadCarve")
        road_fields = client.call(
            "nodes.add", graph_type="terrain", graph_name=TERRAIN,
            type_id="TerrainV2.RoadFieldsOutput")
        client.call("nodes.set_property", graph_type="terrain", graph_name=TERRAIN,
                    node_id=curve_input, property="splineObject", value=spline_name)
        for property_name, value in {
            "roadWidthMeters": 6.0,
            "shoulderWidthMeters": 2.0,
            "gradingFalloffMeters": 2.0,
            "foliageExclusionMarginMeters": 2.0,
            "maxGradePercent": 10.0,
            "ditchWidthMeters": 1.5,
            "ditchDepthMeters": 0.6,
            "crownMeters": 0.15,
            # Flat input still produces measurable fill, making this fixture
            # independent of whichever terrain preset happens to be loaded.
            "elevationOffsetMeters": 1.0,
        }.items():
            client.call("nodes.set_property", graph_type="terrain", graph_name=TERRAIN,
                        node_id=road, property=property_name, value=value)

        client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                    from_node=height_source["id"], from_output="height",
                    to_node=road, to_input="height")
        client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                    from_node=curve_input, from_output="curve",
                    to_node=road, to_input="curve")

        for output_key in FIELDS:
            client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                        from_node=road, from_output=output_key,
                        to_node=road_fields, to_input=output_key)
        client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                    from_node=road, from_output="snapshot_revision",
                    to_node=road_fields, to_input="snapshot_revision")

        stage = "terrain evaluation"
        client.call("terrain.evaluate", name=TERRAIN)
        wait_for_evaluation(client)
        stage = "published field measurement"
        published = set(client.call("terrain.list_fields", terrain=TERRAIN))
        require(set(FIELDS.values()) <= published,
                f"missing infrastructure fields: {set(FIELDS.values()) - published}")
        stats = {name: client.call("terrain.field_stats", terrain=TERRAIN, field=name)
                 for name in FIELDS.values()}

        core = stats["infrastructure.road_core"]
        shoulder = stats["infrastructure.shoulder"]
        ditch = stats["infrastructure.ditch"]
        exclusion = stats["infrastructure.foliage_exclusion"]
        fill = stats["infrastructure.fill"]
        require(core["non_finite_count"] == 0 and core["nonzero_count"] > 0, core)
        require(shoulder["non_finite_count"] == 0 and shoulder["nonzero_count"] > 0,
                shoulder)
        require(exclusion["nonzero_count"] >= core["nonzero_count"], exclusion)
        # A ditch that measures zero everywhere means the cross-section never
        # reached the rasterizer - the road is a flat trench again and nothing
        # about the height field would say so.
        require(ditch["non_finite_count"] == 0 and ditch["nonzero_count"] > 0, ditch)
        require(ditch["max"] <= 1.0 + 1e-5, ditch)
        require(fill["max"] > 0.0, fill)
        for field_name, summary in stats.items():
            require(summary["width"] == 256 and summary["height"] == 256, summary)
            require(summary["non_finite_count"] == 0, (field_name, summary))

        properties = {item["name"] for item in client.call(
            "nodes.list_properties", graph_type="terrain", graph_name=TERRAIN,
            node_id=road)}
        require({"roadWidthMeters", "maxGradePercent", "elevationOffsetMeters"}
                <= properties, properties)
        road_field_ports = client.call(
            "nodes.list_ports", graph_type="terrain", graph_name=TERRAIN,
            node_id=road_fields)
        require({"road_core", "shoulder", "cut", "fill", "ditch",
                 "foliage_exclusion", "snapshot_revision"}
                <= {item["key"] for item in road_field_ports}, road_field_ports)
        print("[terrain road carve IPC] PASS")
        print(f"  spline={spline_name!r} fixture_size={fixture_size} m core_pixels={core['nonzero_count']}")
        print(f"  shoulder_pixels={shoulder['nonzero_count']} fill_max={fill['max']:.6g} m")
        print(f"  ditch_pixels={ditch['nonzero_count']} ditch_max={ditch['max']:.4g}")
        return 0
    except (AssertionError, RtIpcError, StopIteration) as exc:
        print(f"[terrain road carve IPC] FAIL at {stage}: {type(exc).__name__}: {exc!r}")
        return 1
    finally:
        if created:
            ok, result = client.try_call("terrain.remove", name=TERRAIN)
            if not ok:
                print(f"[terrain road carve IPC] cleanup warning: {result}")
        client.close()


if __name__ == "__main__":
    sys.exit(main())
