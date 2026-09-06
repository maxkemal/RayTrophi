#!/usr/bin/env python3
"""Live IPC test: crossing semantics and the road surface mesh, end to end.

Builds its own fixture and removes it: one flat terrain, one road spline running
west-east, and one "river" spline running north-south. The river is rasterized by
Curve to Mask and wired into Road Network's Water pin - the only thing that tells
the solver where a crossing IS.

WHAT IT PROVES
    1. Declaring `bridge` produces bridge samples ON THE CROSSING ONLY. A route
       that comes back entirely bridged would mean the mode replaced the cut/fill
       envelope instead of spanning what the ground cannot carry.
    2. Declaring `ford` produces ford samples and NO bridge samples.
    3. Declaring `terrain` produces neither. Without this the first two checks
       could pass on a solver that bridges everything regardless of the mode.
    4. `infrastructure.ditch` is published and non-empty. Excluding the road from
       channel classification WITHOUT the ditch is the plausible-looking failure:
       no river on the road, and the water that should run beside it goes nowhere.
    5. The surface mesh builds from the solved route, and a REBUILD replaces the
       same object instead of stacking a second road on the first.
    6. clear_mesh removes it and the terrain-only road stays valid.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/test/rt_test_terrain_road_crossings.py
"""

import os
import sys
import time

from rt_ipc import RtIpc, RtIpcError

STAMP = f"{os.getpid()}_{time.time_ns()}"
TERRAIN = f"__TerrainRoadCrossProbe_{STAMP}"
ROAD = f"__probe_road_{STAMP}"
RIVER = f"__probe_river_{STAMP}"
SIZE = 200.0
FIELDS = ["road_core", "shoulder", "cut", "fill", "foliage_exclusion", "ditch"]


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def wait_for_evaluation(client):
    deadline = time.monotonic() + 60.0
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


def draw_spline(client, name, points):
    created = client.call("spline.create", primitive="empty", name=name, plane="free")
    actual = created if isinstance(created, str) and created else name
    for point in points:
        client.call("spline.append_point", name=actual, position=list(point))
    return actual


def solve_with_crossing(client, road_name, mode, max_samples=0):
    client.call("terrain.road.set_crossing_mode", spline=road_name, mode=mode)
    client.call("terrain.evaluate", name=TERRAIN)
    wait_for_evaluation(client)
    return client.call("terrain.road.get_route", spline=road_name,
                       max_samples=max_samples)


def main():
    client = RtIpc()
    created_terrain = False
    road_name = ROAD
    river_name = RIVER
    stage = "fixture"
    try:
        stage = "temporary terrain"
        client.call("terrain.create", name=TERRAIN, resolution=256,
                    mesh_resolution=64, size=SIZE, height_scale=32.0)
        created_terrain = True

        stage = "curves"
        # The terrain is centred on the origin, so world x/z run -SIZE/2..+SIZE/2.
        road_name = draw_spline(client, ROAD,
                                [(-80.0, 0.0, 0.0), (-20.0, 0.0, 0.0),
                                 (20.0, 0.0, 0.0), (80.0, 0.0, 0.0)])
        river_name = draw_spline(client, RIVER,
                                 [(0.0, 0.0, -80.0), (0.0, 0.0, 0.0),
                                  (0.0, 0.0, 80.0)])

        stage = "road assignment"
        client.call("terrain.road.assign_profile", spline=road_name, profile="main_road")

        stage = "graph construction"
        nodes = client.call("nodes.list", graph_type="terrain", graph_name=TERRAIN)
        height_source = next((item for item in nodes
                              if item["type_id"] == "TerrainV2.HeightmapInput"), None)
        if height_source is None:
            for item in nodes:
                if item["type_id"].endswith("Output"):
                    continue
                ports = client.call("nodes.list_ports", graph_type="terrain",
                                    graph_name=TERRAIN, node_id=item["id"])
                if any(p["direction"] == "output" and p["key"] == "height" for p in ports):
                    height_source = item
                    break
        require(height_source is not None,
                f"temporary graph has no Height source: {[n['type_id'] for n in nodes]}")

        curve_input = client.call("nodes.add", graph_type="terrain", graph_name=TERRAIN,
                                  type_id="TerrainV2.CurveInput")
        curve_mask = client.call("nodes.add", graph_type="terrain", graph_name=TERRAIN,
                                 type_id="TerrainV2.CurveToMask")
        network = client.call("nodes.add", graph_type="terrain", graph_name=TERRAIN,
                              type_id="TerrainV2.RoadNetwork")
        road_fields = client.call("nodes.add", graph_type="terrain", graph_name=TERRAIN,
                                  type_id="TerrainV2.RoadFieldsOutput")
        client.call("nodes.set_property", graph_type="terrain", graph_name=TERRAIN,
                    node_id=curve_input, property="splineObject", value=river_name)
        client.call("nodes.set_property", graph_type="terrain", graph_name=TERRAIN,
                    node_id=curve_mask, property="widthMeters", value=20.0)
        client.call("nodes.set_property", graph_type="terrain", graph_name=TERRAIN,
                    node_id=curve_mask, property="falloffMeters", value=1.0)
        client.call("nodes.set_property", graph_type="terrain", graph_name=TERRAIN,
                    node_id=curve_mask, property="usePointWidth", value=False)

        client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                    from_node=height_source["id"], from_output="height",
                    to_node=network, to_input="height")
        client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                    from_node=curve_input, from_output="curve",
                    to_node=curve_mask, to_input="curve")
        client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                    from_node=curve_mask, from_output="mask",
                    to_node=network, to_input="water")
        for key in FIELDS:
            client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                        from_node=network, from_output=key,
                        to_node=road_fields, to_input=key)
        client.call("nodes.link_by_key", graph_type="terrain", graph_name=TERRAIN,
                    from_node=network, from_output="snapshot_revision",
                    to_node=road_fields, to_input="snapshot_revision")

        stage = "bridge crossing"
        route = solve_with_crossing(client, road_name, "bridge", max_samples=40)
        require(route["sample_count"] > 0, route)
        require(route["bridge_samples"] > 0,
                f"declaring bridge produced no bridge samples: {route}")
        # The half that matters: a bridge spans the crossing, it does not replace
        # the whole road. An entirely bridged route means the mode dropped the
        # cut/fill envelope instead of spanning what the ground cannot carry.
        require(route["bridge_samples"] < route["sample_count"],
                f"the whole route was bridged: {route}")
        crossings = [s["crossing"] for s in route["samples"]]
        require(crossings[0] == "terrain" and crossings[-1] == "terrain",
                f"the road ends on a bridge: {crossings}")
        require("bridge" in crossings, crossings)
        bridge_route = route

        stage = "ford crossing"
        route = solve_with_crossing(client, road_name, "ford", max_samples=40)
        require(route["ford_samples"] > 0, f"declaring ford produced no ford: {route}")
        require(route["bridge_samples"] == 0,
                f"a ford must not also bridge: {route}")

        stage = "terrain crossing"
        route = solve_with_crossing(client, road_name, "terrain", max_samples=0)
        require(route["bridge_samples"] == 0 and route["ford_samples"] == 0
                and route["tunnel_samples"] == 0,
                f"terrain mode crossed something: {route}")

        stage = "published fields"
        published = set(client.call("terrain.list_fields", terrain=TERRAIN))
        require("infrastructure.ditch" in published,
                f"the ditch was not published: {sorted(published)}")
        ditch = client.call("terrain.field_stats", terrain=TERRAIN,
                            field="infrastructure.ditch")
        require(ditch["non_finite_count"] == 0 and ditch["nonzero_count"] > 0, ditch)
        require(ditch["max"] <= 1.0 + 1e-5, ditch)

        stage = "surface mesh"
        mesh = client.call("terrain.road.build_mesh", spline=road_name,
                           uv_meters_per_tile=6.0)
        require(mesh["triangle_count"] > 0 and mesh["vertex_count"] > 0, mesh)
        require(mesh["span_count"] >= 1, mesh)
        require(client.call("scene.object_exists", name=mesh["object"]) is True, mesh)
        rebuilt = client.call("terrain.road.build_mesh", spline=road_name,
                              uv_meters_per_tile=6.0)
        # A rebuild that publishes a second object is how repeated generation
        # leaves a stack of stale roads, each one looking correct.
        require(rebuilt["object"] == mesh["object"], (mesh, rebuilt))
        require(rebuilt["replaced_existing"] is True, rebuilt)

        stage = "mesh removal"
        client.call("terrain.road.clear_mesh", spline=road_name)
        require(client.call("scene.object_exists", name=mesh["object"]) is False,
                f"{mesh['object']} survived clear_mesh")
        route = client.call("terrain.road.get_route", spline=road_name)
        require(route["sample_count"] > 0,
                "the terrain-only road did not survive deleting its mesh")

        print("[terrain road crossings IPC] PASS")
        print(f"  bridge={bridge_route['bridge_samples']}/{bridge_route['sample_count']}"
              f" samples  length={bridge_route['length_meters']:.1f} m")
        print(f"  ditch_pixels={ditch['nonzero_count']}  mesh_tris={mesh['triangle_count']}"
              f"  spans={mesh['span_count']}")
        return 0
    except (AssertionError, RtIpcError, KeyError, StopIteration) as exc:
        print(f"[terrain road crossings IPC] FAIL at {stage}: "
              f"{type(exc).__name__}: {exc!r}")
        return 1
    finally:
        for name in (road_name, river_name):
            client.try_call("terrain.road.clear_profile", spline=name)
            client.try_call("scene.delete", name=name)
        if created_terrain:
            ok, result = client.try_call("terrain.remove", name=TERRAIN)
            if not ok:
                print(f"[terrain road crossings IPC] cleanup warning: {result}")
        client.close()


if __name__ == "__main__":
    sys.exit(main())
