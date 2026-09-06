#!/usr/bin/env python3
"""Live IPC smoke test for terrain.field_stats.

Creates one small temporary terrain, publishes biome analysis fields, measures
one of them, checks refusal paths, and removes the terrain in a finally block.
No existing scene object is read as the fixture or intentionally modified.
"""

import sys
import time

from rt_ipc import RtIpc, RtIpcError


NAME = "__TerrainFieldStatsProbe"


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    client = RtIpc()
    created = False
    try:
        # A stale probe from an interrupted earlier run is safe to remove; the
        # reserved name is never used as an authoring object.
        client.try_call("terrain.remove", name=NAME)
        client.call(
            "terrain.create", name=NAME, resolution=64, mesh_resolution=64,
            size=256.0, height_scale=64.0)
        created = True
        landform = client.call(
            "terrain.apply_preset", name=NAME, preset="snowy_mountain_valley",
            replace_graph=True)
        require(
            landform["wiring_fault_count"] == 0,
            f"landform wiring faults: {landform}")
        biome = client.call(
            "terrain.apply_preset", name=NAME, preset="biome_temperate",
            replace_graph=False)
        fixture_warnings = list(biome.get("wiring_faults", []))

        client.call("terrain.evaluate", name=NAME)
        deadline = time.monotonic() + 30.0
        while True:
            status = client.call("terrain.evaluation_status", name=NAME)
            state = status.get("state")
            if state in ("completed", "idle"):
                break
            if state in ("failed", "cancelled"):
                raise AssertionError(f"terrain evaluation ended as {status}")
            if time.monotonic() >= deadline:
                raise AssertionError(f"terrain evaluation timed out: {status}")
            time.sleep(0.1)

        fields = client.call("terrain.list_fields", terrain=NAME)
        require(fields, "biome preset published no analysis fields")
        summaries = {
            candidate: client.call(
                "terrain.field_stats", terrain=NAME, field=candidate)
            for candidate in fields
        }
        field = next(
            (candidate for candidate in fields
             if not summaries[candidate]["constant"]), None)
        require(field is not None, f"procedural preset published only constant fields: {summaries}")
        stats = client.call(
            "terrain.field_stats", terrain=NAME, field=field,
            histogram_bins=8, samples=[[0, 0], [32, 32], [63, 63]])

        require(stats["width"] == 64 and stats["height"] == 64, stats)
        require(stats["channels"] == 1, stats)
        require(stats["value_count"] == 64 * 64, stats)
        require(
            stats["finite_count"] + stats["non_finite_count"] ==
            stats["value_count"], stats)
        require(stats["non_finite_count"] == 0, stats)
        require(len(stats["histogram"]) == 8, stats)
        require(sum(stats["histogram"]) == stats["finite_count"], stats)
        require(len(stats["samples"]) == 3, stats)
        require(0.0 <= stats["nonzero_fraction"] <= 1.0, stats)
        require(stats["min"] <= stats["mean"] <= stats["max"], stats)

        ok, message = client.try_call(
            "terrain.field_stats", terrain=NAME, field="__missing_field__")
        require(not ok and "field not found" in message, message)
        ok, message = client.try_call(
            "terrain.field_stats", terrain=NAME, field=field, histogram_bins=1)
        require(not ok and "histogram_bins" in message, message)
        ok, message = client.try_call(
            "terrain.field_stats", terrain=NAME, field=field, samples=[[64, 0]])
        require(not ok and "out of range" in message, message)

        descriptor = client.call("agent.describe", method="terrain.field_stats")
        require(descriptor["capability"] == "Read", descriptor)
        require(descriptor["documented"] is True, descriptor)

        print("[terrain.field_stats IPC] PASS")
        print(f"  field={field!r} fields={len(fields)}")
        print(
            "  min={:.6g} max={:.6g} mean={:.6g} nonzero={:.3%}".format(
                stats["min"], stats["max"], stats["mean"],
                stats["nonzero_fraction"]))
        print(f"  histogram={stats['histogram']}")
        for warning in fixture_warnings:
            print(f"  fixture warning: {warning}")
        return 0
    except (AssertionError, RtIpcError) as exc:
        print(f"[terrain.field_stats IPC] FAIL: {exc}")
        return 1
    finally:
        if created:
            ok, result = client.try_call("terrain.remove", name=NAME)
            if not ok:
                print(f"[terrain.field_stats IPC] cleanup warning: {result}")
        client.close()


if __name__ == "__main__":
    sys.exit(main())
