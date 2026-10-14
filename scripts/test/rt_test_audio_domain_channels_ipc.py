"""Read-only channel acceptance; run outside Studio on the user's new build.

Manually enable Audio > Scene > Collect physics audio telemetry first.
Run during live simulation and again during forward RAM cache replay.
This probe does not drive playback, create sources, or establish audible acceptance.
"""
import argparse
import json
import math

from rt_ipc import RtIpc


def check(rows, require_dem=False):
    expected = {"mixed_contact", "dem_contact", "dem_friction", "dem_rolling",
                "fluid_flow", "gas_flow", "mpm_deformation", "mpm_fracture"}
    groups = {}
    keys = set()
    for row in rows:
        identity = (row["system_id"], row["domain_name"])
        channel = row["domain_channel"]
        group = groups.setdefault(identity, {})
        assert channel in expected and channel not in group, row
        group[channel] = row
        assert row["signal_key"] not in keys, row
        keys.add(row["signal_key"])
        if row["measured"]:
            duration = row["measurement_duration_seconds"]
            energy = row["contact_energy_j"]
            assert math.isfinite(duration) and duration > 0, row
            assert math.isfinite(energy) and energy >= 0, row
            if row["energy_measured"]:
                assert row["power_measured"], row
                assert math.isclose(row["contact_power_w"], energy / duration, rel_tol=1e-6), row
            else:
                assert energy == 0, row
            assert math.isfinite(row["value"]) and row["value"] >= 0, row
            if channel == "mpm_fracture":
                assert row["value_units"] == "1/s" and not row["power_measured"], row
            else:
                assert row["value_units"] == "W" and row["power_measured"], row
            assert len(row["position"]) == 3 and all(map(math.isfinite, row["position"])), row
        if channel == "dem_contact" and row["measured"]:
            assert row["measurement"] == "dem_normal_damping_work_per_published_step", row
            assert row["position_semantics"] == (
                "energy_weighted_dem_contact_center" if row["contact_energy_j"] > 0
                else "domain_center"), row
    assert groups, "No initialized fixture/domain to inspect"
    for group in groups.values():
        assert set(group) == expected, group
    if require_dem:
        assert any(group["dem_contact"]["measured"] for group in groups.values()), (
            "No measured DEM report; simulate/cache a grain frame first")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", help="Restrict acceptance to this domain name")
    parser.add_argument("--require-dem", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    try:
        rows = client.call("audio.domains")
        if args.domain:
            rows = [row for row in rows if row["domain_name"] == args.domain]
        check(rows, args.require_dem)
        print(json.dumps({"passed": True, "inputs": rows}, indent=2))
    finally:
        client.close()


if __name__ == "__main__":
    main()
