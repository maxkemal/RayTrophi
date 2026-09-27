"""Observe a playing kinematic rig and fluid without changing the scene."""

from __future__ import annotations

import math
import sys
import time

from rt_ipc import RtIpc


def distance(left, right):
    return math.sqrt(
        sum((float(left[axis]) - float(right[axis])) ** 2 for axis in range(3))
    )


def perf_snapshot(client):
    return {
        row["name"]: (int(row["count"]), float(row["total_ms"]))
        for row in client.call("perf.list")
    }


def main() -> None:
    client = RtIpc()
    try:
        requested_name = sys.argv[1] if len(sys.argv) > 1 else None
        drive_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        sets = client.call("physics.collider.proxy_set.list")
        selected = next(
            (
                row
                for row in sets
                if requested_name is None or row["name"] == requested_name
            ),
            None,
        )
        if selected is None:
            raise RuntimeError(f"kinematic set not found: {requested_name!r}")
        domains = client.call("fluid.list_domains")["domains"]
        if not domains:
            raise RuntimeError("no fluid domain is visible")
        domain_name = domains[0]["name"]
        set_id = selected["id"]

        control_before = client.call("sim.control_state")
        fluid_before = client.call("fluid.get", domain=domain_name)
        perf_before = perf_snapshot(client)
        samples_before = client.call(
            "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
        )
        if drive_frames > 0:
            start_frame = int(control_before["frame"])
            for offset in range(1, drive_frames + 1):
                client.call("timeline.set_frame", frame=start_frame + offset)
        else:
            time.sleep(1.0)
        samples_after = client.call(
            "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
        )
        perf_after = perf_snapshot(client)
        fluid_after = client.call("fluid.get", domain=domain_name)
        control_after = client.call("sim.control_state")

        before_by_id = {row["proxy_id"]: row for row in samples_before}
        motion = []
        for row in samples_after:
            previous = before_by_id.get(row["proxy_id"])
            if previous and previous["resolved"] and row["resolved"]:
                motion.append((distance(previous["center"], row["center"]), row))
        motion.sort(key=lambda item: item[0], reverse=True)
        feet = [
            (delta, row["bone"], row["center"])
            for delta, row in motion
            if "foot" in row["bone"].lower() or "toe" in row["bone"].lower()
        ]

        perf_name = "sim.fluid.voxelize_colliders"
        count_before, ms_before = perf_before.get(perf_name, (0, 0.0))
        count_after, ms_after = perf_after.get(perf_name, (0, 0.0))
        print(
            f"frame={control_before['frame']}->{control_after['frame']} "
            f"epoch={control_before['epoch']}->{control_after['epoch']} "
            f"driver={control_after['driver']!r}"
        )
        print(
            f"particles={fluid_before['particle_count']}->"
            f"{fluid_after['particle_count']} "
            f"uvw_drift={fluid_before['uvw_drift']:.6f}->"
            f"{fluid_after['uvw_drift']:.6f}"
        )
        print(
            f"voxelize_calls={count_after - count_before} "
            f"voxelize_ms={ms_after - ms_before:.3f}"
        )
        print(f"largest_motion={[(delta, row['bone']) for delta, row in motion[:8]]}")
        print(f"foot_motion={feet}")
        unresolved = [row for row in samples_after if not row["resolved"]]
        if unresolved:
            raise AssertionError(f"unresolved proxies: {unresolved!r}")
        if drive_frames > 0 and control_after["frame"] <= control_before["frame"]:
            raise AssertionError("timeline did not advance during the observation")
        if not motion or motion[0][0] <= 1.0e-5:
            raise AssertionError("kinematic proxies did not follow the rig")
        if drive_frames > 0 and count_after <= count_before:
            raise AssertionError("fluid collider voxelization did not run")
        if drive_frames > 0:
            print("PASS live rig motion and fluid collider voxelization were observed")
        else:
            print("PASS live rig motion observed; solver timeline was not driven")
    finally:
        client.close()


if __name__ == "__main__":
    main()
