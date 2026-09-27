"""Measure what the solver stamps for a rig's feet, step by step.

Usage: rt_probe_kinematic_foot_stamps_ipc.py [set_name] [frames] [--refit]

--refit re-runs auto_fit on the set first (REPLACES its proxies), so the foot
boxes come from the skinned mesh. Without it the scene is only stepped.

Reads physics.collider.proxy_set.solver_stamps after each stepped frame: the
cell counts and velocities are the ones the solver wrote, not a re-sample.
"""

from __future__ import annotations

import math
import sys
import time

from rt_ipc import RtIpc

FOOT_TOKENS = ("foot", "toe")


def is_foot(name: str) -> bool:
    key = name.lower().replace("_", "").replace(" ", "")
    return any(token in key for token in FOOT_TOKENS)


def voxelize_count(client) -> int:
    for row in client.call("perf.list"):
        if row["name"] == "sim.fluid.voxelize_colliders":
            return int(row["count"])
    return 0


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    refit = "--refit" in sys.argv
    requested = args[0] if args else None
    frames = int(args[1]) if len(args) > 1 else 24
    client = RtIpc()
    try:
        sets = client.call("physics.collider.proxy_set.list")
        selected = next(
            (s for s in sets if requested is None or s["name"] == requested), None
        )
        if selected is None:
            raise RuntimeError(f"kinematic set not found: {requested!r}")
        set_id = int(selected["id"])
        if refit:
            print("auto_fit:", client.call(
                "physics.collider.proxy_set.auto_fit", set_id=set_id))
        detail = client.call("physics.collider.proxy_set.get", set_id=set_id)
        print("foot proxies (bone-local):")
        for proxy in detail["proxies"]:
            if is_foot(proxy["name"]):
                print(f"  {proxy['name']}: shape={proxy['shape']} "
                      f"local_position={[round(v, 4) for v in proxy['local_position']]} "
                      f"half_extents={[round(v, 4) for v in proxy['half_extents']]}")

        cells: dict[str, list[int]] = {}
        speeds: dict[str, list[float]] = {}
        stepped = 0
        start = int(client.call("sim.control_state")["frame"])
        for frame in range(start + 1, start + frames + 1):
            before = voxelize_count(client)
            client.call("timeline.set_frame", frame=frame)
            deadline = time.time() + 4.0
            while voxelize_count(client) == before and time.time() < deadline:
                time.sleep(0.02)
            if voxelize_count(client) == before:
                print(f"f{frame}: solver did not step")
                continue
            stepped += 1
            line = []
            for system in client.call("physics.collider.proxy_set.solver_stamps"):
                for stamp in system["stamps"]:
                    if stamp["set_id"] != set_id or not is_foot(stamp["proxy_name"]):
                        continue
                    speed = math.sqrt(sum(v * v for v in stamp["linear_velocity"]))
                    key = f"{stamp['domain']}/{stamp['proxy_name'].split(':')[-1]}"
                    cells.setdefault(key, []).append(int(stamp["stamped_cells"]))
                    speeds.setdefault(key, []).append(speed)
                    line.append(f"{key.split('/')[-1]}:{stamp['stamped_cells']}c "
                                f"{speed:.2f}m/s")
            print(f"f{frame} " + "  ".join(line))
        print(f"frames stepped by the solver: {stepped}/{frames}")
        for key in sorted(cells):
            c = cells[key]
            s = sorted(speeds[key])
            zero = sum(1 for v in c if v == 0)
            print(f"{key}: cells mean {sum(c) / len(c):.1f} max {max(c)} "
                  f"zero-frames {zero}/{len(c)}; solver speed median "
                  f"{s[len(s) // 2]:.2f} max {s[-1]:.2f} m/s")
    finally:
        client.close()


if __name__ == "__main__":
    main()
