"""Leave a smoke domain around the selected kinematic proxy set."""

from __future__ import annotations

import math
import sys
import time

from rt_ipc import RtIpc


def main() -> None:
    client = RtIpc()
    try:
        requested = sys.argv[1] if len(sys.argv) > 1 else ""
        sets = client.call("physics.collider.proxy_set.list")
        selected = next(
            (row for row in sets if requested in row["name"]),
            sets[0] if sets else None,
        )
        if selected is None:
            raise RuntimeError("no kinematic proxy set")
        samples = client.call(
            "physics.collider.proxy_set.sample",
            set_id=selected["id"],
            dt=1.0 / 60.0,
        )
        centers = [sample["center"] for sample in samples if sample["resolved"]]
        if not centers:
            raise RuntimeError("the selected proxy set has no resolved samples")

        minimum = [min(float(point[a]) for point in centers) for a in range(3)]
        maximum = [max(float(point[a]) for point in centers) for a in range(3)]
        extent = [max(maximum[a] - minimum[a], 0.5) for a in range(3)]
        padding = max(0.25, 0.15 * max(extent))
        domain_min = [minimum[a] - padding for a in range(3)]
        domain_max = [maximum[a] + padding for a in range(3)]
        voxel_size = max(0.035, min(0.12, max(
            domain_max[a] - domain_min[a] for a in range(3)
        ) / 64.0))
        center = [(domain_min[a] + domain_max[a]) * 0.5 for a in range(3)]

        suffix = time.time_ns()
        domain_name = f"Kinematic Smoke {suffix}"
        source_name = f"Kinematic Smoke Source {suffix}"
        client.call(
            "fluid.create_domain",
            name=domain_name,
            type="gas",
            domain_min=domain_min,
            domain_max=domain_max,
            voxel_size=voxel_size,
        )
        client.call(
            "fluid.set_param",
            domain=domain_name,
            backend="vulkan",
            boundary="open",
            enabled=True,
        )
        client.call(
            "flow_source.create",
            name=source_name,
            domain=domain_name,
            source_mode="point",
            position=[center[0], domain_min[1] + 0.25 * extent[1], center[2]],
            radius=max(0.15, 0.18 * max(extent[0], extent[2])),
            density=1.0,
            temperature=310.0,
            fuel=0.0,
            velocity=[0.0, 0.2, 0.0],
        )
        consumer_mask = int(selected.get("consumer_mask", 31)) | 2
        client.call(
            "physics.collider.proxy_set.set",
            set_id=selected["id"],
            consumer_mask=consumer_mask,
        )
        print(
            f"READY domain={domain_name!r} source={source_name!r} "
            f"set={selected['name']!r} bounds={domain_min}..{domain_max} "
            f"voxel={voxel_size:.5f}. Play the walk; smoke should part locally "
            "around moving limbs."
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
