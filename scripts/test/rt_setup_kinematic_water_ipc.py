"""Leave a small seeded water domain around the selected proxy set.

Run from a separate terminal while RayTrophi Studio is open. The script does
not advance the timeline: after it prints READY, play the existing walk in the
application and inspect the local leg/foot wake.
"""

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
        resolved = [sample for sample in samples if sample["resolved"]]
        if not resolved:
            raise RuntimeError("the selected proxy set has no resolved samples")

        minimum = [math.inf, math.inf, math.inf]
        maximum = [-math.inf, -math.inf, -math.inf]
        for sample in resolved:
            points = [sample["center"]]
            if sample["shape"] == "capsule":
                points.extend((sample["capsule_start"], sample["capsule_end"]))
            radius = max(float(sample["radius"]), 0.01)
            for point in points:
                for axis in range(3):
                    minimum[axis] = min(minimum[axis], float(point[axis]) - radius)
                    maximum[axis] = max(maximum[axis], float(point[axis]) + radius)

        height = max(maximum[1] - minimum[1], 0.5)
        width = max(maximum[0] - minimum[0], 0.5)
        depth = max(maximum[2] - minimum[2], 0.5)
        padding = max(0.20, 0.15 * max(width, depth))
        domain_min = [
            minimum[0] - padding,
            minimum[1] - 0.10 * height,
            minimum[2] - padding,
        ]
        domain_max = [
            maximum[0] + padding,
            minimum[1] + 0.65 * height,
            maximum[2] + padding,
        ]
        voxel_size = max(0.025, min(0.10, max(
            domain_max[axis] - domain_min[axis] for axis in range(3)
        ) / 64.0))

        domain_name = f"Kinematic Water {time.time_ns()}"
        client.call(
            "fluid.create_domain",
            name=domain_name,
            type="fluid",
            domain_min=domain_min,
            domain_max=domain_max,
            voxel_size=voxel_size,
        )
        client.call(
            "fluid.set_param",
            domain=domain_name,
            backend="vulkan",
            boundary="closed",
            default_substance="Water",
            enabled=True,
        )
        water_top = minimum[1] + 0.38 * height
        client.call(
            "fluid.seed",
            domain=domain_name,
            seed_min=[
                domain_min[0] + voxel_size,
                domain_min[1] + voxel_size,
                domain_min[2] + voxel_size,
            ],
            seed_max=[
                domain_max[0] - voxel_size,
                min(water_top, domain_max[1] - voxel_size),
                domain_max[2] - voxel_size,
            ],
            particles_per_cell=4,
            replace=True,
            persistent=True,
        )
        consumer_mask = int(selected.get("consumer_mask", 31)) | 1
        client.call(
            "physics.collider.proxy_set.set",
            set_id=selected["id"],
            consumer_mask=consumer_mask,
        )
        print(
            f"READY domain={domain_name!r} set={selected['name']!r} "
            f"bounds={domain_min}..{domain_max} voxel={voxel_size:.5f}. "
            "Play the walk; legs and feet should make local wakes without "
            "leaving blocked ghost cells."
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
