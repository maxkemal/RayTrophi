"""Drive, rewind, and replay a live kinematic-fluid scene over IPC."""

from __future__ import annotations

import math
import sys

from rt_ipc import RtIpc


PERF_NAME = "sim.fluid.voxelize_colliders"


def perf_value(client):
    for row in client.call("perf.list"):
        if row["name"] == PERF_NAME:
            return int(row["count"]), float(row["total_ms"])
    return 0, 0.0


def drive(client, first, last):
    for frame in range(first, last + 1):
        client.call("timeline.set_frame", frame=frame)


def finite_vector(value):
    return len(value) == 3 and all(math.isfinite(float(component)) for component in value)


def main() -> None:
    client = RtIpc()
    try:
        set_name = sys.argv[1] if len(sys.argv) > 1 else None
        sets = client.call("physics.collider.proxy_set.list")
        selected = next(
            (row for row in sets if set_name is None or row["name"] == set_name),
            None,
        )
        if selected is None:
            raise RuntimeError(f"kinematic set not found: {set_name!r}")
        domains = client.call("fluid.list_domains")["domains"]
        if not domains:
            raise RuntimeError("no fluid domain is visible")
        domain_name = domains[0]["name"]
        set_id = selected["id"]

        initial = client.call("sim.control_state")
        start_frame = int(initial["frame"])
        particles_initial = client.call("fluid.get", domain=domain_name)[
            "particle_count"
        ]
        count0, ms0 = perf_value(client)

        drive(client, start_frame + 1, start_frame + 6)
        forward = client.call("sim.control_state")
        count_forward, ms_forward = perf_value(client)
        particles_forward = client.call("fluid.get", domain=domain_name)[
            "particle_count"
        ]

        client.call("timeline.set_frame", frame=0)
        rewound = client.call("sim.control_state")
        count_rewound, ms_rewound = perf_value(client)
        particles_rewound = client.call("fluid.get", domain=domain_name)[
            "particle_count"
        ]
        drive(client, 1, 6)

        replayed = client.call("sim.control_state")
        particles_replayed = client.call("fluid.get", domain=domain_name)[
            "particle_count"
        ]
        count1, ms1 = perf_value(client)
        samples = client.call(
            "physics.collider.proxy_set.sample", set_id=set_id, dt=1.0 / 60.0
        )
        invalid = [
            row["bone"]
            for row in samples
            if not row["resolved"]
            or not finite_vector(row["center"])
            or not finite_vector(row["linear_velocity"])
            or not finite_vector(row["angular_velocity"])
        ]

        print(
            f"frames={initial['frame']}->{forward['frame']}->"
            f"{rewound['frame']}->{replayed['frame']}"
        )
        print(
            f"epochs={initial['epoch']}->{forward['epoch']}->"
            f"{rewound['epoch']}->{replayed['epoch']}"
        )
        print(
            f"particles={particles_initial}->{particles_forward}->"
            f"{particles_rewound}->{particles_replayed}"
        )
        print(
            f"dropped_seeds_after_rewind={rewound.get('dropped_seeds', [])} "
            f"after_replay={replayed.get('dropped_seeds', [])}"
        )
        forward_calls = count_forward - count0
        replay_calls = count1 - count_rewound
        forward_ms = ms_forward - ms0
        replay_ms = ms1 - ms_rewound
        print(
            f"voxelize_forward={forward_calls} calls/{forward_ms:.3f} ms "
            f"replay={replay_calls} calls/{replay_ms:.3f} ms "
            f"invalid_samples={invalid}"
        )

        if int(forward["frame"]) != start_frame + 6:
            raise AssertionError("forward phase did not reach its target frame")
        if int(rewound["frame"]) != 0 or int(replayed["frame"]) != 6:
            raise AssertionError("rewind/replay did not reach target frames")
        if rewound.get("dropped_seeds") or replayed.get("dropped_seeds"):
            raise AssertionError("rewind dropped a non-persistent fluid seed")
        if particles_rewound <= 0 or particles_replayed <= 0:
            raise AssertionError("fluid particles were not restored after rewind")
        if invalid:
            raise AssertionError("nonfinite or unresolved proxy after replay")
        if forward_calls < 6 or replay_calls < 6:
            raise AssertionError("not every driven frame voxelized colliders")
        print("PASS rewind restored fluid and replayed finite kinematic colliders")
    finally:
        client.close()


if __name__ == "__main__":
    main()
