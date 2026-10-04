"""Read-only phase-grid probe after the user's combined C3 build and scene step.

python scripts/test/rt_test_matter_phase_grids_ipc.py "Physics Domain 1" --expect-distinct
"""

import argparse
import json
import math
from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--expect-distinct", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    try:
        grids = checked_call(client, "fluid.get_phase_grids", domain=args.domain)
        assert grids["working_bytes"] == sum(grids[phase]["working_bytes"]
                                             for phase in ("gas", "liquid"))
        if grids["budget_enforced"]:
            assert grids["working_bytes"] <= grids["budget_mb"] * 1024 * 1024
        for phase in ("gas", "liquid"):
            value = grids[phase]
            assert value["present"] and value["measured"], (phase, value)
            dimensions = value["resolution"]
            assert math.prod(dimensions) == value["cells"] > 0
            assert value["voxel"] > 0
            for axis in range(3):
                coverage = value["bounds_min"][axis] + dimensions[axis] * value["voxel"]
                assert abs(coverage - value["bounds_max"][axis]) < 1e-4
                assert coverage >= value["requested_bounds_max"][axis] - 1e-4
        liquid = checked_call(client, "fluid.step_stats", domain=args.domain)
        assert liquid["measured"], liquid
        assert liquid["resolution"] == grids["liquid"]["resolution"]
        assert liquid["full_grid_cells"] == grids["liquid"]["cells"]
        if args.expect_distinct:
            assert (grids["gas"]["bounds_max"] != grids["liquid"]["bounds_max"] or
                    grids["gas"]["voxel"] != grids["liquid"]["voxel"])
        print(json.dumps(grids, indent=2))
        print("PASS: phase geometry and liquid solver telemetry")
    finally:
        client.close()


if __name__ == "__main__":
    main()
