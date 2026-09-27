"""Measure a live granular domain before virtualized rendering changes."""

from __future__ import annotations

import json
import sys
import time

from rt_ipc import RtIpc


SECTIONS = (
    "sim.timeline.step",
    "sim.timeline.capture_frame",
    "render.fluid.splat_instances",
    "loop.viewport_render",
    "sim.fluid.voxelize_colliders",
)


def perf_snapshot(client):
    return {
        row["name"]: (int(row["count"]), float(row["total_ms"]))
        for row in client.call("perf.list")
    }


def main() -> None:
    frames = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    client = RtIpc()
    try:
        domains = client.call("fluid.list_domains")["domains"]
        granular = [row for row in domains if row.get("granular_enabled")]
        if not granular:
            inventory = [
                {
                    "name": row.get("name"),
                    "type": row.get("type"),
                    "preset": row.get("preset"),
                    "granular_enabled": row.get("granular_enabled"),
                    "particle_count": row.get("particle_count"),
                    "render_mode": row.get("render_mode"),
                    "backend": row.get("backend"),
                }
                for row in domains
            ]
            raise RuntimeError(
                "no granular-enabled fluid domain is visible; inventory="
                + json.dumps(inventory, ensure_ascii=False)
            )
        domain = granular[0]
        name = domain["name"]
        control_before = client.call("sim.control_state")
        perf_before = perf_snapshot(client)
        wall_start = time.perf_counter()
        start = int(control_before["frame"])
        for frame in range(start + 1, start + frames + 1):
            client.call("timeline.set_frame", frame=frame)
        wall_ms = (time.perf_counter() - wall_start) * 1000.0
        perf_after = perf_snapshot(client)
        state = client.call("fluid.get", domain=name)
        step_stats = client.call("fluid.step_stats", domain=name)
        control_after = client.call("sim.control_state")

        print(
            json.dumps(
                {
                    "domain": name,
                    "frame": [control_before["frame"], control_after["frame"]],
                    "frames_requested": frames,
                    "wall_ms": wall_ms,
                    "particle_count": state.get("particle_count"),
                    "max_particles": state.get("max_particles"),
                    "render_mode": state.get("render_mode"),
                    "backend": state.get("backend"),
                    "voxel_size": state.get("voxel_size"),
                    "granular_solver_substeps": state.get(
                        "granular_solver_substeps"
                    ),
                    "granular_required_substeps": state.get(
                        "granular_required_substeps"
                    ),
                    "step_stats": step_stats,
                    "perf_delta": {
                        section: {
                            "calls": perf_after.get(section, (0, 0.0))[0]
                            - perf_before.get(section, (0, 0.0))[0],
                            "total_ms": perf_after.get(section, (0, 0.0))[1]
                            - perf_before.get(section, (0, 0.0))[1],
                        }
                        for section in SECTIONS
                    },
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()
