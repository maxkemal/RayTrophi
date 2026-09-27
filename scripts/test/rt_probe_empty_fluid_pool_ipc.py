"""A/B probe for a retained fluid splat pool after particles are cleared.

Temporarily changes an empty particle-rendered domain to the surface route so
the transient splat group is released, then restores the authored render mode.
Run from an external terminal with the app open; never run this through the
embedded script workspace.
"""

from __future__ import annotations

import json
import time

from rt_ipc import RtIpc


TELEMETRY_KEYS = (
    "frame_ms",
    "total_instances",
    "full_instances",
    "proxy_instances",
    "visible_triangles",
    "full_triangles",
    "proxy_triangles",
    "draw_calls",
)


def selected_telemetry(client: RtIpc) -> dict:
    telemetry = client.call("viewport.frame_telemetry")
    return {key: telemetry.get(key) for key in TELEMETRY_KEYS}


def main() -> None:
    client = RtIpc()
    domain_name = ""
    original_render_mode = "particles"
    try:
        domains = client.call("fluid.list_domains").get("domains", [])
        empty_domains = [row for row in domains if row.get("particle_count") == 0]
        if not empty_domains:
            raise RuntimeError("no empty fluid domain is available for the pool probe")

        domain = empty_domains[0]
        domain_name = domain["name"]
        original_render_mode = str(domain.get("render_mode", "particles"))
        before = selected_telemetry(client)

        client.call("fluid.set_param", domain=domain_name, render_mode="surface")
        time.sleep(1.0)
        surface = selected_telemetry(client)
    finally:
        if domain_name:
            client.call(
                "fluid.set_param",
                domain=domain_name,
                render_mode=original_render_mode,
            )
            time.sleep(1.0)

    restored = selected_telemetry(client)
    state = client.call("fluid.get", domain=domain_name)
    client.close()
    print(
        json.dumps(
            {
                "domain": domain_name,
                "particle_count": state.get("particle_count"),
                "render_mode_restored": state.get("render_mode"),
                "before": before,
                "surface": surface,
                "restored": restored,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
