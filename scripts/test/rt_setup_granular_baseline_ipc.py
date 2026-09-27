"""Turn the first fluid domain into a wet-sand particle-render baseline."""

from __future__ import annotations

import json

from rt_ipc import RtIpc


def main() -> None:
    client = RtIpc()
    try:
        domains = client.call("fluid.list_domains")["domains"]
        fluid_domains = [row for row in domains if row.get("type") == "fluid"]
        if not fluid_domains:
            raise RuntimeError("no fluid domain is visible")
        name = fluid_domains[0]["name"]
        client.call(
            "fluid.set_param",
            domain=name,
            preset="wet_sand",
            render_mode="particles",
        )
        client.call(
            "fluid.seed",
            domain=name,
            particles_per_cell=4,
            replace=True,
            persistent=True,
        )
        state = client.call("fluid.get", domain=name)
        print(
            json.dumps(
                {
                    "name": name,
                    "preset": state.get("preset"),
                    "granular_enabled": state.get("granular_enabled"),
                    "render_mode": state.get("render_mode"),
                    "backend": state.get("backend"),
                    "particle_count": state.get("particle_count"),
                    "granular_young_modulus": state.get(
                        "granular_young_modulus"
                    ),
                },
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        if not state.get("granular_enabled"):
            raise AssertionError("wet_sand did not enable granular simulation")
    finally:
        client.close()


if __name__ == "__main__":
    main()
