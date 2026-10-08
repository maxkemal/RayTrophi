"""Turn the first fluid domain into a wet-sand particle-render baseline."""

from __future__ import annotations

import json

from rt_ipc import RtIpc
from rt_domain_material import domain_material


def main() -> None:
    client = RtIpc()
    try:
        domains = client.call("fluid.list_domains")["domains"]
        fluid_domains = [row for row in domains if row.get("type") == "fluid"]
        if not fluid_domains:
            raise RuntimeError("no fluid domain is visible")
        name = fluid_domains[0]["name"]
        # Wet sand is sand plus capillary cohesion: a derived substance with the
        # removed Wet Sand preset's values (2026-10-07).
        material = domain_material(
            client.call, "T: wet sand baseline", "Sand",
            granular_friction_angle=37.0, granular_cohesion=1500.0,
            granular_dilatancy=6.0, granular_young_modulus=2.5e5,
            granular_tensile_cutoff=400.0, granular_fracture_strain=0.010,
            granular_damage_rate=14.0, granular_healing_rate=0.5,
            granular_rebonding=True)
        client.call(
            "fluid.set_param",
            domain=name,
            default_substance=material,
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
                    "default_substance": state.get("default_substance"),
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
