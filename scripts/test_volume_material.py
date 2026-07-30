"""RayTrophi volume-material smoke/fire preset test.

Run from RayTrophi's embedded Python console:
    exec(open("scripts/test_volume_material.py", encoding="utf-8").read())

This creates a small gas-backed volume container and configures the production
volume/combustion controls independently from the gas-solver test.
"""

import rt


DOMAIN = "Script_VolumeMaterial_Test"


def remove_if_present(name):
    try:
        rt.gas.remove_domain(name)
    except RuntimeError:
        pass


remove_if_present(DOMAIN)

created = rt.gas.create_domain(
    name=DOMAIN,
    domain_min=(-1.5, 0.0, -1.5),
    domain_max=(1.5, 3.0, 1.5),
    voxel_size=0.075,
)

# General domain/render controls.
rt.gas.set_param(
    DOMAIN,
    backend="vulkan",
    boundary="open",
    render_mode="volume",
    enabled=True,
    visible=True,
)

# Fire-oriented volume preset. These values exercise density/temperature/fuel
# rendering, blackbody emission and multiple-scattering-friendly turbulence.
rt.gas.set_settings(
    DOMAIN,
    quality_profile="preview",
    resource_budget_mb=1024,
    enforce_resource_budget=True,
    use_sparse_tiles=True,
    render_to_nanovdb=True,
    fire_enabled=True,
    ignition_temperature=0.30,
    burn_rate=1.6,
    heat_release=2.4,
    smoke_generation=0.75,
    flame_dissipation=2.5,
    fire_max_temperature=10.0,
    buoyancy_heat=1.15,
    buoyancy_density=0.06,
    vorticity=0.45,
    fire_expansion=0.15,
    turbulence_strength=0.35,
    turbulence_scale=1.4,
    turbulence_octaves=4,
    turbulence_lacunarity=2.0,
    turbulence_persistence=0.5,
    turbulence_speed=0.6,
)

info = rt.gas.get(DOMAIN)
material = rt.gas.get_settings(DOMAIN)

# rt 0.5.0's get() response omitted "type"; create_domain() already reports
# the authoritative type, so retain compatibility with that binary.
domain_type = info.get("type", created.get("type"))
assert domain_type == "gas", {"created": created, "info": info}
assert info["backend"] == "vulkan", info
assert material["fire_enabled"] is True, material
assert material["render_to_nanovdb"] is True, material

print("[volume-material-test] domain:", info)
print("[volume-material-test] settings:", material)
print("[volume-material-test] PASS")
print("Not: Görünür alev/duman için domain'e UI'dan fuel/temperature flow source ekleyin.")
