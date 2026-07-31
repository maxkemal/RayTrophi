"""Vulkan APIC liquid <-> Gas combustion coupling test preset.

Run from RayTrophi's embedded Python console after compiling the shaders:
    exec(open("scripts/test_burning_fuel_spill.py", encoding="utf-8").read())

The preset intentionally uses overlapping domains. The liquid surface owns a
finite fuel reservoir; its exposed cells inject fuel/heat/smoke into the gas
grid, while gas temperature feeds ignition back into the liquid surface state.
"""

import rt


FLUID = "Test_BurningFuel_Fluid"
GAS = "Test_BurningFuel_Gas"


def remove_if_present(name):
    try:
        rt.fluid.remove_domain(name)
    except RuntimeError:
        pass


remove_if_present(FLUID)
remove_if_present(GAS)

rt.fluid.create_domain(
    FLUID,
    domain_min=(-2.5, 0.0, -2.5),
    domain_max=(2.5, 1.8, 2.5),
    voxel_size=0.10,
    type="fluid",
)
rt.fluid.set_param(
    FLUID,
    backend="vulkan",
    boundary="closed",
    preset="oil",
    render_mode="surface",
)
rt.fluid.seed(
    FLUID,
    seed_min=(-1.8, 0.15, -1.8),
    seed_max=(1.8, 0.55, 1.8),
    particles_per_cell=6,
    replace=True,
)
rt.fluid.set_combustion(
    FLUID,
    enabled=True,
    auto_ignite=True,
    ignition_temperature=0.65,
    evaporation_rate=0.45,
    surface_fuel_capacity=5.0,
    heat_release=2.4,
    smoke_yield=0.55,
    surface_cooling=0.30,
)

rt.gas.create_domain(
    GAS,
    domain_min=(-2.5, 0.0, -2.5),
    domain_max=(2.5, 5.0, 2.5),
    voxel_size=0.10,
)
rt.gas.set_param(
    GAS,
    backend="vulkan",
    boundary="open",
    render_mode="volume",
)
rt.gas.set_settings(
    GAS,
    fire_enabled=True,
    ignition_temperature=0.30,
    burn_rate=1.35,
    heat_release=2.2,
    smoke_generation=0.65,
    flame_dissipation=2.6,
    buoyancy_heat=1.15,
    buoyancy_density=0.06,
    vorticity=0.42,
    fire_expansion=0.12,
    turbulence_strength=0.28,
    turbulence_scale=1.35,
    turbulence_octaves=4,
)

coupling = rt.fluid.get_combustion(FLUID)
assert coupling["enabled"] is True, coupling
assert coupling["auto_ignite"] is True, coupling
assert rt.fluid.get(FLUID)["backend"] == "vulkan"
assert rt.gas.get(GAS)["backend"] == "vulkan"

print("[burning-fuel-spill] fluid:", rt.fluid.get(FLUID))
print("[burning-fuel-spill] coupling:", coupling)
print("[burning-fuel-spill] gas:", rt.gas.get_settings(GAS))
print("[burning-fuel-spill] PASS - Play timeline to inspect surface fire.")
