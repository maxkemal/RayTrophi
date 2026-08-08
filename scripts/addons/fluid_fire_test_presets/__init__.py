"""Fluid/Gas combustion validation presets.

Install/run from RayTrophi's embedded Python console:
    exec(open("scripts/addons/fluid_fire_test_presets/__init__.py", encoding="utf-8").read())
    register()

The recipes intentionally use the public rt API so they also serve as IPC/API
Explorer smoke tests. They create fresh, clearly named domains and sources.
"""

import rt

bl_info = {
    "name": "Fluid Fire Test Presets",
    "description": "Vulkan liquid/gas chemistry and extinguishing test scenes.",
    "version": (1, 0, 0),
}

_panel_id = None


def _remove_fluid(name):
    try:
        rt.fluid.remove_domain(name)
    except RuntimeError:
        pass


def _remove_gas(name):
    try:
        rt.gas.remove_domain(name)
    except RuntimeError:
        pass


def _remove_source(name):
    try:
        rt.flow_source.remove(name)
    except RuntimeError:
        pass


def _clear_recipe(fluid, gas, sources):
    for source in sources:
        _remove_source(source)
    _remove_fluid(fluid)
    _remove_gas(gas)


def _make_pair(fluid, gas, chemistry, fluid_min=(-2.5, 0.0, -2.5),
               fluid_max=(2.5, 1.8, 2.5), gas_max=(2.5, 5.0, 2.5)):
    rt.fluid.create_domain(
        fluid, domain_min=fluid_min, domain_max=fluid_max,
        voxel_size=0.10, type="fluid",
    )
    rt.fluid.set_param(
        fluid, backend="vulkan", boundary="closed",
        preset="oil", render_mode="surface",
    )
    rt.fluid.seed(
        fluid, seed_min=(-1.8, 0.15, -1.8),
        seed_max=(1.8, 0.55, 1.8),
        particles_per_cell=6, replace=True,
    )
    rt.fluid.set_combustion(
        fluid,
        chemistry_preset=chemistry,
        enabled=True,
        auto_ignite=False,
        surface_fuel_capacity=5.0,
    )

    rt.gas.create_domain(
        gas, domain_min=fluid_min, domain_max=gas_max, voxel_size=0.10,
    )
    rt.gas.set_param(gas, backend="vulkan", boundary="open", render_mode="volume")
    rt.gas.set_settings(
        gas, fire_enabled=True, ignition_temperature=0.30,
        burn_rate=1.35, heat_release=2.2, smoke_generation=0.65,
        flame_dissipation=2.6, buoyancy_heat=1.15,
        buoyancy_density=0.06, vorticity=0.42,
        fire_expansion=0.12, turbulence_strength=0.28,
        turbulence_scale=1.35, turbulence_octaves=4,
    )


def _add_pilot(gas, name, position=(1.35, 0.28, 0.0), temperature=5.0):
    rt.flow_source.create(
        name, gas, source_mode="point", position=position,
        velocity=(0.0, 0.85, 0.0), radius=0.55,
        velocity_coupling=12.0, density=0.16,
        temperature=temperature, fuel=2.2, falloff=1.35,
        use_time_limit=True, start_time=0.45, end_time=4.0,
    )


def add_gasoline_flash():
    fluid = "Addon Gasoline Flash Liquid"
    gas = "Addon Gasoline Flash Gas"
    pilot = "Addon Gasoline Flash Pilot"
    _clear_recipe(fluid, gas, [pilot])
    _make_pair(fluid, gas, "gasoline")
    _add_pilot(gas, pilot, temperature=6.0)
    rt.fluid.reset()
    rt.request_render()
    print("[fluid_fire_test_presets] Added Gasoline Flash")


def add_alcohol_flash():
    fluid = "Addon Alcohol Flash Liquid"
    gas = "Addon Alcohol Flash Gas"
    pilot = "Addon Alcohol Flash Pilot"
    _clear_recipe(fluid, gas, [pilot])
    _make_pair(fluid, gas, "alcohol")
    _add_pilot(gas, pilot, temperature=7.0)
    rt.fluid.reset()
    rt.request_render()
    print("[fluid_fire_test_presets] Added Alcohol Flash")


def add_oil_pool_fire():
    fluid = "Addon Oil Pool Liquid"
    gas = "Addon Oil Pool Gas"
    pilot = "Addon Oil Pool Pilot"
    _clear_recipe(fluid, gas, [pilot])
    _make_pair(fluid, gas, "oil")
    _add_pilot(gas, pilot, temperature=5.0)
    rt.fluid.reset()
    rt.request_render()
    print("[fluid_fire_test_presets] Added Oil Pool Fire")


def add_water_extinguish():
    fluid = "Addon Water Extinguish Liquid"
    gas = "Addon Water Extinguish Gas"
    pilot = "Addon Water Extinguish Pilot"
    _clear_recipe(fluid, gas, [pilot])
    _make_pair(fluid, gas, "water")
    _add_pilot(gas, pilot, temperature=6.0)
    rt.fluid.reset()
    rt.request_render()
    print("[fluid_fire_test_presets] Added Water Extinguish")


def _draw_panel():
    rt.ui.text_disabled("Vulkan liquid/gas combustion validation")
    if rt.ui.button("Gasoline Flash", width=-1.0):
        add_gasoline_flash()
    rt.ui.tooltip("Fast vaporization, pilot ignition, surface fuel depletion.")
    if rt.ui.button("Alcohol Flash", width=-1.0):
        add_alcohol_flash()
    rt.ui.tooltip("Very fast vaporization and short flame persistence.")
    if rt.ui.button("Oil Pool Fire", width=-1.0):
        add_oil_pool_fire()
    rt.ui.tooltip("Slow vaporization and persistent surface combustion.")
    if rt.ui.button("Water Extinguish", width=-1.0):
        add_water_extinguish()
    rt.ui.tooltip("Water-like chemistry cooling and quenching an overlapping gas fire.")


def register():
    global _panel_id
    if _panel_id is None:
        _panel_id = rt.ui.register_region(
            "properties.simulation", "Fluid Fire Tests", _draw_panel)
    print("[fluid_fire_test_presets] registered")


def unregister():
    global _panel_id
    if _panel_id is not None:
        rt.ui.unregister_panel(_panel_id)
        _panel_id = None
    print("[fluid_fire_test_presets] unregistered")


register()
