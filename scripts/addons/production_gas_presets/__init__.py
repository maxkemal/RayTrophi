"""Production gas/fluid presets built entirely through RayTrophi's public API."""

import rt

bl_info = {
    "name": "Production Gas & Fluid Presets",
    "description": "Script-authored Vulkan gas/fluid setups.",
    "version": (1, 0, 0),
}

_panel_id = None


def _remove_domain(name):
    try:
        rt.fluid.remove_domain(name)
    except RuntimeError:
        pass


def _remove_collider(name):
    try:
        rt.collider.remove(name)
    except RuntimeError:
        pass


def add_burning_fuel_spill():
    """Create the proven finite oil-surface combustion reference setup."""
    fluid_name = "Addon Burning Fuel Liquid"
    gas_name = "Addon Burning Fuel Gas"
    _remove_domain(fluid_name)
    _remove_domain(gas_name)

    rt.fluid.create_domain(
        fluid_name,
        domain_min=(-2.5, 0.0, -2.5),
        domain_max=(2.5, 1.8, 2.5),
        voxel_size=0.10,
        type="fluid",
    )
    rt.fluid.set_param(
        fluid_name, backend="vulkan", boundary="closed",
        preset="oil", render_mode="surface",
    )
    rt.fluid.seed(
        fluid_name,
        seed_min=(-1.8, 0.15, -1.8),
        seed_max=(1.8, 0.55, 1.8),
        particles_per_cell=6,
        replace=True,
    )
    rt.fluid.set_combustion(
        fluid_name,
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
        gas_name,
        domain_min=(-2.5, 0.0, -2.5),
        domain_max=(2.5, 5.0, 2.5),
        voxel_size=0.10,
    )
    rt.gas.set_param(
        gas_name, backend="vulkan", boundary="open", render_mode="volume",
    )
    rt.gas.set_settings(
        gas_name,
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
    rt.fluid.reset()
    rt.request_render()
    print("[production_gas_presets] Added Burning Fuel Spill")


def add_api_routing_test():
    """Small, cheap scene proving domain routing + shared collider authoring."""
    fluid_name = "Addon Routing Fluid"
    gas_name = "Addon Routing Gas"
    floor_name = "Addon Routing Floor"
    for source_name in ("Addon Liquid Flow", "Addon Gas Pilot"):
        try:
            rt.flow_source.remove(source_name)
        except RuntimeError:
            pass
    _remove_collider(floor_name)
    _remove_domain(fluid_name)
    _remove_domain(gas_name)

    rt.fluid.create_domain(
        fluid_name, domain_min=(-2.0, -0.3, -1.5),
        domain_max=(2.0, 2.5, 1.5), voxel_size=0.12, type="fluid",
    )
    rt.fluid.set_param(
        fluid_name, backend="vulkan", boundary="open",
        preset="oil", render_mode="surface",
    )
    rt.gas.create_domain(
        gas_name, domain_min=(-2.0, -0.3, -1.5),
        domain_max=(2.0, 3.5, 1.5), voxel_size=0.12,
    )
    rt.gas.set_param(gas_name, backend="vulkan", boundary="open")
    rt.gas.set_settings(gas_name, fire_enabled=True)

    rt.collider.create(
        floor_name, source_mode="plane", plane_y=0.0,
        thickness=0.12, friction=0.18, restitution=0.02,
    )
    rt.flow_source.create(
        "Addon Liquid Flow", fluid_name,
        source_mode="point", position=(-1.1, 1.8, 0.0),
        velocity=(1.4, -2.2, 0.0), radius=0.14,
        fluid_particles_per_second=160.0,
        fluid_velocity_spread=0.10,
        use_particle_limit=True, max_emitted_particles=12000,
    )
    rt.flow_source.create(
        "Addon Gas Pilot", gas_name,
        source_mode="point", position=(0.1, 0.2, 0.0),
        velocity=(0.0, 0.3, 0.0), radius=0.25,
        density=0.02, temperature=2.0, fuel=0.12,
    )
    rt.fluid.reset()
    rt.request_render()
    print("[production_gas_presets] Added API Routing Test")


def _draw_panel():
    rt.ui.text_disabled("Script-authored simulation recipes")
    if rt.ui.button("Add Burning Fuel Spill", width=-1.0):
        add_burning_fuel_spill()
    rt.ui.tooltip("Creates the fuel spill domain, emitter and combustion settings.")
    if rt.ui.button("Add API Routing Test", width=-1.0):
        add_api_routing_test()
    rt.ui.tooltip("Minimal domain + flow source used to verify the fluid API wiring.")


def register():
    global _panel_id
    # Mounted into the Simulation tab rather than a floating window: these recipes
    # only make sense next to the domain/emitter controls they drive.
    _panel_id = rt.ui.register_region(
        "properties.simulation", "Gas & Fluid Presets", _draw_panel)
    print("[production_gas_presets] registered")


def unregister():
    global _panel_id
    if _panel_id is not None:
        rt.ui.unregister_panel(_panel_id)
        _panel_id = None
    print("[production_gas_presets] unregistered")
