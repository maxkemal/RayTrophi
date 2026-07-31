"""Smoke test for rt.flow_source and rt.collider CRUD/routing."""

import rt

FLUID = "API Authoring Fluid"
GAS = "API Authoring Gas"
FLOW = "API Authoring Flow"
PILOT = "API Authoring Pilot"
FLOOR = "API Authoring Floor"

for name in (FLOW, PILOT):
    try:
        rt.flow_source.remove(name)
    except RuntimeError:
        pass
try:
    rt.collider.remove(FLOOR)
except RuntimeError:
    pass
for name in (FLUID, GAS):
    try:
        rt.fluid.remove_domain(name)
    except RuntimeError:
        pass

rt.fluid.create_domain(
    FLUID, domain_min=(-1.0, -0.2, -1.0),
    domain_max=(1.0, 2.0, 1.0), voxel_size=0.15, type="fluid",
)
rt.gas.create_domain(
    GAS, domain_min=(-1.0, -0.2, -1.0),
    domain_max=(1.0, 3.0, 1.0), voxel_size=0.15,
)
rt.collider.create(
    FLOOR, source_mode="plane", plane_y=0.0, thickness=0.15,
)
rt.flow_source.create(
    FLOW, FLUID, position=(-0.5, 1.5, 0.0), velocity=(1.0, -1.5, 0.0),
    radius=0.12, fluid_particles_per_second=100.0,
    use_particle_limit=True, max_emitted_particles=2000,
)
rt.flow_source.create(
    PILOT, GAS, position=(0.2, 0.2, 0.0), radius=0.2,
    density=0.02, temperature=2.0, fuel=0.1,
)

assert rt.flow_source.get(FLOW)["domain"] == FLUID
assert rt.flow_source.get(PILOT)["domain"] == GAS
assert rt.collider.get(FLOOR)["source_mode"] == "plane"
rt.flow_source.update(FLOW, fluid_particles_per_second=120.0)
assert abs(rt.flow_source.get(FLOW)["fluid_particles_per_second"] - 120.0) < 1e-5
print("[simulation-authoring-api] PASS")
