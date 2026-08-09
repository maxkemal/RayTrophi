"""Phase 6 chemistry gate: molten iron transfers hot but cannot burn."""

import rt

GAS = "Phase06_IronGas"
LIQUID = "Phase06_IronAPIC"
PILOT = "Phase06_IronHeat"
IRON = "Phase06_Iron"
COLLIDER = "Phase06_IronCollider"

for source in list(rt.flow_source.list()):
    if source["name"] == PILOT:
        rt.flow_source.remove(PILOT)
for collider in list(rt.collider.list()):
    if collider["name"] == COLLIDER:
        rt.collider.remove(COLLIDER)
for domain in (GAS, LIQUID):
    try:
        rt.fluid.remove_domain(domain)
    except RuntimeError:
        pass
if rt.scene.exists(IRON):
    rt.scene.delete(IRON)

iron = rt.scene.add_primitive("cube", IRON, 0.55)
rt.scene.set_transform(iron, translation=(0.0, 0.7, 0.0),
                       scale=(1.0, 0.7, 1.0))
rt.gas.create_domain(name=GAS, domain_min=(-1.4, 0.0, -1.4),
                     domain_max=(1.4, 2.3, 1.4), voxel_size=0.12)
rt.gas.set_param(GAS, backend="vulkan", boundary="open",
                 render_mode="volume")
rt.fluid.create_domain(LIQUID, domain_min=(-1.4, 0.0, -1.4),
                       domain_max=(1.4, 2.3, 1.4), voxel_size=0.12,
                       type="fluid")
rt.fluid.set_param(LIQUID, backend="vulkan", boundary="closed",
                   render_mode="surface")
# Deliberately mark the containing domain flammable. The per-particle chemistry
# must still protect the transferred iron from the plastic fuel lifecycle.
rt.fluid.set_combustion(
    LIQUID, enabled=True, auto_ignite=True,
    ignition_temperature=0.20, evaporation_rate=2.0,
    surface_fuel_capacity=3.0, heat_release=2.0, smoke_yield=0.45,
)
rt.gas.set_settings(GAS, fire_enabled=True, ignition_temperature=0.20,
                    burn_rate=2.0, heat_release=3.0,
                    fire_max_temperature=10.0)
rt.collider.create(
    COLLIDER, source_mode="obb", source_object=iron,
    gas_interaction_enabled=True, gas_temperature_rate=1.0,
    gas_ignite_on_contact=False, msf_substance="Iron",
    msf_mask_resolution=72,
)
rt.flow_source.create(
    PILOT, domain=GAS, source_mode="point", position=(0.0, 0.55, 0.0),
    velocity=(0.0, 0.25, 0.0), radius=0.48, density=0.03,
    temperature=9.0, fuel=1.0, falloff=1.0,
)

field = None
frames = 0
for _block in range(70):
    for _ in range(8):
        rt.fluid.step(1.0 / 30.0)
        frames += 1
    rt.msf.fields()
    rt.fluid.step(1.0 / 30.0)
    frames += 1
    field = {f["object_key"]: f for f in rt.msf.fields()}.get(IRON)
    if field and field["molten_reservoir_mass"] > 0.08:
        break

assert field is not None, "iron MSF field was not created"
assert field["initial_mass"] > 0.0, field
assert field["pyrolyzed_mass"] == 0.0, field
assert field["molten_reservoir_mass"] > 0.0, field

baseline = rt.mass_transfer.stats()
request_mass = min(0.12, field["molten_reservoir_mass"] * 0.5)
sequence = rt.mass_transfer.queue(IRON, request_mass, domain=LIQUID,
                                  particles_per_kg=2048.0,
                                  velocity=(0.0, -0.15, 0.0))
for _ in range(8):
    rt.fluid.step(1.0 / 30.0)
    frames += 1
    if rt.mass_transfer.stats()["completed"] > baseline["completed"]:
        break

start = rt.mass_transfer.stats()
assert start["last_substance"] == "Iron", start
assert start["last_combustible_fraction"] == 0.0, start
assert start["last_temperature_kelvin"] >= 1700.0, start
assert (start["spawned_particles"] - baseline["spawned_particles"]) >= 64, start
assert start["live_tagged_particles"] > 0, start

for _ in range(90):
    rt.fluid.step(1.0 / 30.0)
    frames += 1
end = rt.mass_transfer.stats()
assert end["live_tagged_particles"] > 0, (start, end)
assert abs(end["mean_remaining_mass_fraction"] - 1.0) <= 1e-5, (start, end)

rt.msf.fields()
rt.fluid.step(1.0 / 30.0)
after = {f["object_key"]: f for f in rt.msf.fields()}[IRON]
accounted = (after["solid_mass"] + after["pyrolyzed_mass"] +
             after["molten_reservoir_mass"] + after["transferred_mass"])
tolerance = max(after["initial_mass"] * 1e-4, 1e-5)
assert after["pyrolyzed_mass"] == 0.0, after
assert abs(accounted - after["initial_mass"]) <= tolerance, after

print({"result": "PASS", "phase": "6-metal", "frames": frames,
       "sequence": sequence, "initial_mass": after["initial_mass"],
       "transferred_mass": after["transferred_mass"],
       "particle_start": start, "particle_end": end,
       "accounted_mass": accounted})
