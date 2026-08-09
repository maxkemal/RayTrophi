"""Phase 6 gate: molten surface mass enters APIC exactly once."""

import rt

GAS = "Phase06_PlasticGas"
LIQUID = "Phase06_MoltenAPIC"
PILOT = "Phase06_Pilot"
PLASTIC = "Phase06_Plastic"
COLLIDER = "Phase06_PlasticCollider"

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
if rt.scene.exists(PLASTIC):
    rt.scene.delete(PLASTIC)

plastic = rt.scene.add_primitive("cube", PLASTIC, 0.65)
rt.scene.set_transform(plastic, translation=(0.0, 0.75, 0.0),
                       scale=(1.0, 0.65, 1.0))
rt.gas.create_domain(name=GAS, domain_min=(-1.5, 0.0, -1.5),
                     domain_max=(1.5, 2.5, 1.5), voxel_size=0.12)
rt.gas.set_param(GAS, backend="vulkan", boundary="open",
                 render_mode="volume")
rt.fluid.create_domain(LIQUID, domain_min=(-1.5, 0.0, -1.5),
                       domain_max=(1.5, 2.5, 1.5), voxel_size=0.12,
                       type="fluid")
rt.fluid.set_param(LIQUID, backend="vulkan", boundary="closed",
                   render_mode="surface")
rt.fluid.set_combustion(
    LIQUID, enabled=True, auto_ignite=True,
    ignition_temperature=0.20, evaporation_rate=1.5,
    surface_fuel_capacity=3.0, heat_release=2.0, smoke_yield=0.45,
)
rt.gas.set_settings(GAS, fire_enabled=True, ignition_temperature=0.22,
                    burn_rate=2.2, heat_release=3.0,
                    fire_max_temperature=9.0)
rt.collider.create(
    COLLIDER, source_mode="obb", source_object=plastic,
    # The object remains an active MSF/gas collider, but its own transferred
    # APIC liquid must not be sealed inside the undeformed collision volume.
    fluid_collision_enabled=False,
    gas_interaction_enabled=True, gas_temperature_rate=1.0,
    gas_ignite_on_contact=True, msf_substance="Plastic (PE)",
    msf_mask_resolution=72,
)
rt.flow_source.create(
    PILOT, domain=GAS, source_mode="point", position=(0.0, 0.45, 0.0),
    velocity=(0.0, 0.35, 0.0), radius=0.42, density=0.05,
    temperature=8.0, fuel=1.2, falloff=1.1,
)

field = None
frames = 0
for _block in range(45):
    for _ in range(8):
        rt.fluid.step(1.0 / 30.0)
        frames += 1
    rt.msf.fields()
    rt.fluid.step(1.0 / 30.0)
    frames += 1
    field = {f["object_key"]: f for f in rt.msf.fields()}.get(PLASTIC)
    if field and field["molten_reservoir_mass"] > 0.05:
        break

assert field is not None, "plastic MSF field was not created"
before_reservoir = field["molten_reservoir_mass"]
request_mass = min(0.25, before_reservoir * 0.5)
stats_before = rt.mass_transfer.stats()
sequence = rt.mass_transfer.queue(PLASTIC, request_mass, domain=LIQUID,
                                  particles_per_kg=2048.0,
                                  velocity=(0.0, -0.1, 0.0))
assert sequence > 0

for _ in range(8):
    rt.fluid.step(1.0 / 30.0)
    frames += 1
    if rt.mass_transfer.stats()["completed"] > stats_before["completed"]:
        break

rt.msf.fields()
rt.fluid.step(1.0 / 30.0)
frames += 1
after = {f["object_key"]: f for f in rt.msf.fields()}[PLASTIC]
transfer = rt.mass_transfer.stats()
combustion_start = dict(transfer)

assert transfer["completed"] == stats_before["completed"] + 1, transfer
assert transfer["spawned_particles"] > stats_before["spawned_particles"], transfer
assert transfer["transferred_mass"] > stats_before["transferred_mass"], transfer
assert transfer["last_domain"] == LIQUID, transfer
assert transfer["last_substance"] == "Plastic (PE)", transfer
assert transfer["last_combustible_fraction"] == 1.0, transfer
assert (transfer["spawned_particles"] - stats_before["spawned_particles"]) >= 64, transfer
assert transfer["live_tagged_particles"] > 0, transfer
assert after["transferred_mass"] > 0.0, after
assert after["molten_reservoir_mass"] < before_reservoir, (field, after)
accounted = (after["solid_mass"] + after["pyrolyzed_mass"] +
             after["molten_reservoir_mass"] + after["transferred_mass"])
tolerance = max(after["initial_mass"] * 1e-4, 1e-5)
assert abs(accounted - after["initial_mass"]) <= tolerance, after
assert after["mass_conservation_error"] <= tolerance, after

# The transfer is processed after this frame's fluid/gas coupling, so the first
# snapshot is the unburnt APIC batch. Keep the pilot hot and prove that the
# carried plastic chemistry subsequently consumes particle mass.
for _ in range(90):
    rt.fluid.step(1.0 / 30.0)
    frames += 1
combustion_end = rt.mass_transfer.stats()
assert (combustion_end["live_tagged_particles"] <
        combustion_start["live_tagged_particles"] or
        combustion_end["mean_remaining_mass_fraction"] <
        combustion_start["mean_remaining_mass_fraction"] - 1e-4), (
            combustion_start, combustion_end)

# Leave an inspectable batch in the scene after the destructive combustion
# assertion. Without this hold the workspace returns at frame 110 with zero
# tagged particles, so a successful chemistry test looks visually identical to
# a failed transfer. Particle render mode is deliberate here: it proves source
# placement independently of the surface-SDF reconstruction threshold.
rt.fluid.set_combustion(LIQUID, enabled=False)
rt.fluid.set_param(LIQUID, render_mode="particles")
rt.msf.fields()
rt.fluid.step(1.0 / 30.0)
frames += 1
preview_field = {f["object_key"]: f for f in rt.msf.fields()}[PLASTIC]
preview_mass = min(0.04, preview_field["molten_reservoir_mass"] * 0.5)
preview_sequence = rt.mass_transfer.queue(
    PLASTIC, preview_mass, domain=LIQUID, particles_per_kg=4096.0,
    velocity=(0.0, -0.05, 0.0))
for _ in range(3):
    rt.fluid.step(1.0 / 30.0)
    frames += 1
preview = rt.mass_transfer.stats()
assert preview_sequence > sequence, preview
assert preview["live_tagged_particles"] > 0, preview

print({"result": "PASS", "phase": 6, "frames": frames,
       "sequence": sequence, "before_reservoir": before_reservoir,
       "after_reservoir": after["molten_reservoir_mass"],
       "accounted_mass": accounted, "transfer": transfer,
       "combustion_end": combustion_end,
       "inspection_preview": preview})
