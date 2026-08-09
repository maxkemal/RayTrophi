"""Phase 6 production gate: heat alone drives plastic into an empty APIC domain."""

import rt

GAS = "Phase06Auto_Gas"
LIQUID = "Phase06Auto_Molten"
PLASTIC = "Phase06Auto_Plastic"
COLLIDER = "Phase06Auto_PlasticCollider"
PILOT = "Phase06Auto_Pilot"

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
rt.gas.set_settings(GAS, fire_enabled=True, ignition_temperature=0.22,
                    burn_rate=2.2, heat_release=3.0,
                    fire_max_temperature=9.0)
rt.collider.create(
    COLLIDER, source_mode="obb", source_object=plastic,
    gas_interaction_enabled=True, gas_temperature_rate=1.0,
    gas_ignite_on_contact=True, msf_substance="Plastic (PE)",
    msf_mask_resolution=72, msf_auto_transfer=True,
    msf_transfer_domain=LIQUID, msf_transfer_rate_kg_s=0.10,
    msf_transfer_min_mass_kg=0.01,
    msf_transfer_particles_per_kg=2048.0,
    msf_transfer_max_batch_particles=128,
    msf_transfer_velocity=(0.0, -0.1, 0.0),
    msf_melt_flow_enabled=True, msf_melt_height_loss=0.85,
    msf_melt_sdf_refresh=True, msf_melt_sdf_revision_interval=4,
    msf_melt_sdf_change_threshold=0.025,
    msf_melt_spread=1.50,
)
rt.flow_source.create(
    PILOT, domain=GAS, source_mode="point", position=(0.0, 0.45, 0.0),
    velocity=(0.0, 0.35, 0.0), radius=0.42, density=0.05,
    temperature=8.0, fuel=1.2, falloff=1.1,
)

before = rt.mass_transfer.stats()
frames = 0
for block in range(60):
    for _ in range(8):
        rt.fluid.step(1.0 / 30.0)
        frames += 1
    # Automatic transfer intentionally reads MSF in bounded blocks, not by
    # forcing a render or collider rebuild every frame.
    if rt.mass_transfer.stats()["completed"] > before["completed"]:
        break

transfer = rt.mass_transfer.stats()
domain = rt.fluid.get(LIQUID)
chemistry = rt.fluid.get_combustion(LIQUID)
collider = {c["name"]: c for c in rt.collider.list()}[COLLIDER]

assert transfer["completed"] > before["completed"], transfer
assert transfer["spawned_particles"] > before["spawned_particles"], transfer
assert transfer["last_object"] == PLASTIC, transfer
assert transfer["last_domain"] == LIQUID, transfer
assert transfer["last_substance"] == "Plastic (PE)", transfer
assert domain["particle_count"] > 0, domain
assert domain["render_mode"] == "surface", domain
assert domain["viscosity"] >= 30.0, domain
assert chemistry["chemistry_preset"] == "plastic", chemistry
assert chemistry["enabled"], chemistry
assert chemistry["auto_ignite"], chemistry
assert collider["msf_auto_transfer"], collider
assert collider["msf_transfer_domain"] == LIQUID, collider
assert collider["msf_melt_flow_enabled"], collider
assert abs(collider["msf_melt_height_loss"] - 0.85) < 1e-4, collider
assert collider["msf_melt_sdf_refresh"], collider
assert collider["msf_melt_sdf_revision_interval"] == 4, collider
assert abs(collider["msf_melt_sdf_change_threshold"] - 0.025) < 1e-4, collider
assert abs(collider["msf_melt_spread"] - 1.50) < 1e-4, collider

print({"result": "PASS", "phase": "6-auto", "frames": frames,
       "transfer": transfer, "domain": domain, "chemistry": chemistry})
