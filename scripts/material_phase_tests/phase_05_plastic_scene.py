"""Phase 5 gate: plastic melt and pyrolysis share one mass budget."""

import rt

DOMAIN = "Phase05_PlasticGas"
PILOT = "Phase05_PlasticPilot"
PLASTIC = "Phase05_Plastic"
COLLIDER = "Phase05_PlasticCollider"

for source in list(rt.flow_source.list()):
    if source["name"] == PILOT:
        rt.flow_source.remove(PILOT)
for collider in list(rt.collider.list()):
    if collider["name"] == COLLIDER:
        rt.collider.remove(COLLIDER)
try:
    rt.fluid.remove_domain(DOMAIN)
except RuntimeError:
    pass
if rt.scene.exists(PLASTIC):
    rt.scene.delete(PLASTIC)

plastic = rt.scene.add_primitive("cube", PLASTIC, 0.65)
rt.scene.set_transform(plastic, translation=(0.0, 0.75, 0.0),
                       scale=(1.0, 0.65, 1.0))
rt.gas.create_domain(name=DOMAIN, domain_min=(-1.5, 0.0, -1.5),
                     domain_max=(1.5, 2.5, 1.5), voxel_size=0.12)
rt.gas.set_settings(DOMAIN, fire_enabled=True, ignition_temperature=0.22,
                    burn_rate=2.2, heat_release=3.0,
                    fire_max_temperature=9.0)
rt.collider.create(
    COLLIDER, source_mode="obb", source_object=plastic,
    gas_interaction_enabled=True, gas_temperature_rate=1.0,
    gas_ignite_on_contact=True, msf_substance="Plastic (PE)",
    msf_mask_resolution=72,
)
rt.flow_source.create(
    PILOT, domain=DOMAIN, source_mode="point", position=(0.0, 0.45, 0.0),
    velocity=(0.0, 0.35, 0.0), radius=0.42, density=0.05,
    temperature=8.0, fuel=1.2, falloff=1.1,
)

latest = None
frames = 0
for _block in range(40):
    for _ in range(8):
        rt.fluid.step(1.0 / 30.0)
        frames += 1
    rt.msf.fields()
    rt.fluid.step(1.0 / 30.0)
    frames += 1
    fields = {f["object_key"]: f for f in rt.msf.fields()}
    latest = fields.get(PLASTIC)
    if (latest and latest["molten_reservoir_mass"] > 1e-4 and
            latest["pyrolyzed_mass"] > 1e-4):
        break

assert latest is not None, "plastic MSF field was not created"
assert latest["initial_mass"] > 0.0, latest
assert latest["molten_reservoir_mass"] > 0.0, latest
assert latest["pyrolyzed_mass"] > 0.0, latest
accounted = (latest["solid_mass"] + latest["pyrolyzed_mass"] +
             latest["molten_reservoir_mass"])
tolerance = max(latest["initial_mass"] * 1e-4, 1e-5)
assert abs(accounted - latest["initial_mass"]) <= tolerance, latest
assert latest["mass_conservation_error"] <= tolerance, latest
# ★ The line above is only worth asserting because the error is now measured on
# the RAW field. It used to be derived from the four clamped masses, which made
# it arithmetically incapable of being anything but 0.0 — a green light wired to
# the switch rather than to the solver. These three name the failure mode when
# it does trip.
assert latest["mass_budget_overflow"] <= tolerance, latest
assert latest["mass_negative"] <= tolerance, latest
assert latest["mass_invalid_elements"] == 0, latest

print({"result": "PASS", "phase": 5, "frames": frames,
       "initial_mass": latest["initial_mass"],
       "solid_mass": latest["solid_mass"],
       "pyrolyzed_mass": latest["pyrolyzed_mass"],
       "molten_reservoir_mass": latest["molten_reservoir_mass"],
       "accounted_mass": accounted,
       "mass_conservation_error": latest["mass_conservation_error"]})
