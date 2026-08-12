"""Phase 2 gate: thermal integrity lowers the existing fracture threshold.

The same structural impulse is sent through the contact-event consumer used by
Jolt. Only the pre-burned wood group may break.
"""

import rt

GAS = "Phase02_ThermalGas"
PILOT = "Phase02_BurnPilot"
BURNT = "Phase02_BurntWood"
INTACT = "Phase02_IntactWood"
COLLIDERS = {BURNT: "Phase02_BurntCollider", INTACT: "Phase02_IntactCollider"}

for source in list(rt.flow_source.list()):
    if source["name"] == PILOT:
        rt.flow_source.remove(PILOT)
for collider in list(rt.collider.list()):
    if collider["name"] in COLLIDERS.values():
        rt.collider.remove(collider["name"])
try:
    rt.fluid.remove_domain(GAS)
except RuntimeError:
    pass
for name in (BURNT, INTACT):
    if rt.scene.exists(name):
        rt.scene.delete(name)

burnt = rt.scene.add_primitive("cube", BURNT, 0.62)
intact = rt.scene.add_primitive("cube", INTACT, 0.62)
rt.scene.set_transform(burnt, translation=(-1.0, 0.72, 0.0), scale=(1.0, 0.45, 0.65))
rt.scene.set_transform(intact, translation=(1.0, 0.72, 0.0), scale=(1.0, 0.45, 0.65))

# The intact control deliberately remains outside this domain. Its MSF field is
# still materialised and ambient-stepped, but hot gas cannot diffuse across the
# box and silently turn the control into a second burnt sample.
rt.gas.create_domain(name=GAS, domain_min=(-2.2, 0.0, -1.2),
                     domain_max=(0.0, 2.2, 1.2), voxel_size=0.12)
rt.gas.set_settings(GAS, fire_enabled=True, ignition_temperature=0.22,
                    burn_rate=2.5, heat_release=2.8,
                    fire_max_temperature=9.0, buoyancy_heat=0.5)

for obj in (burnt, intact):
    rt.collider.create(
        COLLIDERS[obj], source_mode="obb", source_object=obj,
        gas_interaction_enabled=True, gas_temperature_rate=1.0,
        gas_ignite_on_contact=True, msf_substance="Wood (Oak)",
        msf_mask_resolution=64,
    )

rt.flow_source.create(
    PILOT, domain=GAS, source_mode="point", position=(-1.0, 0.48, 0.0),
    velocity=(0.0, 0.35, 0.0), radius=0.38, density=0.06,
    temperature=8.0, fuel=1.8, falloff=1.2, velocity_coupling=2.0,
)

# One object is a valid one-shard prepared group for this contract test. The
# production UI's Voronoi generator supplies many objects to the same API.
# ★ break_velocity is in m/s, not N-s. The impulse threshold this test compares
# against is break_velocity * group mass, reported as base/effective_break_impulse.
# Authoring the velocity is what lets one number describe the same material at
# any size; the impulse it works out to is a consequence, not an input.
BASE_TOUGHNESS = 4.0   # m/s
for group, obj in ((BURNT, burnt), (INTACT, intact)):
    info = rt.physics.make_fracture_group(
        group, [obj], break_velocity=BASE_TOUGHNESS,
        integrity_weakening=True, integrity_exponent=1.5,
        minimum_threshold_scale=0.15,
    )
    assert info["shard_count"] == 1, info
    assert info["group_mass_kg"] > 0.0, (
        "a group with no mass has no threshold either", info)

# Derived, not assumed: both bodies must weigh the same for the comparison below
# to isolate thermal weakening rather than a mass difference.
BASE_THRESHOLD = rt.physics.fracture_group(INTACT)["base_break_impulse"]

# Materialise fields and prime the first readback request.
for _ in range(4):
    rt.fluid.step(1.0 / 30.0)
rt.msf.fields()

burn_info = None
intact_info = None
frames = 4
for _block in range(36):
    for _ in range(9):
        rt.fluid.step(1.0 / 30.0)
        frames += 1
    rt.msf.fields()                 # request one sparse coherent snapshot
    rt.fluid.step(1.0 / 30.0)       # fulfil it
    frames += 1
    burn_info = rt.physics.fracture_group(BURNT)
    intact_info = rt.physics.fracture_group(INTACT)
    if (burn_info["effective_break_impulse"] < BASE_THRESHOLD * 0.72 and
            intact_info["effective_break_impulse"] > BASE_THRESHOLD * 0.95):
        break

assert burn_info is not None and intact_info is not None
assert intact_info["mean_integrity"] > 0.95, (
    "intact control was thermally contaminated", intact_info)
assert burn_info["mean_integrity"] < intact_info["mean_integrity"], (
    burn_info, intact_info)
assert burn_info["effective_break_impulse"] < intact_info["effective_break_impulse"], (
    burn_info, intact_info)

# Exactly one impulse, chosen strictly between the two derived thresholds.
impulse = (burn_info["effective_break_impulse"] +
           intact_info["effective_break_impulse"]) * 0.5
intact_triggered = rt.physics.apply_fracture_impulse(
    INTACT, point=(1.0, 0.72, 0.0), direction=(1.0, 0.25, 0.0), impulse=impulse)
burnt_triggered = rt.physics.apply_fracture_impulse(
    BURNT, point=(-1.0, 0.72, 0.0), direction=(-1.0, 0.25, 0.0), impulse=impulse)

assert not intact_triggered, (impulse, intact_info)
assert burnt_triggered, (impulse, burn_info)
assert rt.physics.fracture_group(INTACT)["broken_count"] == 0
assert rt.physics.fracture_group(BURNT)["broken_count"] == 1

print({
    "result": "PASS", "phase": 2, "frames": frames,
    "shared_impulse": impulse,
    "burnt": rt.physics.fracture_group(BURNT),
    "intact": rt.physics.fracture_group(INTACT),
})
