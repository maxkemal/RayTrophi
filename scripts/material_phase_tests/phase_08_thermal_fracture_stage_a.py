"""Phase 8 vertical gate, STAGE A: burn one end of a beam and record the damage.

This is the chain the whole material-transformation roadmap exists to produce:

    burn  ->  integrity loss  ->  damage-guided fracture seeding
          ->  structural clusters  ->  pressure pulse detaches only what it hit

Only the first half is scriptable today: shard GENERATION is still UI-only
(rt.physics.make_fracture_group takes a ready shard list, it does not cut). So
run this, then do the UI step it prints, then run stage B. The scene stays live
between scripts, so nothing needs saving.

Run order:
    1. this script
    2. the UI step printed at the end
    3. phase_08_thermal_fracture_stage_b.py
"""

import rt

GAS = "Phase08_BurnGas"
PILOT = "Phase08_Pilot"
BEAM = "Phase08_Beam"
COLLIDER = "Phase08_BeamCollider"

# The beam runs along X. The gas domain deliberately covers only its -X half, so
# the burn stays LOCALISED — that localisation is the entire input to thermal
# seeding, and a domain covering the whole beam would char it evenly and make
# the test unable to distinguish thermal seeding from uniform seeding.
#
# ★ Everything below is DERIVED from these three numbers. It used to be a set of
# hand-placed coordinates that quietly disagreed with the geometry: add_primitive
# takes an EDGE LENGTH, not a half-extent, so the beam was half the length the
# constants assumed and the pilot flame sat past its end. (Worse, `scale=` was
# being dropped entirely by set_transform — the "beam" was a cube. That is fixed
# in the binding; these numbers are only trustworthy now because of it.)
BEAM_LENGTH = 3.0
BEAM_THICKNESS = 0.3
BEAM_CENTER_Y = 0.9
BEAM_HALF_LENGTH = BEAM_LENGTH * 0.5
# The flame sits just inside the -X end, on the surface of the beam.
BURNT_END_X = -BEAM_HALF_LENGTH + 0.4

for source in list(rt.flow_source.list()):
    if source["name"] == PILOT:
        rt.flow_source.remove(PILOT)
for collider in list(rt.collider.list()):
    if collider["name"] == COLLIDER:
        rt.collider.remove(COLLIDER)
try:
    rt.fluid.remove_domain(GAS)
except RuntimeError:
    pass
if rt.scene.exists(BEAM):
    rt.scene.delete(BEAM)

beam = rt.scene.add_primitive("cube", BEAM, 1.0)
rt.scene.set_transform(beam, translation=(0.0, BEAM_CENTER_Y, 0.0),
                       scale=(BEAM_LENGTH, BEAM_THICKNESS, BEAM_THICKNESS))

# Verify the shape the scene REALLY has before burning it. A scene test whose
# geometry silently differs from its constants measures nothing.
actual_scale = rt.scene.get_transform(BEAM)["scale"]
assert abs(actual_scale[0] - BEAM_LENGTH) < 1e-3, (
    "the beam is not the length this test assumes - set_transform(scale=) did "
    "not take", actual_scale)

# Covers the -X end of the beam only, with headroom above it for the plume.
rt.gas.create_domain(name=GAS,
                     domain_min=(-BEAM_HALF_LENGTH - 0.5, 0.0, -1.0),
                     domain_max=(-BEAM_HALF_LENGTH * 0.35, BEAM_CENTER_Y + 1.5, 1.0),
                     voxel_size=0.11)
rt.gas.set_settings(GAS, fire_enabled=True, ignition_temperature=0.22,
                    burn_rate=2.5, heat_release=2.8,
                    fire_max_temperature=9.0, buoyancy_heat=0.5)

rt.collider.create(
    COLLIDER, source_mode="obb", source_object=beam,
    gas_interaction_enabled=True, gas_temperature_rate=1.0,
    gas_ignite_on_contact=True, msf_substance="Wood (Oak)",
    msf_mask_resolution=96,
)
rt.flow_source.create(
    PILOT, domain=GAS, source_mode="point",
    # Just under the beam's underside, so the plume rises onto it.
    position=(BURNT_END_X, BEAM_CENTER_Y - BEAM_THICKNESS * 0.5 - 0.2, 0.0),
    velocity=(0.0, 0.4, 0.0), radius=0.4, density=0.06,
    temperature=8.0, fuel=1.8, falloff=1.2, velocity_coupling=2.0,
)

for _ in range(4):
    rt.fluid.step(1.0 / 30.0)
rt.msf.fields()

field = None
frames = 4
for _block in range(40):
    for _ in range(9):
        rt.fluid.step(1.0 / 30.0)
        frames += 1
    rt.msf.fields()
    rt.fluid.step(1.0 / 30.0)
    frames += 1
    fields = {f["object_key"]: f for f in rt.msf.fields()}
    field = fields.get(BEAM)
    if field and field["mean_integrity"] < 0.80:
        break

assert field is not None, "beam MSF field was not created"
assert field["mean_integrity"] < 0.95, (
    "the beam did not burn enough to steer seeding", field)

# The raw-field conservation gate from the earlier pass still has to hold while
# all this is going on; a fracture seeded off a broken mass budget is worthless.
tolerance = max(field["initial_mass"] * 1e-4, 1e-5)
assert field["mass_budget_overflow"] <= tolerance, field
assert field["mass_negative"] <= tolerance, field
assert field["mass_invalid_elements"] == 0, field

# Rough guide for the UI threshold, from the impulse the new bridge computes:
#     J [N s] = dp [Pa] * A_projected [m^2] * dt [s] * coupling * falloff
# A cluster of a 6-way split beam presents roughly its cross-section to a blast
# arriving along X. Refine this from `last_projected_area_m2`, which stage B
# reports from the real geometry — that is what the telemetry field is for.
PULSE_KPA = 300.0
PULSE_SECONDS = 0.02
cross_section = BEAM_THICKNESS * BEAM_THICKNESS
suggested = PULSE_KPA * 1000.0 * cross_section * PULSE_SECONDS * 0.7

print({
    "result": "STAGE A OK", "phase": "8a", "frames": frames,
    "mean_integrity": field["mean_integrity"],
    "minimum_integrity": field["minimum_integrity"],
    "mass_loss": field["mass_loss"],
    "suggested_break_impulse_Ns": round(suggested),
})
print("")
print("NEXT — UI step (shard generation is not scriptable yet):")
print("  1. Select '%s' in the outliner." % BEAM)
print("  2. Physics panel -> Fracture (Destruction):")
print("       Shards              = 48")
print("       Pattern             = Thermal (burn-guided)")
print("       Structural Clusters = 6")
print("       -> Generate Shards")
print("     Expect the viewport message to report 6 clusters, and the shards")
print("     to be visibly FINER around x = %.1f (the burnt end)." % BURNT_END_X)
print("  3. Break Toughness = 5 m/s (the default), then -> Make Breakable")
print("     ★ The panel authors a VELOCITY now, not an impulse. The impulse")
print("       threshold is that times the group's mass, and the panel prints")
print("       both under the slider. Compare THAT number - not the toughness -")
print("       against the ~%d N-s this blast delivers." % round(suggested))
print("     ★ If it is far below the blast, every cluster in range breaks no")
print("       matter how strong it is - the split you then see is the radius")
print("       cutoff, not strength, and thermal weakening is never exercised.")
print("       Stage B refuses to report PASS from that situation.")
print("     Expect 'Registered 6 breakable cluster(s)'.")
print("  4. Run phase_08_thermal_fracture_stage_b.py")
