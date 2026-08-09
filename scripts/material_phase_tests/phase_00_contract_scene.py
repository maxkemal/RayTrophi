"""Phase 0 field-contract and scripted mesh-SDF-collider acceptance scene.

Run directly from RayTrophi's Scripts panel after compiling the application.
No addon installation or enable step is required.
"""

import time
import rt

OBJECT = "Phase00_FieldCube"
COLLIDER = "Phase00_FieldCube_SDF"
GAS = "Phase00_FieldGas"
SUBSTANCE = "Plastic (PE)"

available_substances = rt.msf.substances()
assert SUBSTANCE in available_substances, (
    "Required substance is unavailable: %s; available=%r"
    % (SUBSTANCE, available_substances)
)

# Keep the script repeatable without deleting unrelated user content.
for collider in list(rt.collider.list()):
    if collider["name"] == COLLIDER:
        rt.collider.remove(COLLIDER)
try:
    rt.fluid.remove_domain(GAS)
except RuntimeError:
    pass
if rt.scene.exists(OBJECT):
    rt.scene.delete(OBJECT)

cube = rt.scene.add_primitive("cube", OBJECT, 0.65)
rt.scene.set_transform(cube, translation=(0.0, 0.8, 0.0))

rt.gas.create_domain(
    name=GAS,
    domain_min=(-1.5, 0.0, -1.5),
    domain_max=(1.5, 2.5, 1.5),
    voxel_size=0.15,
)
rt.collider.create(
    COLLIDER,
    source_mode="mesh_sdf",
    source_object=cube,
    sdf_resolution_mode=0,  # cheap 32^3 phase-gate cook
    gas_interaction_enabled=True,
    gas_temperature_rate=1.0,
    gas_ignite_on_contact=True,
    msf_substance=SUBSTANCE,
    msf_mask_resolution=64,
)

# Explicit force-rebuild is part of the scripting contract. Creation already
# starts a cook; this second request proves the public command is wired too.
rt.collider.rebuild_sdf(COLLIDER)

deadline = time.time() + 15.0
sdf = rt.collider.get(COLLIDER)
while not sdf["sdf_ready"] and time.time() < deadline:
    time.sleep(0.05)
    sdf = rt.collider.get(COLLIDER)

assert sdf["source_mode"] == "mesh_sdf"
assert sdf["sdf_ready"], "mesh-SDF cook did not finish within 15 seconds"
assert sdf["sdf_resolution"] == 32

# A few deterministic steps materialise the object's MSF. This is intentionally
# short: Phase 0 validates the data contract, not a long combustion bake.
for _ in range(4):
    rt.fluid.step(1.0 / 30.0)

fields = {field["object_key"]: field for field in rt.msf.fields()}
assert cube in fields, (
    "MSF field was not created; collider=%r fields=%r"
    % (rt.collider.get(COLLIDER), list(fields.values()))
)
field = fields[cube]
assert field["substance"] == SUBSTANCE
assert field["element_count"] > 0
assert field["topology_generation"] > 0
assert set(("temperature", "moisture", "fuel_remaining", "char", "melt", "mass_loss")) <= set(field["semantics"])

print({
    "result": "PASS",
    "phase": 0,
    "object_key": field["object_key"],
    "topology_generation": field["topology_generation"],
    "content_generation": field["content_generation"],
    "element_count": field["element_count"],
    "mask_resolution": field["mask_resolution"],
    "sdf_resolution": sdf["sdf_resolution"],
})
