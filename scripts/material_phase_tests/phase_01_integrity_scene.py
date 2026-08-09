"""Phase 1: integrity, paper coverage erosion and wood surface damage.

Run directly from RayTrophi's Scripts panel. This is the single Phase-1 gate;
it intentionally advances enough frames to produce visible damage.
"""

import rt

GAS = "Phase01_BurnGas"
PAPER = "Phase01_Paper"
WOOD = "Phase01_Wood"
PAPER_COL = "Phase01_Paper_Collider"
WOOD_COL = "Phase01_Wood_Collider"
SOURCES = ("Phase01_PaperFlame", "Phase01_WoodFlame")

# Repeatable, but do not erase unrelated systems or scene content.
for source in list(rt.flow_source.list()):
    if source["name"] in SOURCES:
        rt.flow_source.remove(source["name"])
for collider in list(rt.collider.list()):
    if collider["name"] in (PAPER_COL, WOOD_COL):
        rt.collider.remove(collider["name"])
try:
    rt.fluid.remove_domain(GAS)
except RuntimeError:
    pass
for name in (PAPER, WOOD):
    if rt.scene.exists(name):
        rt.scene.delete(name)

paper = rt.scene.add_primitive("cube", PAPER, 0.55)
wood = rt.scene.add_primitive("cube", WOOD, 0.55)
rt.scene.set_transform(paper, translation=(-0.75, 0.7, 0.0), scale=(1.0, 0.08, 0.72))
rt.scene.set_transform(wood, translation=(0.75, 0.7, 0.0), scale=(1.0, 0.30, 0.72))

rt.gas.create_domain(
    name=GAS,
    domain_min=(-2.0, 0.0, -1.3),
    domain_max=(2.0, 2.2, 1.3),
    voxel_size=0.12,
)
rt.gas.set_settings(
    GAS, fire_enabled=True, ignition_temperature=0.22, burn_rate=2.4,
    heat_release=2.8, smoke_generation=0.45, fire_max_temperature=9.0,
    buoyancy_heat=0.55, vorticity=0.25,
)
rt.gas.set_shader(GAS, preset="fire", blackbody_intensity=5.0,
                  temperature_min=780.0, temperature_max=1900.0)

def add_surface(collider, obj, substance):
    rt.collider.create(
        collider, source_mode="obb", source_object=obj,
        gas_interaction_enabled=True,
        gas_temperature_rate=1.0, gas_ignite_on_contact=True,
        msf_substance=substance, msf_mask_resolution=96,
    )

add_surface(PAPER_COL, paper, "Paper")
add_surface(WOOD_COL, wood, "Wood (Oak)")

for name, x in zip(SOURCES, (-0.75, 0.75)):
    rt.flow_source.create(
        name, domain=GAS, source_mode="point", position=(x, 0.45, 0.0),
        velocity=(0.0, 0.35, 0.0), radius=0.32, density=0.08,
        temperature=8.0, fuel=2.0, falloff=1.2, velocity_coupling=2.0,
    )

initial = None
latest = None
for _ in range(10):
    rt.fluid.step(1.0 / 30.0)
    fields = {f["object_key"]: f for f in rt.msf.fields()}
    if paper in fields and wood in fields:
        initial = {paper: fields[paper]["mean_integrity"],
                   wood: fields[wood]["mean_integrity"]}
        break

assert initial is not None, "MSF fields were not produced"
frame = 0
for _block in range(24):
    # Advance mostly without readback. Just before the block's final step,
    # fields() requests one coherent GPU -> host snapshot; the following step
    # fulfils it and the second query consumes the fresh generation.
    for _ in range(9):
        rt.fluid.step(1.0 / 30.0)
        frame += 1
    rt.msf.fields()
    rt.fluid.step(1.0 / 30.0)
    frame += 1
    latest = {f["object_key"]: f for f in rt.msf.fields()}
    if paper in latest and wood in latest:
        if (latest[paper]["minimum_integrity"] < 0.82 and
                latest[wood]["minimum_integrity"] < 0.94):
            break

assert latest is not None, "MSF field telemetry was not refreshed"
for obj in (paper, wood):
    assert "integrity" in latest[obj]["semantics"], latest[obj]
    assert latest[obj]["mean_integrity"] < initial[obj], (
        "surface did not lose integrity: %s initial=%r final=%r"
        % (obj, initial[obj], latest[obj])
    )
    assert latest[obj]["mass_loss"] > 0.0, latest[obj]

print({
    "result": "PASS", "phase": 1,
    "frames": frame,
    "paper": {"mean_integrity": latest[paper]["mean_integrity"],
              "minimum_integrity": latest[paper]["minimum_integrity"],
              "mass_loss": latest[paper]["mass_loss"]},
    "wood": {"mean_integrity": latest[wood]["mean_integrity"],
             "minimum_integrity": latest[wood]["minimum_integrity"],
             "mass_loss": latest[wood]["mass_loss"]},
    "expectation": "paper holes and charred wood cracks; no topology/Jolt rebuild",
})
