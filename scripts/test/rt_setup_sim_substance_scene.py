# A LIVE Material State Field for rt_test_sim_substance.py.
#
# ★★ A primitive collider (sphere/plane) is not enough. MSF elements come from
# the collider's MESH — the resolver hands the field a triangle soup — so an
# analytic collider carries no field at all, and the surface inspector would
# report "unmeasured" forever. The first version of this rig did exactly that
# and the test honestly said NOT VERIFIED.
#
# ★ Non-default authored values on purpose: restoring to the default and
# restoring correctly would look identical otherwise.
import rt

OBJECT = "SubstanceTestBox"
COLLIDER = "SubstanceTestCollider"
GAS = "SubstanceTestGas"

for name in [c["name"] for c in rt.collider.list()]:
    if name == COLLIDER:
        rt.collider.remove(COLLIDER)

if GAS in [d["name"] for d in rt.fluid.list_domains()]:
    rt.fluid.remove_domain(GAS)

if OBJECT not in [o["name"] for o in rt.scene.objects()]:
    OBJECT = rt.scene.add_primitive("cube", OBJECT, 0.6)
print("object: %s" % OBJECT)

rt.collider.create(COLLIDER,
                   source_mode="mesh_sdf",
                   source_object=OBJECT,
                   friction=0.31,
                   restitution=0.42,
                   gas_interaction_enabled=True,
                   gas_ignite_on_contact=True,
                   gas_temperature_rate=6.0,
                   gas_fuel_rate=3.0,
                   msf_substance="Wood (Oak)",
                   msf_burn_rate_scale=1.25,
                   msf_melt_spread=1.75)

# A gas domain that CONTAINS the box. MSF gathers the neighbouring gas cell, so
# an object outside every domain never heats and never builds a field.
rt.gas.create_domain(GAS, domain_min=(-1.5, 0.0, -1.5), domain_max=(1.5, 3.0, 1.5),
                     voxel_size=0.1)
rt.fluid.set_param(GAS, backend="gpu")

for _ in range(12):
    rt.fluid.step(0.0416)

c = rt.collider.get(COLLIDER)
print("collider %s: substance=%r ignite=%s burn_rate_scale=%.3f" % (
    c["name"], c["msf_substance"], c["gas_ignite_on_contact"],
    c["msf_burn_rate_scale"]))
attrs = rt.attr.list("object", COLLIDER)
print("surface attributes: %s" % (attrs,))
print("known substances: %s" % (rt.msf.substances(),))
# ★ Say which of the two states this is. An empty attribute list means the field
# never formed, and that is a SETUP failure, not a passing test.
print("SETUP OK" if attrs else
      "SETUP INCOMPLETE: no MSF field formed; the surface inspector cannot be tested")
