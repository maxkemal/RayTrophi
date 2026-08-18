# Scene for rt_test_sim_render.py: a gas domain to shade, a fluid domain to
# bind a surface material to, and a material that is NOT already bound.
#
# â˜… Both halves matter. Without a fluid domain the liquid-material claim cannot
# fail, and a claim that cannot fail has not passed â€” the test says NOT VERIFIED
# rather than printing green.
import rt

GAS = "RenderTestGas"
FLUID = "RenderTestFluid"
MATERIAL = "RenderTestLiquidMat"

existing_domains = [d["name"] for d in rt.fluid.list_domains()]
if GAS not in existing_domains:
    rt.gas.create_domain(GAS, domain_min=(-1.5, 0.0, -1.5),
                         domain_max=(1.5, 3.0, 1.5), voxel_size=0.1)
if FLUID not in existing_domains:
    rt.fluid.create_domain(FLUID, domain_min=(-1, 0, -1), domain_max=(1, 2, 1),
                           voxel_size=0.1)

# â˜… Give the gas shader a HAND-TUNED value before the test runs. A pristine
# preset would make "restored exactly" pass even if the restore reinstalled the
# recipe and threw the tuning away â€” which is exactly the bug this test caught.
rt.gas.set_shader(GAS, preset="smoke")
rt.gas.set_shader(GAS, scattering_coefficient=0.15)

if MATERIAL not in [m["name"] for m in rt.material.list()]:
    MATERIAL = rt.material.create("principled", MATERIAL)

print("gas=%s fluid=%s material=%s" % (GAS, FLUID, MATERIAL))
print("gas shader: %s" % (rt.gas.get_shader(GAS),))
print("fluid surface material: %r" % (
    [d for d in rt.fluid.list_domains() if d["name"] == FLUID][0]["surface_material"],))
print("SETUP OK")
