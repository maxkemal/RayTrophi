# Minimal seeded+stepped fluid domain for rt_test_sim_graph.py.
#
# ★ The N0 contract ("evaluating the graph must not disturb the solver") cannot
# be tested on an empty domain: resetting an empty sim is indistinguishable from
# not resetting it. This script exists so the graph test has live particles and
# live per-particle attributes to name.
import rt

NAME = "SimGraphTestDomain"

existing = [d["name"] for d in rt.fluid.list_domains()]
if NAME in existing:
    rt.fluid.remove_domain(NAME)

info = rt.fluid.create_domain(NAME,
                              domain_min=(-1, 0, -1),
                              domain_max=(1, 2, 1),
                              voxel_size=0.1)
print("created %s voxel_size=%.3f" % (info["name"], info["voxel_size"]))

rt.fluid.seed(NAME,
              seed_min=(-0.5, 0.8, -0.5),
              seed_max=(0.5, 1.6, 0.5),
              particles_per_cell=4)
rt.fluid.set_param(NAME, backend="gpu", preset="water", boundary="closed")

for _ in range(8):
    rt.fluid.step(0.0166)

# ★ A flow source, so the Emitter node path is actually MEASURED. Without one
# the emitter half of rt_test_sim_graph.py reports NOT VERIFIED -- and a claim
# that never ran is not a passing claim.
EMITTER = "SimGraphTestEmitter"
if any(src["name"] == EMITTER for src in rt.flow_source.list()):
    rt.flow_source.remove(EMITTER)
rt.flow_source.create(EMITTER, NAME,
                      source_mode="point",
                      position=(0.0, 1.2, 0.0),
                      radius=0.3,
                      density=1.0,
                      fluid_particles_per_second=500.0)
print("emitter: %s" % ([src["name"] for src in rt.flow_source.list()],))

d = rt.fluid.get(NAME)
print("particles=%d backend=%s preset=%s live=%s" % (
    d["particle_count"], d["backend"], d["preset"], d.get("live_state")))
print("attributes: %s" % (rt.attr.list("domain", NAME),))

if d["particle_count"] <= 0:
    print("SETUP FAILED: no particles after seeding -- graph test would be vacuous")
else:
    print("SETUP OK")
