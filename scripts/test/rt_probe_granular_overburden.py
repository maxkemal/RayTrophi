# Why does a soft granular pile report "not below load"?
#
# rt_test_granular_soft_stability.py fails on its validity gate: an 800 Pa pile
# ends the run with overburden = 5 Pa and below_load = False, when the whole
# point of the case is that 800 Pa cannot hold a metre of sand.
#
# ★ The discriminating question is not "is the gate wrong" but "WHEN was it
# measured". overburden = column_height * rho * g, and column_height is the
# vertical extent of the MATERIAL. 5 Pa means a column of 0.3 mm -- so either
# the pile flattened completely, or the height measurement is broken. Those two
# look identical in the final number and are told apart only by the trajectory.
#
# If overburden starts high and decays, the gate is measuring a pile that has
# ALREADY collapsed -- i.e. the warning switches itself off exactly when the
# failure it predicts has happened.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("granular_overburden")
log = rt_testlog.log

DOMAIN = "GranularOverburdenProbe"
DT = 1.0 / 24.0
STEPS = 120
DENSITY = 1600.0   # kGranularDensity
GRAVITY = 9.81

for d in rt.fluid.list_domains():
    if d["name"] != DOMAIN and d.get("enabled", True):
        # fluid.step advances every enabled domain; a neighbour would make the
        # numbers below describe a scene, not this pile.
        rt.fluid.set_param(d["name"], enabled=False)
        log("disabled neighbour domain: %s" % d["name"])

if DOMAIN not in [d["name"] for d in rt.fluid.list_domains()]:
    rt.fluid.create_domain(DOMAIN, domain_min=(-1, 0, -1), domain_max=(1, 3, 1),
                           voxel_size=0.05)

rt.fluid.set_param(DOMAIN, backend="vulkan", boundary="closed", preset="sand",
                   enabled=True, granular_enabled=True,
                   granular_friction_angle=35.0, granular_cohesion=0.0,
                   granular_dilatancy=5.0, granular_young_modulus=800.0,
                   granular_poisson_ratio=0.25, granular_tensile_cutoff=0.0,
                   granular_hardening=0.0, granular_rebonding=False,
                   granular_max_solver_substeps=16)
rt.fluid.clear(DOMAIN, clear_seed=True)
rt.fluid.reset()
rt.fluid.seed(DOMAIN, seed_min=(-0.40, 1.20, -0.40), seed_max=(0.40, 2.20, 0.40),
              particles_per_cell=4, replace=True, persistent=False)

log("seeded a column 1.00 m tall; at rho=%.0f that is %.0f Pa of overburden" %
    (DENSITY, 1.00 * DENSITY * GRAVITY))

rows = []
for step in range(1, STEPS + 1):
    rt.fluid.step(DT)
    g = rt.fluid.get(DOMAIN)
    overburden = g["granular_overburden_pressure"]
    rows.append((step,
                 g["particle_count"],
                 overburden,
                 overburden / (DENSITY * GRAVITY),   # implied column height, m
                 g["granular_young_modulus_for_load"],
                 g["granular_stiffness_below_load"]))

log("")
log("step  particles  overburden(Pa)  implied h(m)  E needed(Pa)  below_load")
for r in rows:
    if r[0] % 10 and r[0] != 1:
        continue
    log("%4d  %9d  %14.1f  %12.4f  %12.1f  %s" % r)

first = rows[0]
last = rows[-1]
peak = max(rows, key=lambda r: r[2])
ever_below = any(r[5] for r in rows)

log("")
log("first: h=%.4f m  below_load=%s" % (first[3], first[5]))
log("peak:  h=%.4f m at step %d  below_load=%s" % (peak[3], peak[0], peak[5]))
log("last:  h=%.4f m  below_load=%s" % (last[3], last[5]))
log("below_load was true at some point: %s" % ever_below)

if peak[3] > 0.1 and last[3] < 0.01:
    log("VERDICT: the column COLLAPSED (%.2f m -> %.4f m). The gate is measuring "
        "a pile that has already failed, so it reports 'not below load' exactly "
        "when the failure it predicts has happened. The final-frame assertion in "
        "rt_test_granular_soft_stability is testing the wrong instant." %
        (peak[3], last[3]))
elif peak[3] <= 0.1:
    log("VERDICT: the column height never read above 0.1 m even at the start, "
        "although a 1.00 m column was seeded. That points at the HEIGHT "
        "MEASUREMENT, not at the pile.")
else:
    log("VERDICT: the column did not collapse (%.2f m -> %.2f m), so a near-zero "
        "final overburden is not explained by flattening. Look at measureLoad." %
        (peak[3], last[3]))
