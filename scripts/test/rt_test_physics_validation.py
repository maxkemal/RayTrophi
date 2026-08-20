# Physics validation: is the solver RIGHT, not merely running.
#
# ★★★ Why this rig exists, and why it is different from every other test here.
#
# Every instrument in this repo answers "did it happen?" - did the call land,
# did the frame change, does the summary match the schema. None of them answers
# "is it true?". For a renderer that is fine: the eye judges. For a simulation
# substrate that agents are meant to LEARN from, it is the load-bearing gap -
# an agent that trusts a wrong solver learns wrong physics confidently, and the
# whole instrument stack reports success.
#
# So every case below compares against a number that is known WITHOUT running
# this program: an analytical solution, or a conservation law. A case with no
# independently-known answer does not belong in this file.
#
# ★★ Deliberately NOT routed through the simulation node graph. The node layer
# is the natural place to author these cases and that is the plan - but the
# instrument must not share a failure mode with the subject. An opt-in flag that
# silently fails to apply would make a solver case go red (or, worse, green with
# default parameters). This file is the reference; the node re-expression is the
# differential test that runs against it.
import math
import os
import sys
import time

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("physics_validation")
log = rt_testlog.log

FAIL = []
UNVERIFIED = []


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else
        ("  FAIL " + label + ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def vacuous(label, reason):
    # A check whose preconditions are absent has NOT passed.
    log("  ????  " + label + " -- NOT VERIFIED: " + reason)
    UNVERIFIED.append(label)


def close(label, measured, expected, tol, unit=""):
    """Compare against an independently-known number, and SHOW both."""
    err = abs(measured - expected)
    rel = err / max(abs(expected), 1e-9)
    log("       measured %.5f%s  expected %.5f%s  err %.5f (%.2f%%)"
        % (measured, unit, expected, unit, err, 100.0 * rel))
    check(label, rel <= tol,
          "relative error %.3f%% exceeds %.3f%%" % (100.0 * rel, 100.0 * tol))


G_EARTH = 9.81
G_MOON = 1.62

HAS_WORLD_TRANSFORM = hasattr(rt.scene, "get_world_transform")


def sim_y(name):
    """The SIMULATED height, and whether a solver actually produced it.

    ★★★ rt.scene.get_transform returns the AUTHORED pose, and the rigid solver
    never writes a transform at all - it bakes its motion into the mesh
    vertices. Measured 2026-08-19 over both channels: 240 steps under gravity
    and every transform reader still said y = 50.0, successfully, with entirely
    plausible numbers.

    ★★ The `simulated` flag is not decoration. Without it "fell 0 m" reads
    identically whether the body stood still or nothing was driving it - and the
    second one is not a physics result, it is a missing precondition.
    """
    d = rt.scene.get_world_transform(name)
    return d["translation"][1], bool(d.get("simulated", False))


def drop_object(name):
    try:
        rt.scene.delete(name)
    except Exception:                        # noqa: BLE001
        pass


def drop_domain(name):
    try:
        rt.fluid.remove_domain(name)
    except Exception:                        # noqa: BLE001
        pass


# ★★★ Every body this file spawns gets a run-unique name, and that is not
# tidiness. scene.delete only MARKS an object pending-delete; the frame loop
# does the physical removal - and a script holds the main thread, so that loop
# never runs while this file executes. Re-adding a just-deleted name therefore
# collides with a corpse and produces the half-existing object recorded in
# docs/dev/BUG_DELETED_NAME_REUSE_GHOST.md: add_primitive reports success and
# the very next set_transform says "object not found". Measured here on the
# second run of a session, after the first run had passed.
#
# ★★ Routing around an open bug in a validation file is only honest if the
# routing is visible, so: this suite does NOT test name reuse, and a green run
# says nothing about it.
RUN = str(int(time.time()))[-6:]


def unique(stem):
    return "%s_%s" % (stem, RUN)


def drop(stem, y0, mass, gravity, seconds, dt=1.0 / 240.0):
    """Spawn a body at y0, integrate, return the fall distance."""
    name = unique(stem)
    rt.scene.add_primitive("cube", name=name)
    rt.scene.set_transform(name, translation=(0.0, y0, 0.0))
    rt.physics.set_gravity((0.0, -gravity, 0.0))
    rt.physics.add_body(name, motion_type="dynamic", mass=mass)
    steps = int(round(seconds / dt))
    for _ in range(steps):
        rt.physics.step(dt)
    y, simulated = sim_y(name)
    return y0 - y, steps * dt, simulated


# ---------------------------------------------------------------------------
log("1. free fall against y(t) = y0 - g t^2 / 2")
# ---------------------------------------------------------------------------
# The most basic statement the solver can be wrong about. Damping is left at the
# body default on purpose: if the default damping makes free fall disagree with
# the analytical answer, that is a real thing to know, not something to hide by
# zeroing it first.
if not HAS_WORLD_TRANSFORM:
    vacuous("free fall matches the analytical drop",
            "rt.scene.get_world_transform is missing - this build predates the "
            "simulated-pose reader, and the authored transform cannot see "
            "solver motion")
else:
    fallen, t, simulated = drop("PV_Fall", 50.0, 1.0, G_EARTH, 1.0)
    if not simulated:
        vacuous("free fall matches the analytical drop",
                "no solver contributed to the pose - the body was never driven, "
                "so this measures the spawn point and not gravity")
    else:
        close("free fall matches the analytical drop",
              fallen, 0.5 * G_EARTH * t * t, 0.05, " m")

        # ★★★ The 5% band above is loose on purpose - it asks "is this gravity
        # at all". This second comparison asks the much harder question: is the
        # residual EXPLAINED, or merely small?
        #
        # The vacuum answer is 4.905 m; the solver produces ~4.844 m. A 1.25%
        # miss is exactly the size that gets waved through as "integrator
        # error" - and this repo has paid twice for numbers that were waved
        # through at that size. It is not integrator error. Two independent
        # terms account for it, neither of them fitted:
        #
        #   1. linear damping, declared 0.05 s^-1 in RigidBodySystem.h, whose
        #      closed form is y = (g/c)(t - (1 - e^(-ct))/c);
        #   2. semi-implicit Euler's half-step bias, +g*dt*t/2, which pushes
        #      the discrete answer ABOVE the continuous one.
        #
        # Both are known without running this program, so this belongs here.
        # Verified over two different intervals (1.0 s and 0.8 s) to ~0.02%.
        # ★ If this goes red while the case above stays green, the damping
        # default changed, or the integrator did - and nothing else in the repo
        # would have said so.
        c = 0.05
        predicted = (G_EARTH / c) * (t - (1.0 - math.exp(-c * t)) / c)             + 0.5 * G_EARTH * (1.0 / 240.0) * t
        close("the gap to the vacuum answer is fully explained by the declared "
              "damping, not left as slop", fallen, predicted, 0.005, " m")

# ---------------------------------------------------------------------------
log("2. fall is independent of mass (Galileo)")
# ---------------------------------------------------------------------------
# ★ This one guards a failure class this repo has already paid for twice: a
# force applied without dividing by mass, and an impulse written into a velocity
# field. Both look like "a bit too strong" and both vanish into a calibration
# round. Neither survives this comparison.
if not HAS_WORLD_TRANSFORM:
    vacuous("a 100 kg body falls exactly like a 1 kg body",
            "rt.scene.get_world_transform is missing")
else:
    light, _, sim_l = drop("PV_Light", 50.0, 1.0, G_EARTH, 0.8)
    heavy, _, sim_h = drop("PV_Heavy", 50.0, 100.0, G_EARTH, 0.8)
    log("       1 kg fell %.5f m, 100 kg fell %.5f m" % (light, heavy))
    # ★★★ Two bodies that both fell ZERO are equal, and this check passed on
    # exactly that in the first run (2026-08-19) while case 1 was red. An
    # equality that a total absence of motion satisfies proves nothing about
    # mass. Guard the precondition before comparing.
    if not (sim_l and sim_h):
        vacuous("a 100 kg body falls exactly like a 1 kg body",
                "no solver contributed to the poses")
    elif light <= 1e-6:
        vacuous("a 100 kg body falls exactly like a 1 kg body",
                "the 1 kg body did not move, so the equality is vacuous")
    else:
        close("a 100 kg body falls exactly like a 1 kg body",
              heavy, light, 0.01, " m")

# ---------------------------------------------------------------------------
log("3. gravity is a VALUE, not a baked constant")
# ---------------------------------------------------------------------------
# physics.set_gravity accepts a vector; nothing so far proves the solver reads
# it. Moon gravity must produce exactly the ratio g_moon/g_earth over the same
# interval - a solver with 9.81 hardcoded passes cases 1 and 2 and fails here.
if not HAS_WORLD_TRANSFORM:
    vacuous("moon gravity scales the drop by g_moon/g_earth",
            "rt.scene.get_world_transform is missing")
else:
    earth, _, _ = drop("PV_Earth", 50.0, 1.0, G_EARTH, 0.8)
    moon, _, _ = drop("PV_Moon", 50.0, 1.0, G_MOON, 0.8)
    if earth <= 1e-6:
        vacuous("moon gravity scales the drop by g_moon/g_earth",
                "the earth-gravity drop measured zero, so there is no ratio")
    else:
        close("moon gravity scales the drop by g_moon/g_earth",
              moon / earth, G_MOON / G_EARTH, 0.05)

# ---------------------------------------------------------------------------
log("4. fluid mass is conserved at rest")
# ---------------------------------------------------------------------------
# Conservation of mass. No emitter, no reseed, no outflow: the particle count
# after N steps must equal the count before, exactly. This is the invariant the
# reseed bugs broke - reseed CREATED particles rather than carrying them.
DOM = unique("PV_Tank")
rt.fluid.create_domain(name=DOM, domain_min=(-1.0, 0.0, -1.0),
                       domain_max=(1.0, 2.0, 1.0), voxel_size=0.08)
# ★ The seed region is given EXPLICITLY. The dispatch default is a hardcoded box
# at y 1.0-1.5 that is not derived from the domain, so a domain that does not
# happen to contain it seeds nothing and reports success (measured 2026-08-19).
# A validation case must never depend on a default it did not choose.
rt.fluid.seed(DOM, seed_min=(-0.8, 0.05, -0.8), seed_max=(0.8, 1.0, 0.8),
              particles_per_cell=4)
before = rt.fluid.get(DOM)
n0 = before.get("particle_count", 0)

if n0 <= 0:
    vacuous("particle count is conserved over 120 steps at rest",
            "the seed produced no particles, so there is no mass to conserve")
else:
    for _ in range(120):
        rt.fluid.step(1.0 / 60.0)
    after = rt.fluid.get(DOM)
    n1 = after.get("particle_count", 0)
    log("       %d particles before, %d after; reseed +%d/-%d"
        % (n0, n1, after.get("reseed_added_particles", 0),
           after.get("reseed_removed_particles", 0)))
    check("particle count is conserved over 120 steps at rest", n0 == n1,
          "mass changed by %d particles (%.3f%%)"
          % (n1 - n0, 100.0 * abs(n1 - n0) / n0))

# ---------------------------------------------------------------------------
log("5. seeding outside the domain must not report success")
# ---------------------------------------------------------------------------
# ★★★ OPEN BUG, encoded here so it goes green when fixed.
# The seed region is clamped to the domain, which is right; but when the overlap
# is EMPTY the call still returns success having created nothing. Measured
# 2026-08-19 on three independent zero-overlap regions (above the domain, below
# it, and starting exactly on the top face). For an agent that is silent lost
# work: it seeds, steps, measures nothing, and blames the solver.
#
# ★ Uses the SAME domain as case 4 on purpose - creating a second one destroys
# the first one's particles (case 6), and that would confuse this reading.
rt.fluid.clear(DOM)
outside_reported_ok = True
try:
    rt.fluid.seed(DOM, seed_min=(-0.4, 50.0, -0.4), seed_max=(0.4, 60.0, 0.4),
                  particles_per_cell=4)
except Exception as exc:                     # noqa: BLE001
    outside_reported_ok = False
    log("       seed refused: %s" % str(exc)[:90])
seeded = rt.fluid.get(DOM).get("particle_count", 0)
log("       region entirely above the domain -> %d particles, call %s"
    % (seeded, "succeeded" if outside_reported_ok else "failed"))
check("a seed that creates nothing is reported as a failure",
      not (outside_reported_ok and seeded == 0),
      "seed returned success having created 0 particles - the caller goes on to "
      "step an empty domain and blames the solver")

# ---------------------------------------------------------------------------
log("5b. the default seed region must come from the domain")
# ---------------------------------------------------------------------------
# ★★★ The other half of the same bug, and the half that bites silently.
# Omitting the region used to substitute a fixed box at y 1.0-1.5 that was
# derived from nothing. This domain sits at y 20-22, so the old default missed
# it entirely and seeded zero - with a success return.
#
# ★ The domain is placed deliberately far from the old box. A case that used a
# domain the old default happened to fit would pass either way, which is the
# same vacuity as comparing two bodies that both fell zero.
FAR = unique("PV_Far")
rt.fluid.create_domain(name=FAR, domain_min=(-1.0, 20.0, -1.0),
                       domain_max=(1.0, 22.0, 1.0), voxel_size=0.1)
rt.fluid.seed(FAR, particles_per_cell=4)
far_info = rt.fluid.get(FAR)
far_n = far_info.get("particle_count", 0)
if not far_info.get("live_state", False):
    vacuous("omitting the seed region fills the domain rather than a fixed box",
            "fluid.get could not measure this domain, so zero would not mean empty")
else:
    log("       domain at y 20-22, no region given -> %d particles" % far_n)
    check("omitting the seed region fills the domain rather than a fixed box",
          far_n > 0,
          "the default region produced nothing - it is not derived from the domain")
drop_domain(FAR)

# ---------------------------------------------------------------------------
log("6. a second domain must not destroy the first one's particles")
# ---------------------------------------------------------------------------
# ★★★ OPEN BUG, measured 2026-08-19. Creating a second fluid domain wipes the
# particles of the first. Both readers agree on zero - fluid.get AND
# fluid.list_domains - so this is real loss, not the dead-twin reader family.
#
# This is not an edge case: every coupling scenario this engine is built around
# ("Fuel burns and feeds Smoke") needs two domains at once.
# ★ Runs on its OWN pair of fresh domains, not on the case-4 tank. Measured
# during development: reusing a domain that earlier cases had already exercised
# made this check pass while the isolated sequence failed 5 times out of 5. A
# detector that only fires sometimes is worse than none - it teaches the reader
# that red means "flaky".
FIRST, SECOND = unique("PV_DomA"), unique("PV_DomB")
rt.fluid.create_domain(name=FIRST, domain_min=(-1.0, 0.0, -1.0),
                       domain_max=(1.0, 2.0, 1.0), voxel_size=0.08)
rt.fluid.seed(FIRST, seed_min=(-0.8, 0.05, -0.8), seed_max=(0.8, 1.0, 0.8),
              particles_per_cell=4)
before_second = rt.fluid.get(FIRST).get("particle_count", 0)
if before_second <= 0:
    vacuous("creating a second domain leaves the first one's particles alone",
            "the first domain seeded no particles, so there is nothing to lose")
else:
    rt.fluid.create_domain(name=SECOND, domain_min=(-0.5, 0.0, -0.5),
                           domain_max=(0.5, 1.0, 0.5), voxel_size=0.05)
    after_second = rt.fluid.get(FIRST).get("particle_count", 0)
    log("       first domain had %d particles, has %d after the second was created"
        % (before_second, after_second))
    # ★★★ THIS CHECK IS STRUCTURALLY BLIND TO THE KNOWN FAILURE, and saying so
    # is the point. Measured 2026-08-19: the identical sequence driven over IPC
    # loses every particle (22932 -> 0, five runs out of five), while inside a
    # script it is preserved (two out of two). The difference is the app's frame
    # loop, which runs between IPC calls and never runs while a script holds the
    # main thread.
    #
    # So a green here means "the in-script path is fine", NOT "the engine is
    # fine". The IPC path needs its own probe; a script can never reach this bug.
    # Same lesson as viewport.render_frames not publishing a frame: producer and
    # consumer are in different loops.
    log("       NOTE: script path only - the frame loop never runs here, and the "
        "IPC path loses every particle in the same sequence")
    check("creating a second domain leaves the first one's particles alone "
          "(in-script path only)",
          before_second == after_second,
          "lost %d particles (%.1f%%) to an unrelated domain being created"
          % (before_second - after_second,
             100.0 * (before_second - after_second) / before_second))
drop_domain(FIRST)
drop_domain(SECOND)

# ---------------------------------------------------------------------------
# Tear down this run's entities. They carry a run-unique suffix, so the next run
# spawns fresh names rather than colliding with whatever the frame loop has not
# finished removing yet.
for stem in ("PV_Fall", "PV_Light", "PV_Heavy", "PV_Earth", "PV_Moon"):
    drop_object(unique(stem))
for dom in (DOM, FIRST, SECOND):
    drop_domain(dom)

log("")
if FAIL:
    log("FAILED: " + ", ".join(FAIL))
if UNVERIFIED:
    log("NOT VERIFIED: " + ", ".join(UNVERIFIED))
if not FAIL and not UNVERIFIED:
    log("ALL PASSED")
elif not FAIL:
    # ★ Not "passed". A check whose preconditions were missing measured nothing,
    # and calling that a pass is the exact habit this file exists to break.
    log("NO FAILURES, but %d check(s) could not be verified" % len(UNVERIFIED))

