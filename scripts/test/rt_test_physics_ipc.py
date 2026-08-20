"""Physics validation over IPC - the channel that can see the frame loop.

★★★ This file is not a duplicate of rt_test_physics_validation.py, and the
difference is the whole point.

That file runs INSIDE the application through script.run_file, which means it
holds the main thread: the frame loop never turns while it executes. This one
runs OUTSIDE, over the named pipe, so the application keeps rendering and
stepping between every call.

Measured 2026-08-19: the identical call sequence preserves 6760 particles in
the in-process channel and loses all 22932 here. Same engine, same methods,
opposite answers. Neither channel is the "real" one - they measure different
halves, and a repo maintained by one person needs both, because "works in the
app but not in the test" is otherwise unfalsifiable.

Run it with the app open:

    python scripts/test/rt_test_physics_ipc.py

★ Not copied to x64/Release/scripts. That directory is for scripts the app
LOADS; this one drives the app from outside and would never be read from there.
"""
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc, RtIpcError  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "_physics_ipc_result.txt")
LINES = []
FAIL = []
UNVERIFIED = []


def log(text=""):
    # The file keeps the real text; the console gets an ASCII fallback. A
    # Turkish console codepage cannot encode the star marks and would kill the
    # run mid-case - losing the results to a formatting detail.
    LINES.append(text)
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode("ascii", "replace").decode("ascii"))


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else
        ("  FAIL " + label + ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def vacuous(label, reason):
    """A check whose preconditions are absent has NOT passed."""
    log("  ????  " + label + " -- NOT VERIFIED: " + reason)
    UNVERIFIED.append(label)


def close(label, measured, expected, tol, unit=""):
    err = abs(measured - expected)
    rel = err / max(abs(expected), 1e-9)
    log("       measured %.5f%s  expected %.5f%s  err %.5f (%.2f%%)"
        % (measured, unit, expected, unit, err, 100.0 * rel))
    check(label, rel <= tol,
          "relative error %.3f%% exceeds %.3f%%" % (100.0 * rel, 100.0 * tol))


G_EARTH = 9.81
G_MOON = 1.62
RUN = str(int(time.time()))[-6:]


def unique(stem):
    # Same reason as the in-process rig: scene.delete only MARKS an object
    # pending-delete, so reusing a name races the loop's actual removal.
    return "%s_%s" % (stem, RUN)


rt = RtIpc()


# ---------------------------------------------------------------------------
log("0. the application's frame loop actually turns between calls")
# ---------------------------------------------------------------------------
# ★★★ This runs FIRST and everything below depends on it.
#
# The entire justification for this file is that the loop runs between IPC
# calls. If it does not - the window is minimised, the app is paused, the
# render thread is wedged - then every case below silently degrades into a
# slower copy of the in-process rig, and a green run would mean the opposite
# of what it appears to mean.
#
# viewport.samples advances once per accumulated frame in a path-traced
# shading mode, and is pinned at 0 in 'solid'. So: switch to rendered, reset
# accumulation, and require the counter to move on its own between two reads.
# ★ The check has to be an actual MEASUREMENT of liveness, not the assumption
# that opening a pipe implies a running app.
prior_shading = rt.call("viewport.status").get("shading", "solid")
loop_alive = False
try:
    rt.call("viewport.set_shading", mode="rendered")
    rt.call("reset_accumulation")
    first = rt.call("viewport.status").get("samples", 0)
    deadline = time.time() + 5.0
    last = first
    while time.time() < deadline:
        time.sleep(0.2)
        last = rt.call("viewport.status").get("samples", 0)
        if last > first:
            loop_alive = True
            break
    log("       viewport samples %d -> %d over %s"
        % (first, last, "up to 5 s"))
except RtIpcError as exc:
    log("       liveness probe could not run: %s" % exc)

check("the frame loop advances while this script waits", loop_alive,
      "samples did not move - the app is not rendering, so every case below "
      "measures the same thing the in-process rig already measures, and this "
      "file's reason to exist is absent")

if not loop_alive:
    log("")
    log("ABORTED: without a turning frame loop this channel proves nothing "
        "the in-process rig did not already prove. Not reporting case results.")
    rt.call("viewport.set_shading", mode=prior_shading)
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(LINES) + "\n")
    sys.exit(1)


def sim_y(name):
    d = rt.call("scene.get_world_transform", name=name)
    return d["translation"][1], bool(d.get("simulated", False))


def spawn(stem, y0, gravity, mass=1.0):
    name = unique(stem)
    rt.call("scene.add_primitive", type="cube", name=name)
    rt.call("scene.set_transform", name=name, translation=[0.0, y0, 0.0])
    rt.call("physics.set_gravity", gravity=[0.0, -gravity, 0.0])
    rt.call("physics.add_body", object=name, motion_type="dynamic", mass=mass)
    SPAWNED.append(name)
    return name


SPAWNED = []
DOMAINS = []

# ---------------------------------------------------------------------------
log("1. ★★★ does physics.step SURVIVE to the next call?")
# ---------------------------------------------------------------------------
# ★★★★ This is the case the in-process rig structurally cannot run, and it
# is the reason this file exists.
#
# In-process, 240 x physics.step drops a body 4.84 m and the rig reports a
# clean analytical match. Over IPC the same sequence measures y = 50.00000 and
# simulated = false - the body never moved at all.
#
# It DID move. Measured 2026-08-19 by putting the read inside the same batch as
# the steps, so no frame passes between them: y = 49.70980 after 58 steps,
# which is 0.29 m in 0.24 s and matches gravity exactly. Read one call later,
# it is back at 50.0 with simulated = false.
#
# So the frame loop reverts the scripted step before the caller can observe it.
# The step is not ignored - it is UNDONE, which is worse, because the call
# returns success. An agent driving this engine does real work, gets told it
# worked, and measures nothing.
#
# The check compares the two reads directly rather than asserting a fall
# distance: what is broken is not the physics, it is whether a scripted step
# means anything from outside.
name = spawn("IPC_Survive", 50.0, G_EARTH)
STEPS = 58
inside = None
batch = [{"method": "physics.step", "params": {"dt": 1.0 / 240.0}}
         for _ in range(STEPS)]
batch.append({"method": "scene.get_world_transform", "params": {"name": name}})
res = rt.call("batch", calls=batch)
rows = res["results"] if isinstance(res, dict) and "results" in res else res
tail = rows[-1]
payload = tail.get("result", tail) if isinstance(tail, dict) else tail
if isinstance(payload, dict) and "translation" in payload:
    inside = payload["translation"][1]
after, after_sim = sim_y(name)

if inside is None:
    vacuous("a scripted physics.step survives to the next call",
            "the in-batch read returned no transform, so there is nothing to "
            "compare the post-batch read against")
else:
    moved_inside = 50.0 - inside
    log("       read INSIDE the batch : y=%.5f (fell %.5f m in %.3f s)"
        % (inside, moved_inside, STEPS / 240.0))
    log("       read AFTER  the batch : y=%.5f simulated=%s" % (after, after_sim))
    expected = 0.5 * G_EARTH * (STEPS / 240.0) ** 2
    if moved_inside < 0.5 * expected:
        vacuous("a scripted physics.step survives to the next call",
                "the body did not move even within the batch, so this is not "
                "the loop reverting it - it is a stepping failure")
    else:
        # ★★★ The tolerance is a fraction of the MOTION, not of the height.
        # Written first as 1% of the absolute y, this check passed while the
        # body was being fully reverted: 0.29 m of revert is nothing next to
        # 49.7 m of altitude. A tolerance scaled to the wrong quantity is how
        # a test reports green on the exact failure it was written for.
        reverted = abs(after - inside)
        check("a scripted physics.step survives to the next call",
              reverted < 0.05 * moved_inside,
              "the step moved the body %.5f m and the frame loop put %.5f m of "
              "it back (%.0f%%): %.5f -> %.5f. The call returned success, so a "
              "caller driving this engine does real work and measures nothing"
              % (moved_inside, reverted, 100.0 * reverted / moved_inside,
                 inside, after))

# ---------------------------------------------------------------------------
log("1b. physics.step moves the PLAYHEAD, and the epoch says who is driving")
# ---------------------------------------------------------------------------
# ★★★ The contract that replaced the fight, decided 2026-08-19.
#
# Two clocks disagreed - the script at t = 0.24 s, the loop at frame 0 - and
# the loop corrected the world to match itself. The fix is not an arbiter, it
# is one clock: stepping advances the playhead too, so there is no
# disagreement left to correct.
#
# ★★ The timeline stays the USER's. This checks NOTICE, not ownership: after
# a scrub the epoch must have moved and the driver must read "user", so a
# script can tell "my measurement was invalidated" from "the body did not
# move". Those two used to be identical, which is what made this expensive.
ok_state, state0 = rt.try_call("sim.control_state")
if not ok_state:
    vacuous("physics.step advances the playhead and reports its driver",
            "sim.control_state is missing - this build predates the "
            "control-state contract")
else:
    name = spawn("IPC_Claim", 50.0, G_EARTH)
    frame_before = rt.call("timeline.get_frame")
    epoch_before = rt.call("sim.control_state")["epoch"]
    for _ in range(48):                       # 0.2 s at 1/240 -> ~4 frames @24
        rt.call("physics.step", dt=1.0 / 240.0)
    st = rt.call("sim.control_state")
    frame_after = rt.call("timeline.get_frame")
    log("       frame %s -> %s, driver=%s script_driving=%s epoch %s -> %s"
        % (frame_before, frame_after, st["driver"], st["script_driving"],
           epoch_before, st["epoch"]))
    check("physics.step advances the playhead and reports its driver",
          frame_after > frame_before and st["driver"] == "script"
          and st["epoch"] > epoch_before,
          "the playhead did not follow the solver, or the driver was not "
          "reported as the script - which is the disagreement the frame loop "
          "used to resolve by erasing the scripted motion")

    # ★ The user taking the timeline back must be VISIBLE to the script.
    rt.call("timeline.set_frame", frame=0)
    st2 = rt.call("sim.control_state")
    log("       after a scrub: driver=%s script_driving=%s epoch=%s"
        % (st2["driver"], st2["script_driving"], st2["epoch"]))
    check("a scrub takes the timeline back and the script can see it",
          st2["epoch"] > st["epoch"] and not st2["script_driving"],
          "the epoch did not move on a scrub, so a script cannot tell an "
          "invalidated measurement from a body that never moved")

# ---------------------------------------------------------------------------
log("2. driving physics through the TIMELINE instead")
# ---------------------------------------------------------------------------
# ★★ If the loop owns the rigid state, the timeline is the way to drive it,
# and the descriptor says so: "Playhead; stepping it advances the solvers".
# Measured: this DOES survive - the pose is still there on the next call.
#
# But it stops advancing. Frames 6 and 12 move the body; 24 and 48 return the
# same number as 12. Either a per-call step budget or a stale rigid frame cache
# is serving an old pose, and a caller reading only the final frame would see a
# body that "settled" in mid-air - a plausible, completely wrong observation.
name = spawn("IPC_Timeline", 50.0, G_EARTH)
seen = []
for frame in (6, 12, 24, 48):
    rt.call("timeline.set_frame", frame=frame)
    time.sleep(0.4)
    y, sim = sim_y(name)
    seen.append((frame, y, sim))
    log("       frame %3d -> y=%.5f simulated=%s" % (frame, y, sim))
rt.call("timeline.set_frame", frame=0)

if not any(s for _, _, s in seen):
    vacuous("the timeline keeps advancing the body, frame after frame",
            "no solver contributed at any frame - the timeline is not driving "
            "physics at all here")
else:
    # ★★ STRICTLY decreasing. Written first with a 1e-4 slack this passed on
    # three identical readings - the plateau IS the defect, and a tolerance
    # that admits equality cannot see it. Later frames must be strictly lower:
    # a body under gravity does not stop falling in mid-air.
    advancing = all(seen[i + 1][1] < seen[i][1] - 1e-3
                    for i in range(len(seen) - 1))
    check("the timeline keeps advancing the body, frame after frame",
          advancing,
          "the fall stops: frames %s report heights %s. A caller reading only "
          "the last frame sees a body that settled in mid-air"
          % ([f for f, _, _ in seen],
             ["%.5f" % y for _, y, _ in seen]))

# ---------------------------------------------------------------------------
log("3. seeding honesty over the IPC dispatch (its own code path)")
# ---------------------------------------------------------------------------
# ★★ Not a duplicate of the in-process case. The optional-region logic lives
# in TWO places - RtIpc.cpp decides from params.contains(), RtPython.cpp from
# py::none() - and they are separate code. A fix verified on one says nothing
# about the other, which is the ordinary way a two-channel API drifts.
DOM = unique("IPC_Tank")
rt.call("fluid.create_domain", name=DOM, domain_min=[-1.0, 0.0, -1.0],
        domain_max=[1.0, 2.0, 1.0], voxel_size=0.08)
DOMAINS.append(DOM)
refused, message = rt.try_call(
    "fluid.seed", domain=DOM, seed_min=[-0.4, 50.0, -0.4],
    seed_max=[0.4, 60.0, 0.4], particles_per_cell=4)
seeded = rt.call("fluid.get", domain=DOM).get("particle_count", 0)
log("       zero-overlap region -> %d particles, call %s"
    % (seeded, "succeeded" if refused else "refused"))
if refused:
    check("a seed that creates nothing is refused", False,
          "the IPC dispatch still reports success having created nothing")
else:
    log("       %s" % message[:150])
    # ★ The refusal must carry PARSEABLE numbers. setlocale(LC_ALL,"Turkish")
    # makes the printf family emit a decimal COMMA; this message came out as
    # "region (-0,400 5,000 ..." before it was imbued with the classic locale.
    # A number a script has to read must not change shape with the UI language.
    # ★★ Look for a comma BETWEEN DIGITS, not for any comma. Written first as
    # `"," not in message` this failed on the English prose comma in "does not
    # overlap the domain, so it would create no particles" - a red result on a
    # message whose numbers were all correct. A check that bans a character
    # instead of the pattern it cares about reports the wrong thing broken.
    check("the refusal reports numbers a script can parse",
          re.search(r"\d,\d", message) is None,
          "the message carries a decimal comma, so its numbers are "
          "locale-dependent: " + message[:120])

# ---------------------------------------------------------------------------
log("4. the default seed region comes from the domain, over IPC")
# ---------------------------------------------------------------------------
FAR = unique("IPC_Far")
rt.call("fluid.create_domain", name=FAR, domain_min=[-1.0, 20.0, -1.0],
        domain_max=[1.0, 22.0, 1.0], voxel_size=0.1)
DOMAINS.append(FAR)
rt.call("fluid.seed", domain=FAR, particles_per_cell=4)
far = rt.call("fluid.get", domain=FAR)
if not far.get("live_state", False):
    vacuous("omitting the seed region fills the domain rather than a fixed box",
            "fluid.get could not measure this domain, so zero would not mean "
            "empty")
else:
    log("       domain at y 20-22, no region given -> %d particles"
        % far.get("particle_count", 0))
    check("omitting the seed region fills the domain rather than a fixed box",
          far.get("particle_count", 0) > 0,
          "the default region produced nothing - the old hardcoded box at "
          "y 1.0-1.5 does not reach this domain")

# ---------------------------------------------------------------------------
log("5. ★★★ a second domain must not destroy the first one's particles")
# ---------------------------------------------------------------------------
# ★★★★ THIS IS WHY THIS FILE EXISTS. The in-process rig runs this exact
# sequence and passes; here it lost every particle, 5 runs out of 5. The
# difference is the frame loop, which turns between these calls and never turns
# inside a script.
#
# Both readers are consulted on purpose. fluid.get resolves through the ACTIVE
# particle system only and has a documented history of reporting 0 for a domain
# that list_domains reported as populated at the same instant - absence of
# measurement read as a measured zero. If the two disagree, this is a reader
# fault and NOT particle loss, and the case says so instead of guessing.
FIRST, SECOND = unique("IPC_DomA"), unique("IPC_DomB")
rt.call("fluid.create_domain", name=FIRST, domain_min=[-1.0, 0.0, -1.0],
        domain_max=[1.0, 2.0, 1.0], voxel_size=0.08)
DOMAINS.append(FIRST)
rt.call("fluid.seed", domain=FIRST, seed_min=[-0.8, 0.05, -0.8],
        seed_max=[0.8, 1.0, 0.8], particles_per_cell=4)


def particles(domain):
    """(get_count, list_count, live) - two independent readers."""
    info = rt.call("fluid.get", domain=domain)
    listed = None
    for d in rt.call("fluid.list_domains").get("domains", []):
        if d.get("name") == domain:
            listed = d.get("particle_count")
            break
    return info.get("particle_count", 0), listed, info.get("live_state", False)


g0, l0, live0 = particles(FIRST)
log("       first domain seeded: fluid.get %d, list_domains %s" % (g0, l0))
if not live0 or g0 <= 0:
    vacuous("creating a second domain leaves the first one's particles alone",
            "the first domain reports no measurable particles, so there is "
            "nothing to lose")
else:
    rt.call("fluid.create_domain", name=SECOND, domain_min=[-0.5, 0.0, -0.5],
            domain_max=[0.5, 1.0, 0.5], voxel_size=0.05)
    DOMAINS.append(SECOND)
    g1, l1, live1 = particles(FIRST)
    log("       after the second domain: fluid.get %d, list_domains %s" % (g1, l1))
    if l1 is not None and l1 != g1:
        vacuous("creating a second domain leaves the first one's particles alone",
                "the two readers disagree (%d vs %d) - that is a reader fault, "
                "not measured particle loss" % (g1, l1))
    else:
        check("creating a second domain leaves the first one's particles alone",
              g1 == g0,
              "lost %d of %d particles (%.1f%%) to an unrelated domain being "
              "created - the in-process channel does NOT show this"
              % (g0 - g1, g0, 100.0 * (g0 - g1) / g0))

# ---------------------------------------------------------------------------
for name in SPAWNED:
    rt.try_call("scene.delete", name=name)
for dom in DOMAINS:
    rt.try_call("fluid.remove_domain", domain=dom)
rt.try_call("viewport.set_shading", mode=prior_shading)

log("")
if FAIL:
    log("FAILED: " + ", ".join(FAIL))
if UNVERIFIED:
    log("NOT VERIFIED: " + ", ".join(UNVERIFIED))
if not FAIL and not UNVERIFIED:
    log("ALL PASSED")

with open(OUT, "w", encoding="utf-8") as fh:
    fh.write("\n".join(LINES) + "\n")
log("")
log("written to %s" % OUT)
rt.close()
sys.exit(1 if FAIL else 0)
