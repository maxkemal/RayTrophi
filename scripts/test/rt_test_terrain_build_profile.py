# Terrain build cost profile — the 0th item of the SatMap roadmap checklist.
#
# ★★★ This script does not optimize anything. It answers ONE question that has
# to be answered before any terrain optimization is worth writing:
#
#     of the time spent producing a terrain, how much is the node graph, how
#     much is filling the mesh, and how much is building the acceleration
#     structure?
#
# The roadmap (docs/dev/TERRAIN_SATMAP_COLORIZER_ROADMAP.md, Ek A) predicts that
# at 4k the BVH/BLAS build dominates the mesh fill. If that is true, further
# mesh-fill work is wasted effort — and "the terrain is slow" is a symptom, not
# a measurement.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("terrain_build_profile")
log = rt_testlog.log

FAIL = []
UNVERIFIED = []


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else ("  FAIL " + label +
        ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def vacuous(label, reason):
    """A claim whose preconditions never appeared. NOT a pass."""
    log("  ????  " + label + " -- NOT VERIFIED: " + reason)
    UNVERIFIED.append(label)


# --------------------------------------------------------------------------
# 1. The measurement surface itself
# --------------------------------------------------------------------------
log("-- perf surface --")

rt.perf.reset()
check("perf.list is empty right after reset", len(rt.perf.list()) == 0,
      "%d section(s) left" % len(rt.perf.list()))

# ★ A missing section must read as missing, not as zero. This is the single
# most important property of the whole registry: a zeroed timing reads as
# "measured, and it cost nothing", which is how an absent measurement quietly
# becomes a false one.
check("an unrecorded section reads as None, not as zeros",
      rt.perf.get("terrain.no.such.section") is None)


# --------------------------------------------------------------------------
# 2. Build a terrain and read where the time went
# --------------------------------------------------------------------------
RESOLUTION = int(os.environ.get("RT_TERRAIN_PROFILE_RES", "2048"))
NAME = "ProfileTerrain"

log("")
log("-- building a %dx%d terrain --" % (RESOLUTION, RESOLUTION))

for existing in rt.terrain.list():
    if existing["name"] == NAME or existing["name"] == NAME + "_2":
        rt.terrain.remove(existing["name"])

rt.perf.reset()
rt.terrain.create(name=NAME, resolution=RESOLUTION, size=1000.0, height_scale=120.0)

sections = {s["name"]: s for s in rt.perf.list()}
for name in sorted(sections):
    s = sections[name]
    log("    %-46s %8.1f ms  x%-3d  RSS %+7.0f MB" %
        (name, s["last_ms"], s["count"], s["last_rss_delta_mb"]))

check("terrain.mesh_fill was recorded", "terrain.mesh_fill" in sections)
check("the create branch ran (not the in-place update branch)",
      "terrain.mesh_fill.create" in sections,
      "sections: %s" % sorted(sections))

# --------------------------------------------------------------------------
# 3. The actual question: which phase dominates?
# --------------------------------------------------------------------------
log("")
log("-- where the time went --")

mesh_ms = sections.get("terrain.mesh_fill", {}).get("last_ms", 0.0)
graph_ms = sections.get("terrain.graph.evaluate", {}).get("last_ms", 0.0)

# The acceleration structure is built by the FRAME LOOP, not by the call above:
# updateTerrainMesh only raises g_bvh_rebuild_pending. So its section cannot
# exist yet at this point in a synchronous script — it appears one or more
# frames later.
#
# ★★★ This is the producer≠consumer split that this repository keeps paying
# for, and the reason this check reports NOT VERIFIED rather than 0.0: a script
# is blind to the frame loop, and "the BVH section is missing" means "ask
# again after a frame", never "the BVH was free".
bvh = None
for name, s in sections.items():
    if "rebuildBVH" in name or "rebuildBackendGeometry" in name:
        if bvh is None or s["last_ms"] > bvh["last_ms"]:
            bvh = s

if bvh is None:
    vacuous("BVH/backend build cost was compared against mesh fill",
            "no acceleration-structure section recorded yet -- it is built by "
            "the frame loop, so re-read rt.perf.list() after the viewport has "
            "drawn a frame (or call perf.list over IPC once the app is idle)")
    log("    mesh fill: %.1f ms, graph: %.1f ms, accel: not yet recorded" %
        (mesh_ms, graph_ms))
else:
    log("    mesh fill: %.1f ms, graph: %.1f ms, accel(%s): %.1f ms" %
        (mesh_ms, graph_ms, bvh["name"], bvh["last_ms"]))
    if bvh["last_ms"] > mesh_ms:
        log("    -> ACCELERATION STRUCTURE DOMINATES. Further mesh-fill")
        log("       optimization buys little; the lever is mesh_resolution")
        log("       (roadmap Faz 0), because it cuts triangle count directly.")
    else:
        log("    -> mesh fill dominates; the allocation path is the lever.")

# A pure sanity floor: a 2k+ terrain that reports sub-millisecond mesh fill
# means the timer is not wrapping the work it claims to wrap.
if RESOLUTION >= 1024:
    check("mesh fill cost is plausible for the resolution (> 1 ms)",
          mesh_ms > 1.0, "%.3f ms at %d^2" % (mesh_ms, RESOLUTION))
else:
    vacuous("mesh fill cost is plausible for the resolution",
            "resolution %d is too small for the floor to mean anything" % RESOLUTION)

# --------------------------------------------------------------------------
# 4. Counters accumulate rather than overwrite
# --------------------------------------------------------------------------
log("")
log("-- counter semantics --")

before = rt.perf.get("terrain.mesh_fill")
# A second, deliberately small terrain: the point is that the counter counts,
# not that the second build is expensive.
rt.terrain.create(name=NAME + "_2", resolution=256, size=200.0, height_scale=20.0)
after = rt.perf.get("terrain.mesh_fill")

if before is None or after is None:
    vacuous("repeated work increments the section count",
            "terrain.mesh_fill missing before/after the second build")
else:
    check("repeated work increments the section count",
          after["count"] > before["count"],
          "%d -> %d" % (before["count"], after["count"]))
    check("total_ms accumulates across calls",
          after["total_ms"] >= before["total_ms"],
          "%.1f -> %.1f" % (before["total_ms"], after["total_ms"]))
    check("seq advances so the newer record is identifiable",
          after["seq"] > before["seq"],
          "%d -> %d" % (before["seq"], after["seq"]))

rt.terrain.remove(NAME + "_2")
rt.terrain.remove(NAME)

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
else:
    log("RESULT: ALL PASSED")
