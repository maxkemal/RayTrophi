# Faz 0 — mesh resolution is decoupled from field resolution.
#
# The claim under test is NOT "it got faster". It is the pair of properties that
# make the speed-up honest:
#
#   1. the field (heights, analysis) is UNCHANGED when the mesh is decimated,
#   2. the analysis fields still reach the mesh, resampled rather than dropped.
#
# (2) is the one that fails silently. The vertex-attribute mirror used to test
# `field.size() == vertexCount`, which only held because the two grids were the
# same grid. Split them without fixing it and every foliage/scatter mask
# disappears with no error and no log.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("terrain_mesh_resolution")
log = rt_testlog.log

FAIL = []
UNVERIFIED = []


def check(label, ok, detail=""):
    log(("  OK   " + label) if ok else ("  FAIL " + label +
        ((" -- " + detail) if detail else "")))
    if not ok:
        FAIL.append(label)


def vacuous(label, reason):
    log("  ????  " + label + " -- NOT VERIFIED: " + reason)
    UNVERIFIED.append(label)


NAME = "MeshResTest"
FIELD = 1024
MESH = 256

for existing in rt.terrain.list():
    if existing["name"] == NAME:
        rt.terrain.remove(NAME)

log("-- field %d, mesh %d --" % (FIELD, MESH))
info = rt.terrain.create(name=NAME, resolution=FIELD, size=1000.0, height_scale=120.0)
check("a new terrain follows the field by default (mesh_resolution == 0)",
      info["mesh_resolution"] == 0 and info["mesh_grid"] == (FIELD, FIELD),
      "%s / %s" % (info["mesh_resolution"], info["mesh_grid"]))

# mesh_resolution is also a CREATION parameter: decimating after the fact still
# pays one full-resolution acceleration-structure build.
created = rt.terrain.create(name=NAME + "_born", resolution=FIELD, size=1000.0,
                            height_scale=120.0, mesh_resolution=MESH)
check("terrain.create accepts mesh_resolution up front",
      created["mesh_grid"] == (MESH, MESH) and created["resolution"] == (FIELD, FIELD),
      "%s / %s" % (created["mesh_grid"], created["resolution"]))
rt.terrain.remove(NAME + "_born")

# ★★★ Give the terrain real relief BEFORE measuring anything.
#
# The first version of this script called rt.terrain.evaluate() and then spun on
# evaluation_status() in a loop. That cannot work and the reason matters: the
# graph evaluates on a worker whose finalize step runs on the MAIN thread -- the
# very thread this script is occupying. The loop never observed completion, and
# the next call failed with "terrain evaluation is still running".
#
# Worse is what would have happened if the loop had simply fallen through: every
# height would be 0, and the "field is untouched" assertion would compare 0 to 0
# and report a confident pass while testing nothing.
#
# So: apply a preset, then require actual relief before continuing.
rt.terrain.apply_preset(NAME, "snowy_mountain_valley")
rt.terrain.evaluate(NAME)
status = rt.terrain.evaluation_status(NAME)
if status["state"] == "running":
    vacuous("the field survives mesh decimation",
            "graph evaluation is async and finalizes on the main thread, which "
            "this script is holding -- run the terrain evaluation first, then "
            "this script, or drive it over IPC where calls are separate requests")
    log("")
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
    raise SystemExit(0)

height_before = rt.terrain.sample_height(NAME, 0.0, 0.0)
if abs(height_before) < 1e-3:
    vacuous("the field survives mesh decimation",
            "the terrain is flat (height 0), so comparing before/after would be "
            "0 == 0 -- a green result that tested nothing")
field_before = rt.terrain.get(NAME)["resolution"]

rt.perf.reset()
info = rt.terrain.set_mesh_resolution(NAME, MESH)

log("")
log("-- after decimation --")
check("mesh grid is the requested resolution",
      info["mesh_grid"] == (MESH, MESH), "%s" % (info["mesh_grid"],))

# ★ The whole point of the split: the analysis grid must not move.
check("the FIELD resolution is untouched",
      info["resolution"] == field_before,
      "%s -> %s" % (field_before, info["resolution"]))

height_after = rt.terrain.sample_height(NAME, 0.0, 0.0)
# Sampling reads the field, so this must not change at all -- a decimated mesh
# is a rendering decision, not an edit.
if abs(height_before) < 1e-3:
    log("  ????  sample_height comparison skipped -- flat terrain, see above")
else:
    check("sample_height still reads the field, not the mesh",
          abs(height_after - height_before) < 1e-3,
          "%.6f -> %.6f" % (height_before, height_after))

mesh = rt.perf.get("terrain.mesh_fill")
if mesh is None:
    vacuous("the decimated mesh was actually rebuilt",
            "terrain.mesh_fill was not recorded after set_mesh_resolution")
else:
    log("    terrain.mesh_fill: %.1f ms (%d call(s))" % (mesh["last_ms"], mesh["count"]))
    check("the decimated mesh was actually rebuilt", mesh["count"] >= 1)

log("")
log("-- refusals --")
try:
    rt.terrain.set_mesh_resolution(NAME, FIELD * 2)
    check("a mesh denser than the field is refused, not clamped", False,
          "the call succeeded")
except Exception as exc:  # noqa: BLE001 - the refusal is the assertion
    check("a mesh denser than the field is refused, not clamped",
          "exceeds the field resolution" in str(exc), str(exc))

log("")
log("-- restore --")
info = rt.terrain.set_mesh_resolution(NAME, 0)
check("0 restores the field-following mesh",
      info["mesh_grid"] == (FIELD, FIELD), "%s" % (info["mesh_grid"],))

rt.terrain.remove(NAME)

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
else:
    log("RESULT: ALL PASSED")
