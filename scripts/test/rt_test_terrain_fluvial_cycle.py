# Landscape-evolution (fluvial) cycle acceptance test.
#
# The claim under test is NOT "erosion ran" and it is NOT "the render looks
# better". It is four properties, each of which fails SILENTLY -- every one of
# them produces a heightfield that looks entirely plausible:
#
#   1. The sediment ledger CLOSES. eroded == deposited + exported + carried.
#      A leak in downstream transport shows up nowhere else; the terrain just
#      quietly has a delta smaller than the mountain it came from.
#   2. Standing water DRAINS. A high lake fraction on sloping terrain means the
#      depression fill did not converge, so basins that should have spilled and
#      incised their outlet are still full.
#   3. A river HIERARCHY exists. Drainage-area feedback is the whole reason the
#      cycle was written; without it erosion is statistically isotropic, which
#      is what "too symmetric" meant.
#   4. Turning the cycle OFF is measurably different. If it is not, the cycle
#      never ran and every green result above is vacuous.
#
# ★ Every threshold below is a SHAPE threshold, not a magnitude. Absolute
#   erosion depends on terrain size, height scale and incision_k; asserting on
#   it would make the test a calibration snapshot instead of a physics check.
import os
import sys

import rt

sys.path.insert(0, os.path.join("scripts", "test"))
import rt_testlog  # noqa: E402

rt_testlog.start("terrain_fluvial_cycle")
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


NAME = "FluvialCycleTest"
FIELD = 1024
SIZE = 4000.0
HEIGHT = 900.0

for existing in rt.terrain.list():
    if existing["name"] == NAME:
        rt.terrain.remove(NAME)

log("-- setup: %d field, %.0f m across, %.0f m relief --" % (FIELD, SIZE, HEIGHT))
rt.terrain.create(name=NAME, resolution=FIELD, size=SIZE, height_scale=HEIGHT)

# ★★★ Relief FIRST. Eroding a flat plane produces zero of everything, and every
# assertion below would then compare 0 to 0 and report a confident pass.
rt.terrain.apply_preset(NAME, "snowy_mountain_valley")
rt.terrain.evaluate(NAME)
status = rt.terrain.evaluation_status(NAME)
if status["state"] == "running":
    vacuous("the fluvial cycle shapes a landscape",
            "graph evaluation is async and finalizes on the main thread, which "
            "this script is holding -- drive this over IPC, where each call is "
            "a separate request")
    log("")
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
    raise SystemExit(0)

relief_probe = max(abs(rt.terrain.sample_height(NAME, x, z))
                   for x in (-SIZE * 0.3, 0.0, SIZE * 0.3)
                   for z in (-SIZE * 0.3, 0.0, SIZE * 0.3))
if relief_probe < 1.0:
    vacuous("the fluvial cycle shapes a landscape",
            "the terrain is flat (max |height| %.4f m), so every ledger and "
            "shape assertion would be 0 == 0" % relief_probe)
    log("")
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
    raise SystemExit(0)
log("    relief probe: %.1f m" % relief_probe)


def erode(backend, **kwargs):
    rt.terrain.erode(NAME, "hydraulic", backend, seed=1337, **kwargs)
    return rt.terrain.erosion_stats()


def report(tag, s):
    log("    [%s] path=%s iters=%d" %
        (tag, "GPU" if s["gpu_path"] else "CPU", s["cycle_iterations"]))
    log("         eroded %.6g  deposited %.6g  exported %.6g  carried %.6g" %
        (s["eroded"], s["deposited"], s["exported"], s["carried"]))
    log("         unaccounted %.4f%%  lakes %.3f%%  drainage density %.3f%%  "
        "largest catchment %.4f km2" %
        (s["mass_error_fraction"] * 100.0, s["lake_area_fraction"] * 100.0,
         s["drainage_density"] * 100.0, s["max_drainage_area_km2"]))
    log("         trunk %.3f%% of map  deep lakes %.3f%%  deposit footprint %.3f%%" %
        (s["max_drainage_area_fraction"] * 100.0,
         s["deep_lake_area_fraction"] * 100.0,
         s["deposited_area_fraction"] * 100.0))
    log("         deepest lake %.3f m" % s["deepest_lake_meters"])
    if s["mean_deposit_meters"] > 0.0:
        log("         deposit deepest/mean %.3f m / %.3f m = %.2f" %
            (s["deepest_deposit_meters"], s["mean_deposit_meters"],
             s["deepest_deposit_meters"] / s["mean_deposit_meters"]))


# ---------------------------------------------------------------- GPU cycle
log("")
log("-- 1. GPU fluvial cycle --")
gpu = erode("gpu", fluvial_cycle=1, fluvial_iterations=24, iterations=200000)
report("gpu", gpu)

if gpu["cycle_iterations"] == 0:
    vacuous("the sediment ledger closes",
            "cycle_iterations is 0, so the cycle did not run at all -- the "
            "ledger is a cleared default, not a measurement")
elif gpu["eroded"] <= 0.0:
    vacuous("the sediment ledger closes",
            "nothing was eroded, so eroded == deposited == 0 would pass "
            "without testing transport")
else:
    # Half a percent sits above float accumulation noise on a million cells and
    # far below anything a real transport leak produces.
    check("the sediment ledger closes (GPU)",
          abs(gpu["mass_error_fraction"]) <= 0.005,
          "%.4f%% unaccounted" % (gpu["mass_error_fraction"] * 100.0))

    # The load must actually go somewhere downstream. If everything deposits in
    # the cell it came from, transport is a no-op with a closing ledger.
    moved = gpu["deposited"] + gpu["exported"]
    check("sediment reaches downstream, not just the cell it left",
          moved > gpu["eroded"] * 0.05,
          "only %.6g of %.6g moved" % (moved, gpu["eroded"]))

    check("standing water drains (physical lake coverage stays low on slopes)",
          gpu["deep_lake_area_fraction"] < 0.05,
          "%.3f%% of cells deeper than 10 cm -- raise drainage_fill_passes"
          % (gpu["deep_lake_area_fraction"] * 100.0))
    check("no numerical lake shaft consumes the terrain relief",
          gpu["deepest_lake_meters"] < HEIGHT * 0.20,
          "deepest lake %.2f m on %.2f m relief" %
          (gpu["deepest_lake_meters"], HEIGHT))

    # A river hierarchy means SOME cells carry a catchment far larger than a
    # cell. The map is 4000 m across, so a real trunk drains whole km2.
    check("a drainage hierarchy formed (a trunk with a large catchment)",
          gpu["max_drainage_area_km2"] > 0.5,
          "largest catchment only %.4f km2" % gpu["max_drainage_area_km2"])
    check("the trunk captures a meaningful share of the domain",
          gpu["max_drainage_area_fraction"] > 0.10,
          "largest catchment is only %.3f%% of the map -- flow graph is fragmented"
          % (gpu["max_drainage_area_fraction"] * 100.0))

    check("channels are a minority of the surface, not everywhere",
          0.0005 < gpu["drainage_density"] < 0.5,
          "drainage density %.4f" % gpu["drainage_density"])

    if gpu["deposited"] > 0.0 and gpu["mean_deposit_meters"] > 0.0:
        deposit_ratio = (gpu["deepest_deposit_meters"] /
                         gpu["mean_deposit_meters"])
        check("deposition spreads into a fan/apron instead of one narrow ridge",
              gpu["deposited_area_fraction"] > 0.00001 and deposit_ratio < 10.0,
              "footprint %.5f%%, deepest/mean %.2f" %
              (gpu["deposited_area_fraction"] * 100.0, deposit_ratio))

# ------------------------------------------------------------- cycle is off
log("")
log("-- 2. cycle disabled: the control --")
off = erode("gpu", fluvial_cycle=0, iterations=200000)
report("off", off)
check("disabling the cycle is observable (no ledger, no hydrology)",
      off["cycle_iterations"] == 0 and off["eroded"] == 0.0,
      "cycle_iterations=%d eroded=%.6g" % (off["cycle_iterations"], off["eroded"]))

# ------------------------------------------------------------------ CPU path
log("")
log("-- 3. CPU reference --")
log("    NOTE: the CPU path solves the drainage EXACTLY (priority-flood plus a")
log("    topological sweep); the GPU relaxes toward that answer. Structure is")
log("    expected to agree, digits are not.")
cpu = erode("cpu", fluvial_cycle=1, fluvial_iterations=8,
            sediment_route_steps=32, iterations=40000)
report("cpu", cpu)

if cpu["cycle_iterations"] == 0 or cpu["eroded"] <= 0.0:
    vacuous("the CPU reference agrees structurally with the GPU path",
            "the CPU cycle produced no erosion, so there is nothing to compare")
else:
    check("the sediment ledger closes (CPU)",
          abs(cpu["mass_error_fraction"]) <= 0.005,
          "%.4f%% unaccounted" % (cpu["mass_error_fraction"] * 100.0))
    check("the CPU reference also drains its basins",
          cpu["deep_lake_area_fraction"] < 0.05,
          "%.3f%%" % (cpu["deep_lake_area_fraction"] * 100.0))
    if gpu["max_drainage_area_km2"] > 0.0:
        ratio = cpu["max_drainage_area_km2"] / gpu["max_drainage_area_km2"]
        # ★ The GPU accumulation is a bounded relaxation, so it UNDER-counts
        #   long trunks rather than over-counting them. A GPU trunk far smaller
        #   than the CPU one means drainage_accumulate_passes is too low.
        check("GPU and CPU select trunks of the same order",
              0.4 < ratio < 2.5,
              "cpu/gpu largest catchment ratio %.3f -- if far below 1, raise "
              "drainage_accumulate_passes" % ratio)

# ------------------------------------------------- numerical-limit behaviour
log("")
log("-- 4. numerical limits do something --")
log("    Anti-pit and anti-spike limits are the difference between a landscape")
log("    and a field of spikes, so a run with them wide open must differ.")
loose = erode("gpu", fluvial_cycle=1, fluvial_iterations=24,
              incision_safety=0.9, deposition_safety=0.9, iterations=200000)
report("loose", loose)
if loose["cycle_iterations"] == 0 or gpu["cycle_iterations"] == 0:
    vacuous("the safety limits change the result",
            "one of the two runs did not execute the cycle")
else:
    check("the safety limits change the result (they are wired, not decorative)",
          abs(loose["eroded"] - gpu["eroded"]) > gpu["eroded"] * 1e-4,
          "eroded %.6g vs %.6g" % (loose["eroded"], gpu["eroded"]))
    check("the ledger still closes with the limits opened up",
          abs(loose["mass_error_fraction"]) <= 0.005,
          "%.4f%%" % (loose["mass_error_fraction"] * 100.0))

rt.terrain.remove(NAME)

log("")
if FAIL:
    log("RESULT: %d FAILED: %s" % (len(FAIL), FAIL))
elif UNVERIFIED:
    log("RESULT: PASSED SO FAR, but %d claim(s) NOT VERIFIED: %s" %
        (len(UNVERIFIED), UNVERIFIED))
else:
    log("RESULT: ALL PASSED")
