"""Volume instrumentation smoke test + a three-configuration cost probe.

Two jobs:

  1. Prove the counters are reachable from script at all. Until this batch they
     lived only in the Volume Performance panel, so every volume-cost question
     ended with a human reading numbers off a panel and typing them back.

  2. Give the discriminating measurement for the "plane under the domain makes
     Vulkan RT 50x slower" report. The three readings below separate the three
     candidate mechanisms, which is the whole point of taking three.

Run with the app open and a fluid/gas domain in view:

    python rt_test_volume_stats.py

★ Read `available` and `enabled` before any number. An all-zero snapshot means
one of those is false far more often than it means the volume was free.
"""

import sys
import time

import rt


def snapshot(label, settle_seconds=2.0):
    """Zero the counters, let the renderer run, then read them back."""
    rt.render.volume_counters(enabled=True)
    time.sleep(settle_seconds)
    s = rt.render.volume_stats()
    if not s["available"]:
        print("  FAIL: no Vulkan backend active — counters are Vulkan-only")
        return None
    if not s["enabled"]:
        print("  FAIL: counters did not stay enabled")
        return None
    rays = s["volume_rays"]
    if rays == 0:
        # Not a pass and not a solver bug: no ray entered a volume. Saying
        # "0 samples per ray" here would report a measurement that was never made.
        print(f"  [{label}] volume_rays = 0 — no ray entered a volume.")
        print("           Wrong camera, wrong frame, or the domain is not in the TLAS.")
        return s

    def per_ray(key):
        return s[key] / rays

    print(f"  [{label}]")
    print(f"    volume_rays              {rays}")
    print(f"    density samples / ray    {per_ray('density_samples'):.2f}")
    print(f"    shadow samples / ray     {per_ray('shadow_density_samples'):.2f}")
    print(f"    step budget exhausted    "
          f"{100.0 * s['step_budget_exhausted'] / rays:.1f}%")
    print(f"    solid probes run/hit     {s['solid_probe_runs']} / {s['solid_probe_hits']}")
    if s["solid_probe_runs"] == 0:
        print("    ^ gate suppressed EVERY probe: surfaces inside a volume "
              "cannot be found at all")
    if s["majorant_queries"] == 0:
        print("    ^ no majorant queries — no dense-gas block acceleration on this path")
    return s


def main():
    print("Volume instrumentation reachability")
    first = snapshot("baseline")
    if first is None:
        return 1

    print()
    print("Now change ONE thing and re-run this script:")
    print("  a) move the ground plane away from the domain")
    print("  b) pull the camera back out of the domain box")
    print()
    print("How to read the difference — each points at a different mechanism:")
    print("  shadow samples/ray collapses  -> the plane's NEE marches through the")
    print("                                   volume; fix belongs in the shadow march")
    print("  volume_rays collapses but     -> ray COUNT is the cost: the free-pass")
    print("  density/ray is flat              handoff loop in raygen.rgen")
    print("  step budget exhausted falls   -> the ACTIVE box grew and the step budget")
    print("                                   capped. ★ This one is the quiet failure:")
    print("                                   it does not crash, it just gets slower")
    print("                                   AND softer, and nobody files that as a bug.")

    # Leave the counters off: the atomics have a cost and a scene left with them
    # enabled quietly contaminates every later frame-time measurement.
    rt.render.volume_counters(enabled=False)
    print()
    print("Counters disabled again (they perturb frame-time measurements).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
