"""IPC test: liquid Volumetric Fog render mode.

Run while RayTrophi Studio is open with a liquid domain holding particles:

    python scripts\\test\\rt_test_fluid_fog_mode_ipc.py [domain]

Checks:
  * render_mode 'fog' is accepted and reads back as 'fog'.
  * the fog mode's input exists: active_density_cells > 0.
  * the mode SURVIVES frame changes (the reported bug: a fog look chosen in
    the VDB panel fell back to the refractive surface on the next frame).
  * 'volume' on a liquid now means fog, not a silent isosurface.
  * fluid.set_fog spread_voxels reads back, and out-of-range is refused.
The domain's render mode and fog spread are restored on exit.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_test_fluid_splat_raster_ipc import Ipc  # same pipe client

DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "Physics Domain 1"
failures = []


def check(ok, what, detail=""):
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", what,
                               " -- " + detail if detail else ""))
    if not ok:
        failures.append(what)


def domain_info(rt):
    for d in rt.call("fluid.list_domains")["domains"]:
        if d["name"] == DOMAIN:
            return d
    raise SystemExit("fluid domain not found: " + DOMAIN)


def main():
    rt = Ipc()
    before = domain_info(rt)
    if before["particle_count"] <= 0:
        raise SystemExit("domain has no particles -- run the sim first")
    frame_result = rt.call("timeline.get_frame")
    frame = int(frame_result.get("frame", 0) if isinstance(frame_result, dict)
                else frame_result)
    try:
        print("1. fog mode read-back")
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "fog"})
        info = domain_info(rt)
        check(info["render_mode"] == "fog", "render_mode reads back as 'fog'",
              "got " + info["render_mode"])

        print("2. producer: splatted density")
        rt.call("timeline.set_frame", {"frame": frame + 1})
        time.sleep(1.5)
        info = domain_info(rt)
        check(info.get("active_density_cells", 0) > 0,
              "active_density_cells > 0",
              "cells={} max={} (0 = producer, not render, problem)".format(
                  info.get("active_density_cells"), info.get("max_density")))

        print("3. mode survives frame changes")
        for k in range(2, 5):
            rt.call("timeline.set_frame", {"frame": frame + k})
            time.sleep(1.0)
            mode = domain_info(rt)["render_mode"]
            check(mode == "fog", "frame +{} still 'fog'".format(k), "got " + mode)

        print("4. fog spread")
        rt.call("fluid.set_fog", {"domain": DOMAIN, "spread_voxels": 2.5})
        check(abs(domain_info(rt).get("fog_spread_voxels", -1) - 2.5) < 1e-4,
              "spread_voxels 2.5 reads back")
        try:
            rt.call("fluid.set_fog", {"domain": DOMAIN, "spread_voxels": 7.0})
            check(False, "spread_voxels 7 is rejected", "call was ACCEPTED")
        except RuntimeError as exc:
            check(True, "spread_voxels 7 is rejected", str(exc)[:80])

        info = domain_info(rt)
        check(info.get("particle_kelvin_measured") is True,
              "particles carry a temperature for the fog emission",
              "min={} max={} K (false = every parcel unwritten, 0 K)".format(
                  info.get("particle_min_kelvin"), info.get("particle_max_kelvin")))

        print("5. 'volume' on a liquid means fog")
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "surface"})
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "volume"})
        check(domain_info(rt)["render_mode"] == "fog", "'volume' -> 'fog'")
    finally:
        restore = before["render_mode"]
        if restore not in ("particles", "surface", "fog"):
            restore = "surface"
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": restore})
        if before.get("fog_spread_voxels") is not None:
            rt.call("fluid.set_fog", {"domain": DOMAIN,
                                      "spread_voxels": before["fog_spread_voxels"]})

    print()
    if failures:
        print("FAIL: {} check(s): {}".format(len(failures), "; ".join(failures)))
        sys.exit(1)
    print("PASS. Visual follow-up: Material/Rendered show the liquid as fog shaded by "
          "the domain Volume Material; scrubbing the timeline keeps it fog.")


if __name__ == "__main__":
    main()
