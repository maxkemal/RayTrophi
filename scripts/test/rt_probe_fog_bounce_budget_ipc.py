"""IPC probe: why a liquid surface embedded in fog renders black.

Faz 1 visual check (NEXT_BUILD_CHECKS.md, 9th batch item 4). A fog-default
liquid domain with a "steam" substance drawn as sdf: where the jet spreads into
the fog pool the surface goes near-black. Raising Total Bounces to ~51 lights
it, so paths are dying on a budget. This probe measures WHICH budget and WHICH
kind of bounce spends it, on the dark pixels only (render.volume_counters
region), instead of on the whole image.

Arms (same scene, same frame, one variable each):
  bounces10      max_bounces 10 (the default)
  bounces51      max_bounces 51 (the user's fix)
  fog_off        max_bounces 10, fog density_multiplier 0
and a reference region on plain fog (no liquid) at 10 bounces.

Reading the table:
  capped%   paths that were STILL scattering when the budget ran out.
            High on the dark patch and low at 51 = budget starvation confirmed.
  pass%     free-pass cap (maxBounces + 32) instead of the bounce cap.
  medium/path  straight fog crossings per path. They are free (inside
            free/path) since BOUNCE_MEDIUM_PASS; ~1-2 is healthy, many per
            path is the short-hop re-entry the intersection clamp caused.
            (Before that fix the column was gas/path = fog segments CHARGED
            as bounces: 6.44 of 7.92 on the dark patch at 10 bounces.)
  in/found  arbiter walks that began INSIDE the liquid, and how many found the
            exit. found << in = a refracted ray cannot find its way out.

Self-contained: builds domain 'BudgetProbe', removes it and restores the
render settings on exit. Run with the app open:

    python scripts\\test\\rt_probe_fog_bounce_budget_ipc.py
"""

import math
import os
import struct
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_test_volume_slot_identity_ipc import Ipc  # noqa: E402

DOMAIN = "BudgetProbe"
SOURCE = "BudgetProbeSteam"
CAM_POS = (3.2, 2.2, 4.0)
CAM_TARGET = (0.0, 0.6, 0.0)
FRAMES = 42
SPP = 16
# World boxes whose screen rectangles become the counter regions.
DARK_PATCH = ((0.05, 0.0, 0.05), (0.55, 0.12, 0.55))     # where the jet spreads
FOG_ONLY = ((-0.95, 0.05, 0.55), (-0.55, 0.3, 0.95))     # pool corner, no liquid


def sub(a, b): return (a[0] - b[0], a[1] - b[1], a[2] - b[2])
def dot(a, b): return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def cross(a, b): return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
def norm(a):
    l = math.sqrt(dot(a, a))
    return (a[0] / l, a[1] / l, a[2] / l)


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    return struct.unpack(">II", head[16:24])


def project_box(box, fov_deg, aspect, flip_y):
    """Screen rectangle (normalized launch coords) of a world AABB."""
    fwd = norm(sub(CAM_TARGET, CAM_POS))
    right = norm(cross(fwd, (0.0, 1.0, 0.0)))
    up = cross(right, fwd)
    t = math.tan(math.radians(fov_deg) * 0.5)
    xs, ys = [], []
    lo, hi = box
    for cx in (lo[0], hi[0]):
        for cy in (lo[1], hi[1]):
            for cz in (lo[2], hi[2]):
                d = sub((cx, cy, cz), CAM_POS)
                z = dot(d, fwd)
                u = dot(d, right) / (z * t * aspect)
                v = dot(d, up) / (z * t)
                xs.append(0.5 + 0.5 * u)
                ys.append(0.5 - 0.5 * v if not flip_y else 0.5 + 0.5 * v)
    clamp = lambda x: min(max(x, 0.0), 1.0)
    return [clamp(min(xs)), clamp(min(ys)), clamp(max(xs)), clamp(max(ys))]


def render(rt, path):
    rt.call("render.start", {"output_path": path, "spp": SPP})
    for _ in range(300):
        s = rt.call("render.status")
        if s.get("state") == "completed" and os.path.exists(path):
            return
        if s.get("error"):
            raise RuntimeError("render failed: " + s["error"])
        time.sleep(0.5)
    raise RuntimeError("render did not complete")


def measure(rt, region, path):
    rt.call("render.volume_counters", {"enabled": True, "region": region})
    render(rt, path)
    s = rt.call("render.volume_stats")
    rt.call("render.volume_counters", {"enabled": False})
    return s


def row(name, s):
    p = max(s["paths_traced"], 1)
    return ("{:<12} paths {:>8}  capped {:5.1f}%  pass {:5.1f}%  spec/path {:5.2f}  "
            "medium/path {:5.2f}  trans/path {:5.2f}  diff/path {:5.2f}  free/path {:5.2f}  "
            "in/found {}/{}").format(
        name, s["paths_traced"], 100.0 * s["paths_bounce_capped"] / p,
        100.0 * s["paths_pass_capped"] / p, s["charged_specular"] / p,
        s["medium_passes"] / p, s["charged_transmission"] / p,
        s["charged_diffuse"] / p, s["free_passes"] / p,
        s["arbiter_started_inside"], s["arbiter_inside_found"])


def main():
    rt = Ipc()
    out_dir = tempfile.mkdtemp(prefix="rt_budget_")
    settings_before = rt.call("render.get_settings")
    shading_before = rt.call("viewport.shading").get("mode", "solid")
    cam_before = rt.call("camera.get")
    try:
        rt.call("viewport.set_shading", {"mode": "solid"})
        rt.call("timeline.set_frame", {"frame": 0})
        rt.call("fluid.create_domain", {"name": DOMAIN, "type": "fluid",
                                        "domain_min": [-1, 0, -1], "domain_max": [1, 2, 1],
                                        "voxel_size": 0.04})
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "fog"})
        rt.call("fluid.set_substance_material",
                {"domain": DOMAIN, "substance": "steam", "representation": "sdf"})
        rt.call("flow_source.create", {"name": SOURCE, "domain": DOMAIN,
                                       "position": [0.3, 1.5, 0.3], "radius": 0.15,
                                       "fluid_particles_per_second": 8000,
                                       "fluid_substance": "steam"})
        # After the source: adding a source resets the domain.
        rt.call("fluid.seed", {"domain": DOMAIN, "seed_min": [-0.9, 0.02, -0.9],
                               "seed_max": [0.9, 0.5, 0.9], "particles_per_cell": 4})
        for f in range(1, FRAMES + 1):
            rt.call("timeline.set_frame", {"frame": f})
            time.sleep(0.3)
        rt.call("camera.set_position", {"position": list(CAM_POS)})
        rt.call("camera.set_target", {"target": list(CAM_TARGET)})
        rt.call("viewport.set_shading", {"mode": "rendered"})
        time.sleep(2.0)
        views = {v["view"]: v.get("live") for v in rt.call("fluid.get", {"domain": DOMAIN})["views"]}
        print("views live:", views)
        if not (views.get("sdf") and views.get("fog")):
            print("WARNING: sdf and fog are not both live -- the scene is not the one under test")

        rt.call("render.set_settings", {"max_bounces": 10})
        base = os.path.join(out_dir, "base.png")
        render(rt, base)
        w, h = png_size(base)
        fov = rt.call("camera.get").get("fov", 40.0)

        # ★ Instrument self-check. The first version of the region gate counted
        # the WHOLE image for every region (reset and read landed on different
        # Vulkan devices) and every row below came out identical. A corner of
        # 2% x 2% must count ~0.04% of the full image; if it does not, the
        # region is not being applied and no number below means anything.
        full = measure(rt, [0.0, 0.0, 1.0, 1.0], os.path.join(out_dir, "full.png"))
        corner = measure(rt, [0.0, 0.0, 0.02, 0.02], os.path.join(out_dir, "corner.png"))
        ratio = corner["paths_traced"] / float(max(full["paths_traced"], 1))
        print("region self-check: full {} paths, 2% corner {} ({:.4f})".format(
            full["paths_traced"], corner["paths_traced"], ratio))
        if full["paths_traced"] == 0 or ratio > 0.01:
            print("FAIL: counter region is not applied -- measurement aborted")
            sys.exit(1)

        # Launch y is TOP-DOWN (gl_LaunchIDEXT.y = 0 is the top row). Measured
        # 2026-09-28 with the fog in the lower half: volume rays top 1.76M /
        # bottom 7.99M. It used to be re-derived here from that same ratio, and
        # the fog fix inverted it (fewer fog rays; the jet column dominates the
        # top) -- the regions landed on the column and measured nothing asked.
        flip = False
        print("image {}x{}, launch y top-down".format(w, h))

        dark = project_box(DARK_PATCH, fov, w / float(h), flip)
        fog = project_box(FOG_ONLY, fov, w / float(h), flip)
        print("dark-patch region", [round(x, 3) for x in dark],
              " fog-only region", [round(x, 3) for x in fog])

        print()
        rt.call("render.set_settings", {"max_bounces": 10})
        print(row("bounces10", measure(rt, dark, os.path.join(out_dir, "b10.png"))))
        print(row("fog-only@10", measure(rt, fog, os.path.join(out_dir, "fog10.png"))))
        rt.call("render.set_settings", {"max_bounces": 51})
        print(row("bounces51", measure(rt, dark, os.path.join(out_dir, "b51.png"))))
        rt.call("render.set_settings", {"max_bounces": 10})
        fog_before = rt.call("fluid.get_fog_shader", {"domain": DOMAIN})
        rt.call("fluid.set_fog_shader", {"domain": DOMAIN, "density_multiplier": 0.0})
        print(row("fog_off@10", measure(rt, dark, os.path.join(out_dir, "fogoff.png"))))
        rt.call("fluid.set_fog_shader", {"domain": DOMAIN,
                                         "density_multiplier": fog_before["density_multiplier"]})
        print()
        print("images:", out_dir)
    finally:
        for method, params in (
                ("render.volume_counters", {"enabled": False}),
                ("render.set_settings", {"max_bounces": settings_before["max_bounces"],
                                         "diffuse_bounces": settings_before["diffuse_bounces"],
                                         "transmission_bounces": settings_before["transmission_bounces"],
                                         "debug_view": settings_before["debug_view"]}),
                ("flow_source.remove", {"name": SOURCE}),
                ("fluid.remove_domain", {"domain": DOMAIN}),
                ("camera.set_position", {"position": cam_before["position"]}),
                ("camera.set_target", {"target": cam_before["target"]}),
                ("viewport.set_shading", {"mode": shading_before})):
            try:
                rt.call(method, params)
            except RuntimeError:
                pass


if __name__ == "__main__":
    main()
