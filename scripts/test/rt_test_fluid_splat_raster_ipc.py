"""IPC test: fluid splat spheres reach raster, and the retired mode is gone.

Run from a separate terminal while RayTrophi Studio is open with a liquid
domain that already holds particles:

    python scripts\\test\\rt_test_fluid_splat_raster_ipc.py [domain]

Covers the 2026-09-27 batch:
  * 'virtual_particles' is refused, not aliased.
  * fluid.set_splat_geometry validates, and fluid.get reads it back.
  * Solid: the pool is handed to the sphere impostor and actually DRAWN
    (sphere_impostors_drawn == sphere_impostors_uploaded > 0).
  * Material: the pool comes back as instanced geometry after the switch
    (the switch used to be dropped by the raster generation gate), and none
    of it is demoted to the foliage card proxy (proxy_instances == 0).
The domain's render mode, splat subdivisions and the viewport shading are
restored on exit.
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import sys
import time


PIPE_NAME = r"\\.\pipe\RayTrophiStudio"
DOMAIN = sys.argv[1] if len(sys.argv) > 1 else "Physics Domain 1"
SETTLE_S = 2.0


class Ipc:
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit(
                "Cannot connect to {} (error {}). Is RayTrophi Studio running?"
                .format(PIPE_NAME, self.k32.GetLastError()))
        mode = wintypes.DWORD(0x00000002)
        self.k32.SetNamedPipeHandleState(self.handle, ctypes.byref(mode), None, None)
        self.request_id = 0

    def call(self, method, params=None):
        self.request_id += 1
        request = {"id": self.request_id, "method": method}
        if params:
            request["params"] = params
        payload = json.dumps(request).encode("utf-8")
        written = wintypes.DWORD(0)
        if not self.k32.WriteFile(self.handle, payload, len(payload),
                                  ctypes.byref(written), None):
            raise OSError("WriteFile failed ({})".format(self.k32.GetLastError()))
        chunks = []
        while True:
            buf = ctypes.create_string_buffer(65536)
            read = wintypes.DWORD(0)
            ok = self.k32.ReadFile(self.handle, buf, len(buf),
                                   ctypes.byref(read), None)
            chunks.append(buf.raw[:read.value])
            if ok:
                break
            if self.k32.GetLastError() != 234:
                raise OSError("ReadFile failed ({})".format(self.k32.GetLastError()))
        response = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in response:
            raise RuntimeError("{} failed: {}".format(method, response["error"]))
        return response.get("result")


failures = []


def check(ok, what, detail=""):
    print("  [{}] {}{}".format("PASS" if ok else "FAIL", what,
                               " -- " + detail if detail else ""))
    if not ok:
        failures.append(what)


def expect_refused(rt, method, params, what):
    try:
        rt.call(method, params)
    except RuntimeError as exc:
        check(True, what, str(exc)[:120])
        return
    check(False, what, "call was ACCEPTED")


def domain_info(rt):
    for d in rt.call("fluid.list_domains")["domains"]:
        if d["name"] == DOMAIN:
            return d
    raise SystemExit("fluid domain not found: " + DOMAIN)


def telemetry_after_frames(rt, frame):
    # A timeline step moves the particles, which drives the bridge's motion
    # path (the one that used to leave new slots out of raster); the sleep
    # lets the main loop run the raster rebuild and draw.
    rt.call("timeline.set_frame", {"frame": frame})
    time.sleep(SETTLE_S)
    return rt.call("viewport.frame_telemetry")


def main():
    rt = Ipc()
    before = domain_info(rt)
    shading_before = rt.call("viewport.shading").get("mode", "solid")
    frame_result = rt.call("timeline.get_frame")
    frame = int(frame_result.get("frame", 0) if isinstance(frame_result, dict)
                else frame_result)
    print("domain '{}': {} particles, mode {}, subdiv {}".format(
        DOMAIN, before["particle_count"], before["render_mode"],
        before.get("splat_subdivisions")))
    if before["particle_count"] <= 0:
        raise SystemExit("domain has no particles -- run the sim first")

    try:
        print("1. retired mode")
        expect_refused(rt, "fluid.set_param",
                       {"domain": DOMAIN, "render_mode": "virtual_particles"},
                       "render_mode 'virtual_particles' is refused")
        check(domain_info(rt)["render_mode"] == before["render_mode"],
              "refusal left the render mode untouched")

        print("2. splat geometry validation and read-back")
        expect_refused(rt, "fluid.set_splat_geometry",
                       {"domain": DOMAIN, "subdivisions": 4},
                       "subdivisions 4 is rejected, not clamped")
        expect_refused(rt, "fluid.set_splat_geometry",
                       {"domain": DOMAIN, "geometry": "scene_object",
                        "geometry_source": ""},
                       "scene_object without a live source is rejected")
        rt.call("fluid.set_splat_geometry", {"domain": DOMAIN, "subdivisions": 0})
        info = domain_info(rt)
        check(info.get("splat_subdivisions") == 0 and info.get("splat_triangles") == 20,
              "subdivisions 0 reads back as 20 triangles per splat",
              "got subdiv={} tris={}".format(info.get("splat_subdivisions"),
                                             info.get("splat_triangles")))

        print("3. Solid: sphere impostor")
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": "particles"})
        rt.call("viewport.set_shading", {"mode": "solid"})
        t = telemetry_after_frames(rt, frame + 1)
        up, drawn = t.get("sphere_impostors_uploaded", 0), t.get("sphere_impostors_drawn", 0)
        check(t.get("sphere_impostor_ready") is True, "impostor pipeline ready")
        check(t.get("raster_sphere_groups", 0) >= 1, "splat pool handed to the impostor",
              "raster_sphere_groups={}".format(t.get("raster_sphere_groups")))
        check(up > 0, "impostor spheres uploaded", "uploaded={}".format(up))
        check(drawn == up and drawn > 0, "every uploaded sphere drawn",
              "uploaded={} drawn={} (drawn 0 = a draw gate closed)".format(up, drawn))

        print("4. Material: pool returns as instanced geometry")
        rt.call("viewport.set_shading", {"mode": "material"})
        t = telemetry_after_frames(rt, frame + 2)
        check(t.get("raster_sphere_groups", 1) == 0, "no pool left on the impostor")
        # full_instances, not total_instances: empty pool slots now stay in
        # the raster list masked off, so total counts slots, not particles.
        # full_instances is what survived culling and was drawn.
        check(t.get("full_instances", 0) > 1,
              "splat instances drawn from the raster list",
              "full_instances={} total_instances={} (full 1 = only the scene)".format(
                  t.get("full_instances"), t.get("total_instances")))
        check(t.get("proxy_instances", 0) == 0,
              "no splat demoted to the card proxy",
              "proxy_instances={}".format(t.get("proxy_instances")))
        first = t.get("full_instances", 0)
        t = telemetry_after_frames(rt, frame + 3)
        check(t.get("full_instances", 0) >= first * 0.5,
              "pool stays drawn across a sim step",
              "{} -> {}".format(first, t.get("full_instances")))
    finally:
        rt.call("fluid.set_param", {"domain": DOMAIN, "render_mode": before["render_mode"]
                                    if before["render_mode"] != "virtual_particles"
                                    else "particles"})
        if before.get("splat_subdivisions") is not None:
            rt.call("fluid.set_splat_geometry",
                    {"domain": DOMAIN, "subdivisions": before["splat_subdivisions"]})
        rt.call("viewport.set_shading", {"mode": shading_before})

    print()
    if failures:
        print("FAIL: {} check(s): {}".format(len(failures), "; ".join(failures)))
        sys.exit(1)
    print("PASS. Visual follow-up: Solid shows round shaded splats; Material shows "
          "icospheres with no flat cards; RayFusion does not flicker while the sim plays.")


if __name__ == "__main__":
    main()
