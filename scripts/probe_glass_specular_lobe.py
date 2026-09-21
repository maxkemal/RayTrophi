#!/usr/bin/env python3
"""
Probe: does a GLASS surface show a specular reflection lobe from a scene light?

WHY THIS EXISTS
    Reported 2026-09-06: "Vulkan RT'de cam malzemede specular bir yansima lobu
    gozleyemedim."  The cause is structural, not a tuning issue:

        * closesthit.rchit takes the glass lobe and RETURNS before the direct
          lighting (NEE) block - correct in itself, a Fresnel split is a
          specular decision, not a BRDF evaluation.
        * scene lights in this renderer are ANALYTIC. lights.l[] carries no
          geometry in the TLAS, so nothing the mirror lobe can ever hit.

    Those two facts together mean the glass reflection was not dim, it was
    STRUCTURALLY UNREACHABLE: a glass ball under a lamp could only reflect the
    environment (a Physical Sky sun disk, an emissive mesh) and never the lamp,
    no matter how many bounces or samples it was given.  The fix gives glass the
    same explicit-light estimator water already had
    (addDielectricDirectLighting, bsdf_scatter.glsl).

WHAT IT MEASURES
    A black world (solid mode, background 0,0,0) with ONE point light.  With no
    environment there is nothing else a surface can be lit by, so every non-black
    pixel in the frame is a direct-light response and render.probe measures it
    without needing to know where on screen the highlight landed.

    Two passes over the SAME sphere, same camera, same lamp:

      1. CONTROL  - metal (metallic 1). Metal is shaded through the generic NEE
                    block, so it must be bright BEFORE and AFTER the fix.
      2. SUBJECT  - glass (transmission 1, ior 1.5).

    ★ The control is the whole point of the instrument. "The glass is black" and
      "the lamp is off / the camera is aimed at nothing / the viewport never
      converged" produce the IDENTICAL measurement, and the second family is far
      more likely in an automated run. Only a bright control makes a dark subject
      mean what it looks like it means.

    ★ The subtle failure this catches is the opposite one: a glass ball that
      lights up but is now too bright, i.e. the highlight double-counted. So the
      script also reports the glass/metal ratio instead of a bare pass/fail. A
      dielectric at ior 1.5 reflects ~4% at normal incidence, so glass reading
      BRIGHTER than a mirror is wrong even though it "passes" a "not black" test.

WHAT IT DOES NOT MEASURE
    Refraction, dispersion, caustics, or the environment reflection - all of
    those worked before and none of them go through this estimator.  It also
    says nothing about OptiX: this drives the viewport, which is Vulkan.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/probe_glass_specular_lobe.py

    Creates one sphere and one light, then deletes both and restores the world
    mode it found. Run it on a scratch scene anyway - it repoints the camera.

EXIT CODE
    0 = PASS, 1 = FAIL (glass has no lobe, or an implausible one), 2 = could not
    run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
CONVERGE_FRAMES = 24          # enough for a stochastic 4% lobe to show up
BALL = 'GlassLobeProbeBall'

_kernel32 = ctypes.windll.kernel32
_failures = []


def connect():
    handle = _kernel32.CreateFileW(PIPE_NAME, 0xC0000000, 0, None, 3, 0, None)
    if handle == -1 or handle == 0xFFFFFFFFFFFFFFFF:
        print("FAIL(setup): RayTrophi Studio is not running "
              "(no \\\\.\\pipe\\RayTrophiStudio). Start it first.")
        sys.exit(2)
    return handle


def call(pipe, method, params=None, request_id=[0]):
    request_id[0] += 1
    msg = {"id": request_id[0], "method": method}
    if params:
        msg["params"] = params
    data = json.dumps(msg).encode("utf-8")
    written = wintypes.DWORD(0)
    if not _kernel32.WriteFile(pipe, data, len(data), ctypes.byref(written), None):
        raise OSError("WriteFile failed (%d)" % _kernel32.GetLastError())
    chunks = []
    while True:
        buf = ctypes.create_string_buffer(65536)
        read = wintypes.DWORD(0)
        ok = _kernel32.ReadFile(pipe, buf, 65536, ctypes.byref(read), None)
        chunks.append(buf.raw[:read.value])
        if not ok and _kernel32.GetLastError() != 234:  # ERROR_MORE_DATA
            raise OSError("ReadFile failed (%d)" % _kernel32.GetLastError())
        try:
            return json.loads(b"".join(chunks).decode("utf-8"))
        except ValueError:
            if ok and read.value == 0:
                raise


def refused(response):
    if isinstance(response, dict) and "error" in response:
        return True
    return (isinstance(response, dict) and isinstance(response.get("result"), dict)
            and "__error" in response["result"])


def result_of(pipe, method, params, what):
    response = call(pipe, method, params)
    if refused(response) or not isinstance(response, dict) or "result" not in response:
        print("FAIL(setup): %s -> %s" % (what, json.dumps(response)[:300]))
        sys.exit(2)
    return response["result"]


def check(label, condition, detail=""):
    if condition:
        print("  ok   %-40s %s" % (label, detail))
    else:
        print("  FAIL %-40s %s" % (label, detail))
        _failures.append(label)


def measure(pipe, label):
    """Converge the viewport, then read the whole frame back."""
    result_of(pipe, "viewport.render_frames", {"count": CONVERGE_FRAMES},
              "viewport.render_frames")
    probe = result_of(pipe, "render.probe", {}, "render.probe")
    if not probe.get("available", False):
        # An unavailable probe is NOT a dark scene, and reading it as one is
        # exactly how a missing measurement becomes a confident zero.
        print("FAIL(setup): render.probe reports no captured frame after %s. "
              "Capture is on but no viewport frame arrived." % label)
        sys.exit(2)
    print("  %-8s mean=%.5f  max=%.5f  black=%.3f  nan=%.5f"
          % (label, probe.get("mean_luminance", 0.0), probe.get("max_luminance", 0.0),
             probe.get("black_fraction", 1.0), probe.get("nan_fraction", 0.0)))
    return probe


def main():
    pipe = connect()
    print("Glass specular lobe probe — black world, one point light.\n")

    light_index = None
    world_before = result_of(pipe, "world.get", {}, "world.get")

    # ── Scene: black world so the lamp is the ONLY possible light ────────────
    result_of(pipe, "world.set_mode", {"mode": "solid"}, "world.set_mode")
    result_of(pipe, "world.set_background_color", {"background_color": [0.0, 0.0, 0.0]},
              "world.set_background_color")

    # The returned name is authoritative — a suffix is added when BALL is taken.
    ball = result_of(pipe, "scene.add_primitive",
                     {"type": "sphere", "name": BALL, "size": 1.0},
                     "scene.add_primitive")
    print("  ball: %s" % ball)

    # Lamp beside the camera: the mirror direction off the ball's front face
    # points back near the eye, so the lobe lands where the camera can see it.
    light_name = result_of(pipe, "lights.add",
                           {"type": "point", "position": [3.0, 3.0, 4.0]},
                           "lights.add")
    # lights.* addresses by INDEX, lights.add answers with a NAME. Resolve once
    # here rather than assuming the new light is last: a delete elsewhere
    # renumbers the list and an assumed index silently drives another light.
    light_index = None
    for entry in result_of(pipe, "lights.list", {}, "lights.list"):
        if entry.get("name") == light_name:
            light_index = entry.get("index")
            break
    if light_index is None:
        print("FAIL(setup): lights.add returned %r but lights.list does not "
              "carry it — cannot address the lamp." % (light_name,))
        sys.exit(2)
    print("  lamp: %s (index %d)" % (light_name, light_index))
    result_of(pipe, "lights.set_intensity", {"index": light_index, "intensity": 200.0},
              "lights.set_intensity")

    result_of(pipe, "camera.set_position", {"position": [0.0, 0.5, 6.0]},
              "camera.set_position")
    result_of(pipe, "camera.set_target", {"target": [0.0, 0.0, 0.0]},
              "camera.set_target")

    result_of(pipe, "viewport.capture", {"enabled": True}, "viewport.capture")
    result_of(pipe, "viewport.set_shading", {"mode": "rendered"}, "viewport.set_shading")

    try:
        # ── PASS 1: metal control ────────────────────────────────────────────
        for param, value in (("transmission", 0.0), ("metallic", 1.0),
                             ("roughness", 0.1), ("base_color", [1.0, 1.0, 1.0])):
            result_of(pipe, "material.set",
                      {"param": param, "object_name": ball, "value": value},
                      "material.set %s" % param)
        control = measure(pipe, "metal")

        # ── PASS 2: glass subject ────────────────────────────────────────────
        for param, value in (("metallic", 0.0), ("transmission", 1.0),
                             ("ior", 1.5), ("roughness", 0.1)):
            result_of(pipe, "material.set",
                      {"param": param, "object_name": ball, "value": value},
                      "material.set %s" % param)
        glass = measure(pipe, "glass")

        print()
        control_max = control.get("max_luminance", 0.0)
        glass_max = glass.get("max_luminance", 0.0)

        check("control (metal) is lit",
              control_max > 0.01,
              "max=%.5f — if this fails the LAMP or the CAMERA is the problem, "
              "not glass; nothing below means anything" % control_max)
        if _failures:
            return

        check("glass shows a specular lobe",
              glass_max > 0.01,
              "max=%.5f (was structurally 0: analytic lights are invisible to a "
              "lobe that skips NEE)" % glass_max)
        ratio = glass_max / control_max if control_max > 1e-6 else 0.0
        check("lobe is dielectric-plausible, not double-counted",
              ratio <= 1.0,
              "glass/metal = %.3f — a dielectric reflects a FRACTION of what a "
              "mirror does; >1 means the estimator is being added twice" % ratio)
        check("no invalid pixels",
              glass.get("nan_fraction", 0.0) <= 1e-6,
              "nan_fraction=%.6f" % glass.get("nan_fraction", 0.0))
    finally:
        # ── Restore ─────────────────────────────────────────────────────────
        call(pipe, "scene.delete", {"name": ball})
        if light_index is not None:
            call(pipe, "lights.delete", {"index": light_index})
        mode = world_before.get("mode") if isinstance(world_before, dict) else None
        if isinstance(mode, str):
            call(pipe, "world.set_mode", {"mode": mode})
        colour = world_before.get("background_color") if isinstance(world_before, dict) else None
        if isinstance(colour, list) and len(colour) == 3:
            call(pipe, "world.set_background_color", {"background_color": colour})


if __name__ == "__main__":
    main()
    print()
    if _failures:
        print("RESULT: FAIL (%d) — %s" % (len(_failures), ", ".join(_failures)))
        sys.exit(1)
    print("RESULT: PASS")
    sys.exit(0)
