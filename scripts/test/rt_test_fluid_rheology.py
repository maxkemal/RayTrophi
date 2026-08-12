"""Visual regression rig for the fluid rheology rework.

★ RUN THIS FROM A SEPARATE TERMINAL, NOT INSIDE THE APP:

    python scripts\\test\\rt_test_fluid_rheology.py

Do NOT run it with `rt.ps1 -File` / script.run_file. An in-app script executes
INLINE on the UI thread (RtIpc.cpp: g_inline_dispatch_context), and render.start
is asynchronous — the render job is advanced by that same loop. Polling
render.status from inside therefore blocks the only thread that could ever
change it: the app sits busy at zero CPU and zero GPU, the output folder gets
created, and no image is ever written. Producer and consumer on one thread.
Driving it over the pipe from outside keeps the app's loop free to render.

Builds ONE pour setup and replays it once per material, rendering the same three
frames each time, so the outputs can be flipped through side by side. Every
question this rig answers is answered by comparing two images, not by reading a
number - which is the point: the failure this batch fixes (viscosity cancelled
out of the FLIP delta) produced numbers that all looked reasonable.

WHAT TO LOOK FOR, per output:

  f020 (mid-air, before the stream lands)
      Water and Honey must have fallen THE SAME DISTANCE. This is the headline
      check. Before this batch honey hit a terminal ~1.8 m/s and lagged far
      behind; a viscous liquid resists shear, not gravity, so in free fall it
      accelerates exactly like water. If honey is still visibly higher than
      water at f020, per-particle friction is back in charge and the viscous
      solve is not doing the work.

  f040 (landing / column build-up)
      Water splashes outward and flattens. Honey and Chocolate should pile into
      a heap, and the incoming stream should THICKEN and buckle where it meets
      the surface. A thick preset that lands and spreads like water means the
      solve is under-converged: raise viscosity_sweeps and re-run.

  f060 (settled)
      Chocolate/Honey should still be a mound, with the cube WETTED - liquid
      clinging to its vertical faces rather than sliding off. That is the
      no-slip wall condition; if the mound is right but the cube is bare,
      viscosity_wall_slip did not reach the solver.

  Sand is the KNOWN-APPROXIMATE control. It has no yield criterion, so it will
  still fall slower than g and its pile will still flatten more than real sand.
  Do not tune the others to match it.
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import os
import sys
import time

PIPE_NAME = r"\\.\pipe\RayTrophiStudio"

# ── Rig ─────────────────────────────────────────────────────────────────────
# Deliberately small and coarse: this has to be re-runnable in a couple of
# minutes per material, and every effect above is visible at 5 cm voxels.
DOMAIN_MIN = [-1.5, 0.0, -1.5]
DOMAIN_MAX = [1.5, 4.0, 1.5]
VOXEL = 0.05

# High enough that free fall is unmistakable by f020, landing before f040.
# ~2.5 m of drop is about 0.7 s, i.e. frame 42 at 60 fps.
SEED_MIN = [-0.25, 3.2, -0.25]
SEED_MAX = [0.25, 3.8, 0.25]

CAPTURE_FRAMES = [20, 40, 60]
SPP = 48
DOMAIN = "RheologyTest"

MATERIALS = [
    ("water",     "baseline: thin, splashy, free-slip walls"),
    ("honey",     "nu=7e-3, no-slip - must fall like water, pile like honey"),
    ("chocolate", "nu=4e-3, no-slip - the target look for this batch"),
    ("mud",       "nu=2e-3, mostly no-slip"),
    ("sand",      "KNOWN APPROXIMATION - control, not a target"),
]


# ── Pipe transport ──────────────────────────────────────────────────────────
class Ipc(object):
    def __init__(self):
        self.k32 = ctypes.windll.kernel32
        self.handle = self.k32.CreateFileW(
            PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
        invalid = wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == invalid:
            raise SystemExit(
                "Cannot connect to {} (error {}). Is RayTrophi Studio "
                "running?".format(PIPE_NAME, self.k32.GetLastError()))
        mode = wintypes.DWORD(0x00000002)  # PIPE_READMODE_MESSAGE
        self.k32.SetNamedPipeHandleState(self.handle, ctypes.byref(mode), None, None)
        self._id = 0

    def call(self, method, params=None):
        self._id += 1
        msg = {"id": self._id, "method": method}
        if params:
            msg["params"] = params
        data = json.dumps(msg).encode("utf-8")

        written = wintypes.DWORD(0)
        if not self.k32.WriteFile(self.handle, data, len(data),
                                  ctypes.byref(written), None):
            raise OSError("WriteFile failed ({})".format(self.k32.GetLastError()))

        chunks = []
        while True:
            buf = ctypes.create_string_buffer(65536)
            read = wintypes.DWORD(0)
            ok = self.k32.ReadFile(self.handle, buf, 65536, ctypes.byref(read), None)
            chunks.append(buf.raw[:read.value])
            if ok:
                break
            if self.k32.GetLastError() != 234:  # ERROR_MORE_DATA
                raise OSError("ReadFile failed ({})".format(self.k32.GetLastError()))
        resp = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in resp:
            raise RuntimeError("{} failed: {}".format(method, resp["error"]))
        return resp.get("result")


# ── Rig construction ────────────────────────────────────────────────────────
def build_rig(rt):
    if not rt.call("scene.object_exists", {"name": "PourGround"}):
        rt.call("scene.add_primitive", {"type": "plane", "name": "PourGround",
                                        "size": 6.0})
        rt.call("scene.set_transform", {"name": "PourGround",
                                        "translation": [0.0, 0.0, 0.0]})
    # An obstacle with vertical faces. A free-slip liquid slides straight off it;
    # a no-slip one leaves a film. That contrast is the whole reason it is here.
    if not rt.call("scene.object_exists", {"name": "PourObstacle"}):
        rt.call("scene.add_primitive", {"type": "cube", "name": "PourObstacle",
                                        "size": 0.7})
        rt.call("scene.set_transform", {"name": "PourObstacle",
                                        "translation": [0.0, 0.35, 0.0]})

    names = [d["name"] for d in rt.call("fluid.list_domains")["domains"]]
    if DOMAIN not in names:
        rt.call("fluid.create_domain", {"name": DOMAIN, "type": "fluid",
                                        "domain_min": DOMAIN_MIN,
                                        "domain_max": DOMAIN_MAX,
                                        "voxel_size": VOXEL})
    # Vulkan is the primary path for the new viscosity kernel; on any other
    # backend the solve falls back to the (correct but far slower) host
    # reference, and this rig would then be measuring the CPU.
    rt.call("fluid.set_param", {"domain": DOMAIN, "backend": "vulkan",
                                "render_mode": "surface"})


def wait_for_render(rt, timeout_s=900.0):
    """render.start is asynchronous. Safe to poll from HERE - we are outside the
    app, so its loop is free to actually advance the job."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = rt.call("render.status")
        state = status["state"]
        if state == "completed":
            return status["output_path"]
        if state in ("failed", "cancelled"):
            raise RuntimeError("render {}: {}".format(state, status.get("error", "")))
        time.sleep(0.25)
    raise RuntimeError("render did not finish within the timeout")


def run_material(rt, out_dir, preset, note):
    print("\n=== {} :: {} ===".format(preset, note))
    rt.call("fluid.reset")
    rt.call("fluid.clear", {"domain": DOMAIN})
    rt.call("fluid.set_param", {"domain": DOMAIN, "preset": preset})

    info = rt.call("fluid.get", {"domain": DOMAIN})
    # Echo what the solver actually holds, not what we asked for. A preset that
    # failed to reach the grid domain descriptor used to return success and
    # change nothing; printing the read-back is what makes that visible.
    print("    preset={} nu={:.2e} m^2/s sweeps={} wall_slip={:.2f}".format(
        info["preset"], info["kinematic_viscosity"],
        info["viscosity_sweeps"], info["viscosity_wall_slip"]))
    if info["preset"] != preset:
        raise RuntimeError(
            "asked for '{}' but the domain reports '{}' - the preset did not "
            "reach the grid domain descriptor".format(preset, info["preset"]))

    rt.call("fluid.seed", {"domain": DOMAIN, "seed_min": SEED_MIN,
                           "seed_max": SEED_MAX, "particles_per_cell": 8,
                           "replace": True})

    frame = 0
    for target in CAPTURE_FRAMES:
        while frame < target:
            rt.call("fluid.step", {"dt": 1.0 / 60.0})
            frame += 1
        out = os.path.join(out_dir, "{}_f{:03d}.png".format(preset, target))
        rt.call("render.start", {"output_path": out, "spp": SPP})
        wait_for_render(rt)
        print("    frame {:3d} -> {}".format(target, out))


def main():
    # Renders land next to the app, which is where render.start resolves a
    # relative path; make it absolute here so both ends agree.
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "renders", "fluid_rheology"))
    os.makedirs(out_dir, exist_ok=True)

    rt = Ipc()
    print("Connected to RayTrophi Studio.")
    build_rig(rt)
    for preset, note in MATERIALS:
        run_material(rt, out_dir, preset, note)
    print("\nDone. Compare across materials at the SAME frame:\n  " + out_dir)
    print("f020 water vs honey height is the headline check - they must match.")


def refuse_if_running_inside_the_app():
    """The embedded interpreter exposes `rt`; a standalone one does not.

    Running this file in-app deadlocks in a way that does NOT name itself: the
    script holds the main thread, the pipe request it sends is enqueued FOR the
    main thread, and the dispatcher eventually reports
        'main thread dispatch timeout; request cancelled before execution'
    which reads like an IPC fault rather than 'you ran this in the wrong place'.
    Catch it here instead, while we can still say why.
    """
    try:
        import rt  # noqa: F401
    except ImportError:
        return  # standalone interpreter - this is the supported way
    raise SystemExit(
        "\n  This rig must run OUTSIDE the app, from a separate terminal:\n"
        "      python scripts\\test\\rt_test_fluid_rheology.py\n\n"
        "  It is currently running inside RayTrophi Studio (the `rt` module is\n"
        "  present), where it can only deadlock: render.start is asynchronous and\n"
        "  the loop that advances it is the same one executing this script, so\n"
        "  every call it makes waits for a thread it is itself holding.\n")


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("This rig talks to the Windows named pipe.")
    refuse_if_running_inside_the_app()
    main()
