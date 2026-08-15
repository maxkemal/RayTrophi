"""Focused visual/read-back test for geometric SurfaceSDF fullness.

Run from a separate terminal while RayTrophi Studio is open:

    python scripts\test\rt_test_fluid_surface_fullness.py

The particle state is simulated once, then held fixed for three renders. Only
the zero level set moves, so the silhouette must expand monotonically from
0.00 -> 0.65 -> 1.00 voxels while dielectric colour/extinction stays stable.
"""

import os
import sys

from rt_test_fluid_surface_material import (
    CAPTURE_FRAMES,
    DOMAIN,
    Ipc,
    SEED_MAX,
    SEED_MIN,
    SPP,
    build_rig,
    refuse_if_running_inside_the_app,
    set_porosity,
    wait_for_render,
)


def set_fullness(rt, offset):
    rt.call("fluid.set_param", {
        "domain": DOMAIN,
        "surface_offset_voxels": offset,
    })
    got = rt.call("fluid.get", {"domain": DOMAIN}).get(
        "surface_offset_voxels", -99.0)
    if abs(got - offset) > 1e-4:
        raise RuntimeError(
            "surface fullness write/read mismatch: asked {}, got {}"
            .format(offset, got))
    print("  read-back: {:.2f} vx".format(got))


def main():
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "renders",
        "fluid_surface_fullness"))
    os.makedirs(out_dir, exist_ok=True)

    rt = Ipc()
    build_rig(rt)
    rt.call("fluid.set_param", {
        "domain": DOMAIN,
        "surface_material": "",
        "render_mode": "surface",
        "backend": "vulkan",
        "preset": "water",
    })
    set_porosity(rt, 0.0)

    rt.call("fluid.reset")
    rt.call("fluid.clear", {"domain": DOMAIN})
    rt.call("fluid.seed", {
        "domain": DOMAIN,
        "seed_min": SEED_MIN,
        "seed_max": SEED_MAX,
        "particles_per_cell": 8,
        "replace": True,
    })
    for _ in range(CAPTURE_FRAMES[-1]):
        rt.call("fluid.step", {"dt": 1.0 / 60.0})

    for offset in (0.0, 0.65, 1.0):
        set_fullness(rt, offset)
        out = os.path.join(out_dir, "fullness_{:.2f}.png".format(offset))
        rt.call("render.start", {"output_path": out, "spp": SPP})
        wait_for_render(rt)
        print("  render: " + out)

    # Leave the scene at the production default after the diagnostic sweep.
    set_fullness(rt, 0.65)
    print("PASS (API/read-back). Visual comparison: " + out_dir)


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("This rig talks to the Windows named pipe.")
    refuse_if_running_inside_the_app()
    main()
