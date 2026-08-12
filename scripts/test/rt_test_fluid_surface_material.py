"""Visual rig for the SDF isosurface -> Principled BSDF binding.

★ RUN THIS FROM A SEPARATE TERMINAL, NOT INSIDE THE APP:

    python scripts\\test\\rt_test_fluid_surface_material.py

(Same reason as rt_test_fluid_rheology.py: render.start is asynchronous and an
in-app script holds the very thread that advances it. The guard at the bottom
catches that case and says so.)

WHAT THIS EXERCISES

Until this batch the fluid isosurface had exactly one look: a hand-rolled
Fresnel + Beer-Lambert dielectric. Water and glass were expressible; molten
metal, lava, mud and chocolate were not, at any setting. The surface now runs
the SAME scatterPrincipled() a triangle runs, selected by vol.iso_material_index.

The rig puts each material on TWO surfaces at once:

    MatSwatch   - an ordinary cube (the reference: the mesh BSDF path)
    the liquid  - the reconstructed SDF isosurface (the new path)

so every output answers one question by comparison rather than by a number:

  ★ DO THE CUBE AND THE LIQUID READ AS THE SAME SUBSTANCE?

    They will not be pixel-identical and should not be - the liquid is curved,
    thin at the edges, and carries depth absorption the cube does not. But the
    metal must be metal on both, the glass must transmit on both, and the
    chocolate must be a dense dark brown on both. If the cube changes with the
    material and the liquid does not, the binding never reached the volume:
    check fluid_surface_material_id -> render_isosurface_material_id ->
    GpuVDBVolume::iso_material_id -> VkVolumeInstance::iso_material_index.

  ★ THE SNEAKY FAILURE: a liquid that changes but always looks like TINTED
    GLASS whatever the material. That means the fall-through dielectric is
    still running and only the colour is getting through - the branch was
    entered but scatterPrincipled did not take over. Nobody reports that as a
    bug; it just looks like "the material is a bit weak".

  ★ 'none' IS THE REGRESSION CASE, and it runs first on purpose. With no
    material bound the liquid must look exactly as it did before this batch:
    clear, refractive water. If 'none' changed, the fall-through path was
    disturbed and every pre-existing water scene is affected.

THIN-SHELL AND RESIN (added this batch)

The last three cases cover the branches that used to be dispatched ahead of
scatterPrincipled on the triangle path ONLY, so the liquid never read them:

  ★ thin_shell - the liquid must go TRANSPARENT WITH A BRIGHT RIM, and the rim
    must carry a colour shift from the film thickness. If it instead looks like
    the old dielectric, the MAT_FLAG_BUBBLE branch was not entered.

    ★ THE SNEAKY FAILURE HERE IS A BLACK OR MISSING LIQUID. The pass-through
      lobe leaves along the INCOMING direction; if its origin is not pushed
      clear of the level-set band it re-hits the same film every bounce and
      burns the path budget without ever escaping. A film that reflects fine
      but goes dark where it should be see-through is exactly that.

  ★ resin_clear - a glossy amber SKIN over an opaque base. The liquid must
    stop transmitting and read as coated, not as tinted glass. Note the base
    will be NOISIER on the liquid than on the cube at the same spp: the
    isosurface branch has no NEE, so the base is lit by indirect only. Grain
    that cleans up with more samples is expected; a base that stays FLAT BLACK
    is not.

  ★ resin_inclusions - dust wisps and specks suspended in the coat. Compare
    against the cube: same structure scale, same colours.

    ★ Then STEP THE SIM AND WATCH THE SPECKS. The mesh anchors its interior to
      the object; the liquid can only anchor to the DOMAIN, so the fluid flows
      through a pattern that stays put. Stationary specks in a moving liquid is
      the documented limit, not a bug - but if they are stationary in the CUBE
      too, resin_object_space stopped being honoured on the mesh path.
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import os
import sys
import time

PIPE_NAME = r"\\.\pipe\RayTrophiStudio"

DOMAIN_MIN = [-1.5, 0.0, -1.5]
DOMAIN_MAX = [1.5, 4.0, 1.5]
VOXEL = 0.05
SEED_MIN = [-0.25, 2.6, -0.25]
SEED_MAX = [0.25, 3.2, 0.25]

# One mid-flight frame and one settled frame. The settled one is the important
# view: a thick material only distinguishes itself once it has piled up.
CAPTURE_FRAMES = [30, 55]
SPP = 64
DOMAIN = "SurfaceMatTest"
SWATCH = "MatSwatch"

# (label, material name, type, scalar params, colour params)
# Clearcoat and subsurface still have no script setter; author those in the
# panel if you want to check those lobes (they are wired in the shader).
CASES = [
    ("none", None, None, {}, {}),
    ("molten_metal", "MoltenMetal", "principled",
     {"metallic": 1.0, "roughness": 0.25},
     {"base_color": [0.95, 0.72, 0.35]}),
    ("molten_glass", "MoltenGlass", "principled",
     {"metallic": 0.0, "roughness": 0.05, "transmission": 1.0, "ior": 1.52},
     {"base_color": [0.85, 0.95, 0.90]}),
    ("chocolate", "MoltenChocolate", "principled",
     {"metallic": 0.0, "roughness": 0.35, "transmission": 0.0, "specular": 0.5},
     {"base_color": [0.13, 0.06, 0.03]}),

    # ── Thin-shell + resin: the branches this batch carried to the isosurface.
    # Both used to be dispatched BEFORE scatterPrincipled on the triangle path
    # only, so a bound material's bubble/resin fields were simply not read by
    # the liquid. The cube is the reference for both.
    ("thin_shell", "SoapFilm", "principled",
     # is_bubble is a bool riding the scalar (>0.5 = true).
     {"is_bubble": 1.0, "bubble_ior": 1.33, "bubble_film": 0.65},
     {"base_color": [0.85, 0.92, 1.0]}),

    # Clear amber coat: no inclusions, so this exercises the analytic
    # absorption fallback and the Fresnel coat split on its own.
    ("resin_clear", "AmberCoat", "principled",
     {"transmission": 0.0, "resin_density": 0.6, "resin_roughness": 0.08},
     {"base_color": [0.35, 0.20, 0.08], "resin_color": [0.92, 0.58, 0.20]}),

    # Inclusions on: this is the branch that runs resinMarchInterior, i.e. the
    # one whose ANCHOR differs from the mesh (domain-local, not object-local).
    ("resin_inclusions", "ResinNebula", "principled",
     {"transmission": 0.0, "resin_density": 0.8, "resin_roughness": 0.12,
      "resin_inclusion": 0.55, "resin_dirt": 0.12, "resin_shard": 0.30,
      "resin_inclusion_scale": 8.0, "dust_style": 0.0, "shard_shape": 0.0,
      "resin_object_space": 1.0},
     {"base_color": [0.30, 0.16, 0.06], "resin_color": [0.88, 0.62, 0.30],
      "resin_dirt_color": [0.18, 0.14, 0.10]}),
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


def build_rig(rt):
    if not rt.call("scene.object_exists", {"name": "PourGround"}):
        rt.call("scene.add_primitive", {"type": "plane", "name": "PourGround",
                                        "size": 6.0})
        rt.call("scene.set_transform", {"name": "PourGround",
                                        "translation": [0.0, 0.0, 0.0]})
    # The swatch is deliberately OUT of the pour line: it must show the material
    # on a mesh without liquid running over it, so the two paths stay separable.
    if not rt.call("scene.object_exists", {"name": SWATCH}):
        rt.call("scene.add_primitive", {"type": "cube", "name": SWATCH,
                                        "size": 0.6})
    rt.call("scene.set_transform", {"name": SWATCH,
                                    "translation": [1.1, 0.3, 0.0]})

    names = [d["name"] for d in rt.call("fluid.list_domains")["domains"]]
    if DOMAIN not in names:
        rt.call("fluid.create_domain", {"name": DOMAIN, "type": "fluid",
                                        "domain_min": DOMAIN_MIN,
                                        "domain_max": DOMAIN_MAX,
                                        "voxel_size": VOXEL})
    # SurfaceSDF is the whole point - the binding does not exist in splat mode.
    # Vulkan because that is where the shader branch lives.
    rt.call("fluid.set_param", {"domain": DOMAIN, "backend": "vulkan",
                                "render_mode": "surface", "preset": "chocolate"})


def ensure_material(rt, name, mat_type, scalars, colors):
    """Create if absent, then push params THROUGH THE SWATCH.

    material.set is object-scoped (setMaterialParam takes an object name), so
    the material has to be on the cube before it can be tuned. That ordering is
    not incidental: it is also what guarantees the cube and the liquid are
    carrying the very same material and not two look-alikes.
    """
    existing = rt.call("material.list") or []
    known = [m if isinstance(m, str) else m.get("name") for m in existing]
    if name not in known:
        rt.call("material.create", {"type": mat_type, "name": name})
    rt.call("material.assign", {"object_name": SWATCH, "material_name": name})
    for key, value in scalars.items():
        rt.call("material.set", {"object_name": SWATCH, "param": key,
                                 "value": value})
    for key, value in colors.items():
        rt.call("material.set", {"object_name": SWATCH, "param": key,
                                 "value": value})


def wait_for_render(rt, timeout_s=900.0):
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


def set_porosity(rt, amount, scale=0.03, detail=0.6):
    """Procedural porosity lives on the DOMAIN, not on the material.

    It has to: the ISO threshold is evaluated in TWO places — the shading march
    and nearestSurfaceSDFCrossing, the arbiter deciding where gas hands over to
    liquid — and the arbiter runs for OTHER volumes, with no access to this
    domain's material. So this is a fluid.set_param, not a material.set.
    """
    rt.call("fluid.set_param", {"domain": DOMAIN, "pore_amount": amount,
                                "pore_scale": scale, "pore_detail": detail})
    # Read back rather than trusting the write: a value that never landed and a
    # value that landed and did nothing look identical in the render.
    info = rt.call("fluid.get", {"domain": DOMAIN})
    got = info.get("pore_amount", -1.0)
    if abs(got - amount) > 1e-4:
        raise RuntimeError(
            "asked for pore_amount {} but the domain reports {}".format(amount, got))
    print("    domain reports pore_amount = {:.3f}".format(got))


def run_case(rt, out_dir, label, mat_name, mat_type, scalars, colors):
    print("\n=== {} ===".format(label))
    if mat_name:
        ensure_material(rt, mat_name, mat_type, scalars, colors)
        rt.call("fluid.set_param", {"domain": DOMAIN, "surface_material": mat_name})
    else:
        rt.call("fluid.set_param", {"domain": DOMAIN, "surface_material": ""})

    # Read back what the DOMAIN holds, not what we sent. An update that returns
    # success and binds nothing is the failure mode this echo exists to catch.
    info = rt.call("fluid.get", {"domain": DOMAIN})
    got = info.get("surface_material", "")
    print("    domain reports surface_material = {!r}".format(got))
    expected = mat_name or ""
    if got != expected:
        raise RuntimeError(
            "asked for {!r} but the domain reports {!r} - the binding did not "
            "reach the grid domain descriptor".format(expected, got))

    rt.call("fluid.reset")
    rt.call("fluid.clear", {"domain": DOMAIN})
    rt.call("fluid.seed", {"domain": DOMAIN, "seed_min": SEED_MIN,
                           "seed_max": SEED_MAX, "particles_per_cell": 8,
                           "replace": True})

    frame = 0
    for target in CAPTURE_FRAMES:
        while frame < target:
            rt.call("fluid.step", {"dt": 1.0 / 60.0})
            frame += 1
        out = os.path.join(out_dir, "{}_f{:03d}.png".format(label, target))
        rt.call("render.start", {"output_path": out, "spp": SPP})
        wait_for_render(rt)
        print("    frame {:3d} -> {}".format(target, out))


def main():
    out_dir = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "..", "renders", "fluid_surface_material"))
    os.makedirs(out_dir, exist_ok=True)

    rt = Ipc()
    print("Connected to RayTrophi Studio.")
    build_rig(rt)
    # Porosity off for the material cases: it is a separate axis and mixing the
    # two would make an unreadable image.
    set_porosity(rt, 0.0)
    for label, mat_name, mat_type, scalars, colors in CASES:
        run_case(rt, out_dir, label, mat_name, mat_type, scalars, colors)

    # ── Porosity pass (runs LAST, on the chocolate material) ─────────────────
    # 0.0 first is the regression half: with the feature off the surface must be
    # byte-for-byte the old one. Then the sweep — what to look for is the pore
    # RIMS catching light. Pores that read as flat painted spots mean the
    # gradient never saw them, i.e. some of the six central-difference samples
    # are still reading the raw density instead of the shared pored field.
    print("\n=== porosity sweep ===")
    for amount in (0.0, 0.15, 0.30):
        set_porosity(rt, amount)
        out = os.path.join(out_dir, "porosity_{:.2f}.png".format(amount))
        rt.call("render.start", {"output_path": out, "spp": SPP})
        wait_for_render(rt)
        print("    pore_amount {:.2f} -> {}".format(amount, out))
    set_porosity(rt, 0.0)

    print("\nDone: " + out_dir)
    print("Compare the CUBE against the LIQUID within each image.")
    print("Then compare 'none' against your pre-batch water renders - it must")
    print("be unchanged; that is the regression half of this rig.")


def refuse_if_running_inside_the_app():
    try:
        import rt  # noqa: F401
    except ImportError:
        return
    raise SystemExit(
        "\n  This rig must run OUTSIDE the app, from a separate terminal:\n"
        "      python scripts\\test\\rt_test_fluid_surface_material.py\n\n"
        "  It is currently running inside RayTrophi Studio (the `rt` module is\n"
        "  present), where it can only deadlock: render.start is asynchronous and\n"
        "  the loop that advances it is the same one executing this script.\n")


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("This rig talks to the Windows named pipe.")
    refuse_if_running_inside_the_app()
    main()
