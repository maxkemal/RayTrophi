"""Material coordinates (UVW) on the SDF isosurface.

★ RUN THIS FROM A SEPARATE TERMINAL, NOT INSIDE THE APP:

    python scripts\\test\\rt_test_fluid_material_coords.py

(render.start is asynchronous and an in-app script holds the very thread that
advances it. The guard at the bottom catches that case and says so.)

WHAT THIS EXERCISES

A raymarched isosurface has no UVs and cannot have them: there is no mesh to
unwrap, and the surface is rebuilt from the field every frame. Until this batch
the answer was a tri-planar projection anchored in WORLD space, which means a
flowing liquid slid THROUGH a stationary pattern — the texture belonged to the
room, not to the water.

Each particle now carries a material coordinate: where that parcel of liquid was
born, in world units. It is gathered onto a grid the shader samples, so the
texture is attached to the MATERIAL and travels with it.

★ THE SEED IS THE BIRTH POSITION, WHICH MAKES THE STILL CASE A REGRESSION TEST.
For liquid that has not moved, uvw == position exactly, so the render is
identical to the old world-anchored one. That is why phase 1 asserts drift ~= 0
before anything falls: if it is NOT ~0 at rest, the coordinate is being seeded
from something other than the position and every existing scene just changed.

═══════════════════════════════════════════════════════════════════════════════
PHASE 1 - NUMERIC.  Runs first and needs no renderer, so it fails in seconds
                    instead of after a render queue.
═══════════════════════════════════════════════════════════════════════════════

The measurement is `uvw_drift` from fluid.get: the mean |uvw - position| over
the particles, i.e. how far the average parcel has travelled since birth.

  ★ WHY A TREND AND NOT A THRESHOLD.  Asserting "drift > 0" would pass on a
    coordinate that was seeded once and then frozen — and freezing is the most
    likely way this breaks, because the coordinate has no equation of motion of
    its own; it survives purely by being copied correctly through compaction and
    reseed. So the assertion is that drift RISES while liquid falls.

  ★ THE SNEAKY FAILURE THIS CATCHES, and the reason drift is checked every step
    rather than only at the end: a drift that CLIMBS AND THEN COLLAPSES back
    toward zero. That is reseed stamping fresh coordinates into the middle of
    an existing body — the top-up path creating particles instead of continuing
    them. The end-to-end number would still look "large" and nobody would file
    it, but the texture is being torn every step exactly where the splash is.

  ★ uvw_available == False is a PASS-SHAPED failure. The shader reads a missing
    field as "anchor in world space", which is a silent fall back to the old
    behaviour: everything renders, nothing errors, and the feature is simply
    absent. It is asserted explicitly for that reason.

═══════════════════════════════════════════════════════════════════════════════
PHASE 2 - VISUAL, textured.
═══════════════════════════════════════════════════════════════════════════════

A high-contrast checker is bound to the liquid's albedo (generated here, so the
rig carries no asset). Frames are captured through the pour.

  ★ WHAT PASSES: the checker squares MOVE WITH the liquid. Follow one square on
    the falling column across the frames — it must stay on the same piece of
    water.

  ★ WHAT FAILS: the checker stands still while the liquid pours through it, as
    if projected onto the water from a fixed slide projector. That is the old
    world anchor, and it means the coordinate never reached the shader — check
    uvw_residual_address in VkVolumeInstance, then that the ABI is 608 in ALL FIVE
    declarations (one stale shader copy shifts every volume after the first).

  ★ THE SNEAKY ONE: a checker that travels with the liquid but SMEARS along the
    flow direction into stripes. That is real and expected in a violently
    stretched region — material coordinates stretch with the material — but if
    it happens in the slow, thick pool at the bottom, the coordinate is being
    interpolated from too few supported cells; suspect the extrapolation sweeps.

═══════════════════════════════════════════════════════════════════════════════
PHASE 3 - VISUAL, texture-free.
═══════════════════════════════════════════════════════════════════════════════

Procedural porosity is anchored in the same coordinate, and needs no image at
all. The pores must be carried by the body.

  ★ WHAT FAILS: pores that stay put in the tank while the dough moves past them
    — bubbles appearing and vanishing as material crosses a fixed lattice.

  ★ NOTE the pore rims. They get their normals from the field gradient, which
    now includes the coordinate's gradient, and the coordinate is only C0 across
    cell boundaries. Faceting on the rims that follows the VOXEL GRID rather
    than the pore shape is the thing to watch for here.
"""

import ctypes
import ctypes.wintypes as wintypes
import json
import os
import struct
import sys
import time
import zlib

PIPE_NAME = r"\\.\pipe\RayTrophiStudio"

DOMAIN = "CoordPour"
MATERIAL = "CoordChecker"
SWATCH = "CoordSwatch"

DOMAIN_MIN = [-1.5, 0.0, -1.5]
DOMAIN_MAX = [1.5, 4.0, 1.5]
VOXEL = 0.05
SEED_MIN = [-0.25, 2.6, -0.25]
SEED_MAX = [0.25, 3.2, 0.25]

DT = 1.0 / 60.0
# Long enough for the column to fall, hit the ground and spread — the spread is
# where reseed is busiest, which is where the coordinate is most likely to tear.
NUMERIC_STEPS = 90
CAPTURE_FRAMES = [20, 40, 70]
# One mid-flight moment for the three-way space comparison. Mid-flight matters:
# at rest all three spaces agree by construction, so a settled frame would
# "pass" no matter how badly the spaces were wired.
COORD_SPACE_FRAME = 40
SPP = 48

# Drift is in world units (metres). The column falls from ~2.9 down to ~0, so a
# working coordinate reaches several tenths of a metre. Kept well under the real
# fall distance so the check is about the TREND, not about matching a trajectory
# the solver is free to change.
MIN_FINAL_DRIFT = 0.15
# A single step may dip slightly as new liquid (drift 0) enters the average.
# Tolerate that; do not tolerate a collapse.
MAX_DRIFT_COLLAPSE = 0.35


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


def write_checker_png(path, squares=8, px=256):
    """Minimal PNG writer — no PIL dependency, so the rig carries no asset.

    Deliberately a hard-edged black/white checker rather than something pretty:
    the question this image has to answer is "did this square move with the
    water", and a soft or busy texture makes that a judgement call.
    """
    cell = px // squares
    raw = bytearray()
    for y in range(px):
        raw.append(0)                       # PNG filter type 0 (None) per row
        for x in range(px):
            on = ((x // cell) + (y // cell)) % 2 == 0
            if on:
                raw += b"\xf0\x20\x20"      # red
            else:
                raw += b"\xf5\xf5\xf5"      # near-white

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", px, px, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    png += chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)
    return path


def build_rig(rt):
    if not rt.call("scene.object_exists", {"name": "CoordGround"}):
        rt.call("scene.add_primitive", {"type": "plane", "name": "CoordGround",
                                        "size": 6.0})
        rt.call("scene.set_transform", {"name": "CoordGround",
                                        "translation": [0.0, 0.0, 0.0]})

    # Out of the pour line: it must carry the material WITHOUT liquid running
    # over it, so the mesh path and the isosurface path stay separable.
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
    # SurfaceSDF on Vulkan: the coordinate is sampled in the isosurface branch of
    # volume_closesthit, which neither the splat path nor the fog path reaches.
    rt.call("fluid.set_param", {"domain": DOMAIN, "backend": "vulkan",
                                "render_mode": "surface", "preset": "chocolate"})


def reseed(rt):
    rt.call("fluid.reset")
    rt.call("fluid.clear", {"domain": DOMAIN})
    rt.call("fluid.seed", {"domain": DOMAIN, "seed_min": SEED_MIN,
                           "seed_max": SEED_MAX, "particles_per_cell": 8,
                           "replace": True})


def ensure_material(rt):
    """Create and tune the shared test material, idempotently.

    ★ Its own function because more than one phase needs it and the phases do
    not run in a fixed order — a phase that assumed an earlier one had created
    the material would fail with "material not found" the first time somebody
    reordered them, which reads as an IPC bug rather than as a rig bug.

    material.set is OBJECT-scoped (setMaterialParam takes an object name), so
    the material has to sit on something before it can be tuned. SWATCH also
    doubles as the reference surface: the same material on a mesh, where the
    checker is mapped by real UVs. If the cube shows a checker and the liquid
    shows none, the isosurface never reached the texture at all — a different
    failure from the anchor being wrong, and worth being able to tell apart.
    """
    existing = rt.call("material.list") or []
    known = [m if isinstance(m, str) else m.get("name") for m in existing]
    if MATERIAL not in known:
        rt.call("material.create", {"type": "principled", "name": MATERIAL})
    rt.call("material.assign", {"object_name": SWATCH, "material_name": MATERIAL})
    for key, value in (("roughness", 0.25), ("metallic", 0.0),
                       ("transmission", 0.0),
                       # uv_scale doubles as WORLD UNITS PER TILE on an
                       # isosurface. Coarse on purpose: one square has to be
                       # followable by eye across frames, which is the entire
                       # measurement here.
                       ("uv_scale_x", 1.3), ("uv_scale_y", 1.3)):
        rt.call("material.set", {"object_name": SWATCH, "param": key,
                                 "value": value})


def wait_for_render(rt, timeout_s=900.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        status = rt.call("render.status")
        if not status.get("rendering", False):
            return True
        time.sleep(1.0)
    raise SystemExit("render did not finish within {}s".format(timeout_s))


def phase_numeric(rt):
    print("\n=== phase 1: numeric (no renderer) ===")
    reseed(rt)

    info = rt.call("fluid.get", {"domain": DOMAIN})
    failures = []

    # ── At rest ──────────────────────────────────────────────────────────────
    if not info.get("uvw_available", False):
        failures.append(
            "uvw_available is False right after seeding. The domain published no "
            "material-coordinate field, so the shader is silently anchoring in "
            "world space and this feature is absent while everything still "
            "renders. Check that emit() filled the uvw sidecar.")
    dim = info.get("uvw_dim", [0, 0, 0])
    if min(dim) <= 0:
        failures.append("uvw_dim is {} - a zero axis means the grid is unusable "
                        "and the address will never be published.".format(dim))

    # ── The grid's world placement ───────────────────────────────────────────
    # ★ This is the check that would have caught the smearing bug WITHOUT a
    # render. The shader indexes the coordinate field in world space through
    # exactly these two values; if they do not describe the box the producer
    # actually walked, the field is laid over the wrong region. That does not
    # fail, it warps - and a warp reads as "the coordinate is low quality",
    # which is a diagnosis that sends you off rewriting the wrong thing.
    voxel = float(info.get("uvw_voxel", 0.0))
    origin = info.get("uvw_origin", [0.0, 0.0, 0.0])
    if voxel <= 0.0:
        failures.append(
            "uvw_voxel is {} - without a cell size the shader cannot index the "
            "grid at all and will fall back to world anchoring.".format(voxel))
    elif abs(voxel - VOXEL) > 1e-6:
        failures.append(
            "uvw_voxel is {:.6f} but the domain was created with {:.6f}. The "
            "coordinate grid is being described with a DIFFERENT cell size from "
            "the one it was built at, so the shader will sample it stretched by "
            "the ratio.".format(voxel, VOXEL))
    if voxel > 0.0:
        span = [origin[a] + dim[a] * voxel for a in range(3)]
        # The sim grid may be padded relative to the requested domain, so this is
        # a containment test, not an equality test: whatever else is true, the
        # grid must COVER the domain it belongs to.
        for a, axis in enumerate("xyz"):
            if origin[a] > DOMAIN_MIN[a] + 1e-4 or span[a] < DOMAIN_MAX[a] - 1e-4:
                failures.append(
                    "coordinate grid does not cover the domain on {}: grid spans "
                    "[{:.3f}, {:.3f}] but the domain is [{:.3f}, {:.3f}]. Part of "
                    "the liquid will sample clamped edge cells, which looks like "
                    "the texture being dragged toward the boundary."
                    .format(axis, origin[a], span[a],
                            DOMAIN_MIN[a], DOMAIN_MAX[a]))
        print("  grid: origin=({:.3f}, {:.3f}, {:.3f}) voxel={:.4f}".format(
            origin[0], origin[1], origin[2], voxel))

    rest_drift = float(info.get("uvw_drift", -1.0))
    print("  at rest: particles={} dim={} drift={:.5f}".format(
        info.get("uvw_particles", 0), dim, rest_drift))
    if rest_drift > 1e-3:
        failures.append(
            "drift at rest is {:.5f}, expected ~0. The coordinate is NOT being "
            "seeded from the birth position, which means the still case no "
            "longer matches the old world-anchored render and every existing "
            "liquid scene just changed appearance.".format(rest_drift))

    # ── Falling ──────────────────────────────────────────────────────────────
    history = [rest_drift]
    peak = rest_drift
    for step in range(NUMERIC_STEPS):
        rt.call("fluid.step", {"dt": DT})
        d = float(rt.call("fluid.get", {"domain": DOMAIN}).get("uvw_drift", 0.0))
        history.append(d)
        if d > peak:
            peak = d
        # A collapse means the coordinate is being re-stamped mid-body. Caught
        # per-step because the final value alone still looks healthy.
        elif peak > 0.05 and d < peak * (1.0 - MAX_DRIFT_COLLAPSE):
            failures.append(
                "drift collapsed from {:.4f} to {:.4f} at step {}. Something is "
                "re-seeding the coordinate on existing liquid - almost certainly "
                "the reseed top-up emitting without inheriting from its donor."
                .format(peak, d, step))
            break
        if (step + 1) % 15 == 0:
            print("  step {:3d}: drift={:.5f}".format(step + 1, d))

    final = history[-1]
    if final < MIN_FINAL_DRIFT:
        failures.append(
            "final drift {:.5f} < {:.5f}. The liquid fell but its coordinate did "
            "not travel with it, so the coordinate is frozen rather than carried."
            .format(final, MIN_FINAL_DRIFT))

    print("  final drift={:.5f} (peak {:.5f})".format(final, peak))
    return failures


def phase_textured(rt, out_dir):
    print("\n=== phase 2: visual, checker texture ===")
    tex = write_checker_png(os.path.join(out_dir, "checker.png"))
    print("  checker written: " + tex)

    # material.set is OBJECT-scoped (setMaterialParam takes an object name), so
    # the material has to sit on something before it can be tuned. SWATCH also
    # doubles as the reference surface: the same material on a mesh, where the
    # checker is mapped by real UVs. If the cube shows a checker and the liquid
    # shows none, the isosurface never reached the texture at all — a different
    # failure from the anchor being wrong, and worth being able to tell apart.
    ensure_material(rt)
    rt.call("material.set_texture", {"material_name": MATERIAL,
                                     "slot": "albedo", "path": tex})
    rt.call("fluid.set_param", {"domain": DOMAIN, "surface_material": MATERIAL,
                                "pore_amount": 0.0})

    reseed(rt)
    captured = 0
    for step in range(max(CAPTURE_FRAMES) + 1):
        if step in CAPTURE_FRAMES:
            out = os.path.join(out_dir, "checker_f{:03d}.png".format(step))
            rt.call("render.start", {"output_path": out, "spp": SPP})
            wait_for_render(rt)
            print("    frame {:3d} -> {}".format(step, out))
            captured += 1
        rt.call("fluid.step", {"dt": DT})
    return captured


def phase_porosity(rt, out_dir):
    print("\n=== phase 3: visual, porosity (no texture) ===")
    rt.call("material.clear_texture", {"material_name": MATERIAL,
                                       "slot": "albedo"})
    rt.call("fluid.set_param", {"domain": DOMAIN, "pore_amount": 0.28,
                                "pore_scale": 0.06, "pore_detail": 0.5})
    reseed(rt)
    captured = 0
    for step in range(max(CAPTURE_FRAMES) + 1):
        if step in CAPTURE_FRAMES:
            out = os.path.join(out_dir, "pores_f{:03d}.png".format(step))
            rt.call("render.start", {"output_path": out, "spp": SPP})
            wait_for_render(rt)
            print("    frame {:3d} -> {}".format(step, out))
            captured += 1
        rt.call("fluid.step", {"dt": DT})
    rt.call("fluid.set_param", {"domain": DOMAIN, "pore_amount": 0.0})
    return captured


def phase_coord_spaces(rt, out_dir):
    """Render the SAME moment in all three coordinate spaces.

    ★ WHAT MAKES THIS READABLE: one frame, three spaces, everything else held
    still. The three images differ ONLY in where the pattern is anchored, so a
    difference that is not anchoring is a bug in something else.

      material : checker rides the falling column
      domain   : checker stands still relative to the tank; liquid pours through
      world    : same as domain here, because the domain is not moving - and
                 THAT is the check. If domain and world differ on a stationary
                 domain, the domain transform is being applied where it should
                 be identity, most likely twice (the resin march had exactly
                 that bug: materialAnchor already does the inverse transform, so
                 an additional MAT_FLAG_RESIN_OBJ_SPACE pass ran it again).
    """
    print("\n=== phase 4: coordinate spaces ===")
    rt.call("fluid.set_param", {"domain": DOMAIN, "surface_material": MATERIAL,
                                "pore_amount": 0.0})
    for space in ("material", "domain", "world"):
        reseed(rt)
        for _ in range(COORD_SPACE_FRAME):
            rt.call("fluid.step", {"dt": DT})
        rt.call("fluid.set_param", {"domain": DOMAIN, "coord_space": space})
        back = rt.call("fluid.get", {"domain": DOMAIN}).get("coord_space")
        if back != space:
            raise SystemExit(
                "coord_space read back as {!r} after setting {!r} - the setter "
                "and the reporter disagree, so every later result in this rig "
                "is describing a state nobody asked for.".format(back, space))
        out = os.path.join(out_dir, "space_{}.png".format(space))
        rt.call("render.start", {"output_path": out, "spp": SPP})
        wait_for_render(rt)
        print("    {:8s} -> {}".format(space, out))
    rt.call("fluid.set_param", {"domain": DOMAIN, "coord_space": "material"})


def phase_alpha(rt, out_dir):
    """Opacity texture as a FIELD cutout - holes with real rims.

    ★ WHAT PASSES: holes THROUGH the liquid whose edges refract and shade like
    geometry, and which travel with the body (they ride the same anchor).

    ★ WHAT FAILS, and it is the whole reason this is not an any-hit shader:
      holes with FLAT edges. If a hole's rim carries the surrounding surface's
      normal - refracting as though the hole were not there, casting no shadow
      onto the liquid behind it - then the mask is being applied at the shading
      point instead of to the field.

    ★ THE SNEAKY ONE: holes that appear at the silhouette and CLOSE where the
      liquid is thick. That means the subtracted amount is not enough to carry a
      fully-interior cell past the ISO threshold, so the mask only erodes the
      surface. It reads as "alpha does not work in deep water", which sounds
      like a limitation rather than a constant that is too small.

    ★ ALSO CHECK THE GAS EDGE if a smoke domain overlaps: the handoff arbiter
      reads this same field, so the holes must clip gas too. Shimmering there
      means someone made the mask stochastic - it cannot be, the field has to be
      a pure function of position.
    """
    print("\n=== phase 5: opacity mask (field cutout) ===")
    mask = write_checker_png(os.path.join(out_dir, "holes.png"), squares=4)
    rt.call("material.clear_texture", {"material_name": MATERIAL, "slot": "albedo"})
    rt.call("material.set_texture", {"material_name": MATERIAL,
                                     "slot": "opacity", "path": mask})
    reseed(rt)
    for step in range(max(CAPTURE_FRAMES) + 1):
        if step in CAPTURE_FRAMES:
            out = os.path.join(out_dir, "alpha_f{:03d}.png".format(step))
            rt.call("render.start", {"output_path": out, "spp": SPP})
            wait_for_render(rt)
            print("    frame {:3d} -> {}".format(step, out))
        rt.call("fluid.step", {"dt": DT})
    rt.call("material.clear_texture", {"material_name": MATERIAL, "slot": "opacity"})


def phase_refresh_schedule(rt):
    """★ The two-generation refresh, measured numerically.

    A Lagrangian coordinate stretches without bound, so two generations are
    reset on staggered phases and blended. The property that makes that
    invisible is that a generation's WEIGHT IS ZERO exactly when it is reset —
    the discontinuity is multiplied away.

    That is directly testable: uvw_drift reports the BLENDED displacement, so it
    must stay continuous straight through a reset. Run with a deliberately short
    period so several resets happen inside the test.

    ★ WHAT FAILS: drift saw-toothing to (near) zero on a fixed cycle. That means
    a generation is being reset while it still carries weight, so the texture
    pops once per period — a rhythmic artefact, which is worse than the smearing
    the refresh exists to cure because the eye locks onto it.

    ★ THE SNEAKY ONE: drift that looks perfectly smooth because the two
    generations have become IDENTICAL. Then there is nothing to crossfade, the
    metric is beautifully continuous, and the stretch cure is doing nothing at
    all. The reseed path is where this happens (a child inheriting generation A
    into both slots), so this phase runs long enough for reseeding to be busy.
    """
    print("\n=== phase 1b: coordinate refresh schedule ===")
    period = 20
    rt.call("fluid.set_param", {"domain": DOMAIN, "uvw_refresh_period": period})
    back = rt.call("fluid.get", {"domain": DOMAIN}).get("uvw_refresh_period")
    failures = []
    if back != period:
        failures.append(
            "uvw_refresh_period read back as {!r} after setting {}. The setter "
            "and the reporter disagree, so nothing measured below describes the "
            "state anybody asked for.".format(back, period))
        return failures

    reseed(rt)
    peak = 0.0
    worst_drop = 0.0
    worst_step = -1
    for step in range(period * 4):
        rt.call("fluid.step", {"dt": DT})
        d = float(rt.call("fluid.get", {"domain": DOMAIN}).get("uvw_drift", 0.0))
        if d > peak:
            peak = d
        elif peak > 0.02:
            drop = (peak - d) / peak
            if drop > worst_drop:
                worst_drop, worst_step = drop, step
    print("  {} steps over {} full periods: peak drift={:.5f}, worst drop={:.1%}"
          .format(period * 4, 4, peak, worst_drop))
    if worst_drop > MAX_DRIFT_COLLAPSE:
        failures.append(
            "blended drift dropped {:.1%} at step {} with a {}-step refresh "
            "period. A generation is being reset while it still carries weight, "
            "which pops the texture once per period."
            .format(worst_drop, worst_step, period))

    # Leave the domain on the default; a short period is a test fixture, not a
    # state the later phases should inherit.
    rt.call("fluid.set_param", {"domain": DOMAIN, "uvw_refresh_period": 240})
    return failures


def read_png_rgb(path):
    """Decode an 8-bit RGB/RGBA PNG to (w, h, bytearray of RGB).

    Written out rather than pulled from PIL because this rig has to run on a
    bare interpreter next to the application. Handles exactly what the renderer
    writes: 8-bit, colour type 2 or 6, no interlace.
    """
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise SystemExit("{} is not a PNG".format(path))
    pos = 8
    w = h = 0
    channels = 0
    idat = []
    while pos + 8 <= len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 8 or ctype not in (2, 6) or interlace != 0:
                raise SystemExit(
                    "{}: unsupported PNG (depth={}, colour={}, interlace={}). "
                    "This decoder is deliberately narrow; widen it here rather "
                    "than skipping the comparison.".format(
                        path, depth, ctype, interlace))
            channels = 3 if ctype == 2 else 4
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    raw = zlib.decompress(b"".join(idat))

    stride = w * channels
    out = bytearray(w * h * 3)
    prev = bytearray(stride)
    src = 0
    for y in range(h):
        ft = raw[src]
        src += 1
        line = bytearray(raw[src:src + stride])
        src += stride
        if ft == 1:
            for i in range(channels, stride):
                line[i] = (line[i] + line[i - channels]) & 0xFF
        elif ft == 2:
            for i in range(stride):
                line[i] = (line[i] + prev[i]) & 0xFF
        elif ft == 3:
            for i in range(stride):
                left = line[i - channels] if i >= channels else 0
                line[i] = (line[i] + ((left + prev[i]) >> 1)) & 0xFF
        elif ft == 4:
            for i in range(stride):
                a = line[i - channels] if i >= channels else 0
                b = prev[i]
                c = prev[i - channels] if i >= channels else 0
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[i] = (line[i] + pr) & 0xFF
        elif ft != 0:
            raise SystemExit("{}: unknown PNG filter {}".format(path, ft))
        for x in range(w):
            s = x * channels
            d = (y * w + x) * 3
            out[d] = line[s]
            out[d + 1] = line[s + 1]
            out[d + 2] = line[s + 2]
        prev = line
    return w, h, out


def mean_abs_diff(a, b):
    if len(a) != len(b):
        raise SystemExit("image sizes differ; nothing to compare")
    total = 0
    for i in range(len(a)):
        total += abs(a[i] - b[i])
    return total / float(len(a))


def phase_still_identity(rt, out_dir):
    """★★★ THE MEASUREMENT. Liquid that has not moved must render IDENTICALLY
    in Material and World mode.

    Why this is the check worth having: every particle's uvw is seeded with its
    birth position, so at rest uvw == position and the stored displacement is
    exactly zero. Material mode therefore reduces to worldPos algebraically —
    not approximately, exactly. Any visible difference is the grid intruding
    where it has no business, which is precisely the "one pixel per cell"
    coarseness this storage change exists to remove.

    ★ Compared against the rig's OWN noise floor rather than a magic threshold:
    two renders of the identical state differ by path-tracing noise alone, and
    that difference is the yardstick. A fixed tolerance would either pass a real
    regression at high spp or fail a clean build at low spp.

    ★ WHAT IT MEANS IF IT FAILS: the coordinate is being reconstructed from the
    grid instead of from worldPos + grid. Look at sampleMaterialCoord first (is
    worldPos still added?), then at the producer (is it storing the displacement
    or the coordinate?). Those two must change together or not at all.

    ★ THE SNEAKY ONE: a diff that is small but consistently ABOVE the noise
    floor, concentrated on the surface and absent in the background. That is not
    noise; that is the residual field carrying a small constant — most likely
    the gather subtracting a cell centre instead of differencing per particle,
    which leaves the particle-distribution centroid behind and looks like a
    faint quilt rather than like a bug.
    """
    print("\n=== phase 6: still-liquid identity (material == world) ===")
    tex = write_checker_png(os.path.join(out_dir, "checker.png"))
    ensure_material(rt)
    rt.call("material.set_texture", {"material_name": MATERIAL,
                                     "slot": "albedo", "path": tex})
    rt.call("fluid.set_param", {"domain": DOMAIN, "surface_material": MATERIAL,
                                "pore_amount": 0.0})
    reseed(rt)   # seeded, NOT stepped: this is the at-rest case by construction

    shots = {}
    for tag, space in (("mat_a", "material"), ("mat_b", "material"),
                       ("world", "world")):
        rt.call("fluid.set_param", {"domain": DOMAIN, "coord_space": space})
        out = os.path.join(out_dir, "still_{}.png".format(tag))
        rt.call("render.start", {"output_path": out, "spp": SPP})
        wait_for_render(rt)
        shots[tag] = read_png_rgb(out)[2]
        print("    {:6s} ({:8s}) -> {}".format(tag, space, out))

    noise = mean_abs_diff(shots["mat_a"], shots["mat_b"])
    signal = mean_abs_diff(shots["mat_a"], shots["world"])
    print("    noise floor (material vs material): {:.4f}/255".format(noise))
    print("    material vs world                 : {:.4f}/255".format(signal))

    rt.call("fluid.set_param", {"domain": DOMAIN, "coord_space": "material"})
    rt.call("material.clear_texture", {"material_name": MATERIAL, "slot": "albedo"})

    # Allowance over the noise floor, plus a small absolute term so a perfectly
    # deterministic renderer (noise floor 0) does not make the test infinitely
    # strict on a single stray pixel.
    budget = noise * 3.0 + 0.5
    if signal > budget:
        return ["still liquid renders DIFFERENTLY in material and world mode "
                "({:.4f} vs a {:.4f} noise floor, budget {:.4f}). At rest the "
                "stored displacement is exactly zero, so these two must agree "
                "algebraically. The grid is leaking into the coordinate - see "
                "sampleMaterialCoord and buildMaterialCoordinateGrid, which "
                "must both be storing/reading a DISPLACEMENT."
                .format(signal, noise, budget)]
    return []


def main():
    out_dir = os.path.join(os.getcwd(), "test_output", "fluid_material_coords")
    os.makedirs(out_dir, exist_ok=True)

    rt = Ipc()
    print("Connected to RayTrophi Studio.")
    build_rig(rt)

    # ★ Numeric first, deliberately. It needs no renderer, so a broken
    # coordinate reports in seconds — and if it IS broken, the two render
    # phases below would only produce images that look plausible.
    failures = phase_numeric(rt)
    failures += phase_refresh_schedule(rt)
    if failures:
        print("\n" + "=" * 72)
        print("NUMERIC PHASE FAILED - not rendering; the images would only")
        print("confirm what these numbers already say.")
        for f in failures:
            print("\n  * " + f)
        print("=" * 72)
        raise SystemExit(1)
    print("  numeric phase PASSED")

    # ★ Runs before the eyeball phases: it is the only rendered phase that
    # produces a PASS/FAIL rather than a picture, so it is the one that can
    # report while nobody is watching.
    still_failures = phase_still_identity(rt, out_dir)
    if still_failures:
        print("\n" + "=" * 72)
        print("STILL-IDENTITY PHASE FAILED")
        for f in still_failures:
            print("\n  * " + f)
        print("=" * 72)
        raise SystemExit(1)
    print("  still-identity phase PASSED")

    phase_textured(rt, out_dir)
    phase_porosity(rt, out_dir)
    phase_coord_spaces(rt, out_dir)
    phase_alpha(rt, out_dir)

    print("\nDone: " + out_dir)
    print("Now LOOK at the images - the numbers proved the coordinate is")
    print("carried, not that the shader is reading it.")
    print("  checker_f*.png : follow ONE square down the pour. It must stay on")
    print("                   the same water. A checker that hangs in the air")
    print("                   while liquid pours through it = world anchor,")
    print("                   i.e. uvw_residual_address never reached the shader.")
    print("  pores_f*.png   : the bubbles must travel with the body. Watch the")
    print("                   rims for faceting that follows the VOXEL GRID")
    print("                   rather than the pore - that is the coordinate's")
    print("                   C0 gradient showing through.")
    print("  space_*.png    : same instant, three anchors. material rides the")
    print("                   liquid; domain and world must look IDENTICAL here")
    print("                   because the domain is not moving - if they differ,")
    print("                   a transform is being applied twice.")
    print("  still_*.png    : at rest, mat_a and world must be the SAME image.")
    print("                   A cell-sized quilt on the material one is the grid")
    print("                   leaking into a coordinate it should not touch.")
    print("  alpha_f*.png   : holes must go THROUGH the body and their rims must")
    print("                   refract. Flat-edged holes = cut at the shading")
    print("                   point instead of in the field. Holes that close in")
    print("                   thick liquid = the subtracted amount is too small.")


def refuse_if_running_inside_the_app():
    try:
        import rt  # noqa: F401
    except ImportError:
        return
    raise SystemExit(
        "\n  This rig must run OUTSIDE the app, from a separate terminal:\n"
        "      python scripts\\test\\rt_test_fluid_material_coords.py\n\n"
        "  It is currently running inside RayTrophi Studio (the `rt` module is\n"
        "  present), where it can only deadlock: render.start is asynchronous and\n"
        "  the loop that advances it is the same one executing this script.\n")


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("This rig talks to the Windows named pipe.")
    refuse_if_running_inside_the_app()
    main()
