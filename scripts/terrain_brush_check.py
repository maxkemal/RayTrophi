#!/usr/bin/env python3
"""
RayTrophi Studio - terrain BRUSH acceptance check (sculpt + splat paint).

★★★ WHY THIS FILE EXISTS

The terrain sculpt and splat brushes were reachable from the panel and from
nowhere else, so nothing could regression-test them. What that cost, measured:
TerrainManager::sculpt clamped every touched cell to 0..1 while the height field
is metres / scale_y - a graph-authored landform of 900 m relief over a 1000 m
scale_y sits near 0.9, and one with a base elevation goes well past 1.0. On such
a terrain the FIRST dab slammed the ground under the brush down to the clamp,
which reads as "the brush flattens the terrain to zero", and Raise/Stamp could
never lift it again: every frame added its delta and the clamp took it back.

★★ THE ASSERTION IS TWO-SIDED, on purpose. "raise moved the ground" passes for a
brush that moves it the wrong way; "the field changed" passes for a brush that
destroys it. So each mode is checked for the direction AND the magnitude it
claims: raise up, lower down, flatten toward the target, and - the reading that
would have caught the clamp on its own - the terrain's own peak must SURVIVE a
dab placed on the high ground.

★ Heights here are metres of world Y, not field units: sample_height reports
world space, which is the number an artist can also read off the viewport.

Usage:
    1. Start RayTrophi Studio (.\\scripts\\ipc\\Start-RayTrophi.ps1).
    2. python scripts/terrain_brush_check.py
"""
import json
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ipc_test_client import send_command, PIPE_NAME  # noqa: E402

TERRAIN = "BrushCheck"
RES, SIZE, HSCALE = 256, 1000.0, 1000.0
# A terrain whose field genuinely leaves 0..1. With scale_y = 1000 m the noise
# generator's relief lands near 0.9 of it, and the preset's base elevation puts
# the peaks over 1.0 - which is exactly the case the old clamp destroyed. A
# gentle terrain would pass this file with the bug still in place.
RELIEF = 900.0

failures = []


def call(pipe, method, params=None):
    response = send_command(pipe, method, params, request_id=call.counter)
    call.counter += 1
    return response


call.counter = 1


def result_of(response):
    if not isinstance(response, dict) or response.get("error"):
        return None
    return response.get("result", response)


def error_of(response):
    if isinstance(response, dict) and response.get("error"):
        return json.dumps(response["error"])
    return ""


def check(label, condition, detail=""):
    if condition:
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  [FAIL] {label}" + (f"\n         {detail}" if detail else ""))
        failures.append(label)


def open_pipe():
    import ctypes
    import ctypes.wintypes as wintypes
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
    if handle is None or handle == (2 ** 64 - 1):
        print(f"[terrain-brush] FAIL: cannot connect to {PIPE_NAME}. Is RayTrophi Studio running?")
        sys.exit(1)
    handle = ctypes.c_void_p(handle)
    kernel32.SetNamedPipeHandleState(handle, ctypes.byref(wintypes.DWORD(2)), None, None)
    return handle


def evaluate(pipe):
    call(pipe, "terrain.evaluate", {"name": TERRAIN})
    for _ in range(300):
        state = result_of(call(pipe, "terrain.evaluation_status", {"name": TERRAIN}))
        if not state or state.get("state") != "running":
            return state
        time.sleep(0.5)
    return {"state": "timeout"}


def build_terrain(pipe):
    call(pipe, "terrain.remove", {"name": TERRAIN})
    created = result_of(call(pipe, "terrain.create", {
        "name": TERRAIN, "resolution": RES, "size": SIZE, "height_scale": HSCALE}))
    check("terrain created", created is not None)
    if created is None:
        return False
    call(pipe, "terrain.apply_preset", {
        "name": TERRAIN, "preset": "default", "replace_graph": True, "add_satmap": False})
    nodes = result_of(call(pipe, "nodes.list",
                           {"graph_type": "terrain", "graph_name": TERRAIN})) or []
    for node in nodes:
        if node.get("type_id") == "TerrainV2.NoiseGenerator":
            call(pipe, "nodes.set_property", {
                "graph_type": "terrain", "graph_name": TERRAIN,
                "node_id": node["id"], "property": "reliefMeters", "value": RELIEF})
    state = evaluate(pipe)
    check("graph evaluated", bool(state) and state.get("state") not in ("running", "timeout"),
          json.dumps(state))
    return True


def stroke(pipe, method, dabs, **kwargs):
    params = {"name": TERRAIN, "dabs": dabs}
    params.update(kwargs)
    response = call(pipe, method, params)
    out = result_of(response)
    if out is None:
        print(f"         {method} failed: {error_of(response)}")
    return out


def dab_line(x, z, count=6, step=4.0):
    """A drag, not a click: the panel never produces a single-dab stroke."""
    return [[x + i * step, z] for i in range(count)]


def find_peak(pipe):
    """The highest of a coarse sample grid, in WORLD metres.

    The clamp bug only shows where the field is high, so the test has to go
    looking for high ground instead of assuming the tile centre is a summit.
    """
    best = None
    span = SIZE * 0.5 - 40.0
    steps = 9
    for iz in range(steps):
        for ix in range(steps):
            x = -span + (2 * span) * ix / (steps - 1)
            z = -span + (2 * span) * iz / (steps - 1)
            height = result_of(call(pipe, "terrain.sample_height",
                                    {"name": TERRAIN, "world_x": x, "world_z": z}))
            if height is None:
                continue
            if best is None or height > best[2]:
                best = (x, z, height)
    return best


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    pipe = open_pipe()
    print("=== Terrain brush acceptance check ===")
    if not build_terrain(pipe):
        return 1

    peak = find_peak(pipe)
    check("terrain has relief to sculpt on", peak is not None and peak[2] > 1.0,
          f"peak {peak[2]:.1f} m at ({peak[0]:.0f}, {peak[1]:.0f})" if peak else "no samples")
    if peak is None:
        return 1
    px, pz, pheight = peak
    print(f"  peak sample: {pheight:.1f} m at ({px:.0f}, {pz:.0f})")

    # ── 1. RAISE on the high ground ──────────────────────────────────────────
    # This is the clamp's grave. Under the old code the dabs first CRUSHED the
    # cell to the clamp (a large negative delta on a tall terrain) and then
    # could add nothing back.
    out = stroke(pipe, "terrain.sculpt", dab_line(px, pz), mode="raise",
                 radius=40.0, strength=200.0, dt=0.05)
    check("raise returns a measurement", out is not None)
    if out is not None:
        delta = out["height_after"] - out["height_before"]
        check("raise RAISES the ground it touches", delta > 0.5,
              f"height {out['height_before']:.2f} -> {out['height_after']:.2f} m "
              f"(delta {delta:+.2f})")
        check("raise does not collapse the field", out["field_max_after"] >= out["field_max_before"],
              f"field max {out['field_max_before']:.3f} -> {out['field_max_after']:.3f}")

    # ── 2. LOWER, same place ────────────────────────────────────────────────
    out = stroke(pipe, "terrain.sculpt", dab_line(px, pz), mode="lower",
                 radius=40.0, strength=200.0, dt=0.05)
    if out is not None:
        delta = out["height_after"] - out["height_before"]
        check("lower LOWERS the ground it touches", delta < -0.5,
              f"height {out['height_before']:.2f} -> {out['height_after']:.2f} m "
              f"(delta {delta:+.2f})")

    # ── 3. FLATTEN toward a pinned altitude ─────────────────────────────────
    target = pheight - 50.0
    before = result_of(call(pipe, "terrain.sample_height",
                            {"name": TERRAIN, "world_x": px, "world_z": pz}))
    out = stroke(pipe, "terrain.sculpt", dab_line(px, pz, count=12), mode="flatten",
                 radius=40.0, strength=200.0, dt=0.05,
                 use_fixed_height=True, flatten_target=target)
    if out is not None and before is not None:
        moved_toward = abs(out["height_after"] - target) < abs(before - target)
        check("flatten moves TOWARD its target altitude", moved_toward,
              f"target {target:.1f} m, height {before:.2f} -> {out['height_after']:.2f} m")

    # ── 4. Refusals are refusals, not silent no-ops ─────────────────────────
    off = result_of(call(pipe, "terrain.sculpt",
                         {"name": TERRAIN, "dabs": [[SIZE * 4, SIZE * 4]], "mode": "raise"}))
    check("a dab off the tile is REFUSED", off is None,
          "a miss reported as a change is how a broken mapping passes a test")
    bad = result_of(call(pipe, "terrain.sculpt",
                         {"name": TERRAIN, "dabs": dab_line(px, pz), "mode": "melt"}))
    check("an unknown sculpt mode is refused", bad is None)

    # ── 5. SPLAT PAINT ───────────────────────────────────────────────────────
    layers = result_of(call(pipe, "terrain.list_layers", {"name": TERRAIN}))
    paint = stroke(pipe, "terrain.paint_splat", dab_line(px, pz, count=10),
                   channel=1, radius=60.0, strength=1.0, dt=0.05)
    if paint is None:
        # A terrain with no splat map is a setup gap, not a paint failure: say
        # which one it is instead of reporting a red brush.
        check("splat paint reachable (terrain has layers)", False,
              "terrain.paint_splat refused - initialise the terrain's layers first; "
              f"list_layers said {json.dumps(layers)[:160]}")
    else:
        check("paint CHANGES the channel it paints",
              paint["coverage_after"] - paint["coverage_before"] > 1e-4,
              f"channel {paint['channel']} coverage {paint['coverage_before']:.5f} -> "
              f"{paint['coverage_after']:.5f}")
        other = stroke(pipe, "terrain.paint_splat", dab_line(px, pz, count=1),
                       channel=9, radius=10.0)
        check("an out-of-range channel is refused", other is None)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s)")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("All terrain brush checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
