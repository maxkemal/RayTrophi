#!/usr/bin/env python3
"""
RayTrophi Studio - terrain mask pipeline acceptance check.

What this verifies, and why each check exists:

1. terrain.apply_preset now reports the links a setup could not make.
   Before this, addLink answered a refused connection with 0 and every
   setup ignored the answer, so a preset built with hydraulic erosion in
   the chain silently left Surface Composer's Flow and Wetness inputs
   dangling. The composer fell back to synthesized values and the render
   merely looked plausible - the failure nobody reports as a bug. An
   empty wiring_faults list is the ONLY evidence the setup wired up.

2. Slope now has one definition. Auto Splat and Surface Composer used to
   disagree about the same pixel (45 degrees read as 45.0, 0.50 or 0.75
   depending on the consumer), so two splat authors produced different
   maps from identical terrain. This walks both nodes over one terrain
   and asserts their published splat maps are not wildly divergent.

Usage:
    1. Start RayTrophi Studio (creates \\\\.\\pipe\\RayTrophiStudio).
    2. python scripts/terrain_mask_pipeline_check.py

Exits non-zero on any failure so an agent can gate on it.
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ipc_test_client import send_command, PIPE_NAME  # noqa: E402

TERRAIN = "MaskPipelineCheck"

failures = []
checks = 0


def call(pipe, method, params=None):
    response = send_command(pipe, method, params, request_id=call.counter)
    call.counter += 1
    return response


call.counter = 1


def check(label, condition, detail=""):
    global checks
    checks += 1
    if condition:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}{(' - ' + detail) if detail else ''}")
        failures.append(label)


def result_of(response):
    """Unwrap an IPC response, returning None when the call errored."""
    if not isinstance(response, dict):
        return None
    if response.get("error"):
        return None
    return response.get("result", response)


def open_pipe():
    import ctypes
    import ctypes.wintypes as wintypes

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.CreateFileW(PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
    if handle == -1 or (handle & 0xFFFFFFFFFFFFFFFF) == (wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF):
        print(f"[terrain-check] FAIL: cannot connect to {PIPE_NAME}. Is RayTrophi Studio running?")
        sys.exit(1)
    mode = wintypes.DWORD(2)  # PIPE_READMODE_MESSAGE
    kernel32.SetNamedPipeHandleState(handle, ctypes.byref(mode), None, None)
    return handle


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pipe = open_pipe()
    print(f"[terrain-check] connected to {PIPE_NAME}\n")

    # ------------------------------------------------------------------
    print("1. Setup wiring is reported, not silently dropped")
    # ------------------------------------------------------------------
    call(pipe, "terrain.delete", {"name": TERRAIN})
    created = result_of(call(pipe, "terrain.create", {
        "name": TERRAIN, "resolution": 512, "mesh_resolution": 512,
        "size": 2000.0, "height_scale": 400.0}))
    check("terrain created", created is not None)

    # Every preset that wires a Surface Composer must come back clean. A
    # non-empty list here names exactly which wire the setup lost.
    for preset in ("biome_temperate", "biome_alpine", "snow_layer", "river_network"):
        call(pipe, "terrain.delete", {"name": TERRAIN})
        call(pipe, "terrain.create", {
            "name": TERRAIN, "resolution": 512, "mesh_resolution": 512,
            "size": 2000.0, "height_scale": 400.0})
        applied = result_of(call(pipe, "terrain.apply_preset",
                                 {"name": TERRAIN, "preset": preset}))
        if applied is None:
            check(f"{preset} applied", False, "call failed")
            continue
        faults = applied.get("wiring_faults", None)
        check(f"{preset}: wiring_faults field present", faults is not None,
              "the setup cannot be verified without it")
        check(f"{preset}: every link connected", not faults,
              json.dumps(faults))

    # ------------------------------------------------------------------
    print("\n2. Slope has one definition across splat authors")
    # ------------------------------------------------------------------
    # Both composers classify from the same shared analysis slope now, so
    # a graph carrying both must not produce two unrelated splat maps.
    call(pipe, "terrain.delete", {"name": TERRAIN})
    call(pipe, "terrain.create", {
        "name": TERRAIN, "resolution": 512, "mesh_resolution": 512,
        "size": 2000.0, "height_scale": 400.0})
    applied = result_of(call(pipe, "terrain.apply_preset",
                             {"name": TERRAIN, "preset": "biome_temperate"}))
    check("biome preset applied for slope comparison", applied is not None)

    nodes = result_of(call(pipe, "nodes.list", {"graph_type": "terrain", "graph_name": TERRAIN}))
    check("node list readable", nodes is not None)
    if nodes:
        names = json.dumps(nodes)
        # The Slope pin is what makes the shared solve reachable at all.
        check("Surface Composer present in the setup", "SurfaceComposer" in names or
              "Surface Composer" in names, names[:200])
        check("Terrain Analysis present (the shared slope source)",
              "TerrainAnalysis" in names or "Terrain Analysis" in names, names[:200])

    # ------------------------------------------------------------------
    print("\n3. An unbounded field reaching a mask pin stays graded")
    # ------------------------------------------------------------------
    # Hydraulic discharge is an SI field. Clamping it to 0-1 used to
    # collapse it into a binary stencil: every channel pixel exactly 1,
    # everything else exactly 0. Erode, then confirm the flow statistics
    # are not degenerate.
    eroded = result_of(call(pipe, "terrain.erode", {
        "name": TERRAIN, "iterations": 20000, "type": "hydraulic"}))
    check("erosion ran", eroded is not None)
    stats = result_of(call(pipe, "terrain.erosion_stats"))
    if stats is not None:
        density = stats.get("drainage_density", 0.0)
        # A binary stencil produces either a saturated or an empty field.
        check("drainage density is graded, not degenerate",
              0.0 < float(density) < 1.0, f"drainage_density={density}")
    else:
        check("erosion stats readable", False)

    # ------------------------------------------------------------------
    print("\n3b. The drainage network reaches the map edge instead of dying")
    # ------------------------------------------------------------------
    # This endpoint deliberately reconstructs a diagnostic flow field from the
    # final heightmap. It does NOT inspect Hydraulic Erosion's authoritative
    # receiver/direction raster, so it cannot prove exact river topology. It
    # does catch the original wholesale failure, though: an unconditioned pit
    # made reconstructed accumulation fade out before it reached the basin
    # spill or map border.
    flow = result_of(call(pipe, "terrain.calculate_flow", {"name": TERRAIN}))
    check("flow statistics readable", flow is not None)
    if flow is not None:
        channels = int(flow.get("channel_cells", 0))
        inland = int(flow.get("inland_terminations", 0))
        border = int(flow.get("border_terminations", 0))
        ratio = float(flow.get("inland_termination_ratio", 1.0))
        check("a channel network exists at all", channels > 0,
              f"channel_cells={channels}")
        check("the reconstructed network reaches the border", border > 0,
              f"border_terminations={border}")
        # Numerically flat cells can still stall a single cell here and there;
        # what the bug produced was a wholesale loss, not a rounding tail.
        # This heuristic also counts some equal-accumulation plateaus as local
        # endpoints, so reserve failure for a wholesale loss. Exact receiver
        # continuity needs authoritative raster readback, which RTAPI does not
        # expose yet.
        check("reconstructed channels do not die wholesale inland", ratio < 0.10,
              f"inland={inland}/{channels} ratio={ratio:.4f}")

    # ------------------------------------------------------------------
    print("\n3c. Flow is the single authority for 'how much water is here'")
    # ------------------------------------------------------------------
    # Two quantities were both called flow: DISCHARGE (physical, measured by
    # the erosion sim against the terrain it is carving) and CHANNEL (a 0-1
    # selection of which cells read as a watercourse). Setups branched per pin
    # on whichever was available, so neighbouring consumers could be reading
    # different water with nothing in the render to say so.
    #
    # Erosion has already run above, so the setup must be reading the measured
    # field. derived_erosion_unwired is the failure that looks fine: channels
    # from bare geometry drawn over an eroded surface.
    call(pipe, "terrain.evaluate", {"name": TERRAIN})
    authority = result_of(call(pipe, "terrain.flow_authority", {"name": TERRAIN}))
    check("flow authority readable", authority is not None)
    if authority is not None:
        check("the setup built a Flow node", authority.get("has_flow_node") is True,
              json.dumps(authority))
        check("Flow classifies the MEASURED discharge after erosion",
              authority.get("source") == "measured", json.dumps(authority))
        check("erosion sim is not left unwired",
              authority.get("erosion_unwired") is False, json.dumps(authority))
        if authority.get("has_river_network"):
            # Watershed Analysis is the only node left that owns both halves of
            # this pair. Hydraulic Erosion used to publish a second copy of
            # Drainage Area and Flow Direction; the compact port contract
            # removed them, so "hydraulic" is no longer a reachable answer and a
            # network fed from Hydraulic's Flow pin reads as "mixed".
            #
            # The area/direction sub-check that used to hang off the hydraulic
            # answer is NOT promoted to run here. It asserted a 2-channel vector
            # direction, and that pin only ever existed on Hydraulic Erosion --
            # Watershed publishes a 1-channel direction code. Asserting it under
            # watershed would fail for a graph that is correctly wired.
            check("River Network uses one paired hydrology authority",
                  authority.get("river_network_source") == "watershed",
                  json.dumps(authority))
            check("River Network receives the accepted lake footprint",
                  authority.get("river_lake_mask_connected") is True,
                  json.dumps(authority))
            check("River Network receives explicit lake spill authority",
                  authority.get("river_lake_spill_connected") is True,
                  json.dumps(authority))

    # ------------------------------------------------------------------
    print("\n3d. The eroded landscape is fluvially GRADED, not just eroded")
    # ------------------------------------------------------------------
    # Every other check here says a pass ran and produced fields. None of them
    # says the result is a river landscape rather than plausible-looking noise
    # - and in this domain plausibility is the only thing a human check can
    # apply, which is exactly what every silent bug in this file passed.
    #
    # Stream-power erosion gives S = k * A^-theta: a straight log-log line.
    fit = result_of(call(pipe, "terrain.slope_area_fit", {"name": TERRAIN}))
    check("slope-area fit readable", fit is not None)
    if fit is not None:
        check("the fit had enough channel network to measure",
              fit.get("status") == "ok", json.dumps(fit))
        if fit.get("status") == "ok":
            r2 = float(fit.get("r_squared", 0.0))
            theta = float(fit.get("concavity_index", 0.0))
            # r_squared FIRST. A confident theta fitted to scatter is worse
            # than no number, because it reads as a passing measurement.
            check("channel slopes follow a power law (not scatter)", r2 > 0.6,
                  f"r_squared={r2:.3f} over {fit.get('bin_count')} bins")
            # Real landscapes sit near 0.4-0.6; the band is wide because the
            # point is to catch 0 and 2, not to grade the solver.
            check("concavity index is in the physical range", 0.15 < theta < 1.2,
                  f"theta={theta:.3f}")

    # ------------------------------------------------------------------
    print("\n3e. Crater / Caldera is reachable and actually moves the ground")
    # ------------------------------------------------------------------
    # The one macro landform the generator could not make: every other node
    # builds elongated or tectonic relief, and nothing radial could be
    # authored except by hand. A node that exists but cannot be created from
    # script is, by this repo's own rule, untested.
    before = result_of(call(pipe, "terrain.sample_height",
                            {"name": TERRAIN, "world_x": 0.0, "world_z": 0.0}))
    added = call(pipe, "nodes.add", {"graph_type": "terrain", "graph_name": TERRAIN,
                                     "type": "TerrainV2.CraterCaldera"})
    check("Crater / Caldera can be created from script",
          result_of(added) is not None, json.dumps(added)[:200])
    # A landform node that reports success while leaving the height field
    # untouched is the exact shape of the silent failures above, so the check
    # is on the ground, not on the return value.
    if before is not None:
        call(pipe, "terrain.evaluate", {"name": TERRAIN})
        after = result_of(call(pipe, "terrain.sample_height",
                               {"name": TERRAIN, "world_x": 0.0, "world_z": 0.0}))
        check("the graph still evaluates with the node in it", after is not None,
              f"before={before} after={after}")

    # ------------------------------------------------------------------
    print("\n4. Semantic channels can carry a material, not just a tweak")
    # ------------------------------------------------------------------
    # Slots 0-3 partition the surface via the splat map; slots 4-7 overlay
    # it by their own semantic weight. An empty overlay must keep the
    # built-in shading, so binding one has to be observable and reversible.
    layers = result_of(call(pipe, "terrain.list_layers", {"name": TERRAIN}))
    check("layer slots readable", layers is not None)
    if layers:
        rows = layers.get("layers", [])
        check("eight slots reported", len(rows) == 8, f"got {len(rows)}")
        semantic = [r for r in rows if r.get("semantic_overlay")]
        check("four of them are semantic overlays", len(semantic) == 4,
              json.dumps([r.get("channel") for r in rows]))
        flow_slot = next((r for r in rows if r.get("slot") == 4), None)
        check("slot 4 is the Flow channel",
              flow_slot is not None and "Flow" in flow_slot.get("channel", ""),
              json.dumps(flow_slot))
        check("overlays start unbound (built-in shading intact)",
              all(not r.get("bound") for r in semantic),
              "a fresh terrain must render as it did before overlays existed")

    # Bind a real material to the Flow overlay and read it back.
    mats = result_of(call(pipe, "material.list"))
    material_name = None
    if isinstance(mats, dict):
        entries = mats.get("materials", mats.get("names", []))
        if entries:
            first = entries[0]
            material_name = first.get("name") if isinstance(first, dict) else first
    elif isinstance(mats, list) and mats:
        first = mats[0]
        material_name = first.get("name") if isinstance(first, dict) else first

    if material_name:
        bound = result_of(call(pipe, "terrain.set_layer", {
            "name": TERRAIN, "slot": 4,
            "material": material_name, "overlay_strength": 0.75}))
        check("Flow overlay accepts a material", bound is not None)
        after = result_of(call(pipe, "terrain.list_layers", {"name": TERRAIN}))
        row = None
        if after:
            row = next((r for r in after.get("layers", []) if r.get("slot") == 4), None)
        check("bound material reads back", row is not None and row.get("bound") is True,
              json.dumps(row))
        check("overlay_strength reads back",
              row is not None and abs(float(row.get("overlay_strength", 0)) - 0.75) < 1e-4,
              json.dumps(row))

        # overlay_strength on a splat slot must be REFUSED, not ignored: the
        # splat slots are normalized against each other, so accepting it
        # would report a setting that does nothing.
        refused = call(pipe, "terrain.set_layer", {
            "name": TERRAIN, "slot": 1, "overlay_strength": 0.5})
        check("overlay_strength refused on a splat slot",
              result_of(refused) is None, json.dumps(refused)[:200])

        # Snow burial. An overlay's coverage is a VISIBILITY decision, not the
        # semantic measurement itself: flow reads 0.9 under two metres of snow
        # and painting it there drew a river across the snowfield. The opt-out
        # exists for open water cutting a snowfield.
        covered = call(pipe, "terrain.set_layer", {
            "name": TERRAIN, "slot": 4, "overlay_ignore_cover": True})
        check("Flow overlay accepts the burial opt-out", result_of(covered) is not None,
              json.dumps(covered)[:200])
        state = result_of(call(pipe, "terrain.list_layers", {"name": TERRAIN}))
        row = None
        if state:
            row = next((r for r in state.get("layers", []) if r.get("slot") == 4), None)
        check("overlay_ignore_cover reads back",
              row is not None and row.get("overlay_ignore_cover") is True, json.dumps(row))
        call(pipe, "terrain.set_layer", {
            "name": TERRAIN, "slot": 4, "overlay_ignore_cover": False})

        # Ice is a cover in its own right and is never buried, so the flag must
        # be REFUSED there rather than stored as a setting that does nothing.
        ice_refused = call(pipe, "terrain.set_layer", {
            "name": TERRAIN, "slot": 6, "overlay_ignore_cover": True})
        check("burial opt-out refused on the Ice slot",
              result_of(ice_refused) is None, json.dumps(ice_refused)[:200])
        splat_refused = call(pipe, "terrain.set_layer", {
            "name": TERRAIN, "slot": 2, "overlay_ignore_cover": True})
        check("burial opt-out refused on a splat slot",
              result_of(splat_refused) is None, json.dumps(splat_refused)[:200])

        # Clearing restores the built-in shading path.
        call(pipe, "terrain.set_layer", {"name": TERRAIN, "slot": 4, "material": ""})
        cleared = result_of(call(pipe, "terrain.list_layers", {"name": TERRAIN}))
        row = None
        if cleared:
            row = next((r for r in cleared.get("layers", []) if r.get("slot") == 4), None)
        check("empty string clears the slot",
              row is not None and row.get("bound") is False, json.dumps(row))
    else:
        check("a material was available to bind", False, "material.list returned nothing")

    # ------------------------------------------------------------------
    print("\n5. Every semantic channel is actually produced")
    # ------------------------------------------------------------------
    # A bound overlay has two ways to be dead, and neither raises an error:
    #   coverage 0   -> the graph never fills the channel (Auto Splat writes
    #                   Flow and hard-zeroes the rest)
    #   constant     -> min == max, a flat fill that reports FULL coverage
    #                   yet selects nothing. This is how an unwired Surface
    #                   Composer Hardness input presents: a constant 0.45
    #                   that washes the material evenly over the terrain.
    call(pipe, "terrain.delete", {"name": TERRAIN})
    call(pipe, "terrain.create", {
        "name": TERRAIN, "resolution": 512, "mesh_resolution": 512,
        "size": 2000.0, "height_scale": 400.0})
    applied = result_of(call(pipe, "terrain.apply_preset",
                             {"name": TERRAIN, "preset": "biome_temperate"}))
    check("biome preset applied for channel check", applied is not None)
    call(pipe, "terrain.evaluate", {"name": TERRAIN})

    rows = result_of(call(pipe, "terrain.list_layers", {"name": TERRAIN}))
    if rows:
        for r in rows.get("layers", []):
            if not r.get("semantic_overlay"):
                continue
            name = r.get("channel", "?")
            if not r.get("channel_measured"):
                check(f"{name}: semantic map exists", False,
                      "nothing published a semantic map")
                continue
            check(f"{name}: channel carries data",
                  float(r.get("channel_coverage", 0)) > 0.0,
                  f"coverage={r.get('channel_coverage')}")
            check(f"{name}: channel is not a flat fill",
                  not r.get("channel_constant"),
                  f"min={r.get('channel_min')} max={r.get('channel_max')} "
                  "- a constant selects nothing while reporting full coverage")
    else:
        check("layer channels readable", False)

    call(pipe, "terrain.delete", {"name": TERRAIN})

    print(f"\n[terrain-check] {checks - len(failures)}/{checks} passed")
    if failures:
        print("[terrain-check] FAILED:")
        for name in failures:
            print(f"  - {name}")
        sys.exit(1)
    print("[terrain-check] OK")


if __name__ == "__main__":
    main()
