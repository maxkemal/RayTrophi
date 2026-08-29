#!/usr/bin/env python3
"""
RayTrophi Studio - Noise Generator landform shape acceptance check.

★★★ EVERY BOUND HERE IS TWO-SIDED, AND THAT IS THE POINT.

The first version of this file asserted only that the terrain was not the thing
it used to be: more flat ground than 4.4%, more relief contrast than 1.92, more
broad growth than 1.05. It passed - while the terrain went past useful and out
the other side. Measured on the build it passed on:

    flat ground (<3 deg)        56.1 %      (target: a mountain tile, not a plain)
    median slope                 2.20 deg
    lowland local relief p10     1.6 m      of 134 m on the tile - a floor
    ground over 40 degrees       0.00 %     no rock anywhere

A one-sided bound cannot fail in the direction you are actually moving. It is
not a test, it is a ratchet.

WHAT THE GENERATOR IS BEING ASKED FOR, in numbers a landscape satisfies:

  flat_fraction        0.15 .. 0.45   somewhere to put a road, not a plain
  median_slope_deg      4    .. 18    a mountain tile, at alpine relief
  cliff_fraction        0.05 .. 0.35  ground too steep to hold soil = rock
  local_relief_ratio    2.5  .. 15    massifs and basins, not one texture and
                                      not a cliff standing on a table
  lowland relief share  > 6 %         the lowland is country, not a floor
  lowland > midland     bottom-heavy hypsometry, the one reading that depends
                        on no tuning constant at all

Usage:
    1. Start RayTrophi Studio.
    2. python scripts/terrain_landform_shape_check.py
"""
import json
import sys

sys.path.insert(0, __file__.rsplit("\\", 1)[0].rsplit("/", 1)[0])
from ipc_test_client import send_command, PIPE_NAME  # noqa: E402

TERRAIN = "LandformShapeCheck"
RES, SIZE, HSCALE = 512, 4096.0, 1000.0
# Alpine relief. The node default of 140 m over a 4 km tile is genuinely gentle
# ground and cannot show rock at any setting - asking it to would be measuring
# the wrong terrain, which is how the first round of this got a false reading.
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


def check(label, condition, detail=""):
    if condition:
        print(f"  [PASS] {label}" + (f"  ({detail})" if detail else ""))
    else:
        print(f"  [FAIL] {label}" + (f"\n         {detail}" if detail else ""))
        failures.append(label)


def between(label, value, low, high, fmt="{:.2f}"):
    check(f"{label} within [{fmt.format(low)}, {fmt.format(high)}]",
          low <= value <= high,
          f"measured {fmt.format(value)} - "
          + ("BELOW the range" if value < low else "ABOVE the range"))


def open_pipe():
    import ctypes
    import ctypes.wintypes as wintypes
    kernel32 = ctypes.windll.kernel32
    kernel32.CreateFileW.restype = ctypes.c_void_p
    handle = kernel32.CreateFileW(PIPE_NAME, 0x80000000 | 0x40000000, 0, None, 3, 0, None)
    if handle is None or handle == (2 ** 64 - 1):
        print(f"[landform-check] FAIL: cannot connect to {PIPE_NAME}. Is RayTrophi Studio running?")
        sys.exit(1)
    handle = ctypes.c_void_p(handle)
    kernel32.SetNamedPipeHandleState(handle, ctypes.byref(wintypes.DWORD(2)), None, None)
    return handle


def evaluate(pipe):
    import time
    call(pipe, "terrain.evaluate", {"name": TERRAIN})
    for _ in range(300):
        state = result_of(call(pipe, "terrain.evaluation_status", {"name": TERRAIN}))
        if not state or state.get("state") != "running":
            return state
        time.sleep(0.5)
    return {"state": "timeout"}


def measure(pipe, node, label, props):
    for key, value in props.items():
        call(pipe, "nodes.set_property", {
            "graph_type": "terrain", "graph_name": TERRAIN,
            "node_id": node, "property": key, "value": value})
    evaluate(pipe)
    st = result_of(call(pipe, "terrain.landform_stats", {"name": TERRAIN}))
    if not st or st.get("status") != "ok":
        print(f"  [{label}] landform_stats unavailable: {json.dumps(st)}")
        return None
    st["lowland_relief_share"] = st["local_relief_p10"] / max(st["relief_meters"], 1e-6)
    print(f"  [{label}] relief {st['relief_meters']:.0f} m over {st['size_meters']:.0f} m")
    print(f"      slope   median {st['median_slope_deg']:5.2f}  p95 {st['p95_slope_deg']:5.1f}"
          f"   >40deg {100 * st['cliff_fraction']:5.2f}%   flat<3deg {100 * st['flat_fraction']:5.1f}%")
    print(f"      rock    steep roughness {st['steep_roughness_meters']:.3f} m vs gentle "
          f"{st['gentle_roughness_meters']:.3f} m  -> ratio {st['roughness_slope_ratio']:.2f}")
    print(f"      relief  local p10 {st['local_relief_p10']:6.1f} / p90 {st['local_relief_p90']:6.1f}"
          f" -> ratio {st['local_relief_ratio']:.2f}   lowland share "
          f"{100 * st['lowland_relief_share']:.1f}%")
    print(f"      hypso   lowland {100 * st['lowland_fraction']:5.1f}%  midland "
          f"{100 * st['midland_fraction']:5.1f}%   broad_growth {st['broad_growth']:.2f}")
    print(f"      scales  H {st['realised_hurst']:.3f} over {st['hurst_sample_count']} octaves"
          f"   kink {st['spectrum_kink']:+.3f} at {st['spectrum_kink_meters']:.0f} m"
          f"  (signed {st['spectrum_kink_signed']:+.3f})")
    return st


def build_graph(pipe):
    call(pipe, "terrain.remove", {"name": TERRAIN})
    created = result_of(call(pipe, "terrain.create", {
        "name": TERRAIN, "resolution": RES, "size": SIZE, "height_scale": HSCALE}))
    check("terrain created", created is not None)
    if created is None:
        return None, None
    call(pipe, "terrain.apply_preset", {
        "name": TERRAIN, "preset": "default", "replace_graph": True, "add_satmap": False})
    nodes = result_of(call(pipe, "nodes.list",
                           {"graph_type": "terrain", "graph_name": TERRAIN})) or []
    ids = {n["type_id"]: n["id"] for n in nodes}
    call(pipe, "nodes.remove", {"graph_type": "terrain", "graph_name": TERRAIN,
                                "node_id": ids["TerrainV2.HeightmapInput"]})
    return ids["TerrainV2.HeightOutput"], ids


def add_and_link(pipe, out_node, type_id, key="height"):
    added = result_of(call(pipe, "nodes.add", {
        "graph_type": "terrain", "graph_name": TERRAIN,
        "type_id": type_id, "x": 40.0, "y": 100.0}))
    node = added["id"] if isinstance(added, dict) else added
    linked = result_of(call(pipe, "nodes.link_by_key", {
        "graph_type": "terrain", "graph_name": TERRAIN,
        "from_node": node, "from_output": key,
        "to_node": out_node, "to_input": "height"}))
    return node, linked


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    pipe = open_pipe()
    print(f"[landform-check] connected to {PIPE_NAME}\n")

    out_node, ids = build_graph(pipe)
    if out_node is None:
        return 1
    node, linked = add_and_link(pipe, out_node, "TerrainV2.NoiseGenerator")
    check("noise generator wired to height output", bool(linked))

    shipped = {"terrainModel": 2, "seed": 1337, "octaves": 10,
               "terrainRoughness": 0.45, "reliefMeters": RELIEF,
               "autoFeatureSize": True, "lowlandFraction": 0.35,
               "slopeLimitDegrees": 60.0}

    print("\n1. Orogenic at alpine relief: is this a landscape with rock in it?")
    base = measure(pipe, node, "defaults", shipped)
    if not base:
        check("landform_stats returned a measurement", False)
        return 1

    between("flat ground fraction", base["flat_fraction"], 0.15, 0.45)
    between("median slope (deg)", base["median_slope_deg"], 4.0, 18.0)
    between("cliff fraction (>40 deg)", base["cliff_fraction"], 0.05, 0.35)
    between("massif / basin relief ratio", base["local_relief_ratio"], 2.5, 15.0)
    check("the lowland is country, not a floor",
          base["lowland_relief_share"] > 0.06,
          f"gentlest tiles hold {100 * base['lowland_relief_share']:.1f}% of the "
          f"tile relief ({base['local_relief_p10']:.1f} m of {base['relief_meters']:.0f} m)")
    # The one reading that leans on no tuning constant: deposition makes real
    # landscapes bottom-heavy, a fitted fBm is gaussian and piles into the middle.
    check("hypsometry is bottom-heavy, not gaussian",
          base["lowland_fraction"] > base["midland_fraction"],
          f"lowland {100 * base['lowland_fraction']:.1f}% vs midland "
          f"{100 * base['midland_fraction']:.1f}%")
    # Rough BECAUSE steep. Only meaningful next to a real cliff fraction, which
    # the bound above already established.
    # Macro-micro coherence. This is the one that no other reading covers:
    # slope, relief and hypsometry all stay healthy while a detail layer sits
    # on the landform disagreeing with it, because none of them looks ACROSS
    # scales. A single power law from the grid to the landform is what "the
    # micro belongs to the macro" means numerically.
    check("the scales are one landscape (no seam in the relief spectrum)",
          base["spectrum_kink"] < 0.12,
          f"worst octave departs {base['spectrum_kink_signed']:+.3f} from H="
          f"{base['realised_hurst']:.3f} at {base['spectrum_kink_meters']:.0f} m - "
          + ("that scale is STARVED of detail" if base["spectrum_kink_signed"] < 0
             else "that scale carries too much"))

    check("steep ground is rougher than gentle ground",
          base["roughness_slope_ratio"] > 2.0,
          f"ratio {base['roughness_slope_ratio']:.2f}")

    print("\n2. CONTROL - is Auto Feature Size what widened the landforms?")
    narrow = measure(pipe, node, "feature 600 m, auto off",
                     dict(shipped, autoFeatureSize=False, featureSizeMeters=600.0))
    if narrow:
        check("pinning Feature Size back to 600 m collapses broad growth",
              narrow["broad_growth"] < base["broad_growth"] * 0.85,
              f"{base['broad_growth']:.2f} -> {narrow['broad_growth']:.2f}")

    print("\n3. CONTROL - is Lowland what created the gentle country?")
    dense = measure(pipe, node, "lowland 0.0", dict(shipped, lowlandFraction=0.0))
    if dense:
        check("removing the lowland removes the gentle ground",
              dense["flat_fraction"] < base["flat_fraction"] * 0.75,
              f"{100 * base['flat_fraction']:.1f}% -> {100 * dense['flat_fraction']:.1f}%")
        check("and flattens the massif/basin contrast",
              dense["local_relief_ratio"] < base["local_relief_ratio"] * 0.8,
              f"{base['local_relief_ratio']:.2f} -> {dense['local_relief_ratio']:.2f}")

    print("\n4. CONTROL - does it come back exactly?")
    restored = measure(pipe, node, "restored", shipped)
    if restored:
        check("restoring the defaults restores the measurement",
              abs(restored["flat_fraction"] - base["flat_fraction"]) < 0.01 and
              abs(restored["local_relief_ratio"] - base["local_relief_ratio"]) < 0.05)

    print("\n5. The other two terrain models must stay usable")
    for model, name in ((1, "Continental"), (3, "Eroded Highlands")):
        st = measure(pipe, node, name, dict(shipped, terrainModel=model))
        if st:
            check(f"{name} delivers its relief", st["relief_meters"] > RELIEF * 0.7,
                  f"{st['relief_meters']:.0f} m of an authored {RELIEF:.0f} m")
            between(f"{name} flat fraction", st["flat_fraction"], 0.10, 0.60)

    print("\n6. Mountain Range: a fast mountain node has to reach the whole tile")
    # Measured before this batch: the default 420 m width on a 4096 m tile gave
    # a median slope of 0.01 degrees - a thin welt on a flat plate, while Length
    # was already authored as a fraction of the terrain. Width is now the same
    # unit, so the node is usable at whatever size the terrain happens to be.
    out_node, _ = build_graph(pipe)
    if out_node is not None:
        mr, linked = add_and_link(pipe, out_node, "TerrainV2.MountainRange")
        check("mountain range wired to height output", bool(linked))
        st = measure(pipe, mr, "mountain range", {"reliefMeters": RELIEF, "seed": 4201})
        if st:
            check("the range covers the tile rather than welting it",
                  st["median_slope_deg"] > 1.0,
                  f"median slope {st['median_slope_deg']:.2f} deg, was 0.01 at 420 m width")
            between("mountain range cliff fraction", st["cliff_fraction"], 0.02, 0.40)

    print("\n7. Math needs BOTH operands")
    out_node, _ = build_graph(pipe)
    if out_node is not None:
        added = result_of(call(pipe, "nodes.add", {
            "graph_type": "terrain", "graph_name": TERRAIN,
            "type_id": "TerrainV2.Math", "x": 300.0, "y": 100.0}))
        math_id = added["id"] if isinstance(added, dict) else added
        ports = result_of(call(pipe, "nodes.list_ports", {
            "graph_type": "terrain", "graph_name": TERRAIN, "node_id": math_id})) or []
        inputs = [p for p in ports if p["direction"] == "input"]
        check("Math draws both operand sockets without anything connected",
              len(inputs) == 2 and all(p.get("visible", True) for p in inputs),
              json.dumps([{k: p.get(k) for k in ("key", "visible", "exposure")}
                          for p in inputs]))
        check("neither operand is marked optional",
              all(p.get("exposure") == "primary" for p in inputs),
              json.dumps([p.get("exposure") for p in inputs]))

    print(f"\n== {len(failures)} failed ==")
    for item in failures:
        print(f"   FAILED: {item}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
