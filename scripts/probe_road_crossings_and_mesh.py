#!/usr/bin/env python3
"""
Probe: road cross-section (crown + ditch), crossing semantics and the surface mesh.

WHAT IT MEASURES
    1. The shipped profiles carry a REAL cross-section, not just a width. A main
       road has a crown and a ditch; a footpath has a crown and NO ditch. This is
       the whole answer to "a carved road reads as a river bed": a ditchless,
       crownless road is a flat linear trench, which is a perfect channel to any
       flow solver that reads the carved height afterwards.
    2. crown_meters / ditch_width / ditch_depth round-trip through
       set_carve_override, and a negative crown is REFUSED rather than clamped.
       A clamped write reads back as a write that landed and did something else.
    3. Every crossing mode round-trips, and an unknown one is refused.
    4. terrain.road.get_route FAILS EXPLICITLY when nothing has solved this road
       yet. "No route" and "the graph was never evaluated" must not read the
       same - an empty sample list would make an unevaluated graph look like a
       road with no length.
    5. terrain.road.build_mesh refuses the same way, and refuses an unassigned
       spline. The mesh is a view of the solved route, never its authority.
    6. terrain.road.clear_mesh refuses when the assignment owns no surface.

WHAT IT DOES NOT MEASURE
    Whether a bridge span actually appears in a solved route. That needs a
    terrain, a graph and an evaluation - see
    scripts/test/rt_test_terrain_road_crossings.py.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/probe_road_crossings_and_mesh.py

    Creates and deletes its own spline; needs no terrain.

EXIT CODE
    0 = PASS, 1 = FAIL, 2 = could not run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
CURVE_NAME = "__probe_road_section"

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


def error_text(response):
    if isinstance(response, dict):
        if isinstance(response.get("error"), str):
            return response["error"]
        if isinstance(response.get("error"), dict):
            return json.dumps(response["error"])
        result = response.get("result")
        if isinstance(result, dict) and "__error" in result:
            return str(result["__error"])
    return json.dumps(response)[:200]


def result_of(pipe, method, params, what):
    response = call(pipe, method, params)
    if refused(response) or not isinstance(response, dict) or "result" not in response:
        print("FAIL(setup): %s -> %s" % (what, json.dumps(response)[:300]))
        sys.exit(2)
    return response["result"]


def check(label, condition, detail=""):
    if condition:
        print("  ok   %-32s %s" % (label, detail))
    else:
        print("  FAIL %-32s %s" % (label, detail))
        _failures.append(label)


def section_cross_section(pipe):
    print("1. the shipped profiles have a real cross-section")
    profiles = result_of(pipe, "terrain.road.list_profiles", None, "list_profiles")
    by_id = {p.get("id"): p for p in profiles} if isinstance(profiles, list) else {}
    if set(by_id) != {"footpath", "dirt_road", "main_road"}:
        check("three profiles", False, ", ".join(sorted(by_id)))
        return
    foot, dirt, main = by_id["footpath"], by_id["dirt_road"], by_id["main_road"]
    for name in ("crown_meters", "ditch_width", "ditch_depth"):
        check("profile carries %s" % name, name in main, repr(main.get(name)))
    check("main road is crowned", main.get("crown_meters", 0.0) > 0.0,
          "%.3f m" % main.get("crown_meters", 0.0))
    check("main road has a ditch",
          main.get("ditch_width", 0.0) > 0.0 and main.get("ditch_depth", 0.0) > 0.0,
          "%.2f x %.2f m" % (main.get("ditch_width", 0.0), main.get("ditch_depth", 0.0)))
    # A hiking trail with a drainage ditch either side is not a trail. The
    # profiles must differ here, not only in width.
    check("footpath has NO ditch",
          foot.get("ditch_width", 1.0) == 0.0 and foot.get("ditch_depth", 1.0) == 0.0,
          "%.2f x %.2f m" % (foot.get("ditch_width", -1.0), foot.get("ditch_depth", -1.0)))
    check("footpath is still crowned", foot.get("crown_meters", 0.0) > 0.0,
          "%.3f m" % foot.get("crown_meters", 0.0))
    check("dirt ditch between the two",
          foot.get("ditch_width", 0.0) < dirt.get("ditch_width", 0.0) < main.get("ditch_width", 0.0),
          "%.2f < %.2f < %.2f" % (foot.get("ditch_width", 0.0),
                                  dirt.get("ditch_width", 0.0),
                                  main.get("ditch_width", 0.0)))


def section_override(pipe, name):
    print("\n2. the new dials round-trip and refuse nonsense")
    result_of(pipe, "terrain.road.assign_profile",
              {"spline": name, "profile": "dirt_road"}, "assign_profile")
    echo = result_of(pipe, "terrain.road.set_carve_override",
                     {"spline": name, "crown_meters": 0.25,
                      "ditch_width": 2.5, "ditch_depth": 0.9},
                     "set_carve_override")
    check("override echoes crown", abs(echo.get("crown_meters", 0.0) - 0.25) < 1e-4,
          repr(echo.get("crown_meters")))
    row = result_of(pipe, "terrain.road.get_assignment", {"spline": name}, "get_assignment")
    effective = row.get("effective", {})
    check("effective crown", abs(effective.get("crown_meters", 0.0) - 0.25) < 1e-4,
          repr(effective.get("crown_meters")))
    check("effective ditch width", abs(effective.get("ditch_width", 0.0) - 2.5) < 1e-4,
          repr(effective.get("ditch_width")))
    check("effective ditch depth", abs(effective.get("ditch_depth", 0.0) - 0.9) < 1e-4,
          repr(effective.get("ditch_depth")))
    # Partial writes must not reset the rest: one dial per call is how a
    # single-variable test stays single-variable.
    result_of(pipe, "terrain.road.set_carve_override",
              {"spline": name, "ditch_depth": 0.4}, "set_carve_override(partial)")
    row = result_of(pipe, "terrain.road.get_assignment", {"spline": name}, "get_assignment")
    effective = row.get("effective", {})
    check("partial write keeps crown",
          abs(effective.get("crown_meters", 0.0) - 0.25) < 1e-4,
          repr(effective.get("crown_meters")))
    check("negative crown refused",
          refused(call(pipe, "terrain.road.set_carve_override",
                       {"spline": name, "crown_meters": -1.0})))
    check("absurd crown refused",
          refused(call(pipe, "terrain.road.set_carve_override",
                       {"spline": name, "crown_meters": 5.0})),
          "a 5 m camber is a roof")
    result_of(pipe, "terrain.road.clear_carve_override", {"spline": name},
              "clear_carve_override")


def section_crossings(pipe, name):
    print("\n3. every crossing mode round-trips")
    for mode in ("terrain", "bridge", "ford", "tunnel", "auto"):
        result_of(pipe, "terrain.road.set_crossing_mode", {"spline": name, "mode": mode},
                  "set_crossing_mode(%s)" % mode)
        row = result_of(pipe, "terrain.road.get_assignment", {"spline": name},
                        "get_assignment")
        check("crossing %s" % mode, row.get("crossing_mode") == mode,
              repr(row.get("crossing_mode")))
    check("unknown mode refused",
          refused(call(pipe, "terrain.road.set_crossing_mode",
                       {"spline": name, "mode": "hovercraft"})))


def section_route_and_mesh(pipe, name):
    print("\n4. an unsolved route and an unbuilt mesh say so")
    response = call(pipe, "terrain.road.get_route", {"spline": name, "max_samples": 8})
    # The important half of this check is the WORDING: an empty sample list would
    # make "the graph was never evaluated" look like "this road has no length".
    check("unsolved route refused", refused(response), error_text(response)[:110])
    check("route error names the cause",
          "Road Network" in error_text(response) or "solved" in error_text(response),
          error_text(response)[:110])
    check("route on unassigned spline refused",
          refused(call(pipe, "terrain.road.get_route", {"spline": "__no_such_spline__"})))

    print("\n5. the mesh refuses what it cannot build")
    check("mesh without a solve refused",
          refused(call(pipe, "terrain.road.build_mesh", {"spline": name})))
    check("mesh on unassigned spline refused",
          refused(call(pipe, "terrain.road.build_mesh", {"spline": "__no_such_spline__"})))
    check("bad surface_offset refused",
          refused(call(pipe, "terrain.road.build_mesh",
                       {"spline": name, "surface_offset": -1.0})))
    check("bad uv tile refused",
          refused(call(pipe, "terrain.road.build_mesh",
                       {"spline": name, "uv_meters_per_tile": 0.0})))
    check("clear_mesh with nothing owned refused",
          refused(call(pipe, "terrain.road.clear_mesh", {"spline": name})),
          "owns no generated surface")


def main():
    pipe = connect()
    name = CURVE_NAME
    try:
        section_cross_section(pipe)
        created = result_of(pipe, "spline.create",
                            {"primitive": "empty", "name": CURVE_NAME, "plane": "free"},
                            "spline.create")
        if isinstance(created, str) and created:
            name = created
        section_override(pipe, name)
        section_crossings(pipe, name)
        section_route_and_mesh(pipe, name)
    finally:
        try:
            call(pipe, "terrain.road.clear_profile", {"spline": name})
            call(pipe, "scene.delete", {"name": name})
        except Exception:
            pass
        _kernel32.CloseHandle(pipe)

    if _failures:
        print("\nFAIL: %d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        sys.exit(1)
    print("\nPASS: the cross-section is real and editable, crossings round-trip, "
          "and route/mesh refuse instead of returning a plausible nothing.")
    sys.exit(0)


if __name__ == "__main__":
    main()
