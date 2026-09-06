#!/usr/bin/env python3
"""
Probe: road assignments are reachable, honest, and store no curve geometry.

WHAT IT MEASURES
    1. terrain.road.list_profiles reports the three shipped profiles with real
       carve numbers, and a footpath is NOT a main road with a different name -
       its cut/fill budget and width differ.
    2. assign_profile REFUSES an unknown profile id and a missing spline. Both
       matter: substituting a default would carve a main road where a footpath
       was asked for, and an assignment naming a deleted curve looks identical to
       a working one in every listing.
    3. The assignment round-trips: assign -> get_assignment -> list_assignments,
       including crossing_mode and enabled.
    4. get_diagnostics counts assignments and surfaces dangling ones. Deleting
       the spline behind an assignment must show up here, not vanish.
    5. There is NO terrain.road.set_points. Curve geometry stays on spline.*, so
       the registry cannot become a second authority for the shape of a curve.

WHY IT IS NEEDED
    This registry is what lets one Road Network node carve N roads. If an
    assignment can be written but not read back, "the road did not carve" and
    "the assignment never landed" are indistinguishable.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/probe_road_assignments.py

    Creates and deletes its own spline; needs no terrain.

EXIT CODE
    0 = PASS, 1 = FAIL, 2 = could not run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
CURVE_NAME = "__probe_road_curve"

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
        print("  ok   %-30s %s" % (label, detail))
    else:
        print("  FAIL %-30s %s" % (label, detail))
        _failures.append(label)


def section_profiles(pipe):
    print("1. profiles are real, and differ from one another")
    profiles = result_of(pipe, "terrain.road.list_profiles", None, "list_profiles")
    by_id = {p.get("id"): p for p in profiles} if isinstance(profiles, list) else {}
    check("three profiles", set(by_id) == {"footpath", "dirt_road", "main_road"},
          ", ".join(sorted(by_id)))
    if len(by_id) < 3:
        return
    foot, main = by_id["footpath"], by_id["main_road"]
    # A footpath that carries a main road's earthworks budget is a highway
    # cutting with a different label. These must not be the same numbers.
    check("footpath narrower", foot["road_width"] < main["road_width"],
          "%.2f < %.2f" % (foot["road_width"], main["road_width"]))
    check("footpath smaller cut", foot["max_cut_meters"] < main["max_cut_meters"],
          "%.2f < %.2f" % (foot["max_cut_meters"], main["max_cut_meters"]))
    check("footpath steeper allowed",
          foot["max_grade_percent"] > main["max_grade_percent"],
          "%.1f%% > %.1f%%" % (foot["max_grade_percent"], main["max_grade_percent"]))


def section_refusals(pipe, name):
    print("\n2. bad input is refused, not defaulted")
    check("unknown profile refused",
          refused(call(pipe, "terrain.road.assign_profile",
                       {"spline": name, "profile": "autobahn"})))
    check("missing spline refused",
          refused(call(pipe, "terrain.road.assign_profile",
                       {"spline": "__no_such_spline__", "profile": "dirt_road"})))
    check("unknown crossing mode refused",
          refused(call(pipe, "terrain.road.set_crossing_mode",
                       {"spline": name, "mode": "teleport"})))
    check("crossing on unassigned refused",
          refused(call(pipe, "terrain.road.set_crossing_mode",
                       {"spline": name, "mode": "bridge"})))


def section_roundtrip(pipe, name):
    print("\n3. assignment round trip")
    result_of(pipe, "terrain.road.assign_profile",
              {"spline": name, "profile": "footpath"}, "assign_profile")
    row = result_of(pipe, "terrain.road.get_assignment", {"spline": name}, "get_assignment")
    check("profile stored", row.get("profile_id") == "footpath", repr(row.get("profile_id")))
    check("curve_exists true", row.get("curve_exists") is True)
    check("default crossing auto", row.get("crossing_mode") == "auto",
          repr(row.get("crossing_mode")))
    check("enabled by default", row.get("enabled") is True)

    result_of(pipe, "terrain.road.set_crossing_mode",
              {"spline": name, "mode": "bridge"}, "set_crossing_mode")
    result_of(pipe, "terrain.road.set_enabled",
              {"spline": name, "enabled": False}, "set_enabled")
    row = result_of(pipe, "terrain.road.get_assignment", {"spline": name}, "get_assignment")
    check("crossing mode kept", row.get("crossing_mode") == "bridge",
          repr(row.get("crossing_mode")))
    check("enabled flag kept", row.get("enabled") is False)
    # Re-assigning must not reset the other fields; a partial write that silently
    # restores defaults makes every single-variable test change two things.
    result_of(pipe, "terrain.road.assign_profile",
              {"spline": name, "profile": "main_road"}, "assign_profile(again)")
    row = result_of(pipe, "terrain.road.get_assignment", {"spline": name}, "get_assignment")
    check("reassign keeps crossing", row.get("crossing_mode") == "bridge",
          repr(row.get("crossing_mode")))
    check("reassign keeps enabled", row.get("enabled") is False)

    rows = result_of(pipe, "terrain.road.list_assignments", None, "list_assignments")
    check("listed once", sum(1 for r in rows if r.get("spline_object") == name) == 1)


def section_diagnostics(pipe, name):
    print("\n4. diagnostics see a deleted curve")
    diagnostics = result_of(pipe, "terrain.road.get_diagnostics", None, "get_diagnostics")
    check("counted", diagnostics.get("assignment_count", 0) >= 1,
          "count=%r" % diagnostics.get("assignment_count"))
    check("not dangling yet", name not in diagnostics.get("dangling", []))

    call(pipe, "scene.delete", {"name": name})
    diagnostics = result_of(pipe, "terrain.road.get_diagnostics", None, "get_diagnostics")
    # This is the point of the whole section: a road whose curve was deleted must
    # be visible. Silently skipping it means the terrain simply comes back
    # different with nothing to explain why.
    check("dangling after delete", name in diagnostics.get("dangling", []),
          repr(diagnostics.get("dangling")))
    row = result_of(pipe, "terrain.road.get_assignment", {"spline": name}, "get_assignment")
    check("curve_exists now false", row.get("curve_exists") is False)


def section_no_second_authority(pipe):
    print("\n5. the registry is not a second curve authority")
    response = call(pipe, "terrain.road.set_points", {"spline": CURVE_NAME, "points": []})
    check("no terrain.road.set_points", refused(response),
          "curve geometry stays on spline.*")


def main():
    pipe = connect()
    name = CURVE_NAME
    try:
        section_profiles(pipe)
        created = result_of(pipe, "spline.create",
                            {"primitive": "empty", "name": CURVE_NAME, "plane": "free"},
                            "spline.create")
        if isinstance(created, str) and created:
            name = created
        section_refusals(pipe, name)
        section_roundtrip(pipe, name)
        section_diagnostics(pipe, name)
        section_no_second_authority(pipe)
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
    print("\nPASS: road assignments write, read back, refuse bad input and report "
          "a curve that went missing.")
    sys.exit(0)


if __name__ == "__main__":
    main()
