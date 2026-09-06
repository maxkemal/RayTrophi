#!/usr/bin/env python3
"""
Probe: drawing a curve on a surface is reachable and honest from script.

WHAT IT MEASURES
    1. scene.raycast reports WHAT was hit, not just where. A caller that only
       learns a position has to guess the rest, and a guessed answer is
       indistinguishable from a correct one.
    2. An unknown filter is REFUSED, not defaulted. Silently widening a
       terrain_only query to mesh_and_terrain would return a mesh hit that looks
       perfectly reasonable.
    3. spline.create accepts primitive="empty" + plane="free", and
       spline.append_point builds that curve from ZERO points. This is the whole
       gap: spline.extrude needs two points to derive a tangent, so before this
       a curve could be EXTENDED from script but never BUILT.
    4. The two compose: raycast a ray at the ground, append what it returns, and
       read the curve back with the point actually on the reported surface.

WHY IT IS NEEDED
    The viewport gained a Draw-on-Surface tool. A tool that can only be driven
    by clicking is untestable by rule 1 - and the placement policy it uses
    (nearest of mesh / terrain / ground plane) is exactly the kind of thing that
    fails quietly: a point landing on a rock instead of the terrain still looks
    like a point.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/probe_surface_curve_authoring.py

    Needs no terrain: the ground-plane fallback carries sections 1-4. If a
    terrain IS present the ray simply lands on it instead, and the probe says
    which surface answered.

EXIT CODE
    0 = PASS, 1 = FAIL, 2 = could not run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
CURVE_NAME = "__probe_surface_curve"

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


def result_of(pipe, method, params, what):
    response = call(pipe, method, params)
    if not isinstance(response, dict) or "result" not in response:
        print("FAIL(setup): %s -> %s" % (what, response))
        sys.exit(2)
    result = response["result"]
    if isinstance(result, dict) and "__error" in result:
        print("FAIL(setup): %s -> %s" % (what, result["__error"]))
        sys.exit(2)
    return result


def check(label, condition, detail=""):
    if condition:
        print("  ok   %-28s %s" % (label, detail))
    else:
        print("  FAIL %-28s %s" % (label, detail))
        _failures.append(label)


def section_raycast(pipe):
    print("1. scene.raycast reports what was hit")
    # Straight down from high above the origin: hits the terrain if one exists,
    # otherwise the Y=0 ground plane. Either answer is correct; a MISSING kind
    # is not.
    hit = result_of(pipe, "scene.raycast",
                    {"origin": [0.0, 500.0, 0.0], "direction": [0.0, -1.0, 0.0]},
                    "scene.raycast")
    check("hit", hit.get("hit") is True, "kind=%r" % hit.get("kind"))
    check("kind is named", hit.get("kind") in ("mesh", "terrain", "ground_plane"),
          repr(hit.get("kind")))
    position = hit.get("position")
    check("position is a vec3", isinstance(position, list) and len(position) == 3,
          repr(position))
    check("normal is a vec3", isinstance(hit.get("normal"), list) and len(hit["normal"]) == 3)
    check("distance is positive", isinstance(hit.get("distance"), (int, float)) and
          hit["distance"] > 0.0, "%.3f" % hit.get("distance", -1))
    return hit


def section_filter_refused(pipe):
    print("\n2. an unknown filter is refused, not defaulted")
    response = call(pipe, "scene.raycast",
                    {"origin": [0.0, 500.0, 0.0], "direction": [0.0, -1.0, 0.0],
                     "filter": "everything"})
    refused = isinstance(response, dict) and "error" in response
    if not refused and isinstance(response, dict) and isinstance(response.get("result"), dict):
        refused = "__error" in response["result"]
    check("unknown filter refused", refused, repr(response)[:120])

    # A zero direction is degenerate and must not normalize into garbage.
    response = call(pipe, "scene.raycast",
                    {"origin": [0.0, 500.0, 0.0], "direction": [0.0, 0.0, 0.0]})
    refused = isinstance(response, dict) and "error" in response
    if not refused and isinstance(response, dict) and isinstance(response.get("result"), dict):
        refused = "__error" in response["result"]
    check("zero direction refused", refused, repr(response)[:120])


def section_build_from_zero(pipe, hit):
    print("\n3. a curve is BUILT from zero points")
    call(pipe, "scene.delete", {"name": CURVE_NAME})
    created = result_of(pipe, "spline.create",
                        {"primitive": "empty", "name": CURVE_NAME, "plane": "free"},
                        "spline.create(empty, free)")
    # spline.create returns the created name as a bare JSON string; the name may
    # differ from the request when a uniquifying suffix was appended.
    name = created if isinstance(created, str) and created else CURVE_NAME
    print("  created %r" % name)

    for row in result_of(pipe, "spline.list", None, "spline.list"):
        if row.get("name") == name:
            check("starts with no points", row.get("point_count") == 0,
                  "point_count=%r" % row.get("point_count"))
            break
    else:
        check("appears in spline.list", False, "not listed")
        return name

    # Three appends, the first two of which extrude cannot do at all.
    base = hit.get("position", [0.0, 0.0, 0.0])
    targets = [
        [base[0], base[1], base[2]],
        [base[0] + 10.0, base[1], base[2]],
        [base[0] + 20.0, base[1], base[2] + 10.0],
    ]
    for index, target in enumerate(targets):
        appended = result_of(pipe, "spline.append_point",
                             {"name": name, "position": target},
                             "spline.append_point #%d" % index)
        check("append %d -> index %d" % (index, index),
              appended.get("index") == index, "got %r" % appended.get("index"))

    for row in result_of(pipe, "spline.list", None, "spline.list"):
        if row.get("name") == name:
            check("point count after 3 appends", row.get("point_count") == 3,
                  "point_count=%r" % row.get("point_count"))
            break
    return name


def section_readback(pipe, name, hit):
    print("\n4. the drawn point is where the raycast said the surface is")
    payload = result_of(pipe, "spline.get", {"name": name}, "spline.get")
    text = payload if isinstance(payload, str) else json.dumps(payload)
    try:
        data = json.loads(text) if isinstance(payload, str) else payload
    except ValueError:
        check("spline.get returns JSON", False, text[:120])
        return
    points = None
    if isinstance(data, dict):
        for key in ("points", "control_points"):
            if isinstance(data.get(key), list):
                points = data[key]
                break
        if points is None and isinstance(data.get("spline"), dict):
            points = data["spline"].get("points")
    if not isinstance(points, list) or not points:
        check("readback has points", False, "payload keys=%r" % (
            list(data.keys()) if isinstance(data, dict) else type(data)))
        return
    check("readback has points", len(points) == 3, "%d point(s)" % len(points))

    first = points[0]
    position = first.get("position") if isinstance(first, dict) else None
    expected = hit.get("position")
    if isinstance(position, list) and len(position) == 3 and isinstance(expected, list):
        delta = max(abs(position[i] - expected[i]) for i in range(3))
        # The curve has an identity transform here, so local == world. A large
        # delta would mean the local/world conversion is dropping the transform.
        check("first point matches the hit", delta < 1e-3, "max delta %.6f" % delta)
    else:
        check("first point matches the hit", False, "position=%r" % (position,))


def main():
    pipe = connect()
    try:
        hit = section_raycast(pipe)
        section_filter_refused(pipe)
        name = section_build_from_zero(pipe, hit)
        section_readback(pipe, name, hit)
    finally:
        try:
            call(pipe, "scene.delete", {"name": CURVE_NAME})
        except Exception:
            pass
        _kernel32.CloseHandle(pipe)

    if _failures:
        print("\nFAIL: %d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        sys.exit(1)
    print("\nPASS: a curve can be drawn on a surface from script, and the "
          "raycast says what it hit.")
    sys.exit(0)


if __name__ == "__main__":
    main()
