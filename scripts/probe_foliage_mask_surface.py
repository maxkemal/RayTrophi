#!/usr/bin/env python3
"""
Probe: the foliage placement-mask surface is reachable and honest from script.

WHAT IT MEASURES
    1. scatter.set_settings writes the exclusion mask, its threshold and the
       splat channels, and scatter.list_groups reads back exactly what was
       written. A setter with no readback cannot be tested: it returns success
       whether or not the value landed.
    2. An out-of-range splat channel is REFUSED, not clamped. Clamping a typo to
       channel 3 masks against alpha and looks like a working setting.
    3. terrain.list_fields reports what a terrain actually publishes, and a
       Publish Field node's name appears there after evaluation - which is the
       whole point of the node: a composed mask you can NAME and then select.

WHY IT IS NEEDED
    Before this, exclusion_mask was panel-only: no IPC method wrote it and
    nothing reported it. A mask pointed at a name the terrain does not publish
    resolves to the shader's neutral fallback, so a broken mask and a permissive
    mask produce the same picture. Nothing fails, nothing is logged.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/probe_foliage_mask_surface.py

    Section 3 needs a terrain in the scene; it is skipped (and said so) if there
    is none. Sections 1-2 need no terrain.

EXIT CODE
    0 = PASS, 1 = FAIL, 2 = could not run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
GROUP_NAME = "__probe_foliage_mask_surface"

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
        if ok:
            break
        if _kernel32.GetLastError() != 234:  # ERROR_MORE_DATA
            raise OSError("ReadFile failed (%d)" % _kernel32.GetLastError())
    return json.loads(b"".join(chunks).decode("utf-8"))


def result_of(pipe, method, params, what):
    response = call(pipe, method, params)
    if not isinstance(response, dict) or "result" not in response:
        print("FAIL(setup): %s -> %s" % (what, response))
        sys.exit(2)
    return response["result"]


def check(label, got, want):
    if got == want:
        print("  ok   %-24s %r" % (label, got))
    else:
        print("  FAIL %-24s got %r, wrote %r" % (label, got, want))
        _failures.append(label)


def group_row(pipe, name):
    for row in result_of(pipe, "scatter.list_groups", None, "scatter.list_groups"):
        if row.get("name") == name:
            return row
    print("FAIL(setup): scatter group %r vanished from list_groups" % name)
    sys.exit(2)


def section_roundtrip(pipe):
    print("1. set_settings -> list_groups round trip")
    call(pipe, "scatter.delete_group", {"group": GROUP_NAME})
    result_of(pipe, "scatter.create_group", {"name": GROUP_NAME}, "create_group")

    written = {
        "density_mask": "biome.forest",
        "exclusion_mask": "hydrology.lake_mask",
        "exclusion_threshold": 0.25,
        "scale_mask": "terrain.wetness",
        "scale_mask_influence": 0.75,
        "splat_include_channel": 1,
        "splat_exclude_channel": 3,
    }
    params = {"group": GROUP_NAME}
    params.update(written)
    result_of(pipe, "scatter.set_settings", params, "set_settings")

    row = group_row(pipe, GROUP_NAME)
    for key, want in written.items():
        got = row.get(key)
        if isinstance(want, float) and isinstance(got, (int, float)):
            got = round(float(got), 4)
            want = round(float(want), 4)
        check(key, got, want)

    # A partial patch must not reset the keys it does not mention. Writing a
    # default for an unmentioned key is the failure mode that makes every
    # single-variable test silently change two things.
    result_of(pipe, "scatter.set_settings",
              {"group": GROUP_NAME, "exclusion_threshold": 0.9}, "set_settings(partial)")
    row = group_row(pipe, GROUP_NAME)
    check("partial: threshold", round(float(row.get("exclusion_threshold", -1)), 4), 0.9)
    check("partial: mask kept", row.get("exclusion_mask"), "hydrology.lake_mask")
    check("partial: density kept", row.get("density_mask"), "biome.forest")


def section_channel_refused(pipe):
    print("\n2. out-of-range splat channel is refused, not clamped")
    response = call(pipe, "scatter.set_settings",
                    {"group": GROUP_NAME, "splat_exclude_channel": 7})
    refused = isinstance(response, dict) and "error" in response
    if not refused and isinstance(response, dict) and isinstance(response.get("result"), dict):
        refused = "__error" in response["result"]
    if refused:
        print("  ok   channel 7 refused")
    else:
        print("  FAIL channel 7 accepted -> %r" % (response,))
        _failures.append("splat channel range")
    row = group_row(pipe, GROUP_NAME)
    check("channel unchanged", row.get("splat_exclude_channel"), 3)


def section_published_fields(pipe):
    print("\n3. terrain.list_fields reports what is actually published")
    terrains = result_of(pipe, "terrain.list", None, "terrain.list")
    if not terrains:
        print("  SKIP no terrain in the scene - create one to measure this section")
        return
    name = terrains[0].get("name")
    fields = result_of(pipe, "terrain.list_fields", {"terrain": name}, "terrain.list_fields")
    if not isinstance(fields, list):
        print("  FAIL terrain.list_fields did not return a list: %r" % (fields,))
        _failures.append("terrain.list_fields")
        return
    print("  terrain %r publishes %d field(s)" % (name, len(fields)))
    for field in fields:
        print("    %s" % field)
    authored = [f for f in fields if f.startswith("mask.")]
    if authored:
        print("  ok   authored mask(s) visible: %s" % ", ".join(authored))
    else:
        print("  note no 'mask.*' field yet - add a Publish Field node, name it, "
              "wire a field in and re-evaluate the graph, then re-run this probe")


def main():
    pipe = connect()
    try:
        section_roundtrip(pipe)
        section_channel_refused(pipe)
        section_published_fields(pipe)
    finally:
        try:
            call(pipe, "scatter.delete_group", {"group": GROUP_NAME})
        except Exception:
            pass
        _kernel32.CloseHandle(pipe)

    if _failures:
        print("\nFAIL: %d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        sys.exit(1)
    print("\nPASS: the mask surface writes, reads back and rejects bad input.")
    sys.exit(0)


if __name__ == "__main__":
    main()
