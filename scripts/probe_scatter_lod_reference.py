#!/usr/bin/env python3
"""
Probe: scatter LOD split against a full-detail reference.

WHY THIS EXISTS
    The raster viewport replaces distant scatter instances with a 96-triangle
    proxy impostor and converges a distance threshold toward a triangle target.
    viewport.frame_telemetry reports full_triangles / proxy_triangles /
    full_instances / proxy_instances for that split.

    ★★★ Those numbers had NO DENOMINATOR until 2026-09-01. Reading
    "full_instances = 4000" tells you nothing on its own: it is a saving only if
    you know what the scene contains, and a script had no way to turn the
    substitution off to find out. Measuring a ratio while unable to observe its
    reference is not measuring.

    viewport.set_quality('full') disables the split (frustum culling stays on),
    so this script can establish the reference and compare against it.

WHAT IT MEASURES
    1. viewport.quality answers and reports scatter_lod_split as a VALUE. A
       caller that instead matched on the preset NAME would break the day a
       preset is added, and would be silently wrong meanwhile.
    2. 'full' really disables the split: proxy_instances falls to 0 and
       full_instances rises to total_instances. If proxy_instances stays > 0
       here, the preset reached render_settings but never reached the mesh
       bindings - the call succeeded and nothing changed.
    3. The reference is non-trivial. If the full-detail scene has no scatter
       geometry at all, every later comparison is 0 vs 0 and would PASS while
       measuring nothing. ★★ The physics rig fell into exactly this trap once
       (0 == 0 reported green), so this is checked, not assumed.
    4. Switching back to a LOD preset actually reduces submitted triangles
       below the reference. This is the claim the whole GPU culling module
       exists to make, and it is the one nobody can see on screen.
    5. Nothing DISAPPEARS. full_instances + proxy_instances must still account
       for the visible set: a split that drops instances instead of demoting
       them looks like a performance win and is a rendering bug.
    6. The threshold converges rather than oscillating. ★★★ In a UNIFORM dense
       cluster (every instance at roughly the same distance) a bare distance
       threshold cannot discriminate, and full_instances flips between total
       and 0 every frame. The stochastic transition band added 2026-09-01
       exists to fix that; this section is what tells you whether it worked.

WHAT IT DOES NOT MEASURE
    Image quality. Whether the proxy LOOKS right is a visual question - use
    viewport.get_screenshot with a vision model, or render.probe for a numeric
    question about specific pixels. This script only measures the split.

    Also: it does not measure frame time. It drives the viewport one IPC call
    per frame, which is not the interactive cadence (see the sibling probe
    probe_viewport_frame_presentation.py for the same caveat).

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. Open a scene that HAS scatter instances (foliage, rocks). An empty scene
       makes section 3 fail on purpose.
    3. python scripts/probe_scatter_lod_reference.py

    Leaves the viewport in Solid mode with the quality preset it found on entry.

EXIT CODE
    0 = PASS, 1 = FAIL, 2 = could not run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
FRAME_BATCH = 8
# Threshold convergence is measured across this many separate observations.
OSCILLATION_SAMPLES = 6

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
        print("  ok   %-42s %s" % (label, detail))
    else:
        print("  FAIL %-42s %s" % (label, detail))
        _failures.append(label)


def telemetry(pipe):
    return result_of(pipe, "viewport.frame_telemetry", None, "frame_telemetry")


def drive(pipe, count):
    """Advance the viewport. Each call returns to the frame loop, which is what
    actually publishes a frame and services the GPU culling readback."""
    for _ in range(count):
        call(pipe, "viewport.render_frames", {"count": 1})


def settle(pipe, preset):
    """Apply a preset and let the threshold converge.

    ★ set_quality rebuilds the raster scene, and the culling counters are read
    back ONE FRAME BEHIND. Reading telemetry immediately after switching
    reports the previous preset's split and looks like the switch did nothing.
    """
    result_of(pipe, "viewport.set_quality", {"preset": preset}, "set_quality " + preset)
    drive(pipe, FRAME_BATCH)
    return telemetry(pipe)


def section_quality_surface(pipe):
    print("1. viewport.quality reports behaviour as a value")
    q = result_of(pipe, "viewport.quality", None, "viewport.quality")
    check("answers", isinstance(q, dict), json.dumps(q)[:120])
    check("carries scatter_lod_split", "scatter_lod_split" in q,
          repr(q.get("scatter_lod_split")))
    available = bool(q.get("raster_viewport_available"))
    check("raster viewport exists", available,
          "true" if available else
          "FALSE - the preset is stored but nothing on this machine reads it")
    return q if available else None


def section_full_disables_split(pipe):
    print("2. 'full' actually disables the split")
    t = settle(pipe, "full")
    if not t.get("available"):
        print("  --   no raster frame presented; switch to Solid and re-run.")
        return None
    q = result_of(pipe, "viewport.quality", None, "viewport.quality")
    check("preset reported back", q.get("preset") == "full", repr(q.get("preset")))
    check("scatter_lod_split false", q.get("scatter_lod_split") is False,
          repr(q.get("scatter_lod_split")))
    proxy_inst = t.get("proxy_instances", 0)
    # ★ The preset reaching render_settings is NOT the same as it reaching the
    # mesh bindings. This is the check that separates the two.
    check("no instance demoted to proxy", proxy_inst == 0,
          "proxy_instances=%d%s" % (proxy_inst,
                                    "" if proxy_inst == 0 else
                                    "  <- preset never reached the cull bindings"))
    return t


def section_reference_is_real(reference):
    print("3. the full-detail reference is not empty")
    total = reference.get("total_instances", 0)
    full_tris = reference.get("full_triangles", 0)
    # 0 vs 0 compares equal and passes every later section while measuring
    # nothing. Refuse to continue instead.
    check("scene carries instances", total > 0, "total_instances=%d" % total)
    check("reference has triangles", full_tris > 0,
          "full_triangles=%d" % full_tris)
    if total == 0 or full_tris == 0:
        print("       Open a scene WITH scatter instances (foliage/rocks); an")
        print("       empty reference makes every comparison below 0 vs 0.")
        return False
    return True


def section_lod_reduces(pipe, reference):
    print("4. a LOD preset submits less than the reference")
    ref_tris = reference.get("full_triangles", 0) + reference.get("proxy_triangles", 0)
    t = settle(pipe, "balanced")
    lod_tris = t.get("full_triangles", 0) + t.get("proxy_triangles", 0)
    check("visible triangles fall below reference", lod_tris < ref_tris,
          "%d -> %d (%.1f%% of full)" %
          (ref_tris, lod_tris, 100.0 * lod_tris / max(1, ref_tris)))
    demoted = t.get("proxy_instances", 0)
    check("instances were demoted, not merely fewer", demoted > 0,
          "proxy_instances=%d" % demoted)
    return t


def section_nothing_vanishes(pipe, reference, lod):
    print("5. demoted instances are accounted for, not dropped")
    ref_total = reference.get("total_instances", 0)
    accounted = lod.get("full_instances", 0) + lod.get("proxy_instances", 0)
    # Frustum culling legitimately removes instances outside the view, so the
    # accounted set is a SUBSET of total - but it must not exceed it, and with
    # the camera unmoved between the two reads it must not collapse either.
    check("accounted <= total", accounted <= ref_total,
          "%d <= %d" % (accounted, ref_total))
    ref_accounted = (reference.get("full_instances", 0) +
                     reference.get("proxy_instances", 0))
    # ★ The camera did not move between the reference and the LOD read, so the
    # VISIBLE set is the same set. A split that loses instances shows up here
    # and nowhere else: on screen it just looks like thinner foliage.
    check("visible set preserved across the split",
          accounted == ref_accounted,
          "reference=%d lod=%d%s" % (ref_accounted, accounted,
                                     "" if accounted == ref_accounted else
                                     "  <- instances lost by the split"))


def section_threshold_converges(pipe):
    print("6. the threshold converges instead of oscillating")
    samples = []
    for _ in range(OSCILLATION_SAMPLES):
        drive(pipe, 2)
        t = telemetry(pipe)
        samples.append((t.get("full_instances", 0), t.get("proxy_instances", 0)))
    fulls = [s[0] for s in samples]
    spread = max(fulls) - min(fulls)
    mean = sum(fulls) / float(len(fulls))
    # ★★★ The signature of a uniform dense cluster under a bare distance
    # threshold is full_instances swinging between the total and 0. A settled
    # threshold wanders by a few percent as the counters lag one frame; it does
    # not swing by most of its own magnitude.
    ok = mean <= 0.0 or spread <= 0.35 * mean
    check("full_instances is settled", ok,
          "samples=%s spread=%d mean=%.0f%s" %
          (fulls, spread, mean,
           "" if ok else "  <- oscillating: uniform cluster, band too narrow"))


def main():
    pipe = connect()
    print("Scatter LOD reference probe")
    print("=" * 68)

    entry = section_quality_surface(pipe)
    if entry is None:
        print("\nCould not measure: no raster viewport on this machine.")
        return 2
    entry_preset = entry.get("preset", "auto")

    # Solid is the cheapest mode that still exercises the scatter split, and it
    # does not depend on the material preview pipeline being supported.
    result_of(pipe, "viewport.set_shading", {"mode": "solid"}, "set_shading solid")
    drive(pipe, FRAME_BATCH)

    try:
        reference = section_full_disables_split(pipe)
        if reference is None:
            return 2
        if not section_reference_is_real(reference):
            return 1
        lod = section_lod_reduces(pipe, reference)
        section_nothing_vanishes(pipe, reference, lod)
        section_threshold_converges(pipe)
    finally:
        # Leave the preset as it was found. A probe that silently parks the
        # viewport in 'full' would make the next person's session mysteriously
        # slow and they would not connect it to running this.
        call(pipe, "viewport.set_quality", {"preset": entry_preset})

    print("=" * 68)
    if _failures:
        print("FAIL - %d check(s): %s" % (len(_failures), ", ".join(_failures)))
        return 1
    print("PASS - the split has a reference, reduces submission, loses nothing,")
    print("       and its threshold is settled.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
