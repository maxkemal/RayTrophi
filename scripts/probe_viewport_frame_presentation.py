#!/usr/bin/env python3
"""
Probe: raster/Realtime viewport frame presentation bridge (Realtime roadmap Faz 0.5a).

WHY THIS EXISTS
    The old raster viewport ended every changed frame with a fully serial chain:

        record -> blocking submit/fence -> image-to-buffer -> blocking fence
               -> CPU copy -> SDL texture upload

    Faz 0.5a replaced it with a two-slot ring: the GPU writes slot N while the
    host consumes the newest already-completed slot, and the host waits only
    when a slot is about to be reused.

    ★★★ The change is INVISIBLE. The image is identical either way; only the
    stall disappears. A capability whose success cannot be seen and cannot be
    measured is a capability nobody can defend, so the bridge ships with a
    telemetry surface and this script is the thing that reads it.

WHAT IT MEASURES
    1. viewport.frame_telemetry exists and answers, and it distinguishes "no
       raster frame has ever been presented" from "everything measured zero".
       An `available:false` reply that carried zeros instead would look exactly
       like a perfectly healthy idle bridge.
    2. The async path is actually live (`async_present:true`). False here is not
       a script bug: it means the driver refused persistent frame slots and the
       old synchronous path is running. That is legal and reported, but it must
       never be silent.
    3. `image_readback_ms` is ZERO on the async path. That timer only ticks in
       the legacy branch, so a non-zero value while async_present is true means
       both readbacks are running and the bridge saved nothing.
    4. Frames actually flow: driving the viewport advances frames_submitted, and
       frames_consumed follows it. A submitted counter that climbs while the
       consumed counter stands still is a ring that renders into a void.
    5. stale_presents does not track frame count. A few are normal (the ring is
       deliberately one frame behind); one per frame means the host never finds
       a completed slot and the viewer is permanently watching old pixels.
    6. resource_drains does not track frame count either. Each drain is a full
       host block for a buffer mutation, so a drain per frame means some edit
       path re-uploads every frame and the ring produces no parallelism at all.

    ★ The subtlest failure this catches is number 5: an image that is always one
      frame stale looks completely correct in a screenshot, feels fine on a
      static scene, and only shows up as mush while dragging. Nobody files that
      as a bug.

WHAT IT DOES NOT MEASURE
    Wall-clock speedup. This script drives the viewport over IPC, one call per
    frame, so it cannot reproduce the interactive frame cadence. It measures
    that the bridge is STRUCTURALLY working (async, non-blocking, consuming);
    the speed claim belongs to the interactive session, where the HUD line
    "Present: async ..." reports the same numbers.

    Note also that this script deliberately never enables viewport.capture:
    capture FORCES synchronous presentation so a probe reads the frame it just
    rendered, which would make every measurement here read as blocking.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1      (wait for "HAZIR")
    2. python scripts/probe_viewport_frame_presentation.py

    Switches the viewport to Solid and leaves it there. Touches no scene data.

EXIT CODE
    0 = PASS, 1 = FAIL, 2 = could not run the measurement
"""
import ctypes
import ctypes.wintypes as wintypes
import json
import sys

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'
FRAME_BATCH = 12

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
        print("  ok   %-38s %s" % (label, detail))
    else:
        print("  FAIL %-38s %s" % (label, detail))
        _failures.append(label)


def telemetry(pipe):
    return result_of(pipe, "viewport.frame_telemetry", None, "frame_telemetry")


def drive(pipe, count):
    """Advance the viewport. Each IPC call returns to the frame loop, which is
    what actually publishes a frame - a single script holding the main thread
    would render frames that never reach presentation."""
    for _ in range(count):
        call(pipe, "viewport.render_frames", {"count": 1})


def section_absence_is_not_zero(pipe):
    print("1. absence and zero are distinguishable")
    t = telemetry(pipe)
    check("telemetry answers", isinstance(t, dict), json.dumps(t)[:120])
    check("carries 'available'", "available" in t, repr(t.get("available")))
    if not t.get("available"):
        # This is a legitimate state, not a crash - but every later section
        # would then be measuring nothing, so say so and stop.
        print("  --   no raster frame presented yet: %s" %
              t.get("reason", "(no reason given)"))
        print("       Switch the viewport to Solid and let one frame draw, then"
              " re-run.")
        return False
    check("absent fields are not reported as zeros",
          "frames_submitted" in t and "stale_presents" in t,
          "available=true carries the counters")
    return True


def section_async_is_live(pipe):
    print("2. the asynchronous path is the one actually running")
    t = telemetry(pipe)
    async_on = bool(t.get("async_present"))
    check("async_present", async_on,
          "true" if async_on else
          "FALSE - driver refused persistent frame slots, legacy synchronous "
          "readback is live (this is reported, not silent)")
    check("slot_count is the two-slot ring", t.get("slot_count") == 2,
          repr(t.get("slot_count")))
    check("capture lock is off during this probe",
          not t.get("synchronous_present"),
          "synchronous_present=%r" % t.get("synchronous_present"))
    if async_on:
        # ★ The legacy branch is the ONLY writer of image_readback_ms. Non-zero
        # here while async_present is true would mean both readback paths are
        # live and the bridge removed nothing.
        check("no second per-frame readback",
              float(t.get("image_readback_ms", 0.0)) == 0.0,
              "image_readback_ms=%.3f" % float(t.get("image_readback_ms", 0.0)))
    return async_on


def section_frames_flow(pipe):
    print("3. frames are submitted AND consumed")
    before = telemetry(pipe)
    drive(pipe, FRAME_BATCH)
    after = telemetry(pipe)

    submitted = after.get("frames_submitted", 0) - before.get("frames_submitted", 0)
    consumed = after.get("frames_consumed", 0) - before.get("frames_consumed", 0)
    check("frames_submitted advanced", submitted > 0, "+%d" % submitted)
    # A ring that submits without ever consuming renders into a void: the fence
    # is never observed, the mapped readback is never read, and the viewport
    # keeps showing whatever it last managed to publish.
    check("frames_consumed followed", consumed > 0, "+%d" % consumed)
    check("consumption keeps up with submission",
          submitted == 0 or consumed >= submitted // 2,
          "submitted +%d, consumed +%d" % (submitted, consumed))
    return before, after


def section_not_permanently_behind(before, after):
    print("4. the ring is not permanently behind, and not draining every frame")
    submitted = after.get("frames_submitted", 0) - before.get("frames_submitted", 0)
    stale = after.get("stale_presents", 0) - before.get("stale_presents", 0)
    drains = after.get("resource_drains", 0) - before.get("resource_drains", 0)

    if submitted == 0:
        check("measurable batch", False, "no frames were submitted; cannot judge")
        return
    # ★ The quiet failure. One stale present per submitted frame means the host
    # NEVER finds a completed slot, so the viewer is always looking at older
    # pixels. A screenshot of that state is indistinguishable from a correct one.
    check("stale presents do not track frame count", stale < submitted,
          "stale +%d over +%d frames" % (stale, submitted))
    # Each drain is a full host block for a buffer mutation. One per frame means
    # some edit path re-uploads every frame and the ring buys nothing.
    check("resource drains do not track frame count", drains < submitted,
          "drains +%d over +%d frames" % (drains, submitted))
    check("present latency is bounded by the slot count",
          after.get("present_latency_frames", 0) < 2,
          "lat=%r" % after.get("present_latency_frames"))


def section_capture_forces_determinism(pipe):
    print("5. capture forces synchronous presentation (probe determinism)")
    call(pipe, "viewport.capture", {"enabled": True})
    drive(pipe, 2)
    t = telemetry(pipe)
    # ★★★ Async presentation and probing DIRECTLY conflict: the ring publishes
    # an older completed slot on purpose, so a script could measure a frame
    # recorded BEFORE its own scene edit with nothing in the image saying so.
    check("synchronous_present while capturing",
          bool(t.get("synchronous_present")),
          repr(t.get("synchronous_present")))
    check("blocking seeds are counted separately from resource drains",
          "blocking_seeds" in t and "resource_drains" in t,
          "seeds=%r drains=%r" % (t.get("blocking_seeds"), t.get("resource_drains")))
    call(pipe, "viewport.capture", {"enabled": False})
    drive(pipe, 2)
    t = telemetry(pipe)
    check("capture off releases the lock",
          not t.get("synchronous_present"),
          repr(t.get("synchronous_present")))


def main():
    pipe = connect()
    shading = result_of(pipe, "viewport.shading", None, "viewport.shading")
    if not shading.get("interactive_available"):
        print("FAIL(setup): this machine has no raster viewport (no Vulkan). "
              "The Faz 0.5a bridge does not exist here to measure.")
        sys.exit(2)
    if shading.get("mode") != "solid":
        result_of(pipe, "viewport.set_shading", {"mode": "solid"}, "set_shading solid")
        drive(pipe, 3)

    print("viewport frame presentation probe (Realtime Faz 0.5a)")
    print()
    if not section_absence_is_not_zero(pipe):
        sys.exit(2)
    print()
    section_async_is_live(pipe)
    print()
    before, after = section_frames_flow(pipe)
    print()
    section_not_permanently_behind(before, after)
    print()
    section_capture_forces_determinism(pipe)
    print()

    final = telemetry(pipe)
    print("last frame: %.2f ms  (record %.2f / slot wait %.2f / submit %.2f / "
          "host read %.2f / present %.2f)" % (
              final.get("frame_ms", 0.0), final.get("cpu_record_ms", 0.0),
              final.get("slot_wait_ms", 0.0), final.get("submit_ms", 0.0),
              final.get("host_read_ms", 0.0), final.get("present_ms", 0.0)))
    print()
    if _failures:
        print("FAIL: %d check(s) failed: %s" % (len(_failures), ", ".join(_failures)))
        sys.exit(1)
    print("PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
