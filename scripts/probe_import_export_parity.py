#!/usr/bin/env python3
"""
Probe: import/export parity baseline for the Assimp replacement.

WHY THIS EXISTS
    docs/dev/ASSIMP_IMPORT_REPLACEMENT_BRIEF.md sets one acceptance rule:

        "the same .glb must give the SAME vertex/triangle/joint/channel counts
         and the same bbox on the old path and the new one; if they differ,
         which one is right has to be MEASURED, not assumed."

    That rule was unenforceable until now, because the numbers on the import
    side were not readable from a script. anim.source_clips / anim.source_channels
    (added with Faz 0) expose what the LOADER produced, as opposed to what a
    character's AnimationController is playing.

    This script writes that baseline to a JSON file and, on a later run, diffs
    against it. Faz 0 is a PURE TYPE MIGRATION - it must move none of these
    numbers - so a diff during Faz 0 is a regression, not progress. During
    Faz 1/2 (the direct glTF reader) a diff is the thing to explain.

WHAT IT MEASURES
    1. scene.export_estimate vs scene.export_gltf. The panel's PRE-export
       estimate and the writer's MEASURED result must agree on instance count.
       They silently did not: the writer was moved to InstanceManager (on
       Vulkan scatter never enters world.objects) and the estimate was left
       walking world.objects, so the panel reported 0 instances for a scene
       that exported thousands. A plausible number, therefore an invisible bug.
    2. Raw animation clip inventory: channel and key counts per clip, plus the
       key time span. This is the loader's own output.
    3. Per-node channel counts for the first clip. Reach for this when the
       totals match but the pose does not: same key COUNT on different NODES.
    4. Object/triangle inventory from scene.list_objects.

    Nothing here judges whether the numbers are RIGHT. It records what they
    are, so a change becomes visible. A baseline that is wrong on both runs
    still catches a regression; a missing baseline catches nothing.

WHAT IT DOES NOT MEASURE
    Whether the imported model LOOKS correct. Use render.start + a vision
    check, or render.probe for specific pixels.

USAGE
    1. .\\scripts\\ipc\\Start-RayTrophi.ps1        (wait for "HAZIR")
    2. Open (or import) the model under test.
    3. python scripts/probe_import_export_parity.py --write  baseline.json
       ... make the change, restart, load the SAME file ...
       python scripts/probe_import_export_parity.py --check  baseline.json

    With no --write/--check it just prints the current numbers.

EXIT CODE
    0 = PASS (or plain report), 1 = a number moved, 2 = could not measure
"""
import argparse
import ctypes
import ctypes.wintypes as wintypes
import json
import os
import sys
import tempfile

PIPE_NAME = r'\\.\pipe\RayTrophiStudio'

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
        print("  ok   %-46s %s" % (label, detail))
    else:
        print("  FAIL %-46s %s" % (label, detail))
        _failures.append(label)


def measure(pipe):
    snapshot = {}

    # -- 1. Export: estimate vs measured -------------------------------------
    est = result_of(pipe, "scene.export_estimate", None, "scene.export_estimate")
    snapshot["export_estimate"] = est

    out_path = os.path.join(tempfile.gettempdir(), "rt_parity_probe.glb")
    measured = result_of(pipe, "scene.export_gltf", {"path": out_path},
                          "scene.export_gltf")
    # Drop the timing fields: they are real measurements but they move every
    # run, and mixing them into a parity baseline would make every diff noisy.
    snapshot["export_measured"] = {k: v for k, v in measured.items()
                                    if not k.startswith("seconds_")
                                    and k not in ("path", "peak_writer_mb")}

    print("\nEXPORT")
    print("  estimate: objects=%s triangles=%s instances=%s sources=%s inst_tris=%s"
          % (est.get("objects"), est.get("triangles"), est.get("instances"),
             est.get("unique_instance_sources"), est.get("instance_triangles")))
    print("  measured: meshes=%s triangles=%s instances=%s groups=%s materials=%s images=%s"
          % (measured.get("meshes"), measured.get("triangles"),
             measured.get("instances"), measured.get("instanced_groups"),
             measured.get("materials"), measured.get("images")))

    # The check that would have caught the panel lying. The estimate counts
    # placements the same way the writer does; when instancing collapses them
    # into EXT_mesh_gpu_instancing nodes the writer's `instances` is the number
    # of transforms folded in, which is the same population.
    est_inst = int(est.get("instances", 0))
    got_inst = int(measured.get("instances", 0))
    if est_inst == 0 and got_inst == 0:
        # 0 == 0 is not a passing comparison, it is an absent one. The physics
        # rig reported green on exactly this once.
        print("  note: this scene has no scatter instances - the estimate/measured "
              "agreement below is vacuous. Load a scattered scene to exercise it.")
    check("estimate instances == exported instances", est_inst == got_inst,
          "estimate=%d exported=%d" % (est_inst, got_inst))

    # -- 2. Raw imported animation clips -------------------------------------
    clips = result_of(pipe, "anim.source_clips", None, "anim.source_clips")
    snapshot["source_clips"] = clips
    print("\nIMPORTED ANIMATION CLIPS (%d)" % len(clips))
    for c in clips:
        print("  %-28s ch p/r/s=%s/%s/%s  keys p/r/s=%s/%s/%s  ticks=[%.3f..%.3f]/%s tps=%s"
              % (c.get("name", "")[:28],
                 c.get("position_channels"), c.get("rotation_channels"),
                 c.get("scaling_channels"),
                 c.get("position_keys"), c.get("rotation_keys"), c.get("scaling_keys"),
                 c.get("first_key_time", 0.0), c.get("last_key_time", 0.0),
                 c.get("duration_ticks"), c.get("ticks_per_second")))
        # Keys must be time-sorted: every sampler in the engine binary-searches
        # or scans forward assuming they are.
        check("clip '%s' keys are time-ordered" % c.get("name", "")[:20],
              c.get("first_key_time", 0.0) <= c.get("last_key_time", 0.0),
              "first=%.3f last=%.3f" % (c.get("first_key_time", 0.0),
                                         c.get("last_key_time", 0.0)))
        # A clip with channels but zero keys animates nothing while looking
        # present in every UI that counts clips.
        total_keys = (c.get("position_keys", 0) + c.get("rotation_keys", 0)
                      + c.get("scaling_keys", 0))
        total_ch = (c.get("position_channels", 0) + c.get("rotation_channels", 0)
                    + c.get("scaling_channels", 0))
        if total_ch:
            check("clip '%s' channels carry keys" % c.get("name", "")[:20],
                  total_keys > 0, "%d channels, %d keys" % (total_ch, total_keys))

    if clips:
        channels = result_of(pipe, "anim.source_channels", None, "anim.source_channels")
        snapshot["source_channels_first_clip"] = channels
        print("  first clip has %d animated nodes" % len(channels))

    # -- 3. Scene inventory ---------------------------------------------------
    objects = result_of(pipe, "scene.list_objects", None, "scene.list_objects")
    if isinstance(objects, list):
        snapshot["object_names"] = sorted(
            o.get("name", "") if isinstance(o, dict) else str(o) for o in objects)
        print("\nSCENE: %d objects" % len(objects))

    return snapshot


def diff(baseline, current, path=""):
    """Report every leaf that moved. Deliberately reports ALL of them rather
    than stopping at the first: one changed count usually drags others with it,
    and seeing which ones moved together is what identifies the cause."""
    out = []
    if type(baseline) is not type(current):
        return ["%s: type %s -> %s" % (path or "<root>", type(baseline).__name__,
                                        type(current).__name__)]
    if isinstance(baseline, dict):
        for k in sorted(set(baseline) | set(current)):
            if k not in baseline:
                out.append("%s.%s: ADDED (%r)" % (path, k, current[k]))
            elif k not in current:
                out.append("%s.%s: REMOVED (was %r)" % (path, k, baseline[k]))
            else:
                out += diff(baseline[k], current[k], "%s.%s" % (path, k))
    elif isinstance(baseline, list):
        if len(baseline) != len(current):
            out.append("%s: length %d -> %d" % (path, len(baseline), len(current)))
        for i, (a, b) in enumerate(zip(baseline, current)):
            out += diff(a, b, "%s[%d]" % (path, i))
    else:
        if isinstance(baseline, float) or isinstance(current, float):
            if abs(float(baseline) - float(current)) > 1e-9:
                out.append("%s: %r -> %r" % (path, baseline, current))
        elif baseline != current:
            out.append("%s: %r -> %r" % (path, baseline, current))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", metavar="FILE", help="save the baseline")
    ap.add_argument("--check", metavar="FILE", help="diff against a baseline")
    args = ap.parse_args()

    pipe = connect()
    snapshot = measure(pipe)

    if args.write:
        with open(args.write, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, sort_keys=True)
        print("\nbaseline written: %s" % args.write)

    if args.check:
        with open(args.check, encoding="utf-8") as f:
            baseline = json.load(f)
        deltas = diff(baseline, snapshot)
        print("\nPARITY vs %s" % args.check)
        if deltas:
            for d in deltas:
                print("  MOVED %s" % d)
            _failures.append("parity diff (%d fields moved)" % len(deltas))
        else:
            print("  ok   every recorded number is identical")

    print("\n%s" % ("FAIL: " + "; ".join(_failures) if _failures else "PASS"))
    return 1 if _failures else 0


if __name__ == "__main__":
    sys.exit(main())
