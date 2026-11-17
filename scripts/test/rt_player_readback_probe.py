"""Live readback diagnosis without project reload/save, quality or shader changes.

Temporarily drives Player continuous rendering, compares async, capture-forced
sync and pacing. Capture stays in RAM; no screenshot is requested.
Restores the Player display/capture flags; camera is read-only in both applications.
"""
import argparse
import json
from pathlib import Path
import time

from rt_ipc import RtIpc, PLAYER_PIPE_NAME, PIPE_NAME, RtIpcError


COUNTERS = ("frames_submitted", "frames_consumed", "stale_presents",
            "slot_waits", "resource_drains", "blocking_seeds")


def sample(ipc):
    return {"telemetry": ipc.call("viewport.frame_telemetry"),
            "display": ipc.call("player.display"),
            "timings": ipc.call("viewport.frame_timings"),
            "camera": ipc.call("camera.get")}


def run(args):
    report = {"phases": [], "scope": "delivery ratios; sync/capture is a diagnostic, not an FPS benchmark"}
    ipc = RtIpc(PLAYER_PIPE_NAME)
    original = {"camera": ipc.call("camera.get"), "display": ipc.call("player.display"),
                "capture": ipc.call("viewport.status")["capture_enabled"],
                "project": ipc.call("project.path")}
    report["original"] = original
    try:
        studio = RtIpc(PIPE_NAME)
        try:
            report["studio"] = {m: studio.call(m) for m in
                                ("project.path", "camera.get", "viewport.frame_telemetry",
                                 "viewport.frame_timings")}
        finally:
            studio.close()
    except RtIpcError as error:
        report["studio_read_error"] = str(error)
    try:
        for name, synchronous, pacing in (
            ("near_async", False, 0),
            ("near_capture_sync", True, 0),
            ("near_async_10hz", False, 10),
        ):
            ipc.call("viewport.capture", enabled=synchronous)
            ipc.call("player.set_display", continuous=True, pacing_hz=pacing)
            time.sleep(1)
            ipc.call("viewport.reset_frame_timings")
            before = sample(ipc)
            time.sleep(args.seconds)
            after = sample(ipc)
            delta = {key: after["telemetry"][key] - before["telemetry"][key]
                     for key in COUNTERS}
            phase = {"name": name, "before": before, "after": after, "delta": delta}
            report["phases"].append(phase)
            print(json.dumps({"phase": name, "delta": delta,
                              "gpu_ms": after["timings"]["frame_gpu_mean_ms"],
                              "rendered_fps": after["display"]["frame"].get("rendered_fps"),
                              "visible_triangles": after["telemetry"]["visible_triangles"]}), flush=True)
    finally:
        errors = []
        for method, params in (
            ("viewport.capture", {"enabled": original["capture"]}),
            ("player.set_display", {"continuous": original["display"]["continuous"],
                                    "pacing_hz": float(original["display"]["pacing_hz"])}),
        ):
            try:
                ipc.call(method, **params)
            except Exception as error:
                errors.append(str(error))
        report["restoration_errors"] = errors
        report["restored_camera"] = ipc.call("camera.get")
        report["restored_display"] = ipc.call("player.display")
        report["restored_capture"] = ipc.call("viewport.status")["capture_enabled"]
        report["camera_unchanged"] = report["restored_camera"] == original["camera"]
        report["async_delivery_ok"] = all(
            phase["delta"]["frames_submitted"] >= 8 and
            phase["delta"]["frames_consumed"] > 0 and
            phase["delta"]["stale_presents"] < phase["delta"]["frames_submitted"] and
            not phase["after"]["telemetry"]["device_lost"]
            for phase in report["phases"] if phase["name"] == "near_async"
        ) and any(phase["name"] == "near_async" for phase in report["phases"])
        ipc.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    assert not report["restoration_errors"], report["restoration_errors"]
    report["camera_unchanged"] = report["restored_camera"] == original["camera"]
    assert report["restored_capture"] == original["capture"]
    if args.assert_delivery:
        assert report["async_delivery_ok"], "GPU submissions starved completed-frame delivery"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=3)
    parser.add_argument("--assert-delivery", action="store_true",
                        help="Post-build regression: async submissions must deliver fresh images")
    parser.add_argument("--output", type=Path,
                        default=Path("tools/perception_bridge/out/readback-probe-2026-11-16.json"))
    args = parser.parse_args()
    if not 1 <= args.seconds <= 15:
        parser.error("seconds must be in 1..15")
    run(args)
