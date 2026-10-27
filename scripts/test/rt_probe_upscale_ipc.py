"""B4 external named-pipe probe, AFTER the single user shader/C++ build.

Default: black-box API validation and dependency errors, then restore settings.
--motion-seconds 4: also move the camera slightly in the open Rendered viewport,
observe real FSR dispatches/extents/jitter and the return to full resolution.
Does not launch the app, build, save a project, rename DLLs or capture images.
Image quality/reference tests remain in docs/dev/NEXT_BUILD_CHECKS.md.
"""
import argparse
import json
import math
import time

from rt_ipc import RtIpc, RtIpcError


def selected(report, names):
    return {name: report[name] for name in names}


def settings(client):
    return selected(client.call("render.upscale_stats", synchronize=False),
                    ("mode", "quality", "sharpness"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion-seconds", type=float, default=0.0)
    args = parser.parse_args()
    if not math.isfinite(args.motion_seconds) or not 0.0 <= args.motion_seconds <= 30.0:
        parser.error("--motion-seconds must be in [0, 30]")
    client = RtIpc()
    original = settings(client)
    svgf = selected(client.call("render.realtime_denoiser_stats", synchronize=False),
                    ("mode", "atrous_iterations", "history_max", "firefly_clip",
                     "still_noise", "debug_history"))
    motion = client.call("render.motion_stats")["enabled"]
    camera = client.call("camera.get") if args.motion_seconds else None
    if camera and isinstance(camera["position"], dict):
        camera["position"] = [camera["position"][k] for k in ("x", "y", "z")]
    report = {"api_validation": "not run", "motion_observations": []}
    try:
        client.call("render.set_upscale", mode="off")
        client.call("render.set_motion", enabled=True)
        client.call("render.set_realtime_denoiser", mode="svgf", debug_history=False)
        client.call("render.set_upscale", mode="fsr", quality="balanced", sharpness=0.2)
        expected = settings(client)
        invalid = [
            {"mode": "taau"}, {"mode": True}, {"quality": "ultra"},
            {"quality": 2}, {"sharpness": -0.01}, {"sharpness": 1.01},
            {"sharpness": True}, {"sharpness": "0.5"},
            {"mode": "off", "quality": "invalid"},
            {"quality": "quality", "sharpness": 2}, {"unknown": 1},
        ]
        for values in invalid:
            try:
                client.call("render.set_upscale", **values)
            except RtIpcError:
                pass
            else:
                raise AssertionError("invalid update accepted: %r" % values)
            assert settings(client) == expected, "invalid update partially committed"
        ok, _ = client.try_call("render.set_motion", enabled=False)
        assert not ok, "motion dependency was disabled while FSR was requested"
        # Provider contract: SVGF is an optional input provider, not a gate.
        client.call("render.set_realtime_denoiser", mode="off")
        assert settings(client)["mode"] == "fsr", "disabling SVGF changed the FSR request"
        row = client.call("render.upscale_stats", synchronize=False)
        assert row["input_provider"] in ("none", "svgf"), row
        client.call("render.set_upscale", mode="off")
        client.call("render.set_upscale", mode="auto")
        assert settings(client)["mode"] == "auto", "FSR refused without SVGF"
        client.call("render.set_upscale", mode="off")
        client.call("render.set_realtime_denoiser", mode="svgf")
        client.call("render.set_upscale", mode="auto", quality="balanced")
        report["api_validation"] = "PASS"
        report["atomic_rejections"] = len(invalid)
        if camera:
            position = camera["position"]
            if isinstance(position, dict):
                position = [position[k] for k in ("x", "y", "z")]
            start = time.monotonic()
            while time.monotonic() - start < args.motion_seconds:
                p = list(position)
                p[0] += 0.002 * math.sin((time.monotonic() - start) * 8.0)
                client.call("camera.set_position", position=p)
                time.sleep(0.05)
                report["motion_observations"].append(
                    client.call("render.upscale_stats", synchronize=False))
            observed = report["motion_observations"]
            assert any(row["active"] and row["dispatches"] > 0 for row in observed), observed
            assert any(row["render_w"] < row["display_w"] and
                       row["render_h"] < row["display_h"] for row in observed)
            assert any(row["jitter_phase_count"] > 8 for row in observed)
            assert all(row["input_provider"] == "svgf" for row in observed
                       if row["active"] and row["dispatches"] > 0), observed
            report["gpu_stats"] = client.call("render.upscale_stats", synchronize=True)
            # Separate process: the application's main loop can settle while we wait.
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline:
                row = client.call("render.upscale_stats", synchronize=False)
                if not row["active"] and row["render_w"] == row["display_w"]:
                    report["settled_stats"] = row
                    break
                time.sleep(0.1)
            else:
                raise AssertionError("FSR did not return to full resolution after settling")
        print(json.dumps(report, indent=2))
    finally:
        client.call("render.set_upscale", mode="off")
        if camera:
            client.call("camera.set_position", position=camera["position"])
        client.call("render.set_realtime_denoiser", mode="off")
        client.call("render.set_motion", enabled=motion)
        client.call("render.set_realtime_denoiser", **svgf)
        client.call("render.set_upscale", **original)


if __name__ == "__main__":
    main()
