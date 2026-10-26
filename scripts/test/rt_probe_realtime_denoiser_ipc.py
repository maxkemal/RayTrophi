"""B3 API validation, from an EXTERNAL terminal after the single B3 build.

Does not launch the app, save a project, move the camera or capture images.
Restores the initial SVGF and motion settings even after a failed assertion.
GPU cost, ghosting and image quality belong to NEXT_BUILD_CHECKS.md's one
end-of-batch acceptance round; this probe does not claim to measure them.
"""
import json

from rt_ipc import RtIpc, RtIpcError


FIELDS = ("mode", "atrous_iterations", "history_max", "firefly_clip", "static_samples")


def settings(client):
    report = client.call("render.realtime_denoiser_stats", synchronize=False)
    return {key: report[key] for key in FIELDS}


def main():
    client = RtIpc()
    original = settings(client)
    motion = client.call("render.motion_stats")["enabled"]
    rejected = []
    try:
        client.call("render.set_realtime_denoiser", mode="off")
        client.call("render.set_motion", enabled=True)
        client.call("render.set_realtime_denoiser", mode="svgf",
                    atrous_iterations=4, history_max=24, firefly_clip=3.0,
                    static_samples=0)
        expected = settings(client)
        invalid = [
            {"mode": "oidn"}, {"atrous_iterations": 0},
            {"atrous_iterations": 6}, {"atrous_iterations": True},
            {"atrous_iterations": 2.5}, {"history_max": 0},
            {"history_max": 65}, {"firefly_clip": 0.0},
            {"firefly_clip": 11.0}, {"firefly_clip": True},
            {"static_samples": -1}, {"static_samples": 2000}, {"static_samples": 1.5},
            {"mode": "off", "history_max": 0}, {"unknown": 1},
        ]
        for params in invalid:
            try:
                client.call("render.set_realtime_denoiser", **params)
            except RtIpcError:
                rejected.append(params)
            else:
                raise AssertionError("invalid update accepted: %r" % params)
            assert settings(client) == expected, "invalid update partially committed"
        ok, error = client.try_call("render.set_motion", enabled=False)
        assert not ok, "motion was disabled while SVGF still depended on it"
        assert client.call("render.motion_stats")["enabled"]
        report = client.call("render.realtime_denoiser_stats", synchronize=True)
        assert 0.0 <= report["history_length_mean"] <= 64.0
        print(json.dumps({"api_validation": "PASS", "rejections": len(rejected),
                          "motion_dependency_error": error, "stats": report}, indent=2))
    finally:
        client.call("render.set_realtime_denoiser", mode="off")
        client.call("render.set_motion", enabled=motion)
        client.call("render.set_realtime_denoiser", **original)


if __name__ == "__main__":
    main()
