"""External reversible zero-optics gas probe: secondary rays and work counters."""
import argparse
import json
import time
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--counts-only", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc()
    original = ipc.call("render.get_settings")
    writable = {k: original[k] for k in ["debug_view", "diffuse_bounces",
                                         "max_bounces", "transmission_bounces"]}
    shader = ipc.call("gas.get_shader", domain=args.domain)
    counters = ipc.call("render.volume_stats")
    frame = ipc.call("timeline.get_frame")
    camera = ipc.call("camera.get")
    report = {"frame": frame, "camera": camera, "render_settings": original,
              "shader": shader, "cases": [], "counter_cases": []}
    assert not counters["enabled"]
    assert not ipc.call("render.gpu_timing_stats")["enabled"]
    assert not ipc.call("sim.control_state")["playing"]
    assert all(shader[k] == 0 for k in ["density_multiplier",
               "scattering_coefficient", "absorption_coefficient"])

    def save():
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def until(predicate):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            assert ipc.call("timeline.get_frame") == frame
            assert ipc.call("camera.get") == camera
            result = predicate()
            if result:
                return result
            time.sleep(0.3)
        raise RuntimeError("Natural render progress timed out")

    def capture_ready():
        stats = ipc.call("render.gpu_timing_stats")
        if stats.get("error"):
            raise RuntimeError(stats["error"])
        return stats if stats["recorded"] == 64 and stats["pending"] == 0 else None

    try:
        ipc.call("gas.set_shader", domain=args.domain, shadow_steps=0)
        # Restrict physical bounces, preserving the medium's free-pass handoffs.
        ipc.call("render.set_settings", max_bounces=1)
        time.sleep(0.5)
        until(lambda: ipc.call("viewport.status")["accumulation_complete"])
        if not args.counts_only:
            ipc.call("render.set_gpu_timing", enabled=True)
            ipc.call("reset_accumulation")
            stats = until(capture_ready)
            ipc.call("render.set_gpu_timing", enabled=False)
            assert [s["sample_index"] for s in stats["samples"]] == list(range(64))
            report["cases"].append({"case": "shadow0_one_physical_bounce",
                                    "settings": ipc.call("render.get_settings"),
                                    "timings": stats})
            save()
            print(json.dumps({"case": "one_physical_bounce",
                              "gpu_mean_ms": stats["gpu_mean_ms"]}), flush=True)
        for label, settings in [("one_bounce", {"max_bounces": 1}),
                                ("authored_bounces", writable)]:
            ipc.call("render.set_settings", **settings)
            ipc.call("render.volume_counters", enabled=True, region=[0, 0, 1, 1])
            ipc.call("reset_accumulation")
            until(lambda: ipc.call("viewport.status")["samples"] >= 2)
            counts = ipc.call("render.volume_stats")
            ipc.call("render.volume_counters", enabled=False, region=counters["region"])
            paths = max(counts["paths_traced"], 1)
            row = {"case": label, "counts": counts,
                   "density_per_path": counts["density_samples"] / paths,
                   "volume_rays_per_path": counts["volume_rays"] / paths,
                   "probes_per_path": counts["solid_probe_runs"] / paths}
            report["counter_cases"].append(row)
            save()
            print(json.dumps({k: v for k, v in row.items() if k != "counts"}), flush=True)
    finally:
        try:
            ipc.call("render.set_gpu_timing", enabled=False)
            ipc.call("render.volume_counters", enabled=False, region=counters["region"])
            ipc.call("render.set_settings", **writable)
            ipc.call("gas.set_shader", domain=args.domain,
                     shadow_steps=shader["shadow_steps"])
            report["render_restored"] = ipc.call("render.get_settings") == original
            report["shader_restored"] = ipc.call("gas.get_shader", domain=args.domain) == shader
            report["counters_restored"] = not ipc.call("render.volume_stats")["enabled"]
            save()
            print(json.dumps({k: report[k] for k in ["render_restored", "shader_restored",
                                                    "counters_restored"]}), flush=True)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
