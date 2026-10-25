"""Reversible external close-camera RT bounce/work probe; no simulation stepping."""
import argparse
import json
import time
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ipc = RtIpc()
    original = ipc.call("render.get_settings")
    writable = {k: original[k] for k in ["debug_view", "diffuse_bounces",
                                         "max_bounces", "transmission_bounces"]}
    counter_state = ipc.call("render.volume_stats")
    camera = ipc.call("camera.get")
    frame = ipc.call("timeline.get_frame")
    report = {"camera": camera, "frame": frame, "original_settings": original,
              "timing_cases": [], "counter_cases": []}
    assert not ipc.call("sim.control_state")["playing"]
    assert ipc.call("viewport.status")["shading"] == "rendered"
    assert not counter_state["enabled"]
    assert not ipc.call("render.gpu_timing_stats")["enabled"]

    def save():
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")

    def check_scene():
        assert ipc.call("timeline.get_frame") == frame, "Timeline changed"
        assert ipc.call("camera.get") == camera, "Camera changed"

    def wait_for(predicate):
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            check_scene()
            result = predicate()
            if result:
                return result
            time.sleep(.1)
        raise RuntimeError("Natural RT progress timed out")

    try:
        for label, settings in [("authored", writable),
                                ("one_bounce", {"max_bounces": 1}),
                                ("two_bounces", {"max_bounces": 2,
                                                 "diffuse_bounces": 2,
                                                 "transmission_bounces": 2}),
                                ("authored_repeat", writable)]:
            ipc.call("render.set_settings", **settings)
            time.sleep(.5)
            ipc.call("render.set_gpu_timing", enabled=True)
            ipc.call("reset_accumulation")

            def captured():
                stats = ipc.call("render.gpu_timing_stats")
                if stats.get("error"):
                    raise RuntimeError(stats["error"])
                rows = stats["samples"]
                # Arming can capture preceding in-flight work before reset.
                # Accept only the subsequent exact contiguous sample window.
                for start, row in enumerate(rows):
                    window = rows[start:start + 8]
                    if [r["sample_index"] for r in window] == list(range(8)):
                        return stats, window
                return None

            stats, rows = wait_for(captured)
            ipc.call("render.set_gpu_timing", enabled=False)
            wait_for(lambda: ipc.call("render.gpu_timing_stats")["pending"] == 0)
            assert len({(r["width"], r["height"]) for r in rows}) == 1
            row = {"case": label, "settings": ipc.call("render.get_settings"),
                   "samples": rows, "gpu_mean_ms": sum(r["gpu_ms"] for r in rows) / 8,
                   "raw_timings": stats}
            report["timing_cases"].append(row)
            save()
            print(json.dumps({"case": label, "gpu_mean_ms": row["gpu_mean_ms"]}), flush=True)

        # Separate short counter captures; atomic overhead never enters timings.
        for label, settings in [("one_bounce", {"max_bounces": 1}),
                                ("authored", writable)]:
            ipc.call("render.set_settings", **settings)
            ipc.call("render.volume_counters", enabled=True, region=[0, 0, 1, 1])
            ipc.call("reset_accumulation")
            wait_for(lambda: ipc.call("viewport.status")["samples"] >= 2)
            stats = ipc.call("render.volume_stats")
            ipc.call("render.volume_counters", enabled=False, region=counter_state["region"])
            paths = max(stats["paths_traced"], 1)
            rays = max(stats["volume_rays"], 1)
            keys = ["volume_rays", "density_samples", "shadow_density_samples",
                    "solid_probe_runs", "gas_handoffs", "layered_handoffs", "free_passes",
                    "medium_passes", "paths_pass_capped", "paths_bounce_capped",
                    "empty_segments_skipped", "arbiter_gate_open", "arbiter_candidates",
                    "walk_events"]
            keys += [k for k in stats if k.startswith("sdf_")]
            report["sdf_counters_available"] = "sdf_main_samples" in stats
            per_path = {k: stats[k] / paths for k in keys}
            row = {"case": label, "stats": stats, "per_path": per_path,
                   "density_per_volume_ray": stats["density_samples"] / rays}
            report["counter_cases"].append(row)
            save()
            print(json.dumps({"case": label, "per_path": per_path}), flush=True)
    finally:
        try:
            ipc.call("render.set_gpu_timing", enabled=False)
            ipc.call("render.volume_counters", enabled=False, region=counter_state["region"])
            ipc.call("render.set_settings", **writable)
            report["settings_restored"] = ipc.call("render.get_settings") == original
            report["diagnostics_disabled"] = not ipc.call("render.volume_stats")["enabled"]
            report["camera_unchanged"] = ipc.call("camera.get") == camera
            report["frame_unchanged"] = ipc.call("timeline.get_frame") == frame
            save()
            print(json.dumps({k: report[k] for k in ["settings_restored",
                              "diagnostics_disabled", "camera_unchanged", "frame_unchanged"]}),
                  flush=True)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
