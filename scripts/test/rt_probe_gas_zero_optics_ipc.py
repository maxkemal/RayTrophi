"""Reversible external GPU ablation of an optically disabled gas domain.

Run from a terminal with Studio open and the timeline paused. No simulation step,
build, render_frames, scene save, or camera mutation is performed.
"""
import argparse
import json
import time
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ipc = RtIpc()
    report = {"domain": args.domain, "cases": []}
    original = None
    domain = None
    touched = False
    output = Path(args.output)

    def save():
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def wait_capture():
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if ipc.call("timeline.get_frame") != report["frame"]:
                raise RuntimeError("Timeline changed; stop comparison")
            stats = ipc.call("render.gpu_timing_stats")
            if stats.get("error"):
                raise RuntimeError(stats["error"])
            if stats["recorded"] >= 64 and stats["pending"] == 0:
                return stats
            time.sleep(0.3)
        raise RuntimeError("GPU capture did not complete within 180 seconds")

    def wait_converged():
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if ipc.call("timeline.get_frame") != report["frame"]:
                raise RuntimeError("Timeline changed; stop comparison")
            if ipc.call("camera.get") != report["camera"]:
                raise RuntimeError("Camera changed; stop comparison")
            if ipc.call("viewport.status")["accumulation_complete"]:
                return
            time.sleep(0.3)
        raise RuntimeError("Natural accumulation did not finish")

    try:
        original = ipc.call("gas.get_shader", domain=args.domain)
        domain = next(d for d in ipc.call("gas.list_domains")["domains"]
                      if d["name"] == args.domain)
        assert all(original[k] == 0 for k in ["density_multiplier",
                   "scattering_coefficient", "absorption_coefficient"])
        assert not ipc.call("render.volume_stats")["enabled"]
        assert not ipc.call("render.gpu_timing_stats")["enabled"]
        assert not ipc.call("sim.control_state")["playing"]
        assert ipc.call("viewport.status")["shading"] == "rendered"
        report.update(frame=ipc.call("timeline.get_frame"), original=original,
                      camera=ipc.call("camera.get"),
                      visible=domain["visible"],
                      active_density_cells=domain["active_density_cells"],
                      method="64 natural camera RT dispatch GPU timestamps; counters off")
        cases = [
            ("authored", original["shadow_steps"], original["max_steps"], domain["visible"]),
            ("shadow0", 0, original["max_steps"], domain["visible"]),
            ("shadow0_steps1", 0, 1, domain["visible"]),
            ("authored_repeat", original["shadow_steps"], original["max_steps"], domain["visible"]),
        ]
        touched = True
        for label, shadows, steps, visible in cases:
            ipc.call("gas.set_shader", domain=args.domain,
                     shadow_steps=shadows, max_steps=steps)
            # Let resource publication and the preceding capture drain naturally.
            time.sleep(0.5)
            wait_converged()
            ipc.call("render.set_gpu_timing", enabled=True)
            ipc.call("reset_accumulation")
            stats = wait_capture()
            ipc.call("render.set_gpu_timing", enabled=False)
            samples = stats["samples"]
            assert [s["sample_index"] for s in samples] == list(range(64)), \
                "Capture is not exactly samples 0..63"
            assert len({(s["width"], s["height"]) for s in samples}) == 1
            assert ipc.call("camera.get") == report["camera"]
            row = {"case": label, "shadow_steps": shadows, "max_steps": steps,
                   "visible": visible, "shader_readback": ipc.call(
                       "gas.get_shader", domain=args.domain), "timings": stats}
            report["cases"].append(row)
            save()
            print(json.dumps({"case": label, "gpu_mean_ms": stats["gpu_mean_ms"],
                              "gpu_max_ms": stats["gpu_max_ms"],
                              "recorded": stats["recorded"]}), flush=True)
    finally:
        try:
            if touched:
                ipc.call("render.set_gpu_timing", enabled=False)
                ipc.call("gas.set_shader", domain=args.domain,
                         shadow_steps=original["shadow_steps"],
                         max_steps=original["max_steps"])
                report["shader_restored"] = ipc.call(
                    "gas.get_shader", domain=args.domain) == original
                restored = next(d for d in ipc.call("gas.list_domains")["domains"]
                                if d["name"] == args.domain)
                report["visibility_restored"] = restored["visible"] == domain["visible"]
                report["final_frame"] = ipc.call("timeline.get_frame")
                save()
                print(json.dumps({k: report[k] for k in ["shader_restored",
                                  "visibility_restored", "final_frame"]}), flush=True)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
