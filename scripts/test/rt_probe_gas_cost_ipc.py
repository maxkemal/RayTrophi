"""External RT gas cost probe. Read-only unless --apply; restores material/counters.
Uses natural Vulkan sample progress, never viewport.render_frames (CPU path).
"""
import argparse
import json
import time
from pathlib import Path
from rt_ipc import RtIpc


def wait_samples(ipc, minimum, frame, timeout=45):
    deadline = time.perf_counter() + timeout
    while True:
        assert ipc.call("timeline.get_frame") == frame, "Timeline changed during probe"
        status = ipc.call("viewport.status")
        if status["samples"] >= minimum:
            return time.perf_counter(), status["samples"]
        if status["accumulation_complete"]:
            raise RuntimeError("Accumulation finished before measurement window")
        if time.perf_counter() > deadline:
            raise RuntimeError("RT sample progress timed out")
        time.sleep(.015)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="gas:Physics Domain 1")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", default="docs/dev/rt_gas_cost_postsolid_20261022.json")
    args = parser.parse_args()
    ipc = RtIpc()
    original = None
    prior = None
    edited = False
    report = {"method": "Natural Vulkan RT sample progress; UI/scheduler-inclusive wall time, not GPU timestamp", "rows": []}
    try:
        original = ipc.call("volume.get_material", target=args.target)
        prior = ipc.call("render.volume_stats")
        frame = ipc.call("timeline.get_frame")
        report.update(frame=frame, target=args.target, original_material=original,
                      shading=ipc.call("viewport.shading"),
                      pipeline=ipc.call("render.rt_pipeline_status"))
        if not args.apply:
            print(json.dumps(report, indent=2))
            return
        assert report["shading"]["mode"] == "rendered", "Rendered viewport required"
        assert report["pipeline"]["rt_pipeline_state"] == "ready", "Ready RT pipeline required"
        cases = [("authored", {}),
                 ("scatter0", {"scatter_coefficient": 0}),
                 ("scatter0001", {"scatter_coefficient": .001}),
                 ("scatter1", {"scatter_coefficient": 1}),
                 ("scatter1_no_shadow", {"scatter_coefficient": 1, "shadow_steps": 0}),
                 ("scatter1_stride16", {"scatter_coefficient": 1, "shadow_stride": 16}),
                 ("scatter1_density0", {"scatter_coefficient": 1, "density_multiplier": 0}),
                 ("authored_repeat", {})]
        assert all(device["enabled"] == prior["enabled"] and
                   device["region"] == prior["region"]
                   for device in prior.get("devices", [])), "Mixed device counter states; cannot restore atomically"
        edited = True
        for label, patch in cases:
            settings = dict(original)
            settings.update(patch)
            ipc.call("render.volume_counters", enabled=False, region=prior["region"])
            ipc.call("volume.set_material", target=args.target, settings=settings)
            start, sample0 = wait_samples(ipc, 8, frame)
            end, sample1 = wait_samples(ipc, sample0 + 32, frame)
            row = {"case": label, "patch": patch, "sample_start": sample0,
                   "sample_end": sample1, "wall_ms_per_rt_sample":
                   (end-start)*1000/(sample1-sample0)}
            ipc.call("volume.set_material", target=args.target, settings=settings)
            ipc.call("render.volume_counters", enabled=True, region=[0, 0, 1, 1])
            wait_samples(ipc, 8, frame)
            stats = ipc.call("render.volume_stats")
            assert stats["enabled"] and stats["paths_traced"] > 0, "No measured Vulkan paths"
            rays = max(stats["volume_rays"], 1)
            row.update(stats=stats,
                       density_per_volume_ray=stats["density_samples"]/rays,
                       shadow_per_volume_ray=stats["shadow_density_samples"]/rays,
                       probes_per_path=stats["solid_probe_runs"]/max(stats["paths_traced"], 1))
            report["rows"].append(row)
            Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps({k: v for k, v in row.items() if k != "stats"}), flush=True)
    finally:
        try:
            if edited:
                ipc.call("volume.set_material", target=args.target, settings=original)
                ipc.call("render.volume_counters", enabled=prior["enabled"], region=prior["region"])
                report["material_restored"] = ipc.call("volume.get_material", target=args.target) == original
                report["counters_restored"] = ipc.call("render.volume_stats")["enabled"] == prior["enabled"]
                report["final_frame"] = ipc.call("timeline.get_frame")
                Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(json.dumps({k: report[k] for k in ["material_restored", "counters_restored", "final_frame"]}), flush=True)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
