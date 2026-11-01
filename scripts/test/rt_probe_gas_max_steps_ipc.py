"""External Vulkan Max Steps A/B. Restores authored limits and instrumentation."""
import json
import time
from pathlib import Path
from rt_ipc import RtIpc
from gas_probe_snapshot import capture, require_same
from gas_probe_sampling import measure


def main():
    c = RtIpc()
    original = {}
    prior = c.call("render.volume_stats")
    timing_prior = c.call("render.gpu_timing_stats")
    rows = []
    baseline = None
    complete = False
    frame = c.call("timeline.get_frame")
    try:
        assert c.call("render.rt_pipeline_status")["backend"] == "vulkan"
        targets = [t["target"] for t in c.call("volume.list_materials")["targets"]
                   if t["target"].startswith("matter:") and ":gas" in t["target"]]
        assert targets, "No Matter gas targets"
        original = {t: c.call("volume.get_material", target=t)["max_steps"] for t in targets}
        baseline = capture(c, targets)
        print(json.dumps({"scene_signature": baseline["signature"],
                          "fingerprint_strength": baseline["fingerprint_strength"],
                          "state_versions": baseline["state_versions"]}), flush=True)
        for label, cap in (("256", 256), ("16", 16), ("256_repeat", 256)):
            for target in targets:
                c.call("volume.set_material", target=target, settings={"max_steps": cap})
            assert c.call("timeline.get_frame") == frame, "Keep timeline fixed"
            before = capture(c, targets)
            require_same(baseline, before)
            timing, stats = measure(c, samples=4)
            after = capture(c, targets)
            require_same(before, after)
            paths = max(stats.get("paths_traced", 0), 1)
            row = {"case": label, "frame": frame, "gpu_ms": timing.get("gpu_mean_ms"),
                   "scene_signature": before["signature"],
                   "sampling_method": timing["sampling_method"],
                   "dimensions": sorted({(x.get("width"), x.get("height"))
                                         for x in timing.get("samples", [])}),
                   "per_path": {k: stats.get(k, 0)/paths for k in
                       ("camera_density_samples", "camera_volume_rays", "secondary_density_samples",
                        "step_budget_exhausted", "radiance_cache_fallbacks")}}
            rows.append(row)
            print(json.dumps(row), flush=True)
        if len({json.dumps(r["dimensions"]) for r in rows}) != 1:
            raise RuntimeError("INVALID benchmark: render dimensions changed")
        complete = True
    finally:
        for target, cap in original.items():
            c.call("volume.set_material", target=target, settings={"max_steps": cap})
        c.call("render.set_gpu_timing", enabled=timing_prior.get("enabled", False))
        c.call("render.volume_counters", enabled=prior["enabled"], region=prior["region"])
        restored = all(c.call("volume.get_material", target=t)["max_steps"] == cap
                       for t, cap in original.items())
        c.close()
        output = Path("docs/dev") / ("rt_gas_max_steps_%d.json" % time.time_ns())
        output.write_text(json.dumps({"baseline": baseline, "rows": rows,
            "restored": restored, "complete_and_comparable": complete}, indent=2), encoding="utf-8")
        print("REPORT", str(output), flush=True)
        print("RESTORED", restored, flush=True)


if __name__ == "__main__":
    main()
