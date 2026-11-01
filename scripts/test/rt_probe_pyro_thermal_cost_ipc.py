"""Vulkan thermal-path cost probe, external IPC only; restores material/instrumentation.

With no spatial emission graph, thermal strength=0 and constant strength=0 both
produce zero volume emission. The difference includes thermal fetch/evaluation
and any emitter/cache policy associated with that mode, not just a tree lookup.
Use one NanoVDB gas volume, a stationary camera and paused timeline. Disable
adaptive sampling in the UI for a fixed-pixel-work comparison. This probe cannot
read that Vulkan switch and reports it as unverified. No screenshot capture.
"""
import argparse
import json
from pathlib import Path
import statistics

from gas_probe_sampling import measure
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", type=Path,
                        default=Path("docs/dev/rt_pyro_thermal_cost_live.json"))
    args = parser.parse_args()
    if not 1 <= args.samples <= 32:
        parser.error("samples must be in 1..32")
    client = RtIpc()
    if client.call("viewport.status")["backend"] != "vulkan":
        client.close()
        raise RuntimeError("Vulkan backend required")
    original = client.call("volume.get_material", target=args.target)
    previous_timing = client.call("render.gpu_timing_stats")
    previous_counters = client.call("render.volume_stats")
    previous_capture = client.call("viewport.status")["capture_enabled"]
    context = {"frame": client.call("timeline.get_frame"),
               "camera": client.call("camera.get"), "material": original}
    keys = ("emission_mode", "blackbody_intensity", "emission_strength")
    report = {"context": context, "cases": [], "complete": False,
              "adaptive_sampling": "not verified; disable in UI",
              "scope": "thermal path including cache/emitter policy, zero emission"}
    try:
        client.call("viewport.capture", enabled=False)
        for repeat, order in enumerate((("thermal", "constant"), ("constant", "thermal"))):
            for label in order:
                settings = dict(zip(keys, (2 if label == "thermal" else 1, 0.0, 0.0)))
                client.call("volume.set_material", target=args.target, settings=settings)
                assert client.call("volume.get_material", target=args.target) == dict(
                    original, **settings), "Unrelated material values changed"
                timing, counters = measure(client, samples=args.samples, warmup=8)
                if (client.call("timeline.get_frame") != context["frame"] or
                        client.call("camera.get") != context["camera"]):
                    raise RuntimeError("Camera or timeline changed; comparison invalid")
                paths = max(counters.get("paths_traced", 0), 1)
                row = {"label": label, "repeat": repeat, "timing": timing,
                       "per_path": {k: counters.get(k, 0) / paths for k in (
                           "camera_density_samples", "secondary_density_samples",
                           "shadow_density_samples", "radiance_cache_fallbacks")}}
                report["cases"].append(row)
                print(label, timing["gpu_mean_ms"], "GPU ms", flush=True)
        means = {label: statistics.mean(row["timing"]["gpu_mean_ms"]
                 for row in report["cases"] if row["label"] == label)
                 for label in ("thermal", "constant")}
        report["gpu_mean_ms"] = means
        report["thermal_minus_constant_gpu_ms"] = means["thermal"] - means["constant"]
        report["complete"] = True
    finally:
        errors = []
        for method, params in (
            ("volume.set_material", {"target": args.target,
                "settings": {k: original[k] for k in keys}}),
            ("render.set_gpu_timing", {"enabled": previous_timing.get("enabled", False)}),
            ("render.volume_counters", {"enabled": previous_counters["enabled"],
                "region": previous_counters["region"]}),
            ("viewport.capture", {"enabled": previous_capture})):
            try:
                client.call(method, **params)
            except Exception as error:
                errors.append(str(error))
        try:
            report["material_restored"] = client.call(
                "volume.get_material", target=args.target) == original
        except Exception as error:
            report["material_restored"] = False
            errors.append(str(error))
        finally:
            client.close()
        report["restore_errors"] = errors
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if errors or not report["material_restored"]:
            raise RuntimeError("Restore failed; inspect " + str(args.output))


if __name__ == "__main__":
    main()
