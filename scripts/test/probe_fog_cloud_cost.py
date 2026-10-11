"""External IPC A/B probe. Restores the two cloud toggles and fog walk toggle."""

import json
from pathlib import Path
import time

from rt_ipc import RtIpc


def main():
    client = RtIpc()
    domain = "Physics Domain 1"
    clouds = client.call("world.get_clouds")
    fog = client.call("fluid.get_fog_shader", domain=domain)
    quality = clouds["quality"]
    original = {key: quality[key] for key in
                ("rt_reference_path_trace", "rt_secondary_full_march")}
    results = {"clouds": clouds, "fog": fog, "cases": []}
    cases = [
        ("reference_full_secondary_fog_walk", True, True, True),
        ("reference_capped_secondary_fog_walk", True, False, True),
        ("production_cloud_fog_walk", False, False, True),
        ("production_cloud_fog_march", False, False, False),
    ]
    output = Path(__file__).resolve().parents[2] / "docs/dev/fog_cloud_cost_live_2026_10_11.json"
    try:
        for name, reference, secondary, walk in cases:
            client.call("world.set_clouds", quality={
                "rt_reference_path_trace": reference,
                "rt_secondary_full_march": secondary,
            })
            client.call("fluid.set_fog_shader", domain=domain, random_walk=walk)
            applied_clouds = client.call("world.get_clouds")["quality"]
            applied_fog = client.call("fluid.get_fog_shader", domain=domain)
            assert applied_clouds["rt_reference_path_trace"] == reference
            assert applied_clouds["rt_secondary_full_march"] == secondary
            assert applied_fog["random_walk"] == walk
            client.call("reset_accumulation")
            deadline = time.perf_counter() + 45.0
            while client.call("viewport.status")["samples"] < 2:
                if time.perf_counter() > deadline:
                    raise RuntimeError("No completed warmup samples")
                time.sleep(0.05)
            before = client.call("viewport.status")
            start = time.perf_counter()
            while True:
                after = client.call("viewport.status")
                completed = after["samples"] - before["samples"]
                if completed >= 6:
                    break
                if after["accumulation_complete"] or time.perf_counter() > deadline:
                    raise RuntimeError("Insufficient completed samples for comparison")
                time.sleep(0.05)
            wall_ms = (time.perf_counter() - start) * 1000.0
            row = {"case": name, "completed_samples": completed,
                   "elapsed_ms": wall_ms, "ms_per_completed_sample": wall_ms / completed,
                   "applied_clouds": applied_clouds, "applied_fog": applied_fog,
                   "before": before, "after": after}
            results["cases"].append(row)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    finally:
        try:
            client.call("world.set_clouds", quality=original)
        finally:
            try:
                client.call("fluid.set_fog_shader", domain=domain,
                            random_walk=fog["random_walk"])
                results["restored"] = {
                    "clouds": client.call("world.get_clouds")["quality"],
                    "fog": client.call("fluid.get_fog_shader", domain=domain),
                }
                output.write_text(json.dumps(results, indent=2), encoding="utf-8")
                print("Original cloud/fog toggles restored.", flush=True)
            finally:
                client.close()


if __name__ == "__main__":
    main()
