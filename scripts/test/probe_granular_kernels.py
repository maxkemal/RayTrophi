"""Measure GPU kernels and transfer traffic through the external named pipe.

Run in a paused, single-domain scene from a separate terminal:
    python scripts/test/probe_granular_kernels.py "Grid Domain 1" --steps 5

This intentionally advances the simulation; it does not reset or change material
settings. Kernel timestamps are global, so other domains are rejected. Run once
with timestamps off (default), then with --gpu-timing to measure instrumentation
overhead separately from the production frame time.
"""

import argparse
import json
from statistics import mean

from rt_ipc import RtIpc


STAGES = {
    "gather": ("sim_fluid_g2p",),
    "stress": ("sim_fluid_granular_stress_update",),
    "stress_p2g": ("sim_fluid_granular_stress_p2g",),
    "settle": ("sim_fluid_granular_settle",),
    "advect": ("sim_fluid_advect_tail",),
    "occupancy": ("sim_fluid_occupancy",),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--dt", type=float, default=1.0 / 24.0)
    parser.add_argument("--gpu-timing", action="store_true")
    parser.add_argument("--output", help="Optional JSON report path")
    args = parser.parse_args()
    if args.steps < 1 or not 0.0 < args.dt <= 1.0:
        parser.error("steps must be positive and dt must be in (0, 1]")

    client = RtIpc()
    prior_enabled = None
    try:
        domains = client.call("fluid.list_domains")["domains"]
        if len(domains) != 1 or domains[0]["name"] != args.domain:
            raise RuntimeError("Use a paused scene containing only the named domain")
        timing = client.call("perf.gpu_kernel_timings", reset=False)
        prior_enabled = bool(timing.get("enabled"))
        if args.gpu_timing and not timing.get("supported"):
            raise RuntimeError("Compute queue has no GPU timestamp support")
        if timing.get("supported"):
            client.call("perf.set_gpu_kernel_timing", enabled=args.gpu_timing)
        samples = []
        for index in range(args.steps):
            if args.gpu_timing:
                client.call("perf.gpu_kernel_timings", reset=True)
            client.call("fluid.step", dt=args.dt)
            stats = client.call("fluid.step_stats", domain=args.domain)
            info = client.call("fluid.get", domain=args.domain)
            if not stats.get("measured") or not info.get("granular_enabled"):
                raise RuntimeError("Expected a freshly measured granular solver step")
            kernels = (client.call("perf.gpu_kernel_timings", reset=True)
                       if args.gpu_timing else None)
            stages = None
            if kernels is not None:
                stages = {
                    stage: {
                        "ms": sum(row["ms"] for row in kernels["kernels"]
                                  if row["kernel"] in names),
                        "calls": sum(row["calls"] for row in kernels["kernels"]
                                     if row["kernel"] in names),
                    }
                    for stage, names in STAGES.items()
                }
            sample = {
                "sample": index + 1,
                "info": {key: info.get(key) for key in (
                    "particle_count", "granular_required_substeps", "granular_solver_substeps",
                    "granular_young_modulus", "granular_effective_young_modulus",
                    "granular_stiffness_capped", "granular_invalid", "granular_max_damage")},
                "stats": stats,
                "gpu_stages": stages,
                "gpu_kernels": kernels,
            }
            samples.append(sample)
            print(json.dumps(sample, ensure_ascii=False))
        keys = ("total_ms", "upload_bytes", "download_bytes", "batch_end_calls",
                "batch_end_ms", "synchronize_calls", "synchronize_ms")
        report = {
            "domain": args.domain,
            "dt": args.dt,
            "gpu_timing": args.gpu_timing,
            "samples": samples,
            "mean": {key: mean(row["stats"][key] for row in samples) for key in keys},
        }
        if args.output:
            from pathlib import Path
            Path(args.output).write_text(
                json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"mean": report["mean"]}, indent=2))
    finally:
        if prior_enabled is not None:
            client.try_call("perf.set_gpu_kernel_timing", enabled=prior_enabled)
        client.close()


if __name__ == "__main__":
    main()
