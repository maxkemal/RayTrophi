"""Run externally against a rebuilt app in an EMPTY, paused scene.

Checks that legacy budgets 1/32/64 cannot soften the material, including a
request above 64 substeps. Compares closed Vulkan residency with the existing
open-boundary transfer path, then exercises CPU subcycling. Creates and removes
only its scratch domain; fluid.step/reset are global, hence the empty-scene gate.
"""

import json
import uuid

from rt_ipc import RtIpc


DT = 1.0 / 24.0
YOUNG = 381300.0


def sample(client, domain, backend, boundary, budget, refresh_period=4):
    client.call(
        "fluid.set_param", domain=domain, backend=backend, boundary=boundary,
        granular_max_solver_substeps=budget, uvw_refresh_period=refresh_period)
    client.call("fluid.reset")
    for _ in range(3):
        client.call("fluid.step", dt=DT)
    info = client.call("fluid.get", domain=domain)
    stats = client.call("fluid.step_stats", domain=domain)
    digest = client.call("fluid.state_digest", domain=domain)
    assert stats.get("measured"), stats
    assert info["granular_required_substeps"] > 64, info
    assert info["granular_solver_substeps"] >= info["granular_required_substeps"], info
    assert abs(info["granular_young_modulus"] - YOUNG) < 0.1, info
    assert abs(info["granular_effective_young_modulus"] - YOUNG) < 1.0, info
    assert not info["granular_stiffness_capped"], info
    assert info["granular_invalid"] == 0, info
    if backend == "vulkan":
        assert stats["g2p_on_gpu"] and stats["p2g_on_gpu"], stats
    print(json.dumps({
        "backend": backend, "boundary": boundary, "legacy_budget": budget,
        "required": info["granular_required_substeps"],
        "run": info["granular_solver_substeps"],
        "effective_young": info["granular_effective_young_modulus"],
        "particles": digest["particles"],
        "upload_bytes": stats.get("upload_bytes"),
        "download_bytes": stats.get("download_bytes"),
        "total_ms": stats.get("total_ms"),
    }, indent=2))
    return info, stats, digest


def compare(a, b):
    assert a["particles"] == b["particles"] > 0
    error = max(abs(float(x) - float(y))
                for x, y in zip(a["centroid"], b["centroid"]))
    assert error < 1.0e-4, ("centroid", error)
    assert abs(float(a["mean_speed"]) - float(b["mean_speed"])) < 1.0e-3


def main():
    client = RtIpc()
    domain = "GranularResidency_" + uuid.uuid4().hex[:8]
    created = False
    try:
        assert not client.call("fluid.list_domains")["domains"], (
            "Use an empty, paused scene: fluid.step/reset affect all domains.")
        client.call(
            "fluid.create_domain", name=domain, type="fluid",
            domain_min=[0.0, -0.64, 0.0], domain_max=[0.32, 0.64, 0.32],
            voxel_size=0.02)
        created = True
        client.call(
            "fluid.set_param", domain=domain, preset="sand", visible=False,
            granular_young_modulus=YOUNG, uvw_refresh_period=4,
            max_particles=4096)
        client.call(
            "fluid.seed", domain=domain, seed_min=[0.10, 0.10, 0.10],
            seed_max=[0.22, 0.22, 0.22], particles_per_cell=2, persistent=True)

        closed_info, closed, closed_digest = sample(client, domain, "vulkan", "closed", 1)
        repeat_info, repeat, repeat_digest = sample(client, domain, "vulkan", "closed", 32)
        compare(closed_digest, repeat_digest)
        open_info, opened, open_digest = sample(client, domain, "vulkan", "open", 64)
        # Seed stays far from walls: changing the transfer path must preserve motion.
        compare(closed_digest, open_digest)
        assert closed_info["uvw_available"] and open_info["uvw_available"]
        assert abs(closed_info["uvw_drift"] - open_info["uvw_drift"]) < 1.0e-5
        assert abs(closed_info["uvw_drift"] - repeat_info["uvw_drift"]) < 1.0e-5
        assert closed["download_bytes"] < 0.5 * opened["download_bytes"]
        assert repeat["upload_bytes"] < 0.5 * opened["upload_bytes"]

        # A long material refresh period leaves most positions on device. A
        # one-step period publishes every substep, exercising the host consumer
        # exception without changing the physical solver or its stiffness.
        timing = client.call("perf.gpu_kernel_timings", reset=False)
        previous_timing = bool(timing.get("enabled"))
        assert timing.get("supported"), "Vulkan timestamp support required for occupancy check"
        try:
            client.call("perf.set_gpu_kernel_timing", enabled=True)
            client.call("perf.gpu_kernel_timings", reset=True)
            long_info, long_stats, long_digest = sample(
                client, domain, "vulkan", "closed", 32, refresh_period=240)
            kernels = client.call("perf.gpu_kernel_timings", reset=True)
            calls = {row["kernel"]: row["calls"] for row in kernels["kernels"]}
            gather_calls = calls.get("sim_fluid_g2p", 0)
            assert gather_calls >= 3 * 65, ("exercise long resident subcycles", calls)
            assert sum(calls.values()) > 512, ("exercise descriptor pool rollover", calls)
            for kernel in ("sim_fluid_granular_stress_update",
                           "sim_fluid_granular_settle", "sim_fluid_advect_tail"):
                assert calls.get(kernel) == gather_calls, ("incomplete GPU timing", calls)
            assert calls.get("sim_fluid_occupancy", 0) >= gather_calls, (
                "occupancy shader missing or GPU mask fell back", calls)
            every_info, every_stats, every_digest = sample(
                client, domain, "vulkan", "closed", 64, refresh_period=1)
            compare(long_digest, every_digest)
            assert every_info["uvw_available"] and long_info["uvw_available"]
            assert every_info["uvw_drift"] < 1.0e-5, every_info
            assert long_stats["download_bytes"] < 0.6 * every_stats["download_bytes"], (
                "intermediate position downloads were not removed", long_stats, every_stats)
            odd_closed_info, _, odd_closed_digest = sample(
                client, domain, "vulkan", "closed", 32, refresh_period=5)
            odd_open_info, _, odd_open_digest = sample(
                client, domain, "vulkan", "open", 32, refresh_period=5)
            compare(odd_closed_digest, odd_open_digest)
            assert abs(odd_closed_info["uvw_drift"] - odd_open_info["uvw_drift"]) < 1.0e-5
        finally:
            client.call("perf.set_gpu_kernel_timing", enabled=previous_timing)
        sample(client, domain, "cpu", "closed", 1)
        print("PASS: authored stiffness preserved; granular particle traffic reduced")
    finally:
        if created:
            client.try_call("fluid.remove_domain", domain=domain)
        client.close()


if __name__ == "__main__":
    main()
