"""External live window/fallback matrix.

Creates and removes one hidden scratch domain. Advances ALL existing fluid
domains three times by 1/120 s; never resets or reseeds existing domains.
"""

import json
import uuid
from pathlib import Path

from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    client = RtIpc()
    name = "ActiveWindowProbe_" + uuid.uuid4().hex[:8]
    created = False
    report = {}
    try:
        checked_call(client, "fluid.create_domain", name=name, type="fluid",
                     domain_min=[20, 0, 20], domain_max=[23.2, 3.2, 23.2], voxel_size=0.1)
        created = True
        checked_call(client, "fluid.set_param", domain=name, backend="vulkan",
                     boundary="closed", default_substance="Water", visible=False)
        checked_call(client, "fluid.set_whitewater", domain=name, enabled=False)
        checked_call(client, "fluid.seed", domain=name, seed_min=[21.2, 1.2, 21.2],
                     seed_max=[21.6, 1.6, 21.6], particles_per_cell=8,
                     replace=True, persistent=False)
        checked_call(client, "fluid.step", dt=1 / 120)
        local = checked_call(client, "fluid.step_stats", domain=name)
        report["localized_closed"] = local
        assert local["measured"] and local["p2g_on_gpu"], local
        assert local["normalize_window_used"], local
        assert 0 < local["normalize_window_cells"] < local["full_grid_cells"], local
        assert local["occupancy_on_gpu"] and local["pressure_on_gpu"], local
        assert local["pressure_window_used"], local
        assert 0 < local["pressure_window_cells"] < local["full_grid_cells"], local
        gpu_digest = checked_call(client, "fluid.state_digest", domain=name)
        checked_call(client, "fluid.set_param", domain=name, backend="cpu")
        checked_call(client, "fluid.seed", domain=name, seed_min=[21.2, 1.2, 21.2],
                     seed_max=[21.6, 1.6, 21.6], particles_per_cell=8,
                     replace=True, persistent=False)
        checked_call(client, "fluid.step", dt=1 / 120)
        cpu_digest = checked_call(client, "fluid.state_digest", domain=name)
        assert gpu_digest["particles"] == cpu_digest["particles"] > 0
        centroid_error = max(abs(a - b) for a, b in
                             zip(gpu_digest["centroid"], cpu_digest["centroid"]))
        speed_error = abs(gpu_digest["mean_speed"] - cpu_digest["mean_speed"])
        assert centroid_error < 1e-4, (gpu_digest, cpu_digest)
        assert speed_error < 1e-3, (gpu_digest, cpu_digest)
        report["cpu_reference"] = {
            "centroid_error": centroid_error,
            "mean_speed_error": speed_error,
            "particles": cpu_digest["particles"],
        }
        checked_call(client, "fluid.set_param", domain=name, backend="vulkan", boundary="periodic")
        # Boundary edits invalidate the domain state; seed only this scratch domain again.
        checked_call(client, "fluid.seed", domain=name, seed_min=[21.2, 1.2, 21.2],
                     seed_max=[21.6, 1.6, 21.6], particles_per_cell=8,
                     replace=True, persistent=False)
        checked_call(client, "fluid.step", dt=1 / 120)
        periodic = checked_call(client, "fluid.step_stats", domain=name)
        report["periodic"] = periodic
        assert periodic["measured"] and periodic["p2g_on_gpu"], periodic
        assert not periodic["normalize_window_used"], periodic
        assert periodic["normalize_window_cells"] == 0, periodic
        assert not periodic["pressure_window_used"], periodic
        assert periodic["pressure_window_cells"] == 0, periodic
        report["passed"] = True
    finally:
        try:
            if created:
                checked_call(client, "fluid.remove_domain", domain=name)
                report["scratch_removed"] = True
        finally:
            client.close()
            path = Path(__file__).resolve().parents[2] / ".tmp" / "fluid_active_window_live.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
