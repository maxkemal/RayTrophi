"""C3 live matrix in a disposable particle runtime, restoring the active system.

Two explicit 1/120s world steps also advance existing systems. Authoring
requests and state resets target only the disposable runtime.
"""
import json
import uuid
from pathlib import Path
from rt_ipc import RtIpc, RtIpcError
from rt_test_fluid_active_window_ipc import checked_call


def main():
    client = RtIpc()
    report = {}
    original = None
    scratch = None
    name = "C3Checkpoint_" + uuid.uuid4().hex[:8]

    def call(method, **params):
        return checked_call(client, method, **params)

    def phase(domain, which, **settings):
        return call("fluid.set_phase_grid", domain=domain, phase=which, **settings)

    try:
        systems = call("particle.list_systems")["systems"]
        original = next(item["id"] for item in systems if item["active"])
        scratch = call("particle.add_system", name=name)["id"]
        call("particle.set_active_system", system_id=scratch)
        call("fluid.create_domain", name=name, type="matter", domain_min=[20, 0, 20],
             domain_max=[26, 6, 26], voxel_size=0.1)
        call("fluid.set_param", domain=name, backend="vulkan", boundary="closed",
             preset="water", visible=False)
        call("fluid.set_whitewater", domain=name, enabled=False)
        phase(name, "gas", bounds_min=[20, 0, 20], bounds_max=[26, 6, 26], voxel=0.3)
        phase(name, "liquid", bounds_min=[21, 0, 21], bounds_max=[23, 2, 23], voxel=0.1)
        before = call("fluid.get_phase_grids", domain=name)
        try:
            phase(name, "liquid", bounds_min=[23, 2, 23], bounds_max=[21, 0, 21], voxel=-1)
        except RtIpcError:
            pass
        else:
            raise AssertionError("invalid phase bounds were accepted")
        after = call("fluid.get_phase_grids", domain=name)
        for which in ("gas", "liquid"):
            for key in ("inherit", "requested_bounds_min", "requested_bounds_max", "requested_voxel"):
                assert before[which][key] == after[which][key], (which, key)
        report["invalid_request_atomic"] = True
        call("flow_source.create", name=name + "Gas", domain=name, phase="gas",
             position=[24, 3, 24], radius=0.3, density=1.0, temperature=0.5)
        call("fluid.seed", domain=name, seed_min=[21.5, 0.5, 21.5],
             seed_max=[21.9, 0.9, 21.9], particles_per_cell=8, replace=True, persistent=False)
        call("fluid.step", dt=1 / 120)
        grids = call("fluid.get_phase_grids", domain=name)
        stats = call("fluid.step_stats", domain=name)
        digest = call("fluid.state_digest", domain=name)
        report["distinct_vulkan"] = {"grids": grids, "stats": stats,
                                      "particles": digest["particles"]}
        assert grids["gas"]["voxel"] > grids["liquid"]["voxel"]
        assert grids["gas"]["bounds_max"] != grids["liquid"]["bounds_max"]
        assert stats["measured"] and stats["p2g_on_gpu"] and stats["pressure_on_gpu"]
        assert stats["resolution"] == grids["liquid"]["resolution"]
        assert stats["full_grid_cells"] == grids["liquid"]["cells"]
        assert digest["particles"] > 0
        report["distinct_vulkan"] = {"grids": grids, "stats": stats,
                                      "particles": digest["particles"]}
        # CPU fallback uses the same layouts and freshly seeded scratch state.
        call("fluid.set_param", domain=name, backend="cpu")
        # Reseeding replaces particles, but retains solved grid velocities.
        # Reset this disposable runtime for an independent backend reference.
        phase(name, "liquid", bounds_min=[21, 0, 21], bounds_max=[23, 2, 23], voxel=0.11)
        phase(name, "liquid", bounds_min=[21, 0, 21], bounds_max=[23, 2, 23], voxel=0.1)
        call("fluid.seed", domain=name, seed_min=[21.5, 0.5, 21.5],
             seed_max=[21.9, 0.9, 21.9], particles_per_cell=8, replace=True, persistent=False)
        call("fluid.step", dt=1 / 120)
        cpu = call("fluid.state_digest", domain=name)
        error = max(abs(a - b) for a, b in zip(cpu["centroid"], digest["centroid"]))
        report["cpu_reference"] = cpu
        assert cpu["particles"] == digest["particles"]
        report["cpu_gpu_parity_passed"] = error < 1e-4
        report["cpu_centroid_error_m"] = error
        call("gas.set_settings", domain=name, resource_budget_mb=128)
        phase(name, "gas", bounds_min=[20, 0, 20], bounds_max=[26, 6, 26], voxel=0.025)
        phase(name, "liquid", bounds_min=[21, 0, 21], bounds_max=[23, 2, 23], voxel=0.015)
        budget = call("fluid.get_phase_grids", domain=name)
        assert budget["estimated_working_bytes"] <= budget["budget_mb"] * 1024 * 1024
        assert budget["gas"]["budget_clamped"] or budget["liquid"]["budget_clamped"]
        report["combined_budget"] = budget
        phase(name, "gas", inherit=True)
        phase(name, "liquid", inherit=True)
        inherited = call("fluid.get_phase_grids", domain=name)
        assert inherited["gas"]["inherit"] and inherited["liquid"]["inherit"]
        report["inherit_reset"] = True
        report["legacy_absent_phases"] = []
        for kind, missing in (("gas", "liquid"), ("fluid", "gas")):
            legacy = name + "_" + kind
            call("fluid.create_domain", name=legacy, type=kind, domain_min=[30, 0, 30],
                 domain_max=[32, 2, 32], voxel_size=0.2)
            query = call("fluid.get_phase_grids", domain=legacy)
            assert not query[missing]["present"]
            try:
                phase(legacy, missing, inherit=True)
            except RtIpcError:
                pass
            else:
                raise AssertionError("absent phase accepted")
            report["legacy_absent_phases"].append(kind)
        report["phase_contracts_passed"] = True
        report["passed"] = report["cpu_gpu_parity_passed"]
    finally:
        if original is not None:
            call("particle.set_active_system", system_id=original)
        if scratch is not None:
            call("particle.remove_system", system_id=scratch)
            report["scratch_removed"] = True
        client.close()
        path = Path(__file__).resolve().parents[2] / ".tmp/matter_checkpoint_live.json"
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
