"""DEM physical-size acceptance. Run externally after a full user build.
Requires a paused empty scene. Creates/removes its own domain/source; no app launch.
"""
import math
import uuid
from rt_ipc import RtIpc, RtIpcError


def main():
    ipc = RtIpc()
    domain = "SizeProbe_" + uuid.uuid4().hex[:8]
    source = domain + "_Source"
    made_domain = made_source = False
    try:
        assert not ipc.call("sim.control_state")["playing"], "Pause timeline"
        assert not ipc.call("fluid.list_domains")["domains"], "Use an empty scene"
        ipc.call("fluid.create_domain", name=domain, type="matter",
                 domain_min=[-2, 0, -2], domain_max=[2, 6, 2], voxel_size=0.1)
        made_domain = True
        ipc.call("fluid.set_param", domain=domain, enabled=True, visible=True,
                 backend="vulkan", boundary="closed", default_substance="Sand",
                 solid_phase=False, thermal_liquid_enabled=False)
        ipc.call("fluid.set_pore_exchange", domain=domain, enabled=False,
                 wet_response_enabled=False, wet_appearance_enabled=False)
        ipc.call("flow_source.create", name=source, domain=domain, phase="liquid",
                 fluid_substance="Sand", source_mode="point", position=[0, 4, 0],
                 radius=0.5, velocity=[0, 0, 0], fluid_velocity_spread=0.0,
                 enabled=False, use_particle_limit=True, max_emitted_particles=24,
                 use_time_limit=False, fluid_particles_per_second=240)
        made_source = True
        results = []
        for maximum in (0.0, 0.02, 0.02):
            ipc.call("fluid.reset")
            ipc.call("fluid.set_grain_settings", domain=domain, radius_m=0.01,
                     radius_max_m=maximum, size_distribution="bimodal", size_seed=1337,
                     large_grain_fraction=0.5, fluid_coupling=False, max_substeps=4096)
            ipc.call("flow_source.update", name=source, enabled=True)
            for _ in range(8):
                ipc.call("fluid.step", dt=1/60)
            inventory = ipc.call("fluid.matter_models", domain=domain)
            assert not inventory["mixed_execution"]["step_held"], inventory
            diag = inventory["grain_diagnostics"]
            assert diag["contact_shader_revision"] >= 28
            size = diag["size_population"]
            assert size["count"] == 24, size
            mass = inventory["acceptance_metrics"]["granular"]["dry_mass_kg"]
            results.append((size, mass))
            before = ipc.call("fluid.grain_settings", domain=domain)
            try:
                ipc.call("fluid.set_grain_settings", domain=domain, size_seed=999)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Live size mutation must require reset")
            assert ipc.call("fluid.grain_settings", domain=domain) == before
        fixed, mixed, replay = results
        assert mixed == replay, "Reset/replay resampled physical sizes or mass"
        size = mixed[0]
        assert size["small_count"] > 0 and size["large_count"] > 0, size
        assert math.isclose(size["radius_min_m"], 0.01, rel_tol=1e-5)
        assert math.isclose(size["radius_max_m"], 0.02, rel_tol=1e-5)
        assert math.isclose(mixed[1] / fixed[1],
                            mixed[0]["sphere_volume_m3"] / fixed[0]["sphere_volume_m3"],
                            rel_tol=1e-5), results
        print("PASS: actual radius variation, cubic mass, deterministic replay, birth lock")
    finally:
        try:
            if made_source:
                ipc.call("flow_source.remove", name=source)
            if made_domain:
                ipc.call("fluid.remove_domain", domain=domain)
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
