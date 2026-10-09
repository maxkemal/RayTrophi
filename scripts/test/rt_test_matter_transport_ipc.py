"""T4 contact smoke; external IPC after the user build, in an empty paused scene.

Checks that Soil MPM carriers are not silently born as DEM spheres in a grain
domain, GPU contact exchanges a nonzero balanced impulse, and changing live
transport ownership is rejected without a partial substance edit. No save.
"""

import uuid
import math
import json
import argparse

from rt_ipc import RtIpc, RtIpcError
from rt_grain_material import install_grain_material  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel-timings", action="store_true")
    parser.add_argument("--owners-only", action="store_true",
                        help="Only DEM/MPM ownership, shared clock and balanced contact; no fluid arm")
    args = parser.parse_args()
    client = RtIpc()
    install_grain_material(client)  # old grain keys -> grain substance
    suffix = uuid.uuid4().hex[:8]
    domain = "T4Owners_" + suffix
    soil = "T4Soil_" + suffix
    created_domain = False
    created_soil = False
    sources = []
    timing_restore = None
    timing_start = None
    try:
        assert not client.call("fluid.list_domains")["domains"], "Use an empty scene"
        assert not client.call("sim.control_state")["playing"], "Pause the timeline"
        client.call("substance.derive", name=soil, based_on="Soil")
        created_soil = True
        client.call("fluid.create_domain", name=domain, type="matter",
                    domain_min=[-0.5, 0, -0.5], domain_max=[0.5, 1, 0.5], voxel_size=0.1)
        created_domain = True
        client.call("fluid.set_param", domain=domain, backend="vulkan", boundary="closed",
                    default_substance=soil, solid_phase=False, visible=False,
                    thermal_liquid_enabled=False)
        client.call("fluid.set_grain_settings", domain=domain)
        for material, x, vx in (("Sand", -0.025, 0.3), (soil, 0.025, -0.3)):
            name = "T4Source_" + material + "_" + suffix
            client.call("flow_source.create", name=name, domain=domain, phase="liquid",
                        source_mode="point", position=[x, 0.6, 0], radius=0.0001,
                        fluid_substance=material, velocity=[vx, 0, 0],
                        fluid_velocity_spread=0.0, fluid_particles_per_second=60.0,
                        use_particle_limit=True, max_emitted_particles=1,
                        fluid_temperature_override=True, fluid_temperature_kelvin=293.15)
            sources.append(name)
        if args.kernel_timings:
            timing_start = client.call("perf.gpu_kernel_timings", reset=False)
            if timing_start["supported"]:
                timing_restore = timing_start["enabled"]
                client.call("perf.set_gpu_kernel_timing", enabled=True)
                timing_start = client.call("perf.gpu_kernel_timings", reset=False)
        client.call("fluid.step", dt=1.0 / 60.0)
        models = client.call("fluid.matter_models", domain=domain)
        owners = models["transport_owners"]
        assert owners["grain"] == 1 and owners["mpm"] == 1, owners
        assert owners["fluid"] == 0 and owners["obstacle"] == 0, owners
        assert owners["ready"], owners
        assert not models["mixed_execution"]["step_held"], models["mixed_execution"]
        contact = models["grain_diagnostics"]["runtime"]["mpm_contact"]
        runtime = models["grain_diagnostics"]["runtime"]
        clock = runtime["common_clock"]
        assert clock["enabled"] and contact["schedule"] == "shared_transport_contact_advection", runtime
        assert 0 < clock["continuum_grid_steps"] <= clock["transport_steps"], clock
        assert contact["parcels"] == 1 and contact["events"] > 0, contact
        first_contact = dict(contact)
        first_cost = {key: runtime[key] for key in ["substeps", "dispatches", "working_set_bytes", "upload_bytes", "download_bytes", "host_ms"]}
        magnitude = contact["grain_impulse_magnitude_n_s"]
        assert math.isfinite(magnitude) and magnitude > 1e-8, contact
        assert contact["momentum_residual_n_s"] < max(1e-7, magnitude * 1e-4), contact
        assert models["grain_diagnostics"]["liquid"]["coupled_grains"] == 0

        # Live ownership is fixed at birth; refusal must also preserve the
        # unrelated density field in the same patch.
        before = client.call("substance.get", name=soil)
        try:
            client.call("substance.set", name=soil,
                        fields={"density": 1500.0, "granular_transport": "dem"})
        except RtIpcError as error:
            assert "reset" in str(error).lower(), str(error)
        else:
            raise AssertionError("live transport ownership changed without reset")
        assert client.call("substance.get", name=soil) == before
        if args.owners_only:
            assert runtime["sleeping_grains"] == 0, runtime
            print(json.dumps({"owners": owners, "contact": first_contact,
                              "cost": first_cost, "common_clock": clock}))
            print("PASS: DEM and MPM in one Matter domain, shared clock and balanced contact impulse")
            return
        # Introduce the third owner. It must not consume the skeleton as water
        # or send fluid drag to MPM carriers. All three counts remain separate.
        water_source = "T4Water_" + suffix
        client.call("flow_source.create", name=water_source, domain=domain, phase="liquid",
                    source_mode="point", position=[0.3, 0.6, 0], radius=0.0001,
                    fluid_substance="Water", velocity=[-4, 0, 0], fluid_velocity_spread=0.0,
                    fluid_particles_per_second=60.0, use_particle_limit=True,
                    max_emitted_particles=1, fluid_temperature_override=True,
                    fluid_temperature_kelvin=293.15)
        sources.append(water_source)
        saw_contact = False
        saw_dry_after_contact = False
        drift_samples = []
        for _ in range(12):
            client.call("fluid.step", dt=1.0 / 60.0)
            models = client.call("fluid.matter_models", domain=domain)
            owners = models["transport_owners"]
            assert (owners["grain"], owners["mpm"], owners["fluid"]) == (1, 1, 1), owners
            assert not models["mixed_execution"]["step_held"], models["mixed_execution"]
            runtime = models["grain_diagnostics"]["runtime"]
            assert runtime["common_clock"]["enabled"], runtime
            assert runtime["common_clock"]["liquid_reaction_on_gpu"], runtime
            assert runtime["common_clock"]["liquid_support"] == "device_rebin_each_tick", runtime
            coupling = models["grain_diagnostics"]["liquid"]
            assert coupling["momentum_residual_n_s"] < 1e-6, coupling
            assert coupling["liquid_parcels"] == 1 and not coupling["volume_exclusion"], coupling
            active = coupling["coupled_grains"] > 0
            drift_samples.append({"active": active,
                                  "drag_impulse": coupling["drag_impulse_n_s"],
                                  "residual": coupling["momentum_residual_n_s"]})
            saw_contact = saw_contact or active
            if saw_contact and not active:
                saw_dry_after_contact = True
                break
        assert saw_contact and saw_dry_after_contact, drift_samples
        native_gpu = None
        if args.kernel_timings and timing_start["supported"]:
            timing_end = client.call("perf.gpu_kernel_timings", reset=False)
            baseline = {row["kernel"]: row for row in timing_start["kernels"]}
            kernels = []
            for row in timing_end["kernels"]:
                before = baseline.get(row["kernel"], {"ms": 0.0, "calls": 0})
                calls = row["calls"] - before["calls"]
                if calls > 0:
                    kernels.append({"kernel": row["kernel"], "calls": calls,
                                    "ms": row["ms"] - before["ms"]})
            native_gpu = {"supported": timing_end["supported"], "kernels": kernels,
                          "timestamp_instrumentation_enabled": True}
        elif args.kernel_timings:
            native_gpu = {"supported": False, "kernels": [],
                          "timestamp_instrumentation_enabled": False}
        print(json.dumps({"contact": first_contact, "cost": first_cost,
                          "drift_samples": drift_samples, "native_gpu": native_gpu,
                          "cost_clock": "host_wall_and_separate_native_gpu_timestamps"}))
        print("PASS: shared clock, dynamic fluid enters/leaves support, impulse balance, three owners")
    finally:
        try:
            if timing_restore is not None:
                client.call("perf.set_gpu_kernel_timing", enabled=timing_restore)
            for name in reversed(sources):
                client.call("flow_source.remove", name=name)
            if created_domain:
                client.call("fluid.remove_domain", domain=domain)
            if created_soil:
                client.call("substance.remove", name=soil)
        finally:
            client.close()


if __name__ == "__main__":
    main()
