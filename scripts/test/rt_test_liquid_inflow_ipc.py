"""External IPC acceptance after the USER build, with Studio paused in an empty scene.

Run from a separate terminal: python scripts/test/rt_test_liquid_inflow_ipc.py
Creates and removes its own domain. Does not launch Studio or compile anything.
"""
import math
import argparse
from rt_ipc import RtIpc, RtIpcError

DOMAIN = "T_LiquidInflow"
SOURCE = "T_LiquidInflow_Source"
DT = 0.01
RADIUS = 0.05


def main(drain_only=False):
    call = RtIpc().call
    assert not call("sim.control_state")["playing"], "Pause first"
    assert not call("fluid.list_domains")["domains"], "Use an empty scene"
    created = False
    source_created = False
    try:
        call("fluid.create_domain", name=DOMAIN, type="fluid",
             domain_min=[-.15, -2, -.15], domain_max=[.15, .3, .15], voxel_size=.01)
        created = True
        call("fluid.set_param", domain=DOMAIN, enabled=True, visible=False,
             backend="vulkan", boundary="closed", default_substance="Water",
             thermal_liquid_enabled=False)
        call("fluid.settle", domain=DOMAIN, on_seed=False, request=False)
        call("flow_source.create", name=SOURCE, domain=DOMAIN, phase="liquid",
             liquid_emission="inflow", position=[0, .1, 0], velocity=[0, -1, 0],
             radius=RADIUS, fluid_velocity_spread=0, fluid_particles_per_second=0)
        source_created = True
        info = call("flow_source.get", name=SOURCE)
        assert info["liquid_emission"] == "inflow", info
        for value in ("typo", None, 1):
            try:
                call("flow_source.update", name=SOURCE, liquid_emission=value)
            except RtIpcError:
                pass
            else:
                raise AssertionError("invalid mode accepted: " + repr(value))
            assert call("flow_source.get", name=SOURCE)["liquid_emission"] == "inflow"
        try:
            call("flow_source.update", name=SOURCE, source_mode="mesh_surface")
        except RtIpcError:
            pass
        else:
            raise AssertionError("mesh inflow accepted")
        assert call("flow_source.get", name=SOURCE)["source_mode"] == "point"
        if not drain_only:
            fluxes = []
            for frame in range(60):
                call("fluid.step", dt=DT)
                info = call("flow_source.get", name=SOURCE)
                assert not info["liquid_emission_status"], info
                assert info["withheld_mass_kg_step"] == 0, info
                assert math.isclose(info["fluid_flux_target_m3_s"], math.pi * RADIUS**2,
                                    rel_tol=1e-5), info
                if frame >= 20:
                    fluxes.append(info["fluid_flux_m3_s"])
            ratio = sum(fluxes) / len(fluxes) / (math.pi * RADIUS**2)
            print("steady inflow Q / (A*speed):", ratio, flush=True)
            assert abs(ratio - 1) <= .05, ratio
        # Update resets/replays the simulation by design. Seed an independent
        # uniform-water inventory for a one-step complete-drain mass oracle.
        call("flow_source.update", name=SOURCE, liquid_emission="outflow",
             position=[0, -1, 0], radius=4)
        call("timeline.set_frame", frame=0)
        call("fluid.seed", domain=DOMAIN, seed_min=[-.04, -.5, -.04],
             seed_max=[.04, -.42, .04], particles_per_cell=8, replace=True)
        before = call("fluid.state_digest", domain=DOMAIN)["particles"]
        probes = call("fluid.particle_extremes", domain=DOMAIN, count=1)
        probe = probes["coldest"][0]
        parcel_mass = probe["rest_mass_kg"] * probe["mass_fraction"] + probe["pore_water_kg"]
        assert before > 0, before
        if parcel_mass <= 0:
            # Seeded parcels can defer rest-mass initialization until the first
            # solver step. The drain still owes their physical inventory.
            grid = call("fluid.sample_solid_phi", domain=DOMAIN, points=[[0, -.46, 0]])
            density = call("substance.get", name="Water")["fields"]["liquid_density"]
            parcel_mass = density * grid["voxel_m"]**3 / 8
            print("Seed rest mass deferred; physical parcel mass:", parcel_mass, flush=True)
        call("fluid.step", dt=DT)
        assert call("fluid.state_digest", domain=DOMAIN)["particles"] == 0
        info = call("flow_source.get", name=SOURCE)
        assert math.isclose(info["drained_mass_kg_total"], before * parcel_mass,
                            rel_tol=1e-5), info
        assert info["total_emitted_particles"] == 0, info
        print("PASS mode roundtrip, transactional refusals, outflow mass", flush=True)
    finally:
        try:
            if source_created:
                call("flow_source.remove", name=SOURCE)
        finally:
            if created:
                call("fluid.remove_domain", domain=DOMAIN)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--drain-only", action="store_true")
    main(parser.parse_args().drain_only)
