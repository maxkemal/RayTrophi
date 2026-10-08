"""Live acceptance for Phase 3 liquid/gas boundary and freezing ledger."""

import uuid

from rt_ipc import RtIpc
from rt_domain_material import domain_material  # noqa: E402


DT = 1.0 / 60.0


def main():
    client = RtIpc()
    suffix = uuid.uuid4().hex[:8]
    fluid = "Phase3Fluid_" + suffix
    gas = "Phase3Gas_" + suffix
    source = "Phase3GasSource_" + suffix
    created_domains = []
    source_created = False
    try:
        existing = client.call("fluid.list_domains")["domains"]
        assert not existing, "Phase 3 acceptance needs an otherwise empty scene"

        for name, kind in ((gas, "gas"), (fluid, "fluid")):
            client.call(
                "fluid.create_domain", name=name, type=kind,
                domain_min=[30.0, 0.0, 0.0],
                domain_max=[31.0, 1.0, 1.0], voxel_size=0.1)
            created_domains.append(name)
            client.call(
                "fluid.set_param", domain=name, backend="vulkan",
                boundary="closed", visible=False)

        client.call(
            "flow_source.create", name=source, domain=gas,
            source_mode="point", position=[30.5, 0.5, 0.5], radius=0.2,
            density=0.02, temperature=0.0, fuel=0.0,
            velocity=[0.0, 0.0, 0.0], velocity_coupling=0.0)
        source_created = True

        material = domain_material(client.call, 'T: test_fluid_phase3_boundary_freeze', 'Wax',
            thermal_freeze_kelvin=400.0)
        client.call(
            "fluid.set_param", domain=fluid, default_substance=material,
            thermal_liquid_enabled=True,
            thermal_air_cooling_rate=0.0,
            thermal_contact_cooling_rate=0.0)
        client.call(
            "fluid.seed", domain=fluid,
            seed_min=[30.2, 0.01, 0.2], seed_max=[30.8, 0.18, 0.8],
            particles_per_cell=4, replace=True, persistent=False)

        boundary_seen = False
        freezing_seen = False
        last_stats = None
        last_exchange = None
        coupling_seen = False
        for step in range(1, 9):
            client.call("fluid.step", dt=DT)
            stats = client.call("gas.step_stats", domain=gas)
            ledger = client.call("matter.exchanges")
            couplings = client.call("sim_graph.couplings")
            assert stats["measured"], "gas solver did not step"

            if int(stats.get("liquid_boundary_cells", 0)) > 0:
                boundary_seen = True
                last_stats = stats
            rows = [row for row in ledger["exchanges"]
                    if row["kind"] == "freezing" and
                    row["source"] == fluid + ":liquid" and
                    row["target"] == fluid + ":solid"]
            if rows:
                freezing_seen = True
                last_exchange = rows[-1]
                assert last_exchange["source_mass_kg"] > 0.0
                assert abs(last_exchange["target_mass_kg"] -
                           last_exchange["source_mass_kg"]) < 1.0e-9
                assert abs(ledger["mass_error_kg"]) < 1.0e-9
                assert abs(ledger["energy_error_j"]) < 1.0e-5
            coupling_seen = coupling_seen or any(
                row.get("coupling") == "liquid_moving_boundary"
                for row in couplings.get("actual", []))
            print(
                "step={} boundary_cells={} frozen={} events={}".format(
                    step, stats.get("liquid_boundary_cells", 0),
                    client.call("fluid.get", domain=fluid).get(
                        "thermal_frozen_particles", 0),
                    len(ledger["exchanges"])))

        assert boundary_seen, "overlapping liquid never became a gas boundary"
        assert coupling_seen, "sim graph never reported liquid_moving_boundary"
        assert freezing_seen, "thermal freeze produced no freezing ledger event"
        assert len(last_stats["liquid_boundary_mean_velocity"]) == 3
        print("PASS: Phase 3 moving boundary + freezing ledger")
        print("boundary_cells:", last_stats["liquid_boundary_cells"])
        print("boundary_mean_velocity:",
              last_stats["liquid_boundary_mean_velocity"])
        print("frozen_mass_kg:", last_exchange["source_mass_kg"])
    finally:
        if source_created:
            client.call("flow_source.remove", name=source)
        for name in reversed(created_domains):
            client.call("fluid.remove_domain", domain=name)
        client.close()


if __name__ == "__main__":
    main()
