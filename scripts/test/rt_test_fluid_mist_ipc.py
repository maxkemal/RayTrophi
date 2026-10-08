"""Live acceptance for mist production, gas transfer and physical transport."""

import json
import uuid
from pathlib import Path

from rt_ipc import RtIpc


DT = 1.0 / 60.0


def main():
    client = RtIpc()
    suffix = uuid.uuid4().hex[:8]
    fluid = "MistFluid_" + suffix
    gas = "MistGas_" + suffix
    source = "MistCarrier_" + suffix
    created = []
    source_created = False
    report = {"fluid": fluid, "gas": gas, "steps": []}
    output = Path(".tmp/fluid_mist_live.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = client.call("fluid.list_domains")["domains"]
        assert not existing, "mist acceptance needs an otherwise empty scene"

        for name, kind in ((gas, "gas"), (fluid, "fluid")):
            client.call("fluid.create_domain", name=name, type=kind,
                        domain_min=[10.0, 0.0, 0.0],
                        domain_max=[12.0, 2.0, 2.0], voxel_size=0.1)
            created.append(name)
            client.call("fluid.set_param", domain=name, backend="vulkan",
                        boundary="closed" if kind == "fluid" else "open",
                        visible=False)

        client.call("gas.set_settings", domain=gas, fire_enabled=True,
                    ignition_temperature=0.05, burn_rate=1.0,
                    heat_release=1.0, smoke_generation=0.2,
                    buoyancy_heat=0.0, buoyancy_density=0.0,
                    turbulence_strength=0.0, vorticity=0.0)
        # The chemistry is the domain's substance (2026-10-07).
        client.call("fluid.set_param", domain=fluid, default_substance="Gasoline")
        client.call("fluid.set_combustion", domain=fluid, enabled=True,
                    auto_ignite=True, ignition_temperature=0.05,
                    evaporation_rate=6.0, surface_fuel_capacity=4.0,
                    heat_release=1.0, smoke_yield=0.2,
                    surface_cooling=0.0)
        client.call("flow_source.create", name=source, domain=gas,
                    source_mode="point", position=[11.0, 0.8, 1.0],
                    radius=1.1, density=0.0, temperature=30.0, fuel=0.0,
                    velocity=[3.0, 0.0, 0.0], velocity_coupling=20.0)
        source_created = True
        client.call("fluid.seed", domain=fluid,
                    seed_min=[10.45, 0.35, 0.45],
                    seed_max=[11.55, 1.15, 1.55],
                    particles_per_cell=4, replace=True, persistent=True)

        max_mist = 0
        fog_live = False
        drag_traced = False
        exchange_seen = False
        mist_exchange_seen = False
        mist_coupling_seen = False
        last_exchange = None
        last_mist_exchange = None
        gas_inventory = None
        first_mist_centroid_x = None
        max_mist_centroid_x = None
        initial_particles = None
        min_particles = None
        for step in range(1, 41):
            client.call("fluid.step", dt=DT)
            info = client.call("fluid.get", domain=fluid)
            if initial_particles is None:
                initial_particles = info["particle_count"]
            min_particles = (info["particle_count"] if min_particles is None
                             else min(min_particles, info["particle_count"]))
            labels = info["particle_labels"]["primary"]
            mist = labels["mist"]
            max_mist = max(max_mist, mist)
            fog_rows = [row for row in info["views"] if row["view"] == "fog"]
            fog_live = fog_live or any(
                row["live"] and "mist" in row["labels"] and
                row["particles"] >= mist > 0 for row in fog_rows)
            couplings = client.call("sim_graph.couplings")
            drag_traced = drag_traced or any(
                row.get("coupling") == "mist_gas_drag"
                for row in couplings.get("actual", []))
            mist_coupling_seen = mist_coupling_seen or any(
                row.get("coupling") == "mist_to_gas"
                for row in couplings.get("actual", []))
            ledger = client.call("matter.exchanges")
            vaporization = [row for row in ledger["exchanges"]
                            if row["kind"] == "vaporization" and
                            row["source"] == fluid and row["target"] == gas]
            if vaporization:
                last_exchange = vaporization[-1]
                assert last_exchange["source_mass_kg"] > 0.0
                assert abs(last_exchange["target_mass_kg"] -
                           last_exchange["source_mass_kg"]) < 1.0e-9
                row_energy_error = (
                    last_exchange["target_energy_j"] +
                    last_exchange["latent_required_j"] -
                    last_exchange["source_energy_j"] -
                    last_exchange["latent_accounted_j"])
                assert abs(row_energy_error) < 1.0e-5
                assert abs(ledger["mass_error_kg"]) < 1.0e-9
                assert abs(ledger["energy_error_j"]) < 1.0e-5
                gas_info = client.call("fluid.get", domain=gas)
                gas_inventory = {
                    "mass_kg": gas_info["gas_phase_mass_kg"],
                    "energy_j": gas_info["gas_phase_energy_j"],
                }
                report["last_exchange"] = last_exchange
                report["gas_inventory"] = gas_inventory
                report["inventory_minus_exchange"] = {
                    "mass_kg": (gas_inventory["mass_kg"] -
                                last_exchange["target_mass_kg"]),
                    "energy_j": (gas_inventory["energy_j"] -
                                  last_exchange["target_energy_j"]),
                }
                mass_rounding = max(
                    1.0e-6, last_exchange["target_mass_kg"] * 2.0e-6)
                assert gas_inventory["mass_kg"] >= (
                    last_exchange["target_mass_kg"] - mass_rounding)
                assert gas_inventory["energy_j"] >= (
                    last_exchange["target_energy_j"] - 1.0e-3)
                exchange_seen = True
            mist_exchanges = [row for row in ledger["exchanges"]
                              if row["kind"] == "mist_to_gas" and
                              row["source"] == fluid and row["target"] == gas]
            if mist_exchanges:
                last_mist_exchange = mist_exchanges[-1]
                assert last_mist_exchange["source_mass_kg"] > 0.0
                assert abs(last_mist_exchange["target_mass_kg"] -
                           last_mist_exchange["source_mass_kg"]) < 1.0e-9
                assert abs(ledger["mass_error_kg"]) < 1.0e-9
                assert abs(ledger["energy_error_j"]) < 1.0e-5
                mist_exchange_seen = True

            gas_info = client.call("fluid.get", domain=gas)
            centroid = gas_info["gas_phase_mass_centroid"]
            if mist_exchange_seen and gas_info["gas_phase_active_cells"] > 0:
                if first_mist_centroid_x is None:
                    first_mist_centroid_x = centroid[0]
                max_mist_centroid_x = max(
                    max_mist_centroid_x or centroid[0], centroid[0])
            report["steps"].append({
                "step": step,
                "particles": info["particle_count"],
                "body": labels["body"],
                "spray": labels["spray"],
                "mist": mist,
                "fog_live": bool(fog_rows and fog_rows[0]["live"]),
                "drag_traced": drag_traced,
                "gas_phase_active_cells": gas_info["gas_phase_active_cells"],
                "gas_phase_mass_centroid": centroid,
            })
            centroid_moved = (first_mist_centroid_x is not None and
                              max_mist_centroid_x is not None and
                              max_mist_centroid_x > first_mist_centroid_x + 0.005)
            if (max_mist > 0 and fog_live and drag_traced and exchange_seen and
                    mist_exchange_seen and mist_coupling_seen and
                    centroid_moved):
                break

        report.update({"max_mist": max_mist, "fog_live": fog_live,
                       "drag_traced": drag_traced,
                       "exchange_seen": exchange_seen,
                       "mist_exchange_seen": mist_exchange_seen,
                       "mist_coupling_seen": mist_coupling_seen,
                       "last_exchange": last_exchange,
                       "last_mist_exchange": last_mist_exchange,
                       "gas_inventory": gas_inventory,
                       "first_mist_centroid_x": first_mist_centroid_x,
                       "max_mist_centroid_x": max_mist_centroid_x})
        assert max_mist > 0, "combustible liquid never produced low-mass mist"
        assert fog_live, "mist did not resolve to a live fog view"
        assert drag_traced, "solver did not report mist_gas_drag"
        assert exchange_seen, "liquid mass loss did not enter the matter ledger"
        assert mist_exchange_seen, "residual mist did not enter the gas phase"
        assert mist_coupling_seen, "solver did not report mist_to_gas"
        assert min_particles < initial_particles, "mist parcels were not removed"
        assert first_mist_centroid_x is not None
        assert max_mist_centroid_x > first_mist_centroid_x + 0.005, (
            "physical gas inventory did not advect with the carrier velocity")
        report["passed"] = True
        print(json.dumps(report, indent=2))
    except Exception as error:
        report["error"] = repr(error)
        raise
    finally:
        cleanup_errors = []
        if source_created:
            try:
                client.call("flow_source.remove", name=source)
            except Exception as error:
                cleanup_errors.append(str(error))
        for name in reversed(created):
            try:
                client.call("fluid.remove_domain", domain=name)
            except Exception as error:
                cleanup_errors.append(str(error))
        report["cleanup_errors"] = cleanup_errors
        report["remaining_domains"] = [
            row["name"] for row in client.call("fluid.list_domains")["domains"]]
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        client.close()
        print("Report:", output, "cleanup errors:", cleanup_errors)


if __name__ == "__main__":
    main()
