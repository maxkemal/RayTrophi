"""Validate the per-step physical matter-exchange ledger contract."""

import argparse
import math
import uuid

from rt_ipc import RtIpc


TOTAL_FIELDS = (
    "source_mass_kg", "target_mass_kg", "mass_error_kg",
    "source_energy_j", "target_energy_j", "latent_required_j",
    "latent_accounted_j", "energy_error_j",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-event", action="store_true")
    parser.add_argument("--step", action="store_true",
                        help="advance the simulation once before reading")
    args = parser.parse_args()

    client = RtIpc()
    temporary_domain = "LedgerProbe_" + uuid.uuid4().hex[:8]
    created = False
    try:
        if args.step:
            existing = client.call("fluid.list_domains")["domains"]
            assert not existing, "--step requires an otherwise empty scene"
            client.call("fluid.create_domain", name=temporary_domain, type="fluid",
                        domain_min=[20.0, 0.0, 0.0],
                        domain_max=[21.0, 1.0, 1.0], voxel_size=0.2)
            created = True
            client.call("fluid.seed", domain=temporary_domain,
                        seed_min=[20.3, 0.2, 0.3],
                        seed_max=[20.7, 0.6, 0.7],
                        particles_per_cell=2, replace=True,
                        persistent=False)
            client.call("fluid.step", dt=1.0 / 60.0)
        report = client.call("matter.exchanges")
        assert type(report["traced"]) is bool
        assert type(report["step"]) is int and report["step"] >= 0
        if args.step:
            assert report["step"] > 0, "ledger did not begin a simulation step"
        assert isinstance(report["exchanges"], list)
        assert all(math.isfinite(report[key]) for key in TOTAL_FIELDS)
        assert math.isclose(
            report["mass_error_kg"],
            report["target_mass_kg"] - report["source_mass_kg"],
            rel_tol=0.0, abs_tol=1e-9)
        expected_energy_error = (
            report["target_energy_j"] + report["latent_required_j"] -
            report["source_energy_j"] - report["latent_accounted_j"])
        assert math.isclose(report["energy_error_j"], expected_energy_error,
                            rel_tol=0.0, abs_tol=1e-6)

        for row in report["exchanges"]:
            assert row["kind"] in {
                "melting", "freezing", "vaporization", "combustion",
                "pyrolysis", "mist_to_gas", "condensation",
            }
            assert row["source_mass_kg"] >= 0.0
            assert row["target_mass_kg"] >= 0.0
            assert len(row["source_momentum_kg_m_s"]) == 3
            assert len(row["target_momentum_kg_m_s"]) == 3
        if args.require_event:
            assert report["exchanges"], "no transfer ran in the latest step"

        print("PASS: matter exchange ledger contract")
        print("step:", report["step"], "events:", len(report["exchanges"]))
        print("mass_error_kg:", report["mass_error_kg"])
        print("energy_error_j:", report["energy_error_j"])
    finally:
        if created:
            client.call("fluid.remove_domain", domain=temporary_domain)
        client.close()


if __name__ == "__main__":
    main()
