"""Read-only post-build probe; run externally with an existing Matter domain.

python scripts/test/rt_test_material_resolution_ipc.py "Domain name"
Does not create/reset/step/reseed domains or change any material settings.
"""

import argparse
import math

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("domain")
    args = parser.parse_args()
    client = RtIpc()
    try:
        result = client.call("fluid.matter_models", domain=args.domain)
        report = result["material_resolution"]
        assert report["schema_version"] == 1
        assert report["execution_measured"] is False
        assert report["backend_executed"] is None
        assert report["mpm_parameter_scope"] == "shared_domain_material"
        assert report["phase_mechanics"]["scope"] == "particle_mechanics_not_surface_object_mechanics"
        viscosity = report["viscosity"]
        for field in ("requested_m2_s", "policy_effective_m2_s", "configured_m2_s"):
            assert math.isfinite(viscosity[field]) and viscosity[field] >= 0
        assert viscosity["reason"] in {
            "active", "granular_skeleton", "zero_requested",
            "below_resolution_threshold", "thermal_curve_base",
        }
        carriers = report["carriers"]
        if report["carriers_measured"]:
            assert sum(row["particles"] for row in carriers) == sum(
                row["particles"] for row in result["models"])
        else:
            assert carriers == []
        for row in carriers:
            assert sum(row["phase_counts"].values()) == row["particles"]
            assert sum(row["phase_basis_counts"].values()) == row["particles"]
            assert sum(row["selection_basis_counts"].values()) == row["particles"]
            assert 0 <= row["model_mismatch_particles"] <= row["particles"]
            assert row["temperature_known_particles"] <= row["particles"]
            assert row["fusion_known_particles"] <= row["particles"]
            if row["mean_temperature_k"] is not None:
                assert row["mean_temperature_k"] > 0
            if row["mean_fusion_solid_fraction"] is not None:
                assert 0 <= row["mean_fusion_solid_fraction"] <= 1
        print("PASS read-only material resolution:", report["default_substance"],
              viscosity["reason"], len(carriers), "carrier groups")
    finally:
        client.close()


if __name__ == "__main__":
    main()
