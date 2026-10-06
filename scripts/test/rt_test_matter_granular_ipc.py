"""G2 mechanics probe for a paused user-built Matter scene; read-only by default.
--step advances all active domains once through external IPC, without changing settings.
This validates publication/CFL, not repose, DEM rolling or real support pressure.
"""
import argparse
import math

from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--expect-load", action="store_true")
    parser.add_argument("--expect-dry", action="store_true")
    parser.add_argument("--expect-unbonded", action="store_true")
    parser.add_argument("--step", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    try:
        control = checked_call(client, "sim.control_state")
        assert not control["playing"], "Pause first"
        if args.step:
            checked_call(client, "fluid.step", dt=1.0 / 60.0)
        inventory = checked_call(client, "fluid.matter_models", domain=args.domain)
        domain = checked_call(client, "fluid.get", domain=args.domain)
        mechanics = inventory["granular_mechanics"]
        assert mechanics["measured"], "Run a fresh solver step; cached/default counters are not proof"
        assert inventory["models"][1]["particles"] > 0
        assert mechanics["load_proxy"] == "granular_extent_rho_g_h"
        assert not mechanics["contact_pressure_measured"]
        for key in ["overburden_estimate_pa", "young_modulus_for_load_pa", "strain_rate_per_s"]:
            assert math.isfinite(mechanics[key]) and mechanics[key] >= 0, key
        assert domain["granular_invalid"] == 0
        assert math.isclose(domain["granular_effective_young_modulus"],
                            domain["granular_requested_young_modulus"], rel_tol=1e-6)
        assert domain["granular_solver_substeps"] >= domain["granular_required_substeps"]
        assert domain["granular_solver_substeps"] >= max(mechanics["wave_substeps"],
                                                         mechanics["strain_substeps"])
        if args.expect_load:
            assert mechanics["overburden_estimate_pa"] > 0
            assert mechanics["young_modulus_for_load_pa"] > 0
        if args.expect_dry:
            assert inventory["pore_exchange"]["pore_water_kg"] == 0
            assert inventory["acceptance_metrics"]["exactly_dry_particles"] == \
                inventory["models"][1]["particles"]
        if args.expect_unbonded:
            assert domain["granular_cohesion"] == 0 and domain["granular_tensile_cutoff"] == 0
            assert not domain["granular_rebonding"]
        assert checked_call(client, "sim.control_state") == control, "Control changed"
        print("PASS granular-only load/CFL publication and authored stiffness")
        print(mechanics)
    finally:
        client.close()


if __name__ == "__main__":
    main()
