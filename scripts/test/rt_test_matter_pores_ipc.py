"""Read-only C5 probe after a user-built, paused Closed Vulkan Water/Sand step.

Run externally, never in the application's script workspace:
python scripts/test/rt_test_matter_pores_ipc.py DOMAIN --expect absorption
Use --expect drainage after enabling drainage; plain invocation permits no contact.
This checks the last publication, not resolution convergence or cache round trips.
"""
import argparse
import math

from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--expect", choices=["absorption", "drainage"])
    args = parser.parse_args()
    client = RtIpc()
    try:
        inventory = checked_call(client, "fluid.matter_models", domain=args.domain)
        pore = inventory["pore_exchange"]
        assert pore["settings"]["enabled"], "Enable Pore Water first"
        assert pore["measured"] and not pore["held"], pore["status"]
        execution = inventory["mixed_execution"]
        assert not execution["step_held"], execution["status"]
        assert all(execution[key] for key in ["p2g_on_gpu", "pressure_on_gpu", "g2p_on_gpu"])
        for key in ["pore_water_kg", "capacity_kg", "maximum_saturation", "absorbed_kg",
                    "drained_kg", "mass_error_kg", "momentum_error_kg_m_s",
                    "thermal_energy_error_j"]:
            assert math.isfinite(pore[key]), key
        assert 0 <= pore["maximum_saturation"] <= 1.00001
        assert 0 <= pore["pore_water_kg"] <= pore["capacity_kg"] * 1.00001 + 1e-12
        assert pore["absorbed_kg"] >= 0 and pore["drained_kg"] >= 0
        assert pore["drainage_births"] >= 0 and pore["drainage_refills"] >= 0
        assert (pore["drained_kg"] > 0) == (
            pore["drainage_births"] + pore["drainage_refills"] > 0)
        if args.expect:
            key = "absorbed_kg" if args.expect == "absorption" else "drained_kg"
            assert pore[key] > 0, f"No {args.expect} in the last published frame"
        print("PASS C5 GPU publication, finite inventory, saturation and drainage batches/refills")
        print(pore)
    finally:
        client.close()


if __name__ == "__main__":
    main()
