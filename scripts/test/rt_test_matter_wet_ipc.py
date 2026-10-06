"""External read-only C6 probe for a user-built, paused wet Matter scene.
Does not prove angle of repose, spatial appearance, or resolution convergence.
"""
import argparse
import math
from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--expect-wet", action="store_true")
    args = parser.parse_args()
    client = RtIpc()
    try:
        before = checked_call(client, "sim.control_state")
        assert not before["playing"], "Pause first"
        info = checked_call(client, "fluid.matter_models", domain=args.domain)
        wet = info["wet_response"]
        assert wet["enabled"], "Enable wet_response_enabled first"
        assert wet["field_sampled"] and wet["gpu_step_measured"], wet
        assert not info["mixed_execution"]["step_held"]
        assert wet["model"] == "empirical_cell_head"
        for key in ["maximum_pore_pressure_pa", "maximum_capillary_cohesion_pa"]:
            assert math.isfinite(wet[key]) and wet[key] >= 0, key
        assert wet["wet_particles"] >= 0
        assert 0 <= info["pore_exchange"]["maximum_saturation"] <= 1.00001
        if args.expect_wet:
            assert wet["wet_particles"] > 0, wet
        after = checked_call(client, "sim.control_state")
        assert before == after, "Simulation control changed during query"
        print("PASS C6 finite canonical wet field and mixed GPU publication")
        print(wet)
    finally:
        client.close()


if __name__ == "__main__":
    main()
