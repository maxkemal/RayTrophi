"""Read-only mixed GPU acceptance probe after user build and simulation steps.

Use two overlapping granular/fluid emitters in one Matter domain on Vulkan.
Run externally, never from the application's embedded script workspace:
    python scripts/test/rt_test_matter_mixed_gpu_ipc.py "Physics Domain 1" --contact
"""
import argparse
import math

from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    parser.add_argument("--contact", action="store_true",
                        help="Require measured contact pairs in an overlapping scene")
    args = parser.parse_args()
    client = RtIpc()
    try:
        data = checked_call(client, "fluid.matter_models", domain=args.domain)
        assert data["measured"], "Synchronize and step the Matter domain first"
        assert data["mixed_transport_available"]
        assert data["models"][0]["particles"] > 0, "Fluid carrier is missing"
        assert data["models"][1]["particles"] > 0, "Granular carrier is missing"
        assert data["models"][2]["particles"] == 0, "Elastic is outside this test scope"
        execution = data["mixed_execution"]
        assert data["mixed_transport_ready"], execution["status"]
        assert not execution["step_held"], execution["status"]
        assert execution["common_substeps"] >= 1
        assert execution["working_set_bytes"] > 0
        assert "Mixed Vulkan" in execution["status"], execution["status"]
        for stage in ["p2g_on_gpu", "pressure_on_gpu", "g2p_on_gpu"]:
            assert execution[stage], f"Expected GPU stage: {stage}"
        if args.contact:
            assert execution["contact_pairs"] > 0, "No closing overlap/contact was measured"
        count = sum(row["particles"] for row in data["models"])
        assert data["particle_id_bytes"] == count * 8
        for row in data["models"]:
            assert math.isfinite(row["mass_kg"]) and row["mass_kg"] >= 0
            assert all(math.isfinite(v) for v in row["momentum_kg_m_s"])
        print("PASS mixed Vulkan transfer/projection/gather, common substeps, canonical IDs")
        print(f"contact_pairs={execution['contact_pairs']} / "
              f"working_set_bytes={execution['working_set_bytes']}")
    finally:
        client.close()


if __name__ == "__main__":
    main()
