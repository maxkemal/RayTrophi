"""Read-only C4a inventory/transfer check in a paused scene after the user's build.

Run externally: python scripts/test/rt_test_matter_models_ipc.py "Physics Domain 1"
"""
import argparse
import math

from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    args = parser.parse_args()
    client = RtIpc()
    try:
        inventory = checked_call(client, "fluid.matter_models", domain=args.domain)
        assert inventory["measured"], "Synchronize/step the domain first"
        assert "mixed_transport_ready" in inventory
        count = sum(model["particles"] for model in inventory["models"])
        assert inventory["particle_id_bytes"] == count * 8
        assert len(inventory["particle_id_hash"]) == 16
        assert int(inventory["next_particle_id"]) >= count + 1
        if count > 250000:
            print("PASS inventory; transfer reference omitted above its CPU limit")
            return
        transfer = checked_call(client, "fluid.matter_models", domain=args.domain,
                                include_transfer=True)
        assert transfer["models"] == inventory["models"]
        assert transfer["particle_id_hash"] == inventory["particle_id_hash"]
        supported_mass = sum(row["mass_kg"] for row in inventory["models"][:2])
        assert math.isclose(transfer["deposited_mass_kg"] + transfer["outside_mass_kg"],
                            supported_mass, rel_tol=1e-10, abs_tol=1e-8)
        print("PASS identities, model totals and bounded transfer conservation")
    finally:
        client.close()


if __name__ == "__main__":
    main()
