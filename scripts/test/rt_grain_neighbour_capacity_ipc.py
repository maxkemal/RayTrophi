"""Continue a paused mixed-size DEM pile after rebuilding C++ and revision-30 shaders.
No reset, emitter changes, scene save, or app launch. Advances up to --steps ticks.
"""
import argparse
import json
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", default="Physics Domain 1")
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    ipc = RtIpc()
    try:
        control = ipc.call("sim.control_state")
        assert not control["playing"], "Pause the timeline first"
        before = ipc.call("fluid.grain_settings", domain=args.domain)
        assert before["radius_max_m"] > before["radius_m"], "Use a mixed-size pile"
        initial = ipc.call("fluid.matter_models", domain=args.domain)
        assert initial["grain_diagnostics"]["contact_shader_revision"] >= 30, "Rebuild first"
        wide_contacts = False
        for index in range(args.steps):
            ipc.call("fluid.step", dt=1/24)
            result = ipc.call("fluid.matter_models", domain=args.domain)
            assert not result["mixed_execution"]["step_held"], result["mixed_execution"]
            runtime = result["grain_diagnostics"]["runtime"]
            assert runtime and runtime["neighbour_list_capacity"] >= 128, runtime
            assert runtime["contact_history_capacity"] >= 64, runtime
            wide_contacts = wide_contacts or runtime["max_contacts_per_grain"] > 24
            print(json.dumps({"step": index + 1, "capacity": runtime["neighbour_list_capacity"],
                              "retries": runtime["neighbour_list_retries"],
                              "contacts": runtime["max_contacts_per_grain"],
                              "history_capacity": runtime["contact_history_capacity"],
                              "history_retries": runtime["contact_history_retries"]}))
        assert ipc.call("fluid.grain_settings", domain=args.domain) == before
        assert ipc.call("sim.control_state") == control
        print("PASS mixed-size pile continuation without changing authored settings")
        print("Contacts above legacy 24 observed:", wide_contacts)
    finally:
        ipc.close()


if __name__ == "__main__":
    main()
