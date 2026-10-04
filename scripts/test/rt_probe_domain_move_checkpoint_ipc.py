"""Temporarily move the active domain and restore its authored position.

Uses external IPC, takes no solver step, and records state after UI/cache ticks.
"""
import argparse
import json
import time
from pathlib import Path
from rt_ipc import RtIpc
from rt_test_fluid_active_window_ipc import checked_call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("domain")
    args = parser.parse_args()
    client = RtIpc()
    report = {}
    moved = False

    def snapshot():
        return {"digest": checked_call(client, "fluid.state_digest", domain=args.domain),
                "grids": checked_call(client, "fluid.get_phase_grids", domain=args.domain),
                "frame": client.call("timeline.get_frame")}

    try:
        report["before"] = snapshot()
        report["move"] = checked_call(client, "sim.move_domain", domain=args.domain,
                                      delta=[0.25, 0, 0])
        moved = True
        report["immediate"] = snapshot()
        time.sleep(0.3)
        report["after_ui_ticks"] = snapshot()
        original = report["before"]["digest"]["centroid"]
        for name in ("immediate", "after_ui_ticks"):
            centroid = report[name]["digest"]["centroid"]
            report[name]["centroid_delta"] = [a - b for a, b in zip(centroid, original)]
    finally:
        if moved:
            report["restore"] = checked_call(client, "sim.move_domain", domain=args.domain,
                                             delta=[-0.25, 0, 0])
            time.sleep(0.3)
            report["restored"] = snapshot()
        client.close()
        destination = Path(__file__).resolve().parents[2] / ".tmp/domain_move_checkpoint.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
