"""Read-only inspection. Does not step, play, clear or edit the current scene.

python scripts/test/rt_inspect_mixed_stage_profile.py [domain name]
Run from an external terminal after the user C++ build and a user-driven step.
"""
import json
import math
import sys
from rt_ipc import RtIpc


def main():
    call = RtIpc().call
    assert not call("sim.control_state")["playing"], "Pause after the user-driven step first"
    names = sys.argv[1:] or [d["name"] for d in call("fluid.list_domains")["domains"]]
    for name in names:
        before = call("fluid.state_digest", domain=name)
        stats = call("fluid.step_stats", domain=name)
        report = stats.get("mixed_stages")
        assert report and report["measured"], "No new measured mixed step for " + name
        assert report["timing_kind"] == "host_wall_including_waits"
        assert report["common_substeps"] > 0
        stages = report["stages"]
        for row in stages.values():
            assert math.isfinite(row["host_wall_ms"]) and row["host_wall_ms"] >= 0
        if stats["transfer_sparse_canonical"]:
            assert stages["flip_snapshot"]["synchronize_calls"] == 0, "explicit FLIP fence remains"
        after = call("fluid.state_digest", domain=name)
        for key in ("particles", "position_hash", "velocity_hash"):
            assert before[key] == after[key], "state changed during paused read-only acceptance"
        assert before["particles"] == stats["particle_count"], "stale particle statistics"
        # Profile excludes outer driver work, so phase counts cannot exceed frame counts.
        for key in ("dispatch_calls", "download_calls", "synchronize_calls", "upload_calls"):
            assert sum(row[key] for row in stages.values()) <= stats[key], (name, key)
        print(json.dumps({"domain": name, "total_ms": stats["total_ms"],
                          "mixed_stages": report}, indent=2))


if __name__ == "__main__":
    main()
