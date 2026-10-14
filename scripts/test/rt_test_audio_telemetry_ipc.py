"""Post-build manual audio telemetry policy acceptance, using external IPC.

With domains present, enabling requires --exercise-enable (use a small fixture).
Never launches Studio, drives simulation, or implicitly enables heavy collection.
"""
import argparse
import json
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise-enable", action="store_true",
                        help="Explicitly exercise enabled policy on the open small fixture")
    args = parser.parse_args()
    client = RtIpc()
    previous = client.call("audio.telemetry_get")
    modified = False
    try:
        assert isinstance(previous["enabled"], bool)
        before = client.call("audio.list")
        for bad in ({"enabled": 1}, {"enabled": "true"}, {"enabled": None},
                    {"enabled": False, "automatic": True}):
            try:
                client.call("audio.telemetry_update", **bad)
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"Invalid policy accepted: {bad}")
            assert client.call("audio.telemetry_get") == previous
            assert client.call("audio.list") == before
        rows = client.call("audio.domains")
        if args.exercise_enable or not rows:
            for enabled in (False, True, False):
                client.call("audio.telemetry_update", enabled=enabled)
                modified = True
                assert client.call("audio.telemetry_get") == {"enabled": enabled}
                assert client.call("audio.list")["telemetry"] == {"enabled": enabled}
                for row in client.call("audio.domains"):
                    assert row["telemetry_enabled"] == enabled
                    if not enabled:
                        assert not row["measured"]
                        assert row["reason"] == "telemetry_disabled"
                        assert row["contact_energy_j"] == row["contact_power_w"] == row["value"] == 0
        elif not previous["enabled"]:
            for row in rows:
                assert not row["measured"] and row["reason"] == "telemetry_disabled"
        print(json.dumps({"passed": True, "checks": ["strict atomic validation",
            "shared scene policy", "disabled measurement semantics"],
            "enable_exercised": args.exercise_enable or not rows}, indent=2))
    finally:
        if modified:
            client.call("audio.telemetry_update", **previous)
        client.close()


if __name__ == "__main__":
    main()
