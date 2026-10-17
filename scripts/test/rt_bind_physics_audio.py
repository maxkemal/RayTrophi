"""Bind a mono asset to real solver telemetry and shared live scene acoustics.

Run externally after the Studio C++ build. Does not start simulation or collect
telemetry implicitly. --enable-telemetry explicitly enables the existing scene
policy; otherwise the source remains quiet until measurements are available.
"""
import argparse
import json
import math

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system-id", type=int, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--asset-id", required=True, help="Existing mono audio asset ID")
    parser.add_argument("--channel", default="fluid_flow", choices=(
        "mixed_contact", "dem_contact", "dem_friction", "dem_rolling", "fluid_flow",
        "gas_flow", "mpm_deformation", "mpm_fracture"))
    parser.add_argument("--input-max", type=float, required=True,
                        help="Measured W at full activity, or 1/s for mpm_fracture")
    parser.add_argument("--gain", type=float, default=0.5)
    parser.add_argument("--name", default="Physics audio source")
    parser.add_argument("--enable-telemetry", action="store_true")
    parser.add_argument("--no-reflections", action="store_true")
    args = parser.parse_args()
    assert args.system_id >= 0 and args.domain.strip() and args.name.strip()
    assert math.isfinite(args.input_max) and 0 < args.input_max <= 1e9
    assert math.isfinite(args.gain) and 0 <= args.gain <= 16
    c = RtIpc()
    source = None
    prior = c.call("audio.telemetry_get")
    policy_changed = False
    try:
        rows = c.call("audio.domains")
        row = next((entry for entry in rows if entry["system_id"] == args.system_id and
                    entry["domain_name"] == args.domain and
                    entry["domain_channel"] == args.channel), None)
        assert row is not None, "Binding not found; inspect audio.domains first"
        asset = next((entry for entry in c.call("audio.list")["assets"]
                      if entry["id"] == args.asset_id), None)
        assert asset and asset["channels"] == 1, "Choose an existing mono asset"
        reaction = {"driver": "domain_contact", "system_id": args.system_id,
                    "domain_name": args.domain, "domain_channel": args.channel,
                    "input_metric": "value" if args.channel == "mpm_fracture" else "power_w",
                    "input_min": 0, "input_max": args.input_max, "gain_min": 0,
                    "gain_max": 1, "response_seconds": 0.08, "follow_position": True,
                    "automatic": False}
        source = c.call("audio.create", name=args.name, asset_id=args.asset_id, spatial=True,
                        loop=True, gain=args.gain, position=row["position"], reaction=reaction)
        c.call("audio.acoustics_preview", id=source["id"], enabled=True, spectral=True,
               reflections=not args.no_reflections)
        if args.enable_telemetry and not prior["enabled"]:
            c.call("audio.telemetry_update", enabled=True)
            policy_changed = True
        configured = source["reaction"]
        configured["automatic"] = True
        source = c.call("audio.update", id=source["id"], reaction=configured)
        print(json.dumps({"source": source,
                          "runtime": c.call("audio.source_runtime", id=source["id"]),
                          "telemetry": c.call("audio.telemetry_get"),
                          "measurement_at_bind": row,
                          "note": "Live solver measurement drives start/stop, gain and world position. "
                                  "No real-time sound synthesis or simulation start was requested."},
                         indent=2))
    except Exception:
        if source:
            c.try_call("audio.remove", id=source["id"])
        if policy_changed:
            c.try_call("audio.telemetry_update", **prior)
        raise
    finally:
        c.close()


if __name__ == "__main__":
    main()
