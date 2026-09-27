"""Read recovery evidence from an external process without changing the scene."""

import json
from rt_ipc import RtIpc


def main():
    client = RtIpc()
    try:
        evidence = {}
        for method in (
            "viewport.device_recovery_status",
            "viewport.status",
            "viewport.frame_telemetry",
            "sim.control_state",
        ):
            ok, result = client.try_call(method)
            evidence[method] = {"ok": ok, "result": result}
        print(json.dumps(evidence, indent=2, ensure_ascii=False))
    finally:
        client.close()


if __name__ == "__main__":
    main()
