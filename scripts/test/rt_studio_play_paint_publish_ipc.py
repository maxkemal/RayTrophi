"""Check GPU-only wet paint publication on an open disposable Studio game scene."""

import argparse
import time

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise-disposable", action="store_true", required=True)
    parser.parse_args()
    ipc = RtIpc()

    def play(operation):
        return ipc.call("game.playtest", operation=operation)

    def demo(operation="status", **config):
        return ipc.call("paint.wet_demo", operation=operation, config=config)

    started_here = False
    try:
        state = demo()
        assert state["prepared"], "Open a prepared paint balloon scene first"
        assert state["balloons"], "Need at least one balloon"
        if not play("status")["active"]:
            started_here = play("start")["active"]
            assert started_here, "Could not start Studio Play"
        before = ipc.call("paint.wet_status")
        ids = [balloon["id"] for balloon in demo()["balloons"] if balloon["inflated"]]
        assert ids, "Need at least one inflated balloon"
        for balloon in ids[:3]:
            demo("pop", balloon=balloon)

        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            time.sleep(0.1)
            after = ipc.call("paint.wet_status")
            for counter in ("publish_host_downloads", "publish_bytes_downloaded"):
                assert after[counter] == before[counter], f"{counter} increased during Play"

        assert play("status")["active"], "Play stopped during the sample"
        copies = after["publish_device_copies"] - before["publish_device_copies"]
        channel_copies = after["pbr_device_copies"] - before["pbr_device_copies"]
        assert copies > 0, "No GPU colour copy observed after a pop"
        assert after["publish_mode"] == "device_copy", after["publish_mode"]
        print(f"PASS {copies} GPU colour copies, {channel_copies} GPU channel copies, no host copies")
    finally:
        try:
            if started_here:
                play("stop")
        finally:
            ipc.close()


if __name__ == "__main__":
    main()
