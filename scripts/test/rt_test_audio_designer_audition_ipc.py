"""Post-build Designer routing regression. Run externally with Studio open.

Auditions a quiet short test tone; temporarily changes mixer controls and restores
them. Stops audition in finally. Does not restore scene voices stopped by audition.
Checks callback meter energy, not merely the playing flag. No files/assets created.
"""
import time
from rt_ipc import RtIpc


def main():
    client = RtIpc()
    mixer = client.call("audio.mixer_get")
    design = client.call("audio.design_status")
    try:
        client.call("audio.mixer_update", bus="Master", gain=1, muted=False)
        client.call("audio.mixer_update", bus="Effects", muted=True)
        client.call("audio.mixer_update", bus="Music", solo=True)
        client.call("audio.design_render", recipe={
            "mode": "synth", "preset": "tone", "duration_seconds": 1,
            "gain": 0.1, "sustain": 1, "noise_mix": 0,
        })
        status = client.call("audio.design_status", details=False)
        assert status["audition_bus"] == "Master"
        assert status["audition_output_gain"] == 1
        # Preview mute remains the final mute; unmute only for this bounded test.
        client.call("audio.preview_mute", muted=False)
        client.call("audio.design_preview", loop=True)
        time.sleep(0.15)
        status = client.call("audio.design_status", details=False)
        assert status["playing"] and status["cursor_seconds"] > 0
        buses = client.call("audio.mixer_get")["buses"]
        master = next(bus for bus in buses if bus["name"] == "Master")
        assert master["meter"]["peak"] > 0.001, master
        client.call("audio.design_stop")
        print("PASS: Designer reaches Master meter despite Effects mute and Music solo")
    finally:
        client.call("audio.design_stop")
        for bus in mixer["buses"]:
            client.call("audio.mixer_update", bus=bus["name"], gain=bus["gain"],
                        muted=bus["muted"], solo=bus["solo"], pan=bus["pan"])
        client.call("audio.preview_mute", muted=mixer["output_muted"])
        if design.get("ready"):
            client.call("audio.design_render", recipe=design["recipe"])
        client.close()


if __name__ == "__main__":
    main()
