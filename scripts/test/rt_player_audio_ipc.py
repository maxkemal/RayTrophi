"""Player scene audio acceptance: the shared audio tick on the game clock.

Never launches an application. Player open:
    python scripts/test/rt_player_audio_ipc.py [--project scene.rtp]

Default project: the installed EchoRoomsLab (an acoustic lab with audio
sources). Checks that the Player now ticks scene audio (it compiled the audio
service but never advanced it): the transport plays and its time follows the
game clock, the source poses are resolved against the scene (audio::syncScene),
and the runtime counters move. Reports the device state and live voices.
"""
import argparse
import json
from pathlib import Path
import sys
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "x64/Release/assets/acoustic_echo_lab/EchoRoomsLab.rtp"


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def run(project):
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    try:
        revision = runtime(ipc)["scene_revision"]
        ipc.call("project.open", path=str(project))
        deadline = time.monotonic() + 120
        state = runtime(ipc)
        while not (state["scene_ready"] and state["scene_revision"] == revision + 1):
            assert not state["error"] and not state["device_lost"], state
            assert time.monotonic() < deadline, state
            time.sleep(0.1)
            state = runtime(ipc)
        listed = ipc.call("audio.list")
        sources = listed.get("emitters", [])
        time.sleep(1.0)
        a_state, a_status = runtime(ipc), ipc.call("audio.status")
        wall_a = time.monotonic()
        time.sleep(3.0)
        b_state, b_status = runtime(ipc), ipc.call("audio.status")
        wall = time.monotonic() - wall_a
        pa, pb = a_status["playback_runtime"], b_status["playback_runtime"]
        sa, sb = a_state["simulation"], b_state["simulation"]
        report = {
            "sources": len(sources), "assets": len(listed.get("assets", [])),
            "transport_playing": pb["playing"], "transport_mode": pb.get("mode"),
            "transport_time": [round(pa["time_seconds"], 2), round(pb["time_seconds"], 2)],
            "wall_seconds": round(wall, 2),
            "audio_ticks": [sa["audio_ticks"], sb["audio_ticks"]],
            "resolved_poses": len(b_status.get("resolved_poses", {})),
            "device": b_status.get("device"),
            "voices": len(b_status.get("clip_runtime", {}).get("voices", [])),
        }
        assert report["sources"] > 0, ("fixture has no audio sources", report)
        assert sb["audio_ticks"] - sa["audio_ticks"] > 20, ("audio not ticked", report)
        assert pb["playing"], ("transport not playing in the Player", report)
        advanced = pb["time_seconds"] - pa["time_seconds"]
        # Background windows pump at 20 Hz with dt clamped, so allow slack.
        assert 0.5 * wall < advanced < 1.5 * wall, ("transport not on the game clock", report)
        assert report["resolved_poses"] > 0, ("source poses never resolved (syncScene)", report)

        # Manual emitters are silent until played (Studio behaves the same):
        # play one over the Player's own IPC and require a live device.
        emitter = sources[0]["id"]
        ipc.call("audio.play", id=emitter)
        time.sleep(1.0)
        played = ipc.call("audio.status")
        report["played_emitter"] = emitter
        report["device_after_play"] = played.get("device")
        report["source_runtime"] = ipc.call("audio.source_runtime", id=emitter)
        ipc.call("audio.stop_all")
        assert report["device_after_play"] == "ready", ("audio device did not open in the Player", report)
        print(json.dumps({"result": "PASS", **report}, indent=2, default=str))
    finally:
        ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", type=Path, default=DEFAULT)
    args = parser.parse_args()
    run(args.project)


if __name__ == "__main__":
    sys.exit(main())
