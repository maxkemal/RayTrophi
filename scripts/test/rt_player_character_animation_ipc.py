"""Read-only walking-character acceptance in an already open, newly built Player.

Checks joint progression and raster skin service calls. Visible skin quality still
requires looking at the Player viewport. Does not launch or build the app.
"""
import argparse
import json
import math
import time
from pathlib import Path

from rt_ipc import RtIpc, PLAYER_PIPE_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = RtIpc(PLAYER_PIPE_NAME)
    report = {"status": "FAIL", "character": args.character}
    try:
        first = client.call("agent.get_state_summary")["runtime"]
        assert first["animation_playback"], "Player has no animation playback hook"
        samples = []
        sources = set()
        for _ in range(20):
            bones = client.call("rig.list_bones", character=args.character)
            assert bones, "character has no skeleton"
            samples.append({bone["name"]: bone["world_position"] for bone in bones})
            sources.update(bone["pose_source"] for bone in bones)
            time.sleep(0.11)
        last = client.call("agent.get_state_summary")["runtime"]
        excursion = max(math.dist(point, sample[name])
                        for sample in samples[1:]
                        for name, point in samples[0].items() if name in sample)
        report.update(max_joint_excursion_m=excursion, pose_sources=sorted(sources),
                      runtime_before=first, runtime_after=last)
        assert excursion > 0.001, "walking joint positions stayed static"
        assert last["animation_updates"] > first["animation_updates"]
        assert last["animation_skin_sync_calls"] > first["animation_skin_sync_calls"], (
            "no raster skin service calls")
        report["status"] = "PASS"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        client.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
