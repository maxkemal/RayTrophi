"""MP4 acceptance game script: natural end and owner-requested graceful shutdown."""
import argparse
import json
from pathlib import Path
import time

try:
    from .game_bridge import BridgeProcess
    from .transport import connect
except ImportError:
    from game_bridge import BridgeProcess
    from transport import connect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--synthetic-depth", type=float)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    options = ["--video", str(args.video.resolve()), "--model", str(args.model.resolve()),
               "--calibration", str(args.calibration.resolve()), "--people", "1",
               "--video-card"]
    if args.offline:
        options.append("--offline")
    if args.synthetic_depth is not None:
        options.extend(["--synthetic-depth", str(args.synthetic_depth)])
    reports = {}
    ipc = None
    before = None
    try:
        if not args.offline:
            ipc = connect()
            before = ipc.call("player.display")
            ipc.call("player.set_display", video_card=True, continuous=True)
        for phase in ("natural_end", "owner_stop"):
            with BridgeProcess(options, python=args.python, worker=args.worker) as owner:
                if phase == "owner_stop":
                    owner.wait_for_pose()
                    time.sleep(0.2)
                    report = owner.stop()
                else:
                    report = owner.wait(timeout=180)
            if not report or report.get("frames_with_pose", 0) == 0 or report.get("card_frames", 0) == 0:
                raise RuntimeError("MP4 replay did not produce pose and masked card frames")
            reports[phase] = report
        if ipc:
            body = ipc.call("input.body_poses")
            reports["cleanup_person_count"] = len(body["persons"])
            if body["persons"]:
                raise RuntimeError("body poses remained after bridge shutdown")
            runtime = ipc.call("agent.get_state_summary", include_probe=False)["runtime"]
            if runtime["device_lost"]:
                raise RuntimeError("Player device lost during replay")
            reports["device_lost"] = runtime["device_lost"]
    finally:
        if ipc:
            try:
                if before:
                    ipc.call("player.set_display", video_card=before["video_card"],
                             continuous=before["continuous"])
            finally:
                ipc.close()
    if reports["owner_stop"]["frames"] >= reports["natural_end"]["frames"]:
        raise RuntimeError("owner stop did not interrupt the video")
    result = {"status": "PASS", "scope": "offline MP4 lifecycle" if args.offline
              else "MP4 lifecycle with external Player IPC", "phases": reports}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
