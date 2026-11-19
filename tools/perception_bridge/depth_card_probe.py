"""External Player synthetic depth probe. No images are encoded or saved.

A solid oval alternates near/far depth halves. Existing scene depth must cover
the far half while the near half covers objects behind it. One balloon is popped
in each phase to exercise the existing GPU droplet pass. Visual acceptance is
reported separately by the observer; process success alone is not a depth PASS.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np

try:
    from .transport import connect
    from .video_card import VideoCardPublisher
except ImportError:
    from transport import connect
    from video_card import VideoCardPublisher


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase-seconds", type=float, default=18)
    args = parser.parse_args()
    ipc = connect()
    before = ipc.call("player.display")
    report = {"image_storage": "memory_only", "visual_acceptance": "PENDING", "phases": []}
    try:
        ipc.call("player.set_display", video_card=True, continuous=True)
        yy, xx = np.mgrid[:128, :96]
        rgba = np.zeros((128, 96, 4), dtype=np.uint8)
        rgba[..., :3] = (245, 50, 200)
        rgba[..., 3] = (((xx-47.5)/47)**2 + ((yy-63.5)/63)**2 < 1) * 255
        with_publisher = VideoCardPublisher(synthetic_depth=0.0)
        try:
            for phase in (0, 1):
                depth = np.where(xx < 48, -2.0, 2.0).astype(np.float32)
                if phase:
                    depth = -depth
                begin = time.monotonic()
                popped = False
                frames = 0
                print(json.dumps({"phase": phase, "near_half": "left" if phase == 0 else "right"}),
                      flush=True)
                while time.monotonic() - begin < args.phase_seconds:
                    with_publisher.publish((rgba, 0.0, 0.0, 3.2, depth))
                    frames += 1
                    if not popped and time.monotonic() - begin > 6:
                        state = ipc.call("paint.wet_demo", operation="pop", config={"balloon": 0})
                        popped = True
                    time.sleep(1/30)
                runtime = ipc.call("agent.get_state_summary", include_probe=False)["runtime"]
                assert not runtime["device_lost"], runtime.get("error")
                display = ipc.call("player.display")
                report["phases"].append({"phase": phase, "published": frames,
                    "popped": popped, "device_lost": runtime["device_lost"],
                    "presented_fps": display["frame"]["presented_fps"]})
        finally:
            with_publisher.close()
        report["status"] = "TRANSPORT_AND_RUNTIME_PASS"
    finally:
        ipc.call("player.set_display", video_card=before["video_card"],
                 continuous=before["continuous"])
        ipc.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
