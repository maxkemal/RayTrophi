"""Check Player scene-load telemetry after opening a disposable scene."""

import argparse
from pathlib import Path
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("--exercise-disposable", required=True, action="store_true")
    args = parser.parse_args()

    ipc = RtIpc(PLAYER_PIPE_NAME)

    def operator(operation="status", **config):
        return ipc.call("player.operator", operation=operation, config=config)

    try:
        before = operator()["scene_load"]["attempt"]
        scheduled = operator("library.open", path=str(args.project.resolve()))
        assert scheduled["scheduled"]
        deadline = time.monotonic() + 60
        last_progress = 0
        while time.monotonic() < deadline:
            status = operator()
            load = status["scene_load"]
            if load["attempt"] <= before or Path(load["path"]).resolve() != args.project.resolve():
                time.sleep(.1)
                continue
            assert load["progress"] >= last_progress, load
            last_progress = load["progress"]
            assert load["state"] != "failed", load["error"]
            if load["state"] == "ready":
                break
            time.sleep(.1)
        else:
            raise AssertionError("Player did not report a ready first frame")

        assert status["project"] == load["key"]
        assert load["progress"] == 100 and load["elapsed_ms"] > 0
        assert all(row["ms"] >= 0 for row in load["stages"])
        assert {"check_project", "project_data", "prepare_viewport", "first_frame"} <= {
            row["stage"] for row in load["stages"]
        }
        print(f'PASS scene ready in {load["elapsed_ms"]:.0f} ms')
        for row in sorted(load["stages"], key=lambda r: r["ms"], reverse=True)[:5]:
            print(f'{row["ms"]:.0f} ms  {row["detail"] or row["stage"]}')
    finally:
        ipc.close()


if __name__ == "__main__":
    main()
