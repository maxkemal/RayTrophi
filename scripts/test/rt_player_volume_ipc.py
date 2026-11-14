"""Player VDB volume acceptance (shared volume table publication).

A cloud VDB imported over the material fixture must load in the Player, reach its
single viewport device through AppRuntime::publishVolumes and change the image.
Never launches an application.

1. Fixture (Studio open, startup scene; uses the new scene.import_vdb IPC):
       python scripts/test/rt_player_materials_ipc.py --make-fixture   (once)
       python scripts/test/rt_player_volume_ipc.py --make-fixture
2. Acceptance (Player open):
       python scripts/test/rt_player_volume_ipc.py
"""
import argparse
import json
from pathlib import Path
import sys
import time

from PIL import ImageChops, ImageStat

from rt_ipc import PLAYER_PIPE_NAME, RtIpc
from rt_player_materials_ipc import FIXTURE as MATERIAL_FIXTURE, screenshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = MATERIAL_FIXTURE.parent.parent / "player-volume" / "volume.rtp"
VDB = ROOT / ("x64/Release/assets/volume/vdb/Cumulonimbus_Field_1/f048/"
              "Cumulonimbus_Field_1.f048.eighth.vdb")


def make_fixture():
    assert MATERIAL_FIXTURE.exists(), "run rt_player_materials_ipc.py --make-fixture first"
    assert VDB.exists(), f"missing sample VDB {VDB}"
    studio = RtIpc()
    try:
        names = studio.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"Studio scene not empty: {names}"
        studio.call("project.open", path=str(MATERIAL_FIXTURE))
        time.sleep(1.0)
        info = studio.call("scene.import_vdb", path=str(VDB))
        assert info["scene_vdb_count"] == 1, info
        # Import contract: a volume larger than 50 units is scaled down to 5.
        extent = max(b - a for a, b in zip(info["local_bounds_min"], info["local_bounds_max"]))
        if extent > 50.0:
            assert info["auto_scaled"] and abs(extent * info["scale"][0] - 5.0) < 0.01, info
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        studio.call("project.save", path=str(FIXTURE))
        print(json.dumps({"fixture": str(FIXTURE), "import": info}, indent=2))
    finally:
        studio.close()


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def open_and_settle(ipc, path):
    revision = runtime(ipc)["scene_revision"]
    ipc.call("project.open", path=str(path))
    deadline = time.monotonic() + 120
    state = runtime(ipc)
    while not (state["scene_ready"] and state["scene_revision"] == revision + 1 and
               state["frame_available"] and not state["render_pending"]):
        assert not state["error"] and not state["device_lost"], state
        assert time.monotonic() < deadline, state
        time.sleep(0.1)
        state = runtime(ipc)
    time.sleep(1.0)
    return state, screenshot(ipc)


def run():
    assert FIXTURE.exists(), "run --make-fixture first"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    capture_before = ipc.call("viewport.status")["capture_enabled"]
    try:
        ipc.call("viewport.capture", enabled=True)
        _, without_volume = open_and_settle(ipc, MATERIAL_FIXTURE)
        state, with_volume = open_and_settle(ipc, FIXTURE)
        diff = ImageChops.difference(without_volume.convert("L"), with_volume.convert("L"))
        mean_diff = ImageStat.Stat(diff).mean[0]
        changed = sum(1 for v in diff.resize((diff.width // 4, diff.height // 4)).getdata() if v > 12)
        report = {"vdb_volumes": state["vdb_volumes"], "gas_volumes": state["gas_volumes"],
                  "mean_luma_diff": round(mean_diff, 2), "changed_pixels_q": changed,
                  "device_lost": state["device_lost"]}
        assert state["vdb_volumes"] == 1 and not state["device_lost"], report
        # Same camera and geometry; only the volume differs.
        assert changed > 500, ("volume not visible in the Player image", report)
        print(json.dumps({"result": "PASS", **report}, indent=2))
    finally:
        try:
            ipc.call("viewport.capture", enabled=capture_before)
        finally:
            ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-fixture", action="store_true")
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture()
    else:
        run()


if __name__ == "__main__":
    sys.exit(main())
