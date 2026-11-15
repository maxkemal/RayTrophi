"""Player clouds acceptance: shared cloud/local-weather push on the game clock.

Never launches an application.

1. Fixture (Studio open, startup scene; built on the material fixture):
       python scripts/test/rt_player_materials_ipc.py --make-fixture   (once)
       python scripts/test/rt_player_clouds_ipc.py --make-fixture
2. Acceptance (Player open):
       python scripts/test/rt_player_clouds_ipc.py

Checks: the Player publishes the cloud state (clouds_enabled, cloud_pushes),
cloud time runs on the game clock, the sky differs from the cloudless fixture,
and the sky changes over time (drift + evolution). Also checks that a cloudless
scene does NOT keep pushing (it must be able to idle).
"""
import argparse
import json
import sys
import time

from PIL import ImageChops, ImageStat

from rt_ipc import PLAYER_PIPE_NAME, RtIpc
from rt_player_materials_ipc import FIXTURE as MATERIAL_FIXTURE, screenshot

FIXTURE = MATERIAL_FIXTURE.parent.parent / "player-clouds" / "clouds.rtp"
PRESET = "broken"


def make_fixture():
    assert MATERIAL_FIXTURE.exists(), "run rt_player_materials_ipc.py --make-fixture first"
    studio = RtIpc()
    try:
        names = studio.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"Studio scene not empty: {names}"
        studio.call("project.open", path=str(MATERIAL_FIXTURE))
        time.sleep(1.0)
        studio.call("world.set_mode", mode="nishita")
        studio.call("world.apply_cloud_preset", preset=PRESET)
        # Strong wind so a few seconds of game time move the sky visibly.
        studio.call("world.set_climate", wind_direction=[1.0, 0.0, 0.3], wind_speed_mps=60.0)
        # Look up a little so the sky fills the upper half.
        studio.call("camera.set_target", target=[0.0, 2.5, 0.0])
        clouds = studio.call("world.get_clouds")
        assert clouds["any_enabled"], clouds
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        studio.call("project.save", path=str(FIXTURE))
        print(json.dumps({"fixture": str(FIXTURE), "preset": PRESET,
                          "any_enabled": clouds["any_enabled"]}, indent=2))
    finally:
        studio.close()


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def open_scene(ipc, path):
    revision = runtime(ipc)["scene_revision"]
    ipc.call("project.open", path=str(path))
    deadline = time.monotonic() + 120
    state = runtime(ipc)
    while not (state["scene_ready"] and state["scene_revision"] == revision + 1 and
               state["frame_available"]):
        assert not state["error"] and not state["device_lost"], state
        assert time.monotonic() < deadline, state
        time.sleep(0.1)
        state = runtime(ipc)
    return state


def sky_diff(a, b):
    box = (0, 0, a.width, int(a.height * 0.35))
    diff = ImageChops.difference(a.crop(box).convert("L"), b.crop(box).convert("L"))
    return ImageStat.Stat(diff).mean[0]


def run():
    assert FIXTURE.exists(), "run --make-fixture first"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    capture_before = ipc.call("viewport.status")["capture_enabled"]
    try:
        ipc.call("viewport.capture", enabled=True)
        # Cloudless baseline: the push must settle (no per-frame republish).
        open_scene(ipc, MATERIAL_FIXTURE)
        time.sleep(1.0)
        sim0 = runtime(ipc)["simulation"]
        time.sleep(2.0)
        sim1 = runtime(ipc)["simulation"]
        assert not sim1["clouds_enabled"], sim1
        assert sim1["cloud_pushes"] - sim0["cloud_pushes"] == 0, \
            ("a cloudless scene keeps republishing clouds (cannot idle)", sim0, sim1)
        clear_sky = screenshot(ipc)

        open_scene(ipc, FIXTURE)
        time.sleep(1.5)
        state_a = runtime(ipc)
        image_a = screenshot(ipc)
        time.sleep(4.0)
        state_b = runtime(ipc)
        image_b = screenshot(ipc)
        sa, sb = state_a["simulation"], state_b["simulation"]
        report = {
            "clouds_enabled": sb["clouds_enabled"],
            "cloud_time_s": [round(sa["cloud_time_s"], 2), round(sb["cloud_time_s"], 2)],
            "cloud_pushes": [sa["cloud_pushes"], sb["cloud_pushes"]],
            "sky_vs_cloudless": round(sky_diff(clear_sky, image_a), 2),
            "sky_change_over_4s": round(sky_diff(image_a, image_b), 3),
            "device_lost": state_b["device_lost"],
        }
        assert sb["clouds_enabled"] and not state_b["device_lost"], report
        assert sb["cloud_time_s"] - sa["cloud_time_s"] > 2.0, ("cloud time not on the game clock", report)
        assert sb["cloud_pushes"] - sa["cloud_pushes"] > 10, ("cloud state not republished", report)
        assert report["sky_vs_cloudless"] > 2.0, ("clouds not visible in the Player sky", report)
        assert report["sky_change_over_4s"] > 0.05, ("sky frozen while cloud time runs", report)
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
