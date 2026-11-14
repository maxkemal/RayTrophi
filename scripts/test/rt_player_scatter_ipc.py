"""Player scatter/instance acceptance (native InstanceGroup raster path).

Yellow pebbles scattered over the material fixture's ground must load in the
Player and be drawn by the same raster scatter path Studio uses. Never launches
an application.

1. Fixture (Studio open, startup scene):
       python scripts/test/rt_player_materials_ipc.py --make-fixture   (once)
       python scripts/test/rt_player_scatter_ipc.py --make-fixture
2. Acceptance (Player open):
       python scripts/test/rt_player_scatter_ipc.py
"""
import argparse
import json
import sys
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc
from rt_player_materials_ipc import FIXTURE as MATERIAL_FIXTURE, screenshot

FIXTURE = MATERIAL_FIXTURE.parent.parent / "player-scatter" / "scatter.rtp"
TARGET_COUNT = 150


def make_fixture():
    assert MATERIAL_FIXTURE.exists(), "run rt_player_materials_ipc.py --make-fixture first"
    studio = RtIpc()
    try:
        names = studio.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"Studio scene not empty: {names}"
        studio.call("project.open", path=str(MATERIAL_FIXTURE))
        time.sleep(1.0)
        # Source mesh parked under the ground: only its scattered copies are seen.
        pebble = studio.call("scene.add_primitive", type="sphere", name="Pebble", size=0.12)
        studio.call("scene.set_transform", name=pebble, translation=[0.0, -20.0, 0.0])
        mat = studio.call("material.create", type="principled", name="PebbleYellow")
        mat = mat if isinstance(mat, str) else mat.get("name")
        studio.call("material.assign", material_name=mat, object_name=pebble)
        studio.call("material.set", object_name=pebble, param="base_color", value=[0.95, 0.85, 0.05])
        group = studio.call("scatter.create_group", name="Pebbles", target_node="MatGround")
        group_name = group if isinstance(group, str) else group.get("name", "Pebbles")
        studio.call("scatter.add_source", group=group_name, mesh=pebble, scale_min=0.8, scale_max=1.2)
        studio.call("scatter.set_settings", group=group_name, target_count=TARGET_COUNT, seed=7,
                    min_distance=0.15)
        filled = studio.call("scatter.fill", group=group_name)
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        studio.call("project.save", path=str(FIXTURE))
        print(json.dumps({"fixture": str(FIXTURE), "group": group, "fill": filled,
                          "groups": studio.call("scatter.list_groups")}, indent=2, default=str))
    finally:
        studio.close()


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def yellow_pixels(image):
    small = image.resize((image.width // 4, image.height // 4))
    return sum(1 for r, g, b in small.getdata() if r > 120 and g > 100 and b < 0.45 * min(r, g))


def run():
    assert FIXTURE.exists(), "run --make-fixture first"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    capture_before = ipc.call("viewport.status")["capture_enabled"]
    try:
        ipc.call("viewport.capture", enabled=True)
        revision = runtime(ipc)["scene_revision"]
        ipc.call("project.open", path=str(FIXTURE))
        deadline = time.monotonic() + 90
        state = runtime(ipc)
        while not (state["scene_ready"] and state["scene_revision"] == revision + 1 and
                   state["frame_available"] and not state["render_pending"]):
            assert not state["error"] and not state["device_lost"], state
            assert time.monotonic() < deadline, state
            time.sleep(0.1)
            state = runtime(ipc)
        report = {"scatter_groups": state.get("scatter_groups"),
                  "scatter_instances": state.get("scatter_instances"),
                  "visible_triangles": state["visible_triangles"],
                  "draw_calls": state["draw_calls"]}
        assert state["scatter_groups"] == 1, report
        assert state["scatter_instances"] >= TARGET_COUNT // 2, report
        time.sleep(0.5)
        report["yellow_pixels"] = yellow_pixels(screenshot(ipc))
        assert report["yellow_pixels"] > 40, ("scattered pebbles not visible in the Player", report)
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
