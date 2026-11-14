"""Player particle acceptance (reject lifted; shared clock + shared billboard pass).

A magenta fountain emitter in front of the material fixture must spawn particles
in the Player and appear in its image. Never launches an application.

1. Fixture (Studio open; built on top of rt_player_materials_ipc.py's fixture):
       python scripts/test/rt_player_materials_ipc.py --make-fixture   (once)
       python scripts/test/rt_player_particles_ipc.py --make-fixture
2. Acceptance (Player open):
       python scripts/test/rt_player_particles_ipc.py
"""
import argparse
import json
from pathlib import Path
import sys
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc
from rt_player_materials_ipc import FIXTURE as MATERIAL_FIXTURE, screenshot

FIXTURE = MATERIAL_FIXTURE.parent.parent / "player-particles" / "particles.rtp"
MAGENTA = [1.0, 0.1, 0.9]


def make_fixture():
    assert MATERIAL_FIXTURE.exists(), "run rt_player_materials_ipc.py --make-fixture first"
    studio = RtIpc()
    try:
        names = studio.call("scene.list_objects")
        assert set(names) <= {"Default_Cube"}, f"Studio scene not empty: {names}"
        studio.call("project.open", path=str(MATERIAL_FIXTURE))
        time.sleep(1.0)
        studio.call("particle.add_system", name="PaintBurst")
        emitter = studio.call("particle.add_emitter", system="PaintBurst", name="Fountain",
                              point=[0.0, 1.2, 1.5], direction=[0.0, 1.0, 0.0],
                              rate_per_second=600.0, speed=3.0, spread=35.0,
                              lifetime_seconds=2.0)
        studio.call("particle.set_appearance", system="PaintBurst",
                    profile_id=int(emitter["appearance_profile_id"]),
                    color_ramp=[[0.0] + MAGENTA, [1.0] + MAGENTA])
        # A new system starts as an emitter gizmo only; the billboard pass skips it.
        studio.call("particle.set_system_emitter_only", system="PaintBurst", emitter_only=False)
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        studio.call("project.save", path=str(FIXTURE))
        print(json.dumps({"fixture": str(FIXTURE),
                          "systems": studio.call("particle.list_systems")}, indent=2))
    finally:
        studio.close()


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def magenta_pixels(image):
    small = image.resize((image.width // 4, image.height // 4))
    return sum(1 for r, g, b in small.getdata() if r > 120 and b > 110 and g < 0.6 * min(r, b))


def run():
    assert FIXTURE.exists(), "run --make-fixture first"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    capture_before = ipc.call("viewport.status")["capture_enabled"]
    try:
        ipc.call("viewport.capture", enabled=True)
        revision = runtime(ipc)["scene_revision"]
        ipc.call("project.open", path=str(FIXTURE))
        deadline = time.monotonic() + 60
        state = runtime(ipc)
        while not (state["scene_ready"] and state["scene_revision"] == revision + 1):
            assert not state["error"] and not state["device_lost"], state
            assert time.monotonic() < deadline, state
            time.sleep(0.05)
            state = runtime(ipc)
        time.sleep(2.5)  # a 2 s lifetime fountain reaches steady state
        state = runtime(ipc)
        sim = state["simulation"]
        report = {"simulation": sim}
        assert not state["error"] and not state["device_lost"], state
        assert sim["particle_systems"] == 1 and not sim["stopped"], report
        assert sim["particles_alive"] > 200, ("fountain did not spawn", report)
        assert sim["particle_draws"] > 0, ("no particle draws handed to the viewport", report)
        frame_calls = state["render_calls"]
        time.sleep(0.5)
        image = screenshot(ipc)
        report["magenta_pixels"] = magenta_pixels(image)
        assert runtime(ipc)["render_calls"] > frame_calls, ("Player stopped rendering", report)
        assert report["magenta_pixels"] > 50, ("particles not visible in the Player image", report)
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
