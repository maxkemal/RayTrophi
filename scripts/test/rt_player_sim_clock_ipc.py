"""Player shared simulation clock acceptance (PLAYER_GAME_PAINT_ROOM.md step 1).

A rigid box dropped onto a static ground must fall and settle in the Player,
driven by the same AppRuntime::SimulationClock that Studio's SceneUI ticks.
Never launches an application.

1. Fixture (once, needs Studio open on an EMPTY scene; the scene is cleaned up
   again, but Studio's project path becomes the fixture path):
       python scripts/test/rt_player_sim_clock_ipc.py --make-fixture
2. Acceptance (Player open):
       python scripts/test/rt_player_sim_clock_ipc.py
"""
import argparse
import json
from pathlib import Path
import sys
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc

FIXTURE = (Path(__file__).resolve().parents[2] /
           "docs/dev/.captures/player-sim-clock/rigid_drop.rtp")
BOX_START_Y = 3.0
BOX_SIZE = 0.5


def make_fixture(path):
    studio = RtIpc()
    try:
        existing = studio.call("scene.list_objects")
        names = existing if isinstance(existing, list) else existing.get("objects", [])
        # A fresh Studio launch carries only its startup cube; anything else is
        # somebody's scene and must not be touched.
        assert set(names) <= {"Default_Cube"},             f"Studio scene is not empty ({len(names)} objects); open an empty scene first"
        for name in names:
            studio.call("scene.delete", name=name)
        # A thin slab, not a plane: a zero-height box shape is not a collider.
        ground = studio.call("scene.add_primitive", type="cube", name="SimGround", size=1.0)
        studio.call("scene.set_transform", name=ground, translation=[0.0, -0.1, 0.0],
                    scale=[10.0, 0.2, 10.0])
        box = studio.call("scene.add_primitive", type="cube", name="SimBox", size=BOX_SIZE)
        studio.call("scene.set_transform", name=box, translation=[0.0, BOX_START_Y, 0.0])
        studio.call("physics.add_body", object=ground, motion_type="static", shape="box")
        studio.call("physics.add_body", object=box, motion_type="dynamic", mass=1.0, shape="box")
        path.parent.mkdir(parents=True, exist_ok=True)
        studio.call("project.save", path=str(path))
        for name in (box, ground):
            studio.call("physics.remove_body", object=name)
            studio.call("scene.delete", name=name)
        print(json.dumps({"fixture": str(path), "ground": ground, "box": box}))
    finally:
        studio.close()


def runtime(ipc):
    return ipc.call("agent.get_state_summary", include_probe=False)["runtime"]


def box_y(ipc, name):
    pose = ipc.call("scene.get_world_transform", name=name)
    return float(pose["translation"][1]), bool(pose["simulated"])


def run(path, seconds):
    assert path.exists(), f"missing fixture {path}; run with --make-fixture first"
    ipc = RtIpc(pipe_name=PLAYER_PIPE_NAME)
    capture_before = ipc.call("viewport.status")["capture_enabled"]
    try:
        # Capture makes the Player render while unfocused, so the pixel
        # signature below follows the simulation even in a background window.
        ipc.call("viewport.capture", enabled=True)
        before = runtime(ipc)
        ipc.call("project.open", path=str(path))
        deadline = time.monotonic() + 60
        state = runtime(ipc)
        while not (state["scene_ready"] and state["scene_revision"] == before["scene_revision"] + 1):
            assert not state["error"] and not state["device_lost"], state
            assert time.monotonic() < deadline, state
            time.sleep(0.05)
            state = runtime(ipc)
        sim0 = state["simulation"]
        assert sim0["runtime_present"] and sim0["rigid_bodies"] >= 2, sim0
        names = ipc.call("scene.list_objects")
        names = names if isinstance(names, list) else names.get("objects", [])
        box = next(n if isinstance(n, str) else n.get("name") for n in names
                   if "SimBox" in (n if isinstance(n, str) else n.get("name", "")))
        y0, _ = box_y(ipc, box)
        signature0 = state["pixel_signature"]

        samples = []
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            time.sleep(0.25)
            y, simulated = box_y(ipc, box)
            samples.append(round(y, 4))
        state = runtime(ipc)
        sim = state["simulation"]
        y_end, simulated = box_y(ipc, box)

        assert not state["error"] and not state["device_lost"], state
        assert not sim["stopped"] and not sim["error"], sim
        assert sim["live_mode"], sim
        assert sim["driven_ticks"] - sim0["driven_ticks"] >= 20, ("clock did not run", sim0, sim)
        assert simulated, "scene.get_world_transform says no solver contributed"
        # The clock starts on open, so y0 may already be below the spawn height.
        assert BOX_START_Y - y_end > 1.5, ("box did not fall", y0, y_end, samples)
        # Settled on the ground plane (half extent 0.25 m), not tunnelled through it.
        assert -0.05 < y_end < 0.6, ("box did not rest on the ground", y_end, samples)
        assert abs(samples[-1] - samples[-2]) < 0.01, ("box still moving at the end", samples)
        assert sim["raster_refits"] + sim["raster_rebuilds"] > 0, ("motion never reached raster", sim)
        assert state["pixel_signature"] != signature0, "displayed image never changed"
        print(json.dumps({"result": "PASS", "y_start": y0, "y_end": y_end,
                          "samples": samples, "simulation": sim}, indent=2))
    finally:
        try:
            ipc.call("viewport.capture", enabled=capture_before)
        finally:
            ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--make-fixture", action="store_true")
    parser.add_argument("--project", type=Path, default=FIXTURE)
    parser.add_argument("--seconds", type=float, default=4.0)
    args = parser.parse_args()
    if args.make_fixture:
        make_fixture(args.project)
    else:
        run(args.project, args.seconds)


if __name__ == "__main__":
    sys.exit(main())
