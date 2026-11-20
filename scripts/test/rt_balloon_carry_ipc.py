"""Post-build external acceptance; use a disposable mesh-balloon scene.

python scripts/test/rt_balloon_carry_ipc.py --player --exercise-disposable
Restores demo config, not transient balloon/paint state. Does not save anything.
Mouse capture/focus/menu behavior additionally requires the manual checklist.
"""
import argparse
import math
import time

from rt_ipc import PIPE_NAME, PLAYER_PIPE_NAME, RtIpc, RtIpcError


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(a):
    length = math.sqrt(dot(a, a))
    return [x / length for x in a]


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def project(point, camera, aspect):
    forward = unit(sub(camera["target"], camera["position"]))
    right = unit(cross(forward, camera["up"]))
    up = unit(cross(right, forward))
    offset = sub(point, camera["position"])
    depth = dot(offset, forward)
    if depth <= 0:
        return None
    scale = 2 * depth * math.tan(math.radians(camera["fov"]) / 2)
    screen = [.5 + dot(offset, right) / (scale * aspect), .5 - dot(offset, up) / scale]
    return screen if all(0 <= x <= 1 for x in screen) else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", action="store_true")
    parser.add_argument("--exercise-disposable", action="store_true", required=True)
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)

    def demo(operation="status", config=None):
        return ipc.call("paint.wet_demo", operation=operation, config=config or {})

    def pointer(action, **fields):
        return demo("pointer", {"action": action, **fields})

    def row(state, id):
        return next(b for b in state["balloons"] if b["id"] == id)

    def pick():
        camera = ipc.call("camera.get")
        view = ipc.call("viewport.status")
        for balloon in demo()["balloons"]:
            screen = project(balloon["position"], camera, view["width"] / view["height"])
            if not balloon["inflated"] or screen is None:
                continue
            result = pointer("down", screen=screen)
            if result["held"] >= 0:
                return result["held"], screen
        raise AssertionError("No visible balloon in camera; frame the room first")

    before = demo()
    assert before["prepared"] and before["visual_mode"] == "flat_mesh"
    assert "interaction" in before, "Build updated Studio/Player first"
    original = dict(before["config"])
    original.pop("surfaces", None)
    original.pop("resolution", None)
    config = dict(original, interaction_mode="carry", birth_seconds=0, autostart=False,
                  respawn_seconds=30, float_amplitude=.08, drift_speed=.12,
                  room_goal=0, idle_reset_seconds=0)
    try:
        config.pop("interaction_mode", None)
        configured = demo("configure", config)
        assert configured["config"]["interaction_mode"] == "carry", "Mesh default is not carry"
        id, screen = pick()
        held = demo()
        serial = held["events_total"]
        assert row(held, id)["inflated"] and row(held, id)["held"]
        assert held["interaction"]["owner"] == "mouse"
        for invalid in ({"action": "up", "extra": True},
                        {"action": "move", "screen": [2, .5]},
                        {"action": "wheel", "wheel": True},
                        {"action": "wheel", "wheel": 101}, {"action": "up", "owner": 5}):
            snapshot = demo()["interaction"]
            try:
                demo("pointer", invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError(f"Invalid pointer accepted: {invalid}")
            assert demo()["interaction"] == snapshot
        pointer("cancel", owner="other-hand")
        assert demo()["interaction"]["held"] == id, "Other owner stole/cancelled hold"
        snapshot = demo()["interaction"]
        try:
            demo("interaction", {"mode": True})
        except RtIpcError:
            pass
        else:
            raise AssertionError("Invalid interaction mode accepted")
        assert demo()["interaction"] == snapshot
        initial = row(demo(), id)["position"]
        pointer("move", screen=[min(.95, screen[0] + .08), screen[1]])
        pointer("wheel", wheel=1)
        moved = row(demo(), id)["position"]
        assert math.dist(initial, moved) > .001, "Carry did not move balloon"
        time.sleep(.3)
        assert math.dist(moved, row(demo(), id)["position"]) < 1e-5, "Held balloon drifted"
        assert demo()["events_total"] == serial, "Holding/moving popped a balloon"
        pointer("cancel")
        cancelled = demo()
        assert cancelled["interaction"]["held"] == -1 and row(cancelled, id)["inflated"]
        assert cancelled["events_total"] == serial, "Cancel popped the balloon"
        print("PASS hold/move/depth/frozen drift/strict validation/owner/cancel")
        id, screen = pick()
        demo("burst", {"count": 3})
        assert row(demo(), id)["inflated"], "Random burst popped the held balloon"
        for _ in range(3):
            pointer("wheel", wheel=100)
        constrained = row(demo(), id)
        if "PaintBack" in ipc.call("scene.list_objects"):
            wall = ipc.call("scene.object_info", name="PaintBack")
            assert constrained["position"][2] >= wall["world_max"][2] - .001, "Passed back wall"
        expected = constrained["position"]
        result = pointer("up")
        after = demo()
        assert result["popped"] == id and result["held"] == -1
        assert not row(after, id)["inflated"]
        event = next(e for e in reversed(after["events"]) if "balloons" in e)
        actual = next(b["position"] for b in event["balloons"] if b["balloon"] == id)
        assert math.dist(actual, expected) < .02, "Release burst moved from its accepted position"
        print("PASS held excluded from random burst / wall constraint / release world position")
        demo("reset_balloons")
        id, screen = pick()
        mode = demo("interaction", {"mode": "pop"})
        assert mode["interaction"]["held"] == -1 and row(mode, id)["inflated"]
        assert mode["config"]["interaction_mode"] == "pop"
        design = ipc.call("paint.wet_design")["design"]
        assert design["balloon_demo"]["interaction_mode"] == "pop"
        camera = ipc.call("camera.get")
        view = ipc.call("viewport.status")
        for b in demo()["balloons"]:
            point = project(b["position"], camera, view["width"] / view["height"])
            if b["inflated"] and point:
                popped = pointer("down", screen=point)["popped"]
                if popped >= 0:
                    assert not row(demo(), popped)["inflated"]
                    break
        else:
            raise AssertionError("Touch/pop regression: no balloon popped")
        print("PASS mode switch cancels without pop / saved design / legacy touch-pop")
    finally:
        pointer("cancel")
        demo("configure", original)
        ipc.close()


if __name__ == "__main__":
    main()
