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
    parser.add_argument("--soft-body", action="store_true")
    parser.add_argument("--exercise-disposable", action="store_true", required=True)
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)

    def demo(operation="status", config=None):
        return ipc.call("paint.wet_demo", operation=operation, config=config or {})

    def pointer(action, **fields):
        return demo("pointer", {"action": action, **fields})

    def row(state, id):
        return next(b for b in state["balloons"] if b["id"] == id)

    def check_opaque(state):
        config = state["config"]
        assert "opaque_balloons" not in config and "transmission" not in config
        for node in config["visual_nodes"]:
            assert ipc.call("material.get", object_name=node, param="transmission") == 0, node
            assert ipc.call("material.get", object_name=node, param="opacity") == 1, node

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
    check_opaque(before)
    original = dict(before["config"])
    original.pop("surfaces", None)
    original.pop("resolution", None)
    config = dict(original, interaction_mode="carry", birth_seconds=0, autostart=False,
                  respawn_seconds=30, float_amplitude=.08, drift_speed=.12,
                  room_goal=0, idle_reset_seconds=0, soft_body=args.soft_body,
                  jolt_motion=False)
    try:
        config.pop("interaction_mode", None)
        configured = demo("configure", dict(config, opaque_balloons=False, transmission=0.9))
        check_opaque(configured)
        assert configured["config"]["interaction_mode"] == "carry", "Mesh default is not carry"
        if args.soft_body:
            assert configured["soft_body"], "Jolt shell did not initialize"
            assert "PaintBack" in ipc.call("scene.list_objects"), "Soft-body probe needs PaintBack"
            try:
                demo("configure", dict(config, soft_body="true"))
            except RtIpcError:
                assert demo()["soft_body"], "Invalid configuration disturbed the live demo"
            else:
                raise AssertionError("Non-boolean soft_body was accepted")
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

        # World pointer input goes through the same carry solver as mouse input.
        # Freeze passive bob/drift while checking the swept contact geometry.
        demo("configure", dict(config, float_amplitude=0, drift_speed=0))
        demo("reset_balloons")
        balloons = [b for b in demo()["balloons"] if b["inflated"]]
        pairs = sorted(
            (math.dist(a["position"], b["position"]), a["id"], b["id"])
            for a in balloons for b in balloons if a["id"] != b["id"]
        )
        def visual_bounds(balloon):
            info = ipc.call("scene.object_info", name=balloon["visual_node"])
            radii = [(high - low) * .5 for low, high in zip(info["world_min"], info["world_max"])]
            center = [(high + low) * .5 for low, high in zip(info["world_min"], info["world_max"])]
            return radii, sub(center, balloon["position"])

        for _, source_id, target_id in pairs:
            state = demo()
            source, target = row(state, source_id), row(state, target_id)
            start, obstacle = source["position"], target["position"]
            source_radii, source_offset = visual_bounds(source)
            target_radii, target_offset = visual_bounds(target)
            radii = [a + b for a, b in zip(source_radii, target_radii)]
            delta = [b + ob - a - oa for a, b, oa, ob in
                     zip(start, obstacle, source_offset, target_offset)]
            separation = math.sqrt(dot(delta, delta))
            if separation < .02:
                continue
            direction = unit(delta)
            reach = 1 / math.sqrt(sum((d / r) ** 2 for d, r in zip(direction, radii)))
            if separation <= reach + .06:
                continue
            near = [x + d * (separation - reach - .04) for x, d in zip(start, direction)]
            end = [x + d * (separation + .25) for x, d in zip(start, direction)]
            fields = {"owner": "carry-contact-probe"}
            try:
                down = pointer("down", balloon=source_id, position=start, **fields)
                assert down["held"] == source_id
                pointer("move", position=near, **fields)
                assert math.dist(row(demo(), target_id)["position"], obstacle) < .002, (
                    "Balloon moved before its visual contact envelope was reached"
                )
                pointer("move", position=end, **fields)
                after = demo()
            finally:
                pointer("cancel", **fields)
            moved_source = row(after, source_id)["position"]
            moved_target = row(after, target_id)["position"]
            center_gap = [a + oa - b - ob for a, b, oa, ob in
                          zip(moved_source, moved_target, source_offset, target_offset)]
            assert sum((d / r) ** 2 for d, r in zip(center_gap, radii)) >= .98
            if dot(sub(moved_target, obstacle), direction) > .02:
                break
        else:
            raise AssertionError("No free balloon pair could be pushed; check room clearance")
        print("PASS swept carry pushes another balloon without crossing it")

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
        if "PaintBack" in ipc.call("scene.list_objects"):
            wall = ipc.call("scene.object_info", name="PaintBack")
            wall_z = wall["world_max"][2]
            mid_x = (wall["world_min"][0] + wall["world_max"][0]) * .5
            candidate = min(
                (b for b in demo()["balloons"] if b["inflated"]),
                key=lambda b: b["position"][2] - wall_z + abs(b["position"][0] - mid_x),
            )
            id = candidate["id"]
            start = candidate["position"]
            fields = {"owner": "wall-slide-probe"}
            assert pointer("down", balloon=id, position=start, **fields)["held"] == id
            try:
                press = [start[0], start[1], wall_z - 1.0]
                assert pointer("move", position=press, **fields)["compression"] > 0
                at_wall = row(demo(), id)["position"]
                sideways = .22 if at_wall[0] < mid_x else -.22
                slide = [at_wall[0] + sideways, at_wall[1], at_wall[2] - .04]
                pointer("move", position=slide, **fields)
                slid = row(demo(), id)["position"]
                assert abs(slid[0] - at_wall[0]) > .12, "Wall contact blocked lateral motion"
                assert slid[2] > wall_z, "Sliding crossed the wall"
                assert row(demo(), id)["inflated"], "A glancing slide popped the balloon"

                press = [slid[0], slid[1], wall_z - 1.0]
                assert pointer("move", position=press, **fields)["compression"] > 0
                deadline = time.monotonic() + 2.0
                while row(demo(), id)["inflated"] and time.monotonic() < deadline:
                    time.sleep(.03)
                crushed = demo()
                assert not row(crushed, id)["inflated"], "Sustained wall pressure did not pop"
                assert crushed["interaction"]["held"] == -1
                if args.soft_body:
                    assert crushed["soft_body_min_axial_scale"] < .995, "Jolt shell did not yield"
                    assert crushed["soft_body_axial_scale"] == 1, "Popped shell remained active"
                    print(f'PASS Jolt shell yielded, peak CPU {crushed["soft_body_ms_max"]:.3f} ms')
                event = next(e for e in reversed(crushed["events"]) if "balloons" in e)
                balloon_pop = next(b for b in event["balloons"] if b["balloon"] == id)
                assert balloon_pop["drops"] > 0, "Squeeze pop emitted no paint"
                assert math.dist(balloon_pop["position"], slid) < .025, (
                    "Squeeze pop shifted the balloon position"
                )
            finally:
                pointer("cancel", **fields)
            print("PASS wall slide and squeeze pop at the balloon position")
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
