"""External acceptance for hidden-object controls; never runs a C++ build.

Default: read-only status. --exercise-disposable changes visibility, collision policy,
room bounds and poses, then restores metadata. Studio also restores transforms; Player retains the test
positions because its IPC policy has no scene authoring. Paint, undo history and
reset counters are transient test effects and are not restored.
"""
import argparse
import copy
import math
import time
from rt_ipc import RtIpc, RtIpcError, PIPE_NAME, PLAYER_PIPE_NAME


def near(a, b):
    return all(abs(x-y) < 1e-5 for x, y in zip(a, b))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", action="store_true")
    parser.add_argument("--exercise-disposable", action="store_true")
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)

    def demo(operation="status", config=None):
        return ipc.call("paint.wet_demo", operation=operation, config=config or {})

    def hidden(operation="status", **fields):
        return demo("hidden", dict(operation=operation, **fields))

    def wait_ready():
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if all(row["ready"] for row in demo().get("colliders", []) if row["enabled"]):
                return
            time.sleep(.1)
        raise AssertionError("Paint collider cook did not finish")

    def design():
        return ipc.call("paint.wet_design")["design"]["hidden_objects"]

    original = None
    poses = {}
    try:
        status = hidden()
        assert status["count"] == len(status["objects"])
        assert len({row["object"] for row in status["objects"]}) == status["count"]
        print("PASS count / unique object names", [row["object"] for row in status["objects"]])
        if not args.exercise_disposable:
            return
        assert status["objects"], "Open a scene with a reveal-opacity wet paint object"
        assert not ipc.call("input.body_poses")["persons"], "Stop body pose senders first"
        if args.player:
            operator = ipc.call("player.operator", operation="status", config={})
            assert not operator["camera"]["active"], "Stop camera input first"
            assert operator["hidden_objects"]["count"] == status["count"]
            assert ipc.call("player.operator", operation="game.hidden.status", config={})["count"] == status["count"]
        session = ipc.call("game.session", operation="status", config={})
        assert not session["configured"] or (session["state"] == "attract" and
            not session["config"].get("auto_start") and not session["config"].get("rules")), "Use an idle disposable scene"
        assert demo()["interaction"]["held"] == -1, "Release the carried balloon first"
        wait_ready()
        original = copy.deepcopy(design())
        poses = {row["object"]: ipc.call("scene.get_world_transform", name=row["object"])["matrix"]
                 for row in original["objects"]}
        target = original["objects"][0]["object"]

        def row():
            return next(item for item in hidden()["objects"] if item["object"] == target)

        for mode in ("hidden", "shown", "paint"):
            wait_ready()
            result = hidden("update", object=target, visibility=mode)
            current = next(item for item in result["objects"] if item["object"] == target)
            assert current["visibility"] == mode and current["visible"] == (mode != "hidden")
            assert any(item["object"] == target for item in result["objects"])
        print("PASS hidden / always shown / paint reveal; registry survives reveal-off")
        for bad in ({"yaw_min": 46}, {"yaw_min": 20, "yaw_max": -20},
                    {"blocks_player": 1}, {"blocks_balloon": "false"},
                    {"randomize_on_reset": 0}, {"typo": True}):
            before = design()
            try:
                hidden("update", object=target, **bad)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Accepted invalid patch: " + repr(bad))
            assert design() == before, "Invalid patch changed design"
        print("PASS invalid values / unknown fields leave design unchanged")
        wait_ready()
        hidden("update", object=target, blocks_player=False, blocks_balloon=False)
        current = row()
        low, high = current["world_min"], current["world_max"]
        center = [(a + b) / 2 for a, b in zip(low, high)]
        assert not hidden("paint_origin", position=center)["allowed"]
        assert hidden("paint_origin", position=[high[0]+1, high[1]+1, high[2]+1])["allowed"]
        print("PASS interior paint origin rejected / exterior origin accepted")
        anchor = [center[0], low[1], center[2]]
        assert near(hidden("player_anchor", owner="hidden-test", position=anchor)["position"], anchor)
        hidden("update", object=target, blocks_player=True, blocks_balloon=True)
        accepted = hidden("player_anchor", owner="hidden-test", position=anchor)["position"]
        assert abs(accepted[0] - anchor[0]) + abs(accepted[2] - anchor[2]) > .01
        # A sweep from outside must not cross the expanded obstacle, even in one large packet.
        start = [low[0] - 1, low[1], center[2]]
        end = [high[0] + 1, low[1], center[2]]
        hidden("player_anchor", owner="sweep-test", position=start)
        accepted = hidden("player_anchor", owner="sweep-test", position=end)["position"]
        assert accepted[0] < low[0], "Player anchor tunneled through the object"
        hidden("update", object=target, blocks_player=False, blocks_balloon=False)
        assert near(hidden("player_anchor", owner="sweep-test", position=end)["position"], end)
        print("PASS nonblocking passage / optional player obstacle / swept jump / balloon flags")
        wait_ready()
        trial = copy.deepcopy(design())
        for item in trial["objects"]:
            item["randomize_on_reset"] = item["object"] == target
            item["yaw_min"] = item["yaw_max"] = 10
        hidden("configure", config=trial)
        fixed = {name: pose for name, pose in poses.items() if name != target}
        positions = []
        for mode in ("shown", "paint", "shown", "paint"):
            wait_ready()
            hidden("update", object=target, visibility=mode)
            current = row()
            center = [(a+b)/2 for a,b in zip(current["world_min"], current["world_max"])]
            radius = min(9.0, math.dist(current["world_min"], current["world_max"]) + .05)
            ipc.call("paint.wet_dab", position=center, radius=radius,
                     color=[.95, .2, .1], amount=1.0, facing=False)
            time.sleep(.1)  # Leave a painted scene / recent asynchronous texture copies at F9.
            prior_paint = ipc.call("paint.wet_status")
            previous_rebakes = prior_paint["geometry_rebakes"]
            previous_round = hidden()["round"]
            demo("reset")  # Same canonical operation used by Player F9.
            result = hidden()
            assert not result["error"], result["error"]
            assert result["round"] == previous_round + 1
            current = next(item for item in result["objects"] if item["object"] == target)
            room = result["room"]
            assert room["valid"] and "boundary_slabs" in room
            project = ipc.call("project.path").replace("\\", "/")
            if project.endswith("pastel-paint-workshop/workshop.rtp"):
                assert {"Cube_1", "Cube_2"}.issubset(room["boundary_slabs"])
                assert room["min"][1] > .03, "Used wall origin instead of floor slab top"
            if room["placement"] == "walls_floor":
                assert abs(current["world_min"][1] - room["min"][1] - .03) < 1e-4
                wall_gap = min(current["world_min"][0] - room["min"][0],
                               room["max"][0] - current["world_max"][0],
                               current["world_min"][2] - room["min"][2],
                               room["max"][2] - current["world_max"][2])
                assert wall_gap <= room["wall_band"] + .031
            camera = ipc.call("camera.get")["position"]
            nearest = [max(current["world_min"][i], min(current["world_max"][i], camera[i]))
                       for i in range(3)]
            assert math.dist(camera, nearest) + 1e-4 >= room["camera_clearance"]
            assert all(item["ready"] for item in demo()["colliders"] if item["enabled"]), "Rigid reset discarded a cooked collider"
            assert all(current["world_min"][i] >= room["min"][i] + .02 and
                       current["world_max"][i] <= room["max"][i] - .02 for i in range(3))
            pose = ipc.call("scene.get_world_transform", name=target)["matrix"]
            positions.append([pose[i][3] for i in range(3)])
            # Distinguishes an SDF issue from an atlas still baked at the old P.
            deadline = time.monotonic() + 5
            while ipc.call("paint.wet_status")["geometry_rebakes"] <= previous_rebakes:
                assert time.monotonic() < deadline, "Instance move did not invalidate paint geometry"
                time.sleep(.05)
            patched = ipc.call("paint.wet_status")
            assert patched["device_rebuilds"] == prior_paint["device_rebuilds"], "Rigid reset recreated GPU atlas buffers"
            assert patched["geometry_uploads"] > prior_paint["geometry_uploads"], "Moved geometry pages were not uploaded"
            telemetry = ipc.call("viewport.frame_telemetry")
            assert not telemetry.get("device_lost", False), telemetry
            center = [(a+b)/2 for a,b in zip(current["world_min"], current["world_max"])]
            radius = min(9.0, math.dist(current["world_min"], current["world_max"]) + .05)
            ipc.call("paint.wet_dab", position=center, radius=radius,
                     color=[.1, .7, .95], amount=1.0, facing=False)
            deadline = time.monotonic() + 5
            while True:
                coverage = ipc.call("paint.wet_coverage")
                surfaces = [surface for surface in coverage["surfaces"] if surface["object"] == target]
                if sum(surface["covered_texels"] for surface in surfaces) > .9 * sum(
                        surface["valid_texels"] for surface in surfaces) and surfaces:
                    break
                assert time.monotonic() < deadline, "World-space dab missed moved object's atlas"
                time.sleep(.05)
            ipc.call("paint.wet_reset")
            baseline = next(item["base_transform"] for item in trial["objects"] if item["object"] == target)
            angle = math.radians(10)
            for col in range(3):
                assert abs(pose[0][col] - (math.cos(angle)*baseline[col] + math.sin(angle)*baseline[8+col])) < 1e-4
                assert abs(pose[2][col] - (-math.sin(angle)*baseline[col] + math.cos(angle)*baseline[8+col])) < 1e-4
            for name, before in fixed.items():
                assert ipc.call("scene.get_world_transform", name=name)["matrix"] == before
        assert positions[0] != positions[1], "Successive F9-equivalent resets repeated the same position"
        print("PASS room bounds include object extents / limited noncumulative yaw / pinned objects / grounded wall band / camera clearance / retained SDF / moved-world atlas dab / painted shown+paint resets retain GPU atlas")
        wait_ready()
        before = ipc.call("scene.get_world_transform", name=target)["matrix"]
        hidden("bounds", mode="manual", min=[0, 0, 0], max=[.11, .11, .11])
        try:
            hidden("randomize")
        except RtIpcError:
            pass
        else:
            raise AssertionError("Impossible room accepted")
        assert ipc.call("scene.get_world_transform", name=target)["matrix"] == before
        assert design()["objects"] == trial["objects"]
        print("PASS failed placement keeps all transforms / metadata persisted in game design")
    finally:
        if original is not None:
            wait_ready()
            if not args.player:
                for name, pose in poses.items():
                    ipc.call("scene.set_transform", name=name, matrix=pose)
            # Re-register collider shapes after restoring positions.
            hidden("configure", config=original)
            for item in original["objects"]:
                if item["paint_collider_mode"] != "none" and item["visibility"] != "hidden":
                    wait_ready()
                    demo("set_collider", {"object": item["object"],
                        "mode": item["paint_collider_mode"],
                        "sdf_resolution_mode": item["sdf_resolution_mode"]})
            print("Restored design; " + ("Player test positions retained" if args.player else
                  "Studio transforms restored") + "; transient paint/history are disposable")
        ipc.close()


if __name__ == "__main__":
    main()
