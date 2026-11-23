"""Verify an updated Player with a disposable mesh-balloon room open."""

import argparse
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exercise-disposable", action="store_true", required=True)
    parser.add_argument("--expect-front-barrier", action="store_true")
    parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME)

    def demo(operation="status", **config):
        return ipc.call("paint.wet_demo", operation=operation, config=config)

    def pointer(action, **config):
        return demo("pointer", action=action, owner="jolt-motion-probe", **config)

    before = demo()
    assert before["prepared"] and before["visual_mode"] == "flat_mesh"
    assert before["interaction"]["held"] < 0, "Release the current mouse hold first"
    original = dict(before["config"])
    original.pop("surfaces", None)
    original.pop("resolution", None)
    config = dict(original, autostart=False, room_goal=0,
                  birth_seconds=0, idle_reset_seconds=0)
    for key in ("jolt_motion", "impact_soft_body", "wind_strength"):
        config.pop(key, None)
    try:
        ready = demo("configure", **config)
        status = ready["jolt_motion"]
        assert status["enabled"] and status["bodies"] == ready["config"]["balloons"], status
        assert status["soft_shells"] == status["bodies"], status
        assert ready["config"]["wind_strength"] == .25
        front = status["front_barrier"]
        if args.expect_front_barrier:
            assert front, "Expected the three-sided Workshop front barrier"
        if front:
            def distance(position):
                return sum((p - q) * inward for p, q, inward in
                           zip(position, front["point"], front["inward"]))

            nearest = min(ready["balloons"], key=lambda row: distance(row["position"]))
            selected = nearest["id"]
            position = nearest["position"]
            assert pointer("down", balloon=selected, position=position)["held"] == selected
            outside = [p - 10 * inward for p, inward in zip(position, front["inward"])]
            result = pointer("move", position=outside)
            assert result["held"] == selected and result["compression"] == 0, result
            assert distance(demo()["balloons"][selected]["position"]) > 0
            assert pointer("cancel")["held"] == -1
        try:
            demo("configure", **dict(config, jolt_motion="true"))
        except RtIpcError:
            assert demo()["jolt_motion"]["enabled"]
        else:
            raise AssertionError("Accepted an invalid jolt_motion value")

        balloon = max(ready["balloons"], key=lambda row: row["position"][1])
        balloon = demo()["balloons"][balloon["id"]]
        balloon_id = balloon["id"]
        assert pointer("down", balloon=balloon_id, position=balloon["position"])["held"] == balloon_id
        assert pointer("cancel")["held"] == -1
        before_wind = demo()["jolt_motion"]["steps"]
        wind_change = demo("wind", wind_strength=1)
        assert wind_change["config"]["wind_strength"] == 1
        assert wind_change["jolt_motion"]["steps"] >= before_wind
        floor_y = balloon["position"][1]
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            state = demo()
            if front:
                assert all(distance(b["position"]) >= -.02
                           for b in state["balloons"] if b["inflated"])
            y = state["balloons"][balloon_id]["position"][1]
            floor_y = min(floor_y, y)
            stats = state["jolt_motion"]
            if stats["room_hits"] and y - floor_y > .35 and stats["peak_shell_squash"] > .01:
                break
            time.sleep(.04)
        else:
            raise AssertionError("No room impact and subsequent rise within 12 seconds")
        assert state["balloons"][balloon_id]["inflated"]
        assert state["events_total"] == ready["events_total"], "Dropping produced a pop"
        assert stats["bodies"] == status["bodies"]
        assert stats["squashes"] > 0 and 0 < stats["peak_squash"] <= .201, stats
        assert stats["rotating_bodies"] > 0 and stats["shell_step_ms_max"] > 0, stats
        print(f'PASS {stats["bodies"]} Jolt balloons, drop and rise, '
              f'{stats["room_hits"]} room contacts, {stats["balloon_hits"]} balloon contacts, '
              f'{stats["squashes"]} impacts, peak {stats["max_ms"]:.3f} ms/tick, '
              f'{stats["shell_step_ms_max"]:.3f} ms/soft-shell step')
    finally:
        pointer("cancel")
        demo("configure", **original)
        ipc.close()


if __name__ == "__main__":
    main()
