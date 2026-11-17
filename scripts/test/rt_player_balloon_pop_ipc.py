"""Click-to-pop on the balloon paint demo (Player, the workshop already open).

    python scripts/test/rt_player_balloon_pop_ipc.py

The Player's left click and this script go through the same core call,
paint.wet_demo operation=pop. Needs the demo prepared and autostart stopped.
Creates nothing; popped balloons come back by themselves. Checks, in order:
  1. bad requests are refused and change nothing (both keys, no key, id out of
     range, screen outside 0..1);
  2. pop by id: that balloon and only that one deflates, one event with
     particles_per_balloon droplets;
  3. popping the same id again does nothing (popped = -1, no new event);
  4. pop by screen point: scanning the window finds a balloon; the reported id
     is the one that deflated; a point that hit nothing reports -1;
  5. respawn: every balloon is inflated again within a few seconds.
"""
import json
import sys
import time

from rt_ipc import PLAYER_PIPE_NAME, RtIpc, RtIpcError


def inflated(state):
    return [b["id"] for b in state["balloons"] if b["inflated"]]


def main():
    s = RtIpc(PLAYER_PIPE_NAME)
    report = {}
    try:
        state = s.call("paint.wet_demo")
        assert state["prepared"] and not state["running"], "prepare the demo and stop autostart"
        total = len(state["balloons"])
        assert len(inflated(state)) == total, "wait until every balloon is back"
        per_balloon = state["config"]["particles_per_balloon"]

        # 1. refused requests
        refused = 0
        for bad in ({}, {"balloon": 0, "screen": [0.5, 0.5]}, {"balloon": total},
                    {"balloon": -1}, {"screen": [1.5, 0.5]}, {"screen": [0.5]}, {"count": 1}):
            try:
                s.call("paint.wet_demo", operation="pop", config=bad)
            except RtIpcError:
                refused += 1
        after = s.call("paint.wet_demo")
        report["refused"] = refused
        assert refused == 7 and after["events_total"] == state["events_total"] and \
            len(inflated(after)) == total, ("a bad pop request was accepted", report)

        # 2. pop by id
        target = total // 2
        out = s.call("paint.wet_demo", operation="pop", config={"balloon": target})
        report["by_id"] = {"popped": out["popped"], "spawned": out["events"][-1]["spawned"],
                           "deflated": sorted(set(range(total)) - set(inflated(out)))}
        assert out["popped"] == target and report["by_id"]["deflated"] == [target], report["by_id"]
        assert out["events"][-1]["spawned"] == per_balloon and \
            out["events_total"] == state["events_total"] + 1, report["by_id"]

        # 3. the same id again
        again = s.call("paint.wet_demo", operation="pop", config={"balloon": target})
        report["same_again"] = {"popped": again["popped"]}
        assert again["popped"] == -1 and again["events_total"] == out["events_total"], report

        # 4. pop by screen point
        hit, misses, tried = None, 0, 0
        for iy in range(1, 16):
            for ix in range(1, 16):
                tried += 1
                before = s.call("paint.wet_demo")
                res = s.call("paint.wet_demo", operation="pop",
                             config={"screen": [ix / 16.0, iy / 16.0]})
                if res["popped"] < 0:
                    misses += 1
                    assert inflated(res) == inflated(before), "a miss changed a balloon"
                    continue
                gone = sorted(set(inflated(before)) - set(inflated(res)))
                hit = {"screen": [ix / 16.0, iy / 16.0], "popped": res["popped"], "deflated": gone}
                break
            if hit:
                break
        report["by_screen"] = {"tried": tried, "misses": misses, "hit": hit}
        assert hit is not None, ("no window point hit a balloon (camera sees none?)", report["by_screen"])
        assert hit["deflated"] == [hit["popped"]], report["by_screen"]

        # 5. respawn
        t0 = time.time()
        while time.time() - t0 < 10.0:
            now = s.call("paint.wet_demo")
            assert not now["error"], now["error"]
            if len(inflated(now)) == total:
                break
            time.sleep(0.1)
        report["respawn_seconds"] = round(time.time() - t0, 2)
        assert len(inflated(now)) == total, ("balloons did not come back", report)
        print(json.dumps({"result": "PASS", **report}, indent=2))
    except AssertionError as e:
        print(json.dumps({"result": "FAIL", "reason": str(e), **report}, indent=2))
        sys.exit(1)
    finally:
        s.close()


if __name__ == "__main__":
    main()
