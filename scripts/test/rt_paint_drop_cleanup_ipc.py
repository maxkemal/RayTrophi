"""External regression: visible paint drops must age out and clear with the atlas.
Requires a freshly built application and a disposable mesh-balloon scene.
Restores demo settings, but transient paint/particle/balloon state is not restored.
"""
import argparse
import sys
import time
from rt_ipc import RtIpc, PLAYER_PIPE_NAME, PIPE_NAME


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", action="store_true")
    parser.add_argument("--exercise-disposable", action="store_true", required=True)
    args = parser.parse_args()
    ipc = RtIpc(PLAYER_PIPE_NAME if args.player else PIPE_NAME)
    original = None

    def demo(operation="status", config=None):
        return ipc.call("paint.wet_demo", operation=operation, config=config or {})

    def pools():
        return demo()["pools"]

    def pop_outside():
        demo("reset_balloons")
        for balloon in demo()["balloons"]:
            allowed = demo("hidden", {"operation": "paint_origin", "position": balloon["position"]})
            if balloon["inflated"] and allowed["allowed"]:
                return demo("pop", {"balloon": balloon["id"]})
        raise AssertionError("No balloon outside reveal-object exclusion boxes")

    try:
        before = demo()
        assert before["prepared"] and before["interaction"]["held"] == -1
        assert not ipc.call("input.body_poses")["persons"], "Stop pose senders"
        if args.player:
            operator = ipc.call("player.operator", operation="status", config={})
            assert not operator["camera"]["active"], "Stop camera input"
        session = ipc.call("game.session", operation="status", config={})
        assert not session["configured"] or (session["state"] == "attract" and
            not session["config"].get("auto_start") and not session["config"].get("rules"))
        assert all(row["ready"] for row in before["colliders"] if row["enabled"]), "Wait for SDF cooks"
        original = dict(before["config"])
        original.pop("surfaces", None)
        original.pop("resolution", None)
        trial = dict(original, autostart=False, body_controls=False, idle_reset_seconds=0,
                     finale_reset_seconds=0, room_goal=0, birth_seconds=0)
        demo("configure", trial)
        ipc.call("paint.wet_reset")
        pop_outside()
        time.sleep(.15)
        active = pools()
        assert sum(row["alive"] for row in active) > 0, "No paint drops spawned"
        assert all("host_consumer_collider_sdf_not_cooked" != row["gpu_status"] for row in active)
        time.sleep(3.2)
        expired = pools()
        assert sum(row["alive"] for row in expired) == 0, expired
        print("PASS emitted drops age out; no frozen host pool")
        pop_outside()
        time.sleep(.1)
        assert sum(row["alive"] for row in pools()) > 0
        ipc.call("paint.wet_reset")
        time.sleep(.25)
        assert sum(row["alive"] for row in pools()) == 0
        first = ipc.call("paint.wet_coverage")
        time.sleep(.5)
        second = ipc.call("paint.wet_coverage")
        assert first["covered_texels"] == second["covered_texels"] == 0, (first, second)
        effects = demo()["effects"]["systems"]
        assert effects["BalloonDropGhosts"]["alive"] == 0
        print("PASS reset clears drops / ghosts / atlas; no post-reset repaint")
    finally:
        if original is not None:
            ipc.call("paint.wet_reset")
            demo("configure", original)
        ipc.close()


if __name__ == "__main__":
    main()
