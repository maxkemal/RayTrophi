"""Camera-free body -> two-hand balloon interaction acceptance.

--plan validates synthetic trajectories offline; never contacts an application.
--run --exercise-disposable streams real input.set_body_poses packets at 30 Hz.
No pointer/pop/burst events are sent: the C++ recognizer must produce all actions.
Requires an updated Player/Studio with a disposable prepared mesh balloon scene.
"""
import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/perception_bridge"))
from schema import JOINTS, validate_pose


def packet(t, center, radius, phase, person=9001):
    x, y, z = center
    joints = {name: [x, y, z, 1.0] for name in JOINTS}
    joints.update(head=[x, y + .75, z, 1.0], neck=[x, y + .5, z, 1.0],
                  pelvis=[x, y - .65, z, 1.0])
    for side, sign in (("l", -1), ("r", 1)):
        for joint, dy, width in (("shoulder", .4, .18), ("elbow", .15, .25),
                                 ("hip", -.65, .12), ("knee", -1.0, .12),
                                 ("ankle", -1.3, .12)):
            joints[f"{side}_{joint}"] = [x + sign * width, y + dy, z, 1.0]
        span = 3.0 if phase == "neutral" else (1.6 if phase == "release" else .7)
        joints[f"{side}_wrist"] = [x + sign * radius * span, y, z, 1.0]
        if phase == "neutral":
            # Deliberately exaggerated withdrawal: keep wrists trustworthy but
            # outside every balloon in the small room, avoiding incidental taps.
            joints[f"{side}_wrist"][2] = z + 10.0
    if phase in ("single", "approach"):
        joints["r_wrist"] = [x + 3 * radius, y, z, 0.0 if phase == "single" else 1.0]
    if phase == "low_confidence":
        joints["r_wrist"][3] = 0.0
    persons = [] if phase == "missing" else [
        {"id": person, "confidence": 1.0, "joints": joints}]
    return validate_pose({"t": t, "persons": persons})


def plan():
    t = 0.0
    count = 0
    for phase, seconds in [("neutral", .3), ("approach", .08), ("grip", .4),
                           ("move", .8), ("release", .4), ("single", .4),
                           ("low_confidence", .25), ("missing", 1.1)]:
        previous = None
        for _ in range(math.ceil(seconds * 30)):
            t += 1 / 30
            value = packet(t, [0.0, 1.5, -2.0], .4, phase)
            validate_pose(value, previous)
            previous = value["t"]
            count += 1
    print(json.dumps({"offline_schema": "PASS", "packets": count, "joints": len(JOINTS),
                      "camera_required": False, "recognizer_executed": False}, indent=2))


def live(player):
    sys.path.insert(0, str(ROOT / "scripts/test"))
    from rt_ipc import RtIpc, PLAYER_PIPE_NAME, PIPE_NAME, RtIpcError
    ipc = RtIpc(PLAYER_PIPE_NAME if player else PIPE_NAME)
    original = None
    changed = False
    sequence = 0
    try:
        def demo(operation="status", config=None):
            return ipc.call("paint.wet_demo", operation=operation, config=config or {})

        before = demo()
        if "body_interaction" not in before:
            raise RuntimeError("Build the new C++ core before running this probe")
        assert before["prepared"] and before["visual_mode"] == "flat_mesh"
        assert before["interaction"]["held"] == -1, "Release the mouse-owned hold first"
        body = ipc.call("input.body_poses")
        assert not body["persons"], "Stop real/synthetic pose senders before this probe"
        if player:
            operator = ipc.call("player.operator", operation="status", config={})
            assert not operator["camera"]["active"], "Stop the camera before synthetic input"
        session = ipc.call("game.session", operation="status", config={})
        assert not session["configured"] or (
            session["state"] == "attract" and not session["config"].get("auto_start") and
            not session["config"].get("rules")), "Use an idle session without active rules"
        original = dict(before["config"])
        original.pop("surfaces", None)
        original.pop("resolution", None)
        config = dict(original, body_controls=True, autostart=False, birth_seconds=0,
                      respawn_seconds=30, float_amplitude=0, drift_speed=0,
                      room_goal=0, idle_reset_seconds=0)
        demo("configure", config)
        changed = True
        ipc.call("input.clear_body_poses")

        def stream(phase, center, radius, seconds, destination=None):
            nonlocal sequence
            frames = math.ceil(seconds * 30)
            for frame in range(frames):
                started = time.monotonic()
                fraction = frame / max(1, frames - 1)
                point = center if destination is None else [
                    a + (b - a) * fraction for a, b in zip(center, destination)]
                sequence += 1
                value = packet(sequence / 30, point, radius, phase)
                ipc.call("input.set_body_poses", **value)
                remaining = 1 / 30 - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
            return demo()

        def target():
            demo("reset_balloons")
            row = next(b for b in demo()["balloons"] if b["inflated"])
            return row["id"], row["position"], .4 * row["size"]

        def balloon(state, id):
            return next(b for b in state["balloons"] if b["id"] == id)

        # Two hands arriving after the first hand must beat the single touch delay.
        id, center, radius = target()
        stream("neutral", center, radius, .3)
        serial = demo()["events_total"]
        stream("approach", center, radius, .08)
        held = stream("grip", center, radius, .4)
        assert held["interaction"]["held"] == id, held["body_interaction"]
        assert held["interaction"]["owner"] == "body:9001:both"
        assert held["events_total"] == serial, "The first hand prematurely popped the balloon"
        destination = [center[0] + .25, center[1] + .12, center[2] - .12]
        moved = stream("move", center, radius, .8, destination)
        accepted = balloon(moved, id)["position"]
        assert math.dist(accepted, center) > .01, "Two-hand midpoint did not carry the balloon"
        assert moved["events_total"] == serial
        released = stream("release", destination, radius, .4)
        assert released["interaction"]["held"] == -1
        assert not balloon(released, id)["inflated"]
        assert released["body_interaction"]["releases"] == 1
        event = next(e for e in reversed(released["events"]) if "balloons" in e)
        position = next(b["position"] for b in event["balloons"] if b["balloon"] == id)
        assert math.dist(position, accepted) < .02, "Release popped at a different world position"
        print("PASS two-hand grip / virtual walking midpoint / separated-hand release position")

        # Exit the previous gesture, then a single visible wrist pops by contact.
        stream("neutral", destination, radius, .3)
        id, center, radius = target()
        stream("neutral", center, radius, .3)
        stream("single", center, radius, .08)
        tapped = stream("neutral", center, radius, .35)
        assert not balloon(tapped, id)["inflated"] and tapped["body_interaction"]["taps"] == 1
        print("PASS one-hand touch; no direct pop/pointer command sent")

        stream("neutral", center, radius, .3)
        id, center, radius = target()
        stream("neutral", center, radius, .3)
        assert stream("grip", center, radius, .4)["interaction"]["held"] == id
        serial = demo()["events_total"]
        low = stream("low_confidence", center, radius, .25)
        assert low["interaction"]["held"] == id and low["events_total"] == serial
        assert stream("grip", center, radius, .25)["interaction"]["held"] == id
        missing = stream("missing", center, radius, 1.1)
        assert missing["interaction"]["held"] == -1
        assert balloon(missing, id)["inflated"] and missing["events_total"] == serial
        assert missing["body_interaction"]["tracking_cancels"] == 1
        print("PASS confidence loss freezes / same identity resumes / missing packets cancel safely")

        snapshot = demo("body")
        for invalid in ({"enabled": 1}, {"enabled": True, "unknown": 0}):
            try:
                demo("body", invalid)
            except RtIpcError:
                pass
            else:
                raise AssertionError("Invalid body-controls input accepted")
            assert demo("body")["enabled"] == snapshot["enabled"]
        saved = ipc.call("paint.wet_design")["design"]
        assert saved["balloon_demo"]["body_controls"] is True
        print("PASS validation / persisted design", json.dumps(demo("body")))
    finally:
        if changed:
            ipc.call("input.clear_body_poses")
            ipc.call("paint.wet_demo", operation="configure", config=original)
        ipc.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--studio", action="store_true")
    parser.add_argument("--exercise-disposable", action="store_true")
    args = parser.parse_args()
    if args.run:
        if not args.exercise_disposable:
            parser.error("--run requires --exercise-disposable; transient paint/balloons are not restored")
        live(not args.studio)
    else:
        plan()


if __name__ == "__main__":
    main()
