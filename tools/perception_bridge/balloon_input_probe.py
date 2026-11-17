"""External body -> arms-up rule -> balloon droplets -> GPU paint acceptance.

Never launches/builds an application or loads a project. The live run requires
the prepared balloon fixture already open, no people, and an attract session.
The demo scheduler is stopped; no direct burst, post_event or paint dab is sent.
This tests gesture-triggered random balloon selection, NOT spatial hand hits.

    python tools/perception_bridge/balloon_input_probe.py --plan
    python tools/perception_bridge/balloon_input_probe.py --run --reset-paint

The run owns transient poses/session/paint in this test fixture. Display pacing
and a previously configured attract session's config are restored. If there was
no session config, the test leaves its isolated config in attract. Demo scheduling
stays stopped. Paint is cleared at the end; counters/live artwork are not restored.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

try:
    from .fake_children import Children
    from .schema import validate_pose
    from .transport import PIPE, connect, summary_ms
except ImportError:
    from fake_children import Children
    from schema import validate_pose
    from transport import PIPE, connect, summary_ms

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/dev/fixtures/balloon-paint-room/balloons.rtp"
RULE = "body_probe_arms_up"


def session_config():
    return {
        "duration_seconds": 180, "finale_seconds": 1, "reset_seconds": 0.5,
        "coverage_goal": 1.0, "auto_start": False,
        "zones": [], "contact_targets": [],
        "rules": [{"id": RULE, "event": "gesture.arms_up", "state": "playing",
                   "cooldown_seconds": 0.5,
                   "action": {"type": "balloon.burst", "count": 1}}],
    }


def packet(sender_t, raised):
    value = copy.deepcopy(Children(1, seed=417).frame(0))
    value["t"] = sender_t
    joints = value["persons"][0]["joints"]
    for side in ("l", "r"):
        joints[f"{side}_wrist"][1] = joints["head"][1] + 0.25 if raised else 0.65
    return validate_pose(value)


def rule_fires(snapshot):
    rules = snapshot["session"]["rules"]
    assert len(rules) == 1 and rules[0]["id"] == RULE, "test rule set was replaced"
    return rules[0]["fires"]


def validate_trial(before, after):
    """Check causal evidence, not just an action-success counter or visible paint."""
    assert not before["demo"]["running"] and not after["demo"]["running"], \
        "automatic demo scheduling confounds the input test"
    assert before["coverage"]["covered_texels"] == 0, "trial did not start on a clean atlas"
    assert after["body"]["gestures"] - before["body"]["gestures"] == 1, \
        "expected exactly one gesture edge, including the held-high interval"
    for name in ("gesture.arms_up", "balloon.pop"):
        delta = (after["session"]["events"].get(name, 0) -
                 before["session"]["events"].get(name, 0))
        assert delta == 1, f"expected exactly one {name}, got {delta}"
    assert rule_fires(after) - rule_fires(before) == 1, "rule fired zero or multiple times"
    assert after["session"]["actions_ok"] - before["session"]["actions_ok"] == 1
    assert after["session"]["actions_failed"] == before["session"]["actions_failed"]
    assert after["session"]["dropped_events"] == before["session"]["dropped_events"]
    assert after["demo"]["events_total"] - before["demo"]["events_total"] == 1
    event = after["demo"]["events"][-1]
    expected_drops = after["demo"]["config"]["particles_per_balloon"]
    assert event["simultaneous"] == 1 and event["spawned"] == expected_drops, event
    assert len(event["balloons"]) == 1 and event["frames"] > 0, event
    assert event["sources_dropped"] == 0, event
    assert after["coverage"]["covered_texels"] > 0, "no physical droplet paint reached the atlas"
    wet_before, wet_after = before["wet"], after["wet"]
    # stamped_texels is a last-tick gauge and can legitimately be zero after
    # the drops drain. Coverage + cumulative device counters establish the chain.
    for key in ("device_event_dispatches", "publish_device_copies"):
        assert wet_after[key] > wet_before[key], f"no GPU progress: {key}"
    for key in ("publish_host_downloads", "publish_bytes_downloaded", "sources_dropped"):
        assert wet_after[key] == wet_before[key], f"unexpected host path/loss: {key}"
    assert not after["runtime"]["device_lost"], after["runtime"]
    assert after["runtime"]["simulation"]["particles_alive"] == len(after["demo"]["balloons"]), \
        "droplets have not drained back to the stationary balloon visuals"
    assert all(balloon["inflated"] for balloon in after["demo"]["balloons"]), \
        "balloon respawn has not completed"
    return {"gesture_edges": 1, "balloons_popped": 1, "drops_spawned": expected_drops,
            "covered_texels": after["coverage"]["covered_texels"],
            "painted_surfaces": sum(s["covered_texels"] > 0
                                    for s in after["coverage"]["surfaces"])}


class Probe:
    def __init__(self, ipc, report):
        self.ipc = ipc
        self.report = report
        self.sender_t = 0.0
        self.rpc_ms = []

    def call(self, method, **params):
        # An allow-list protects this instrument from becoming a synthetic
        # success through a direct paint/burst shortcut.
        if method == "paint.wet_demo":
            assert params.get("operation", "status") in ("status", "stop")
        assert method not in ("game.post_event", "paint.wet_dab")
        started = time.perf_counter()
        value = self.ipc.call(method, **params)
        self.rpc_ms.append((time.perf_counter() - started) * 1000)
        self.report["calls"].append({"method": method, "params": params})
        return value

    def snapshot(self):
        return {
            "body": self.call("input.body_poses"),
            "session": self.call("game.session"),
            "demo": self.call("paint.wet_demo", operation="status"),
            "coverage": self.call("paint.wet_coverage"),
            "wet": self.call("paint.wet_status"),
            "runtime": self.call("agent.get_state_summary", include_probe=False)["runtime"],
        }

    def hold(self, raised, seconds):
        deadline = time.perf_counter() + seconds
        next_send = time.perf_counter()
        while time.perf_counter() < deadline:
            self.call("input.set_body_poses", **packet(self.sender_t, raised))
            self.sender_t += 1 / 30
            next_send += 1 / 30
            time.sleep(max(0, min(next_send - time.perf_counter(), deadline - time.perf_counter())))


def run(args):
    report = {"status": "NOT_RUN", "scope": "synthetic arms-up to random balloon / GPU paint",
              "camera_acceptance": False, "spatial_hand_hit_acceptance": False,
              "visual_pixel_acceptance": False,
              "calls": [], "trials": [], "errors": []}
    ipc = None
    probe = None
    original_session = None
    original_display = None
    owns_fixture = False
    try:
        ipc = connect(args.pipe)
        probe = Probe(ipc, report)
        current = probe.call("project.path")
        if not current or Path(current).resolve() != FIXTURE.resolve():
            raise RuntimeError(f"prepared balloon fixture must already be open; current={current}")
        demo = probe.call("paint.wet_demo", operation="status")
        if not demo["prepared"]:
            raise RuntimeError("balloon fixture is not prepared")
        body = probe.call("input.body_poses")
        if body["persons"]:
            raise RuntimeError("existing body sender/persons must be stopped before this probe")
        original_session = probe.call("game.session")
        if original_session["state"] != "attract":
            raise RuntimeError("active game session must not be replaced")
        original_display = probe.call("player.display")
        report["preflight"] = {"project": current, "demo": demo,
                               "session": original_session, "display": original_display}
        if not args.reset_paint:
            raise RuntimeError("--reset-paint is required: this controlled fixture test clears its atlas")
        owns_fixture = True
        probe.call("paint.wet_demo", operation="stop")
        probe.call("input.clear_body_poses")
        probe.call("player.set_display", continuous=True, pacing_hz=60.0)
        probe.call("game.session", operation="configure", config=session_config())
        probe.call("game.session", operation="start")

        for trial in range(args.trials):
            # Neutral input cannot pop a balloon, even over one full automatic
            # scheduling window (the timer must remain stopped throughout).
            idle = probe.call("paint.wet_demo", operation="status")
            before_idle = idle["events_total"]
            probe.hold(False, 5.2 if trial == 0 else 0.65)
            neutral = probe.call("paint.wet_demo", operation="status")
            assert neutral["time_s"] > idle["time_s"], "shared simulation clock did not advance"
            assert neutral["events_total"] == before_idle, \
                "balloon popped without an arms-up gesture"
            probe.call("paint.wet_reset")
            reset_deadline = time.monotonic() + 5
            while probe.call("paint.wet_coverage")["covered_texels"] != 0:
                assert time.monotonic() < reset_deadline, "atlas reset did not settle"
                probe.hold(False, 0.1)
            before = probe.snapshot()
            assert before["session"]["state"] == "playing", before["session"]

            # Keep sending the same raised pose through the physical drop
            # lifetime + respawn. A held gesture must not fire again.
            probe.hold(True, 4.5)
            after = probe.snapshot()
            evidence = validate_trial(before, after)
            report["trials"].append({"index": trial, "before": before,
                                     "after": after, "evidence": evidence})
        report["status"] = "PASS"
    except Exception as error:
        report["status"] = "FAIL" if owns_fixture else "NOT_READY"
        report["errors"].append(str(error))
    finally:
        if ipc is not None:
            if owns_fixture:
                try:
                    probe.call("input.clear_body_poses")
                    probe.call("game.session", operation="reset")
                    deadline = time.monotonic() + 8
                    while probe.call("game.session")["state"] != "attract":
                        if time.monotonic() > deadline:
                            raise TimeoutError("session cleanup did not return to attract")
                        time.sleep(0.05)
                    probe.call("paint.wet_demo", operation="stop")
                    probe.call("paint.wet_reset")
                    if original_session["configured"]:
                        probe.call("game.session", operation="configure",
                                   config=original_session["config"])
                    probe.call("player.set_display",
                               continuous=original_display["continuous"],
                               pacing_hz=float(original_display["pacing_hz"]))
                    report["cleanup"] = {"session": probe.call("game.session"),
                                         "body": probe.call("input.body_poses"),
                                         "demo": probe.call("paint.wet_demo", operation="status")}
                except Exception as error:
                    report["status"] = "FAIL"
                    report["errors"].append("cleanup: " + str(error))
            ipc.close()
        if probe:
            report["rpc_roundtrip"] = summary_ms(probe.rpc_ms)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"status": report["status"], "trials": len(report["trials"]),
                      "errors": report["errors"], "report": str(args.output)}, ensure_ascii=False))
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="Print the isolated rule, no connection")
    mode.add_argument("--run", action="store_true", help="Mutate only the pre-opened test fixture")
    parser.add_argument("--reset-paint", action="store_true")
    parser.add_argument("--trials", type=int, choices=range(1, 4), default=3)
    parser.add_argument("--pipe", default=PIPE)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "tools/perception_bridge/out/balloon-input-probe.json")
    args = parser.parse_args()
    if args.plan:
        print(json.dumps({"fixture": str(FIXTURE), "session": session_config(),
                          "scope": "arms-up -> random balloon; no spatial hand hit or camera claim"},
                         indent=2))
        return 0
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
