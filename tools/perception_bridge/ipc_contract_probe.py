"""External IPC contract acceptance. Never builds or launches an application.

Body/invalid-request probes work in an empty Studio. Full Player probes require
the prepared balloon fixture and an idle (attract) game session.
"""
import argparse
import copy
import json
from pathlib import Path
import time

try:
    from .fake_children import Children
    from .transport import PIPE, connect
except ImportError:
    from fake_children import Children
    from transport import PIPE, connect


def run(args):
    report = {"status": "RUNNING", "pipe": args.pipe, "checks": {}, "errors": [],
              "scope": "body and validation" if args.body_only else "Player live contracts"}
    ipc = None
    original = None
    mutated_session = False
    owns_input = False

    def check(name, condition, evidence=None):
        report["checks"][name] = {"pass": bool(condition), "evidence": evidence}
        if not condition:
            raise AssertionError(name)

    def rejected(name, method, params):
        # A disconnected pipe is never accepted as an engine validation result.
        from rt_ipc import RtIpcError
        try:
            ipc.call(method, **params)
        except RtIpcError as exc:
            text = str(exc)
            check(name, text.startswith(method + ":"), text)
        else:
            check(name, False, "request accepted")

    def wait_until(reader, predicate, timeout=5):
        deadline = time.monotonic() + timeout
        while True:
            value = reader()
            if predicate(value):
                return value
            if time.monotonic() >= deadline:
                raise TimeoutError("condition not reached: " + str(value))
            time.sleep(0.03)

    def session():
        return ipc.call("game.session")

    def configure(config):
        nonlocal mutated_session
        value = ipc.call("game.session", operation="configure", config=config)
        mutated_session = True
        return value

    def post(target, person):
        ipc.call("game.post_event", event={"type": "contact", "target": target,
                                          "person": person, "value": 1})

    try:
        ipc = connect(args.pipe)
        original = session()
        if not args.body_only and (not original["configured"] or original["state"] != "attract"):
            raise RuntimeError("full probe requires a configured attract session")
        active = ipc.call("input.body_poses")
        check("no_existing_people", not active["persons"])
        owns_input = True
        ipc.call("input.clear_body_poses")
        packet = Children(1).frame(0)
        ipc.call("input.set_body_poses", **packet)
        before = ipc.call("input.body_poses")
        check("body_accepted", len(before["persons"]) == 1 and before["t"] == 0)
        invalid = []
        value = copy.deepcopy(packet)
        value["t"] = True
        invalid.append(("boolean_time", value))
        value = copy.deepcopy(packet)
        value["t"] = 0.1
        value["persons"] *= 2
        invalid.append(("duplicate_ids", value))
        for name, field, bad in (("boolean_id", "id", True),
                                 ("overflow_id", "id", 2**31),
                                 ("bad_confidence", "confidence", 1.1)):
            value = copy.deepcopy(packet)
            value["t"] = 0.1
            value["persons"][0][field] = bad
            invalid.append((name, value))
        value = copy.deepcopy(packet)
        value["t"] = 0.1
        del value["persons"][0]["joints"]["head"]
        invalid.append(("missing_joint", value))
        value = copy.deepcopy(packet)
        value["t"] = 0.1
        value["persons"][0]["joints"]["head"][0] = 10001
        invalid.append(("coordinate_limit", value))
        invalid.append(("replayed_time", packet))
        for name, value in invalid:
            rejected(name, "input.set_body_poses", value)
        after = ipc.call("input.body_poses")
        check("invalid_packets_atomic", before["packets"] == after["packets"] and
              before["t"] == after["t"])

        ipc.call("input.clear_body_poses")
        baseline = ipc.call("input.body_poses")["gestures"]
        ipc.call("input.set_body_poses", **packet)
        for t, arms, foot, reach in ((0.1, True, 0, False), (0.2, True, 0, False),
                                     (0.3, False, 0.22, False),
                                     (0.4, False, 0, False), (0.5, False, 0, True)):
            value = copy.deepcopy(packet)
            value["t"] = t
            joints = value["persons"][0]["joints"]
            if arms:
                joints["l_wrist"][1] = joints["r_wrist"][1] = 1.55
            joints["r_ankle"][1] = foot
            if reach:
                joints["l_wrist"][2] = joints["pelvis"][2] - 0.6
            ipc.call("input.set_body_poses", **value)
        gestures = ipc.call("input.body_poses")["gestures"] - baseline
        check("three_gestures_with_arms_edge", gestures == 3, gestures)

        ipc.call("input.clear_body_poses")
        ipc.call("input.set_body_poses", **packet)
        begin = time.monotonic()
        ipc.call("input.set_body_poses", t=1000000, persons=[])
        check("sender_jump_preserves_grace", len(ipc.call("input.body_poses")["persons"]) == 1)
        wait_until(lambda: ipc.call("input.body_poses"), lambda v: not v["persons"], 2)
        elapsed = time.monotonic() - begin
        check("receiver_timeout", 0.40 < elapsed < 1.5, {"observed_seconds": elapsed})
        ipc.call("input.clear_body_poses")
        ipc.call("input.set_body_poses", **packet)
        check("clear_restarts_sender_epoch", ipc.call("input.body_poses")["t"] == 0)
        ipc.call("input.clear_body_poses")

        previous = session()
        for name, config in (("unknown_config_field", {"typo": 1}),
                              ("negative_duration", {"duration_seconds": -1}),
                              ("bad_rule_action", {"rules": [{"id": "bad", "event": "contact",
                                               "action": {"type": "not_supported"}}]})):
            rejected(name, "game.session", {"operation": "configure", "config": config})
        current = session()
        check("invalid_session_config_atomic", current["config"] == previous["config"] and
              current["rules"] == previous["rules"])
        rejected("invalid_event_type", "game.post_event", {"event": {"type": "unknown"}})
        rejected("invalid_event_vector", "game.post_event",
                 {"event": {"type": "contact", "position": [0, 0]}})

        if not args.body_only:
            wait_until(session, lambda v: v["state"] == "attract", 5)
            configure({"duration_seconds": 20, "coverage_goal": 1,
                "rules": [{"id": "group", "event": "contact", "state": "playing",
                           "min_persons": 2, "group_window_seconds": 0.75,
                           "cooldown_seconds": 1,
                           "action": {"type": "paint.dab", "position": [0, 0, 1.5]}}]})
            ipc.call("game.session", operation="start")
            rejected("start_requires_attract", "game.session", {"operation": "start"})
            for target, person in (("QA_A", 1), ("QA_A", 1), ("QA_B", 2)):
                post(target, person)
            check("same_person_and_distinct_target_not_group", session()["rules"][0]["fires"] == 0)
            post("QA_A", 2)
            wait_until(session, lambda v: v["rules"][0]["fires"] == 1)
            post("QA_B", 1)
            grouped = wait_until(session, lambda v: v["rules"][0]["fires"] == 2)
            check("independent_target_groups", grouped["actions_ok"] == 2 and
                  grouped["actions_failed"] == 0, grouped)
            post("QA_A", 1)
            post("QA_A", 2)
            check("group_cooldown", session()["rules"][0]["fires"] == 2)

            configure({"duration_seconds": 10, "coverage_goal": 1,
                       "contact_targets": ["PaintFloor"], "rules": [
                           {"id": "floor_contact", "event": "contact", "state": "playing",
                            "min_persons": 2, "cooldown_seconds": 1,
                            "action": {"type": "paint.dab", "position": [0, 0, 1.5]}}]})
            ipc.call("game.session", operation="start")
            ipc.call("input.set_body_poses", **Children(2).frame(0))
            contact = wait_until(session, lambda v: v["rules"][0]["fires"] >= 1)
            check("real_flat_floor_proxy_group", contact["events"].get("contact", 0) >= 2 and
                  contact["actions_failed"] == 0, contact)
            ipc.call("input.clear_body_poses")

            configure({"duration_seconds": 0.4, "finale_seconds": 0.4,
                       "reset_seconds": 0.4, "coverage_goal": 1})
            states = ["attract"]
            ipc.call("game.session", operation="start")
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                value = session()
                if states[-1] != value["state"]:
                    states.append(value["state"])
                if value["completed_sessions"] == 1:
                    break
                time.sleep(0.03)
            check("timer_cycle", states == ["attract", "playing", "finale", "resetting", "attract"], states)

            ipc.call("paint.wet_reset")
            configure({"duration_seconds": 30, "finale_seconds": 1,
                       "reset_seconds": 0.5, "coverage_goal": 0.001})
            ipc.call("game.session", operation="start")
            ipc.call("paint.wet_dab", position=[0, 0, 0], color=[1, 0, 0], radius=1.0, amount=1.0)
            final = wait_until(session, lambda v: v["state"] == "finale", 5)
            check("coverage_triggers_early_final", final["coverage"] >= 0.001 and
                  final["time_s"] < 5 and final["events"].get("coverage.threshold", 0) == 1, final)
        report["status"] = "PASS"
    except Exception as exc:
        report["status"] = "FAIL"
        report["errors"].append(str(exc))
    finally:
        if ipc is not None:
            try:
                if owns_input:
                    ipc.call("input.clear_body_poses")
                if mutated_session and original and original["configured"]:
                    ipc.call("game.session", operation="configure", config=original["config"])
                    ipc.call("paint.wet_reset")
                report["final_body"] = ipc.call("input.body_poses")
                report["final_session"] = session()
            except Exception as exc:
                report["errors"].append("cleanup: " + str(exc))
                report["status"] = "FAIL"
            ipc.close()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"status": report["status"], "checks": len(report["checks"]),
                      "errors": report["errors"], "report": str(args.output)}))
    return int(report["status"] != "PASS")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipe", default=PIPE)
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    raise SystemExit(run(parser.parse_args()))
