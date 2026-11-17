"""Finite Python-binding smoke, dispatched by external script.run_file.

No IPC calls, waits, simulation steps or loops over frames inside the app.
Live timing and session acceptance belong exclusively to the external probes.
"""
import json
from pathlib import Path

import rt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "tools/perception_bridge/out/live-build-2026-11-15/studio-bindings.json"


def run():
    result = {"status": "RUNNING", "checks": {}, "errors": []}
    owns_input = False

    def check(name, condition):
        result["checks"][name] = bool(condition)
        if not condition:
            raise AssertionError(name)

    def reject(name, callback):
        try:
            callback()
        except RuntimeError as exc:
            result["checks"][name] = str(exc)
        else:
            check(name, False)

    try:
        check("no_existing_people", not rt.input.body_poses()["persons"])
        owns_input = True
        session_before = rt.game.session()
        rules_before = rt.game.rules()
        rt.input.clear_body_poses()
        names = ["head", "neck", "pelvis"] + [
            side + "_" + joint for side in ("l", "r")
            for joint in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle")]
        persons = [{"id": 1, "confidence": 1.0,
                    "joints": {name: [0, 0.6, 1.5, 1] for name in names}}]
        rt.input.set_body_poses(t=0, persons=persons)
        check("python_pose_round_trip", rt.input.body_poses()["persons"][0]["id"] == 1)
        reject("python_boolean_time_rejected", lambda: rt.input.set_body_poses(True, persons))
        reject("python_replay_rejected", lambda: rt.input.set_body_poses(0, persons))
        reject("python_invalid_config_rejected", lambda: rt.game.session("configure", {"typo": 1}))
        reject("python_invalid_event_rejected", lambda: rt.game.post_event({"type": "unknown"}))
        check("python_rules_read", rt.game.rules() == rules_before)
        check("python_invalid_config_atomic", rt.game.session()["config"] == session_before["config"])
        rt.input.clear_body_poses()
        check("python_clear", not rt.input.body_poses()["persons"])
        result["status"] = "PASS"
    except Exception as exc:
        result["status"] = "FAIL"
        result["errors"].append(str(exc))
    finally:
        if owns_input:
            rt.input.clear_body_poses()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(result, indent=2), encoding="utf-8")


run()
