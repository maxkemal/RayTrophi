"""Offline negative controls for the live acceptance instrument, not engine acceptance."""
import copy
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.perception_bridge.balloon_input_probe import RULE, packet, validate_trial
from tools.perception_bridge import balloon_input_probe


def evidence():
    before = {
        "demo": {"running": False, "events_total": 4},
        "coverage": {"covered_texels": 0}, "body": {"gestures": 10},
        "session": {"rules": [{"id": RULE, "fires": 2}], "actions_ok": 2,
                    "actions_failed": 0, "dropped_events": 0, "events": {}},
        "wet": {"device_event_dispatches": 20, "publish_device_copies": 100,
                "publish_host_downloads": 0, "publish_bytes_downloaded": 0,
                "sources_dropped": 0, "stamped_texels": 0},
    }
    after = copy.deepcopy(before)
    after["body"]["gestures"] += 1
    after["session"].update(actions_ok=3, rules=[{"id": RULE, "fires": 3}],
                             events={"gesture.arms_up": 1, "balloon.pop": 1})
    after["demo"].update(events_total=5, config={"particles_per_balloon": 1800},
                          balloons=[{"inflated": True}] * 12,
                          events=[{"simultaneous": 1, "spawned": 1800, "frames": 60,
                                   "sources_dropped": 0, "balloons": [{"balloon": 3}]}])
    after["wet"].update(device_event_dispatches=40, publish_device_copies=150)
    after["coverage"].update(covered_texels=200, surfaces=[{"covered_texels": 200}])
    after["runtime"] = {"device_lost": False, "simulation": {"particles_alive": 12}}
    return before, after


class BalloonInputProbeTests(unittest.TestCase):
    def test_wrong_project_preflight_sends_no_mutating_requests(self):
        class ReadOnlyClient:
            def __init__(self):
                self.calls = []
                self.closed = False

            def call(self, method, **params):
                self.calls.append(method)
                if method != "project.path":
                    raise AssertionError("preflight unexpectedly continued")
                return str(Path(__file__).resolve())

            def close(self):
                self.closed = True

        client = ReadOnlyClient()
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(pipe="unused", reset_paint=True, trials=1,
                                   output=Path(directory) / "report.json")
            with patch.object(balloon_input_probe, "connect", return_value=client), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(balloon_input_probe.run(args), 1)
            result = json.loads(args.output.read_text(encoding="utf-8"))
        self.assertEqual(result["status"], "NOT_READY")
        self.assertEqual(client.calls, ["project.path"])
        self.assertTrue(client.closed)

    def test_full_chain_accepts_zero_last_tick_stamp_gauge(self):
        before, after = evidence()
        self.assertEqual(validate_trial(before, after)["drops_spawned"], 1800)

    def test_auto_scheduler_and_held_gesture_repeats_are_rejected(self):
        for mutate in (lambda a: a["demo"].update(running=True),
                       lambda a: a["body"].update(gestures=12),
                       lambda a: a["session"]["events"].update({"balloon.pop": 2})):
            before, after = evidence()
            mutate(after)
            with self.assertRaises(AssertionError):
                validate_trial(before, after)

    def test_action_success_alone_does_not_prove_physical_paint(self):
        for mutate in (lambda a: a["coverage"].update(covered_texels=0),
                       lambda a: a["demo"]["events"][0].update(spawned=0),
                       lambda a: a["wet"].update(device_event_dispatches=20),
                       lambda a: a["wet"].update(publish_device_copies=100)):
            before, after = evidence()
            mutate(after)
            with self.assertRaises(AssertionError):
                validate_trial(before, after)

    def test_host_fallback_loss_and_undrained_drops_are_rejected(self):
        for mutate in (lambda a: a["wet"].update(publish_bytes_downloaded=1024),
                       lambda a: a["wet"].update(sources_dropped=1),
                       lambda a: a["runtime"]["simulation"].update(particles_alive=200),
                       lambda a: a["runtime"].update(device_lost=True)):
            before, after = evidence()
            mutate(after)
            with self.assertRaises(AssertionError):
                validate_trial(before, after)

    def test_pose_changes_only_wrist_height_and_timestamp(self):
        low, high = packet(0, False), packet(1, True)
        self.assertEqual(low["persons"][0]["id"], high["persons"][0]["id"])
        low_joints, high_joints = low["persons"][0]["joints"], high["persons"][0]["joints"]
        for name in low_joints:
            if name.endswith("_wrist"):
                self.assertGreater(high_joints[name][1], high_joints["head"][1] + 0.1)
                self.assertEqual(low_joints[name][::2], high_joints[name][::2])
            else:
                self.assertEqual(low_joints[name], high_joints[name])


if __name__ == "__main__":
    unittest.main()
