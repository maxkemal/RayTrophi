import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.perception_bridge.bridge import confidence, convert_pose, should_send_pose_packet
from tools.perception_bridge.calibration import GroundCalibration
from tools.perception_bridge.fake_children import Children, load_plan, run
from tools.perception_bridge.schema import JOINTS, validate_pose
from tools.perception_bridge.tracking import OneEuroFilter3D, PersonTracker
from tools.perception_bridge.video_card import cutout, mask_input

try:
    import cv2
    import numpy as np
except ModuleNotFoundError:
    cv2 = None
    np = None


def config():
    return {"image_size": [1000, 1000], "points": [
        {"pixel": [100, 100], "room_xz": [0, 0]},
        {"pixel": [900, 100], "room_xz": [4, 0]},
        {"pixel": [900, 900], "room_xz": [4, 4]},
        {"pixel": [100, 900], "room_xz": [0, 4]}]}


class CalibrationTests(unittest.TestCase):
    def test_corners_and_interior(self):
        calibration = GroundCalibration(config())
        for u, v, x, z in ((100, 100, 0, 0), (900, 900, 4, 4), (500, 500, 2, 2)):
            actual = calibration.project(u, v)
            self.assertAlmostEqual(actual[0], x)
            self.assertAlmostEqual(actual[1], z)

    def test_perspective_independent_point(self):
        # Known forward camera projection u=(100X+100)/(1+0.1Z),
        # v=(100Z+100)/(1+0.1Z); validate a point not used in fitting.
        cfg = config()
        for pair in cfg["points"]:
            x, z = pair["room_xz"]
            pair["pixel"] = [(100*x+100)/(1+0.1*z), (100*z+100)/(1+0.1*z)]
        c = GroundCalibration(cfg)
        x, z = c.project(300/1.1, 200/1.1)
        self.assertAlmostEqual(x, 2)
        self.assertAlmostEqual(z, 1)

    def test_degenerate(self):
        cfg = config()
        for p in cfg["points"]:
            p["pixel"] = [100, 100]
        with self.assertRaises(ValueError):
            GroundCalibration(cfg)

    def test_nonfinite_and_resolution(self):
        cfg = config()
        cfg["points"][0]["room_xz"][0] = float("nan")
        with self.assertRaises(ValueError):
            GroundCalibration(cfg)
        cfg = config()
        cfg["image_size"][0] = 0
        with self.assertRaises(ValueError):
            GroundCalibration(cfg)
        with self.assertRaises(ValueError):
            GroundCalibration(config()).project(float("inf"), 0)

    def test_rotation(self):
        cfg = config()
        cfg["camera_yaw_degrees"] = 90
        x, y, z = GroundCalibration(cfg).rotate(1, 2, 0)
        self.assertAlmostEqual(x, 0)
        self.assertEqual(y, 2)
        self.assertAlmostEqual(z, -1)

    def test_pitch_and_degenerate_room(self):
        cfg = config()
        cfg["camera_pitch_degrees"] = 90
        x, y, z = GroundCalibration(cfg).rotate(0, 1, 0)
        self.assertAlmostEqual(y, 0)
        self.assertAlmostEqual(z, 1)
        cfg = config()
        for pair in cfg["points"]:
            pair["room_xz"][1] = 0
        with self.assertRaises(ValueError):
            GroundCalibration(cfg)


class TrackingTests(unittest.TestCase):
    def test_detection_reordering(self):
        tracker = PersonTracker()
        self.assertEqual(tracker.update([(0, 0), (3, 0)], 0), [1, 2])
        self.assertEqual(tracker.update([(3.1, 0), (0.1, 0)], 0.1), [2, 1])

    def test_short_and_long_occlusion(self):
        tracker = PersonTracker()
        tracker.update([(0, 0)], 0)
        self.assertEqual(tracker.update([], 0.1), [])
        self.assertEqual(tracker.update([(0, 0)], 0.3), [1])
        self.assertEqual(tracker.update([(0, 0)], 0.81), [2])

    def test_prediction_through_crossing(self):
        tracker = PersonTracker()
        tracker.update([(-0.6, 0), (0.6, 0)], 0)
        tracker.update([(-0.2, 0), (0.2, 0)], 0.2)
        self.assertEqual(tracker.update([(-0.2, 0), (0.2, 0)], 0.4), [2, 1])

    def test_global_assignment_avoids_greedy_loss(self):
        tracker = PersonTracker(gate_metres=0.9)
        tracker.update([(0, 0), (0.8, 0)], 0)
        self.assertEqual(tracker.update([(0.1, 0), (-0.8, 0)], 0.1), [2, 1])

    def test_gate_and_invalid_time(self):
        tracker = PersonTracker()
        tracker.update([(0, 0)], 0)
        self.assertEqual(tracker.update([(5, 0)], 0.1), [2])
        with self.assertRaises(ValueError):
            tracker.update([], 0.1)
        with self.assertRaises(ValueError):
            tracker.update([(float("nan"), 0)], 0.2)

    def test_ten_people(self):
        tracker = PersonTracker()
        points = [(i*2, 0) for i in range(10)]
        self.assertEqual(tracker.update(points, 0), list(range(1, 11)))
        self.assertEqual(tracker.update(points[::-1], 0.1), list(range(10, 0, -1)))

    def test_one_euro_filter_stationary_vs_fast(self):
        flt = OneEuroFilter3D()
        first = flt.filter([1.0, 2.0, 3.0], 0.0)
        self.assertEqual(first, [1.0, 2.0, 3.0])

        # Stationary noisy jitter should be damped substantially.
        raw_x = []
        filtered_x = []
        for step in range(1, 30):
            jitter = 0.04 if step % 2 else -0.04
            raw_x.append(1.0 + jitter)
            filtered_x.append(flt.filter([1.0 + jitter, 2.0, 3.0], step / 30.0)[0])
        self.assertLess(max(abs(v - 1.0) for v in filtered_x), 0.025)

        # Fast swing should track closely without heavy fixed-filter lag.
        swing_flt = OneEuroFilter3D()
        swing_flt.filter([0.0, 1.0, 2.0], 0.0)
        last_y = 1.0
        for step in range(1, 10):
            target_y = 1.0 + 0.2 * step  # 6 m/s at 30 Hz
            last_y = swing_flt.filter([0.0, target_y, 2.0], step / 30.0)[1]
        self.assertGreater(last_y, 2.55)

    def test_tracker_pose_smoothing_and_expiry_cleanup(self):
        tracker = PersonTracker(max_gap=0.2)
        pose1 = {
            "id": 0,
            "confidence": 0.9,
            "joints": {name: [1.0, 0.5, 2.0, 0.77] for name in JOINTS},
        }
        tracker.update_poses([pose1], 0.0)
        self.assertEqual(pose1["id"], 1)
        self.assertIn(1, tracker.joint_filters)
        self.assertEqual(pose1["joints"]["l_wrist"], [1.0, 0.5, 2.0, 0.77])

        pose2 = {
            "id": 0,
            "confidence": 0.9,
            "joints": {name: [1.0, 0.6, 2.0, 0.77] for name in JOINTS},
        }
        tracker.update_poses([pose2], 1 / 30.0)
        self.assertEqual(pose2["id"], 1)
        self.assertLess(pose2["joints"]["l_wrist"][1], 0.6)
        self.assertGreater(pose2["joints"]["l_wrist"][1], 0.5)
        self.assertEqual(pose2["joints"]["l_wrist"][3], 0.77)

        # After max_gap expires, old filter state is cleaned up.
        tracker.update_poses([], 0.5)
        self.assertEqual(tracker.joint_filters, {})


class SchemaTests(unittest.TestCase):
    def test_complete_seeded_all_counts(self):
        self.assertEqual(len(JOINTS), 15)
        for count in range(1, 11):
            children = Children(count, 417)
            other = Children(count, 417)
            previous = None
            for frame in range(24*30):
                message = children.frame(frame)
                self.assertEqual(message, other.frame(frame))
                validate_pose(message, previous)
                previous = message["t"]
            self.assertEqual(len(children.frame(0)["persons"]), count)

    def test_reject_mutations(self):
        valid = Children(1).frame(0)
        mutations = []
        for key, value in (("t", float("nan")), ("t", True), ("persons", {})):
            packet = copy.deepcopy(valid)
            packet[key] = value
            mutations.append(packet)
        packet = copy.deepcopy(valid)
        packet["persons"] *= 2
        mutations.append(packet)
        for bad in (True, -1, "1", 2**31):
            packet = copy.deepcopy(valid)
            packet["persons"][0]["id"] = bad
            mutations.append(packet)
        for bad in ([0, 0, float("inf"), 1], [0, 0, 0, 2], [0, 0, 0], [10001, 0, 0, 1]):
            packet = copy.deepcopy(valid)
            packet["persons"][0]["joints"]["head"] = bad
            mutations.append(packet)
        packet = copy.deepcopy(valid)
        del packet["persons"][0]["joints"]["head"]
        mutations.append(packet)
        for packet in mutations:
            with self.subTest(packet=packet), self.assertRaises(ValueError):
                validate_pose(packet)

    def test_empty_and_monotonic(self):
        validate_pose({"t": 1, "persons": []}, 0)
        with self.assertRaises(ValueError):
            validate_pose({"t": 1, "persons": []}, 1)

    def test_scripted_loss_and_gestures(self):
        c = Children(1)
        self.assertEqual(c.frame(19*30)["persons"][0]["id"], 1)
        self.assertEqual(c.frame(round(19.1*30))["persons"], [])
        self.assertEqual(c.frame(round(19.4*30))["persons"][0]["id"], 1)
        self.assertEqual(c.frame(22*30)["persons"], [])
        self.assertEqual(c.frame(23*30)["persons"][0]["id"], 2)
        self.assertGreater(c.frame(13*30)["persons"][0]["joints"]["l_wrist"][1], 1.4)
        self.assertGreater(c.frame(round(6.25*30))["persons"][0]["joints"]["l_ankle"][1], 0.3)
        self.assertGreater(c.frame(round(16.5*30))["persons"][0]["joints"]["pelvis"][0], 5)


class ConversionTests(unittest.TestCase):
    def points(self):
        return [SimpleNamespace(x=0.5, y=0.5, z=0, visibility=1.0, presence=1.0)
                for _ in range(33)]

    def test_mapping_foot_anchor_and_y_up(self):
        image, world = self.points(), self.points()
        for i in (27, 28):
            world[i].y = 1.0
        person = convert_pose(image, world, GroundCalibration(config()))
        validate_pose({"t": 0, "persons": [person]})
        self.assertAlmostEqual(person["joints"]["head"][0], 2)
        self.assertAlmostEqual(person["joints"]["head"][1], 0.5)
        self.assertAlmostEqual(person["joints"]["l_ankle"][1], 0)

    def test_hidden_feet_rejected(self):
        image = self.points()
        for i in (27, 28):
            image[i].visibility = 0
        self.assertIsNone(convert_pose(image, self.points(), GroundCalibration(config())))

    def test_bad_world_rejected(self):
        world = self.points()
        world[0].x = float("nan")
        self.assertIsNone(convert_pose(self.points(), world, GroundCalibration(config())))

    def _standing_pose(self):
        image, world = self.points(), self.points()
        image_y = {
            0: 0.15, 11: 0.28, 12: 0.28, 13: 0.38, 14: 0.38,
            15: 0.45, 16: 0.45, 23: 0.50, 24: 0.50,
            25: 0.70, 26: 0.70, 27: 0.90, 28: 0.90,
        }
        world_y = {
            0: -0.60, 11: -0.40, 12: -0.40, 13: -0.20, 14: -0.20,
            15: 0.00, 16: 0.00, 23: 0.00, 24: 0.00,
            25: 0.40, 26: 0.40, 27: 0.85, 28: 0.85,
        }
        for idx, val in image_y.items():
            image[idx].y = val
        for idx, val in world_y.items():
            world[idx].y = val
        return image, world

    def test_ankle_occluded_knee_fallback(self):
        image, world = self._standing_pose()
        for i in (27, 28):
            image[i].visibility = 0.0
            world[i].visibility = 0.0
        person = convert_pose(image, world, GroundCalibration(config()))
        self.assertIsNotNone(person)
        validate_pose({"t": 0, "persons": [person]})
        self.assertGreaterEqual(person["confidence"], 0.5)
        # Knee at world y=0.40 + 0.45 fallback offset => knee is 0.45m above floor.
        self.assertAlmostEqual(person["joints"]["l_knee"][1], 0.45)
        self.assertAlmostEqual(person["joints"]["head"][1], 1.45)

    def test_ankle_out_of_frame_uses_world_ankle_y(self):
        image, world = self._standing_pose()
        for i in (27, 28):
            image[i].y = 1.15  # cropped below frame
            world[i].y = 0.92
        person = convert_pose(image, world, GroundCalibration(config()))
        self.assertIsNotNone(person)
        validate_pose({"t": 0, "persons": [person]})
        self.assertAlmostEqual(person["joints"]["l_knee"][1], 0.52)

    def test_ankle_and_knee_occluded_hip_fallback(self):
        image, world = self._standing_pose()
        for i in (25, 26, 27, 28):
            image[i].visibility = 0.0
            world[i].visibility = 0.0
        person = convert_pose(image, world, GroundCalibration(config()))
        self.assertIsNotNone(person)
        validate_pose({"t": 0, "persons": [person]})
        self.assertGreaterEqual(person["confidence"], 0.5)
        # Hip at world y=0.0 + 0.85 fallback offset => pelvis is 0.85m above floor.
        self.assertAlmostEqual(person["joints"]["pelvis"][1], 0.85)

    def test_no_ankles_knees_or_hips_returns_none(self):
        image, world = self._standing_pose()
        for i in (23, 24, 25, 26, 27, 28):
            image[i].visibility = 0.0
        self.assertIsNone(convert_pose(image, world, GroundCalibration(config())))

    def test_empty_frame_ipc_throttling(self):
        # Non-empty always sends.
        self.assertTrue(should_send_pose_packet([{"id": 1}], 1.0, True, 0.99))
        # Transition non-empty -> empty sends immediately.
        self.assertTrue(should_send_pose_packet([], 1.01, False, None))
        # Repeated empty within 0.25s is suppressed.
        self.assertFalse(should_send_pose_packet([], 1.10, True, 1.01))
        self.assertFalse(should_send_pose_packet([], 1.25, True, 1.01))
        # Empty after >= 0.25s sends again.
        self.assertTrue(should_send_pose_packet([], 1.26, True, 1.01))


@unittest.skipIf(cv2 is None or np is None, "video_card tests require cv2 and numpy")
class VideoCardTests(unittest.TestCase):
    def test_cutout_matches_mask_input_16px_aligned_grid(self):
        # 1920x1080 scales to 640x360 unaligned, vs 640x352 (16-px aligned) in mask_input.
        rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
        model_rgb = mask_input(rgb, cv2)
        self.assertEqual(model_rgb.shape[:2], (352, 640))

        # Paint a sharp green person box in source pixels that aligns with (352, 640) mask rows.
        y0_src = round(80 / 352 * 1080)
        y1_src = round(300 / 352 * 1080)
        x0_src = round(200 / 640 * 1920)
        x1_src = round(400 / 640 * 1920)
        rgb[y0_src:y1_src, x0_src:x1_src] = (0, 255, 0)

        mask = np.zeros((352, 640), dtype=np.float32)
        mask[80:300, 200:400] = 1.0

        pts = [SimpleNamespace(x=0.5, y=0.5, z=0, visibility=1.0, presence=1.0)
               for _ in range(33)]
        pts[0].y = 85 / 352
        pts[27].y = 295 / 352
        pts[28].y = 295 / 352
        person = {
            "confidence": 0.9,
            "joints": {
                "head": [0.0, 1.4, 2.0, 0.9],
                "l_ankle": [-0.1, 0.0, 2.0, 0.9],
                "r_ankle": [0.1, 0.0, 2.0, 0.9],
            },
        }
        result = cutout(rgb, mask, pts, person, confidence, cv2, np)
        self.assertIsNotNone(result)
        rgba, x, z, card_height = result
        self.assertEqual(rgba.shape, (220, 200, 4))
        self.assertTrue((rgba[:, :, 3] == 255).all())
        self.assertTrue((rgba[:, :, 1] > 200).all())
        self.assertTrue((rgba[:, :, 0] == 0).all())
        self.assertAlmostEqual(x, 0.0)
        self.assertAlmostEqual(z, 2.0)
        self.assertGreater(card_height, 0.3)


class RunnerTests(unittest.TestCase):
    def plan(self):
        # Mock-only protocol: these method names are not a proposed engine API.
        return {"setup": [], "start": [{"method": "mock.start"}], "cleanup": [],
                "state_path": "state", "display_fps_path": "frames",
                "expected_states": ["attract", "play", "finale", "reset", "attract"]}

    def test_live_cycle_and_cleanup_with_mock_transport(self):
        class Engine:
            def __init__(self):
                self.polls = 0
                self.methods = []
                self.closed = False

            def call(self, method, **params):
                self.methods.append(method)
                if method == "game.session":
                    self.polls += 1
                    phases = ["attract", "play", "finale", "reset", "attract"]
                    index = 0 if self.polls == 1 else min(4, 1 + (self.polls-2)//30)
                    return {"state": phases[index]}
                if method == "paint.wet_coverage":
                    return {"valid_texels": 100, "ratio": min(1, self.polls/100)}
                if method == "player.display":
                    return {"frames": self.polls}

            def close(self):
                self.closed = True

        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp)/"plan.json"
            plan_path.write_text(json.dumps(self.plan()))
            args = SimpleNamespace(seconds=24, session_plan=plan_path, dry_run=False,
                                   output=Path(tmp), seed=417, children=1, pipe="mock")
            engine = Engine()
            with patch("tools.perception_bridge.fake_children.connect", return_value=engine), \
                    patch("tools.perception_bridge.fake_children.wait_frame"):
                self.assertEqual(run(args), 0)
            report = json.loads((Path(tmp)/"acceptance.json").read_text())
            self.assertEqual(report["engine_acceptance"], "PASS")
            self.assertEqual(engine.methods[-1], "input.clear_body_poses")
            self.assertTrue(engine.closed)

    def test_connection_failure_writes_failed_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan_path = Path(tmp)/"plan.json"
            plan_path.write_text(json.dumps(self.plan()))
            args = SimpleNamespace(seconds=24, session_plan=plan_path, dry_run=False,
                                   output=Path(tmp), seed=417, children=1, pipe="mock")
            with patch("tools.perception_bridge.fake_children.connect",
                       side_effect=RuntimeError("unavailable")):
                self.assertEqual(run(args), 1)
            report = json.loads((Path(tmp)/"acceptance.json").read_text())
            self.assertEqual(report["status"], "FAIL")
            self.assertIn("unavailable", report["errors"])

    def test_dry_run_without_ipc(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = SimpleNamespace(seconds=24, session_plan=None, dry_run=True,
                                   output=Path(tmp), seed=417, children=2)
            with patch("tools.perception_bridge.fake_children.connect",
                       side_effect=AssertionError("dry run opened IPC")):
                self.assertEqual(run(args), 0)
            report = json.loads((Path(tmp)/"acceptance.json").read_text())
            self.assertEqual(report["engine_acceptance"], "NOT_RUN")
            self.assertEqual(report["frames"], 720)
            messages = [json.loads(line) for line in
                        (Path(tmp)/"messages.jsonl").read_text().splitlines()]
            self.assertEqual([m["id"] for m in messages], list(range(1, len(messages)+1)))
            self.assertIn("game.session", {m["method"] for m in messages})
            poses = [m["params"] for m in messages if m["method"] == "input.set_body_poses"]
            previous = None
            for pose in poses:
                validate_pose(pose, previous)
                previous = pose["t"]
            self.assertEqual(poses[-1]["persons"], [])

    def test_live_requires_contract(self):
        args = SimpleNamespace(seconds=24, session_plan=None, dry_run=False)
        with self.assertRaisesRegex(ValueError, "session-plan"):
            run(args)


if __name__ == "__main__":
    unittest.main()
