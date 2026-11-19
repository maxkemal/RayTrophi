import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from tools.perception_bridge import operator_camera as worker


def request():
    return {"device_id": "stable-device-link", "width": 1280, "height": 720,
            "fps": 30, "people": 4, "confidence": 0.5, "tracking": False,
            "mirror": False, "video_card": False}


class RequestTests(unittest.TestCase):
    def test_preview_does_not_require_model_or_calibration(self):
        self.assertIs(worker.validate_request(value := request()), value)

    def test_rejects_types_ranges_nonfinite_and_missing_identity(self):
        for key, value in (("device_id", ""), ("device_id", 0), ("width", True),
                           ("height", 0), ("people", 11), ("fps", float("nan")),
                           ("confidence", 1.1), ("tracking", 1), ("mirror", "yes")):
            with self.subTest(key=key, value=value):
                config = request()
                config[key] = value
                with self.assertRaises(ValueError):
                    worker.validate_request(config)

    def test_tracking_requires_both_real_files(self):
        config = request()
        config["tracking"] = True
        with self.assertRaisesRegex(ValueError, "calibration"):
            worker.validate_request(config)
        with tempfile.TemporaryDirectory() as directory:
            calibration = Path(directory) / "floor.json"
            calibration.write_text("{}", encoding="utf-8")
            config["calibration"] = str(calibration)
            with self.assertRaisesRegex(ValueError, "model"):
                worker.validate_request(config)
            model = Path(directory) / "pose.task"
            model.write_bytes(b"fixture")
            config["model"] = str(model)
            worker.validate_request(config)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.config = request()
        self.config["status_path"] = str(self.directory / "status.json")
        self.cv = MagicMock()
        self.cv.CAP_MSMF = 1400
        self.capture = self.cv.VideoCapture.return_value
        self.capture.isOpened.return_value = True
        self.preview = MagicMock()

    def status(self):
        return json.loads(Path(self.config["status_path"]).read_text(encoding="utf-8"))

    def run_worker(self, identity=3):
        with patch.dict("sys.modules", {"cv2": self.cv}), \
                patch.object(worker, "camera_index", return_value=identity), \
                patch.object(worker, "Preview", return_value=self.preview):
            return worker.run(self.config)

    def test_preview_uses_exact_mf_device_and_releases_on_stop(self):
        self.preview.stopped.side_effect = [False, True]
        frame = object()
        self.capture.read.return_value = (True, frame)
        self.assertEqual(self.run_worker(identity=5), 0)
        self.cv.VideoCapture.assert_called_once_with(5, self.cv.CAP_MSMF)
        self.preview.frame.assert_called_once_with(frame, self.cv)
        self.capture.release.assert_called_once()
        self.preview.close.assert_called_once()
        self.assertEqual(self.status()["state"], "stopped")
        self.assertEqual([p.name for p in self.directory.iterdir()], ["status.json"])

    def test_disconnected_device_never_falls_back_to_default_camera(self):
        with patch.dict("sys.modules", {"cv2": self.cv}), \
                patch.object(worker, "camera_index", side_effect=ValueError("disconnected")):
            self.assertEqual(worker.run(self.config), 1)
        self.cv.VideoCapture.assert_not_called()
        self.assertIn("disconnected", self.status()["error"])

    def test_busy_device_reports_error_and_releases(self):
        self.capture.isOpened.return_value = False
        self.assertEqual(self.run_worker(), 1)
        self.assertIn("permission", self.status()["error"])
        self.capture.release.assert_called_once()
        self.preview.close.assert_called_once()

    def test_unplug_during_read_reports_error_and_releases(self):
        self.preview.stopped.return_value = False
        self.capture.read.return_value = (False, None)
        self.assertEqual(self.run_worker(), 1)
        self.assertEqual(self.status()["state"], "error")
        self.capture.release.assert_called_once()
        self.preview.close.assert_called_once()

    def test_invalid_request_reports_error_without_opening_device(self):
        self.config["width"] = True
        self.assertEqual(self.run_worker(), 1)
        self.cv.VideoCapture.assert_not_called()
        self.assertEqual(self.status()["state"], "error")

    def test_tracking_reuses_existing_bridge_and_callbacks(self):
        self.config["tracking"] = True
        self.config["video_card"] = True
        for key in ("model", "calibration"):
            path = self.directory / key
            path.write_text("{}", encoding="utf-8")
            self.config[key] = str(path)
        self.config["pipe"] = "fixture-pipe"
        with patch("tools.perception_bridge.bridge.run", return_value={"status": "STOPPED"}) as bridge:
            self.assertEqual(self.run_worker(), 0)
            args = bridge.call_args.args[0]
            self.assertEqual(args.camera_backend, self.cv.CAP_MSMF)
            self.assertEqual(args.camera, 3)
            self.assertEqual(args.pipe, "fixture-pipe")
            self.assertTrue(args.video_card)
            args.on_operator_frame("frame", 2)
            self.preview.frame.assert_called_once_with("frame", self.cv, 2)
        self.cv.VideoCapture.assert_not_called()
        self.preview.close.assert_called_once()


class PreviewContractTests(unittest.TestCase):
    def test_rejects_wrong_actual_resolution_before_publishing(self):
        preview = worker.Preview.__new__(worker.Preview)
        preview.request = request()
        image = SimpleNamespace(shape=(480, 640, 3))
        with self.assertRaisesRegex(ValueError, "different resolution"):
            preview.frame(image, MagicMock())

    def test_status_write_contains_only_supplied_aggregate_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "status.json"
            value = {"state": "preview", "width": 1280, "height": 720}
            worker.write_status(path, value)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), value)
            self.assertEqual(len(list(Path(directory).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
