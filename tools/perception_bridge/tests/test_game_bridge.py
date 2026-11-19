import pathlib
import sys
import tempfile
import unittest

from tools.perception_bridge.game_bridge import BridgeProcess


class LifecycleTests(unittest.TestCase):
    def worker(self, code):
        root = tempfile.TemporaryDirectory()
        self.addCleanup(root.cleanup)
        path = pathlib.Path(root.name) / "worker.py"
        path.write_text(code, encoding="utf-8")
        return path

    def test_owner_closes_control_pipe_and_waits_for_cleanup(self):
        path = self.worker(
            "import json,sys\n"
            "print(json.dumps({'event':'ready'}),flush=True)\n"
            "sys.stdin.readline()\n"
            "print(json.dumps({'status':'STOPPED','cleanup':True}),flush=True)\n")
        with BridgeProcess([], python=sys.executable, worker=path) as owner:
            self.assertTrue(owner.running)
            with self.assertRaises(RuntimeError):
                owner.start()
        self.assertFalse(owner.running)
        self.assertTrue(owner.report["cleanup"])
        self.assertEqual(owner.stop(), owner.report)

    def test_natural_end_keeps_aggregate_report(self):
        path = self.worker(
            "import json\n"
            "print(json.dumps({'event':'ready'}),flush=True)\n"
            "print(json.dumps({'status':'STOPPED','frames':7}),flush=True)\n")
        owner = BridgeProcess([], python=sys.executable, worker=path).start()
        self.assertEqual(owner.wait(timeout=10)["frames"], 7)
        owner.stop()

    def test_failure_before_ready_is_not_accepted(self):
        path = self.worker("import json; print(json.dumps({'status':'FAIL','errors':['ModelError']}))")
        with self.assertRaisesRegex(RuntimeError, "ModelError"):
            BridgeProcess([], python=sys.executable, worker=path).start()

    def test_wait_for_first_pose_and_stop(self):
        path = self.worker(
            "import json,sys\n"
            "print(json.dumps({'event':'ready'}),flush=True)\n"
            "print(json.dumps({'event':'first_pose'}),flush=True)\n"
            "sys.stdin.readline()\n"
            "print(json.dumps({'status':'STOPPED'}),flush=True)\n")
        with BridgeProcess([], python=sys.executable, worker=path) as owner:
            owner.wait_for_pose(timeout=5)

    def test_ready_timeout_requests_graceful_cleanup(self):
        path = self.worker(
            "import json,sys\n"
            "sys.stdin.readline()\n"
            "print(json.dumps({'status':'STOPPED'}),flush=True)\n")
        owner = BridgeProcess([], python=sys.executable, worker=path, startup_timeout=0.2)
        with self.assertRaisesRegex(TimeoutError, "startup"):
            owner.start()
        self.assertFalse(owner.running)


if __name__ == "__main__":
    unittest.main()
