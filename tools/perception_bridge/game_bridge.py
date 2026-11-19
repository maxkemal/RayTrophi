"""External game-script owner for the perception worker (no embedded Python).

Use BridgeProcess as a context manager. A closed parent pipe also stops the worker.
Only aggregate status is read from stdout; images never cross this control pipe.
"""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading


class BridgeProcess:
    def __init__(self, arguments, *, python=None, worker=None, startup_timeout=60):
        self.python = Path(python or sys.executable).resolve()
        self.worker = Path(worker or Path(__file__).with_name("bridge.py")).resolve()
        self.arguments = list(map(str, arguments))
        self.startup_timeout = startup_timeout
        self.process = None
        self.report = None
        self._messages = queue.Queue()
        self._reader = None
        self._error_reader = None
        self._pose_ready = threading.Event()

    def _discard_errors(self):
        # Native ML libraries require a real redirected handle on Windows.
        # Drain their diagnostics without retaining or persisting their content.
        while self.process.stderr.read(4096):
            pass

    def _read(self):
        try:
            for line in self.process.stdout:
                try:
                    message = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if isinstance(message, dict):
                    if message.get("event") == "first_pose":
                        self._pose_ready.set()
                    if "status" in message:
                        self.report = message
                    self._messages.put(message)
        finally:
            self._messages.put({"event": "exit"})

    def start(self):
        if self.process is not None:
            raise RuntimeError("bridge already started; create a new owner to restart")
        creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        self.process = subprocess.Popen(
            [str(self.python), "-s", "-u", str(self.worker), *self.arguments, "--stop-stdin"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace", creationflags=creationflags,
            cwd=str(self.worker.parent),
        )
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()
        self._error_reader = threading.Thread(target=self._discard_errors, daemon=True)
        self._error_reader.start()
        try:
            while True:
                message = self._messages.get(timeout=self.startup_timeout)
                if message.get("event") == "ready":
                    return self
                if message.get("event") == "exit":
                    raise RuntimeError("bridge failed before ready: " +
                                       str((self.report or {}).get("errors", [])))
        except queue.Empty as exc:
            self.stop()
            raise TimeoutError("bridge startup did not report ready") from exc
        except BaseException:
            try:
                self.stop()
            except RuntimeError:
                pass  # Keep the original startup failure and its error classes.
            raise

    @property
    def running(self):
        return self.process is not None and self.process.poll() is None

    def wait(self, timeout=None):
        if self.process is None:
            raise RuntimeError("bridge has not started")
        code = self.process.wait(timeout=timeout)
        self._reader.join(timeout=2)
        self._error_reader.join(timeout=2)
        if not self._error_reader.is_alive():
            self.process.stderr.close()
        if not self._reader.is_alive():
            self.process.stdout.close()
        if not self.process.stdin.closed:
            self.process.stdin.close()
        if code != 0 or not self.report or self.report.get("status") != "STOPPED":
            raise RuntimeError("bridge failed: " + str(self.report))
        return self.report

    def wait_for_pose(self, timeout=60):
        import time
        deadline = time.monotonic() + timeout
        while not self._pose_ready.wait(0.1):
            if not self.running:
                self.wait(timeout=2)
                raise RuntimeError("bridge ended without a valid pose/card")
            if time.monotonic() >= deadline:
                raise TimeoutError("bridge did not produce a valid pose/card")

    def stop(self, timeout=15):
        if self.process is None:
            return None
        if self.running:
            try:
                self.process.stdin.close()  # EOF requests the worker's normal finally path.
            except (BrokenPipeError, OSError):
                pass
            try:
                self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
                self._reader.join(timeout=2)
                raise TimeoutError("bridge required forced shutdown; cleanup is unverified")
        return self.wait(timeout=2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.stop()
