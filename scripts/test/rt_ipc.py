"""Minimal JSON-RPC client for the RayTrophi Studio named pipe.

★★★ Why this exists next to the in-process rt module.

A script run through script.run_file holds the application's MAIN THREAD, so
the frame loop never turns while it executes. Every test written that way is
structurally blind to anything the loop does - measured 2026-08-19: the exact
same call sequence preserves 6760 particles inside a script and loses all 22932
over IPC. The bug is real and no in-process script can reach it.

So this repo needs two channels and they do NOT substitute for each other:

  in-process (rt module)  - core logic, fast, blind to the loop
  over IPC (this module)  - drives the real application, sees loop faults

Local pipe access needs no token; the security boundary is the pipe ACL.
"""
import ctypes
import ctypes.wintypes as wintypes
import json

PIPE_NAME = r"\\.\pipe\RayTrophiStudio"

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_PIPE_READMODE_MESSAGE = 0x00000002
_ERROR_MORE_DATA = 234


class RtIpcError(RuntimeError):
    """The engine refused the call. Carries the message the engine gave."""


class RtIpc:
    def __init__(self):
        self._k32 = ctypes.windll.kernel32
        self._id = 0
        self.handle = self._k32.CreateFileW(
            PIPE_NAME, _GENERIC_READ | _GENERIC_WRITE, 0, None,
            _OPEN_EXISTING, 0, None)
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == \
                (wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF):
            raise RtIpcError(
                "cannot open %s (error %d) - is RayTrophi Studio running?"
                % (PIPE_NAME, self._k32.GetLastError()))
        mode = wintypes.DWORD(_PIPE_READMODE_MESSAGE)
        self._k32.SetNamedPipeHandleState(
            self.handle, ctypes.byref(mode), None, None)

    def close(self):
        if self.handle:
            self._k32.CloseHandle(self.handle)
            self.handle = None

    def call(self, method, /, **params):
        """Send one request. Raises RtIpcError on an engine-reported failure.

        ★ `method` is POSITIONAL-ONLY. Several engine methods take a parameter
        literally named "method" - agent.describe is one - and without the `/`
        the call collides with its own first argument. A test client that
        cannot address part of the API is a blind spot in the instrument.

        ★ Raising rather than returning the envelope is deliberate: the agent
        tool layer once let an error branch fall through without returning,
        so the engine did the work and the caller was told it failed. A test
        harness that can silently drop an error is not an instrument.
        """
        self._id += 1
        msg = {"id": self._id, "method": method}
        if params:
            msg["params"] = params
        data = json.dumps(msg).encode("utf-8")
        written = wintypes.DWORD(0)
        if not self._k32.WriteFile(self.handle, data, len(data),
                                   ctypes.byref(written), None):
            raise RtIpcError("WriteFile failed (error %d)"
                             % self._k32.GetLastError())
        chunks = []
        while True:
            buf = ctypes.create_string_buffer(65536)
            read = wintypes.DWORD(0)
            ok = self._k32.ReadFile(self.handle, buf, 65536,
                                    ctypes.byref(read), None)
            chunks.append(buf.raw[:read.value])
            if ok:
                break
            err = self._k32.GetLastError()
            if err != _ERROR_MORE_DATA:
                raise RtIpcError("ReadFile failed (error %d)" % err)
        resp = json.loads(b"".join(chunks).decode("utf-8"))
        if "error" in resp:
            raise RtIpcError("%s: %s" % (method, resp["error"]))
        return resp.get("result")

    def try_call(self, method, /, **params):
        """(ok, result_or_message). For cases where the REFUSAL is the result."""
        try:
            return True, self.call(method, **params)
        except RtIpcError as exc:
            return False, str(exc)
