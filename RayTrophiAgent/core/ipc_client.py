"""Named-pipe JSON-RPC client for RayTrophi Studio.

One instance owns one pipe connection and must be used from ONE thread. The
engine serves several pipe instances, so give each thread its own client rather
than sharing one (the chat/heartbeat loop and the tool worker each hold one).
"""

import ctypes
import json
import logging
import time
from ctypes import wintypes

GENERIC_READ = 0x80000000
GENERIC_WRITE = 0x40000000
OPEN_EXISTING = 3
PIPE_READMODE_MESSAGE = 2
PIPE_WAIT = 0
ERROR_PIPE_BUSY = 231
ERROR_MORE_DATA = 234

INVALID_HANDLE_VALUE = -1
READ_CHUNK = 65536

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)


class IPCClient:
    def __init__(self, pipe_name=r"\\.\pipe\RayTrophiStudio", agent_id="agent_01",
                 session_id="default_session", label="ipc"):
        self.pipe_name = pipe_name
        self.agent_id = agent_id
        self.session_id = session_id
        self.label = label
        self.handle = INVALID_HANDLE_VALUE
        self.request_counter = 1

    # -- connection ---------------------------------------------------------

    def connect(self, timeout_sec=10):
        start = time.time()
        while time.time() - start < timeout_sec:
            handle = kernel32.CreateFileW(
                self.pipe_name,
                GENERIC_READ | GENERIC_WRITE,
                0, None, OPEN_EXISTING, 0, None)
            if handle != INVALID_HANDLE_VALUE:
                self.handle = handle
                mode = wintypes.DWORD(PIPE_READMODE_MESSAGE | PIPE_WAIT)
                kernel32.SetNamedPipeHandleState(self.handle, ctypes.byref(mode),
                                                 None, None)
                logging.info("[%s] connected to %s", self.label, self.pipe_name)
                return True

            error = ctypes.get_last_error()
            if error == ERROR_PIPE_BUSY:
                kernel32.WaitNamedPipeW(self.pipe_name, 1000)
            else:
                time.sleep(0.5)

        logging.error("[%s] could not connect to %s within %ss",
                      self.label, self.pipe_name, timeout_sec)
        return False

    def close(self):
        if self.handle != INVALID_HANDLE_VALUE:
            kernel32.CloseHandle(self.handle)
            self.handle = INVALID_HANDLE_VALUE

    @property
    def connected(self):
        return self.handle != INVALID_HANDLE_VALUE

    # -- request/response ---------------------------------------------------

    def _read_message(self):
        """Read one whole pipe message.

        ★ The message can be larger than any single buffer. In message mode
        ReadFile then returns FALSE with ERROR_MORE_DATA and the REST OF THE
        MESSAGE STAYS IN THE PIPE. Parsing the truncated buffer and moving on -
        which is what the first version did - leaves that remainder to be read
        as the next response, so from then on every reply belongs to the
        previous request. Both are valid JSON, so nothing looks wrong.
        """
        chunks = []
        buffer = ctypes.create_string_buffer(READ_CHUNK)
        read = wintypes.DWORD(0)
        while True:
            ok = kernel32.ReadFile(self.handle, buffer, READ_CHUNK,
                                   ctypes.byref(read), None)
            error = 0 if ok else ctypes.get_last_error()
            if read.value:
                chunks.append(buffer.raw[:read.value])
            if ok:
                break
            if error == ERROR_MORE_DATA:
                continue
            raise IOError("read failed with error %d" % error)
        return b"".join(chunks)

    def call(self, method, params=None):
        """Send one request and return the decoded response envelope.

        Returns {"result": ...} or {"error": "..."}; never raises.
        """
        if not self.connected and not self.connect():
            return {"error": "not connected to RayTrophi (is the app running?)"}

        request_id = self.request_counter
        self.request_counter += 1
        payload = json.dumps({
            "jsonrpc": "2.0",
            "id": request_id,
            "method": method,
            "params": params or {},
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }).encode("utf-8")

        written = wintypes.DWORD(0)
        if not kernel32.WriteFile(self.handle, payload, len(payload),
                                  ctypes.byref(written), None):
            error = ctypes.get_last_error()
            self.close()
            return {"error": "write failed with error %d (connection dropped)" % error}

        try:
            raw = self._read_message()
        except IOError as exc:
            self.close()
            return {"error": "%s (connection dropped)" % exc}

        try:
            response = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001 - report, do not crash the agent
            # A parse failure means the stream is no longer trustworthy: drop
            # the connection rather than answering the next request with this
            # one's leftovers.
            self.close()
            return {"error": "could not decode response: %s" % exc}

        if isinstance(response, dict) and response.get("id") not in (None, request_id):
            self.close()
            return {"error": "response id %r did not match request %d; connection reset"
                             % (response.get("id"), request_id)}
        return response
