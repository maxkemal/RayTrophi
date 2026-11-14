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

PLAYER_PIPE_NAME = r"\\.\pipe\RayTrophiPlayer"

PIPE_NAME = r"\\.\pipe\RayTrophiStudio"

_GENERIC_READ = 0x80000000
_GENERIC_WRITE = 0x40000000
_OPEN_EXISTING = 3
_PIPE_READMODE_MESSAGE = 0x00000002
_ERROR_MORE_DATA = 234


class RtIpcError(RuntimeError):
    """The engine refused the call. Carries the message the engine gave."""


def _running_inside_the_app():
    # `rt` is a PYBIND11_EMBEDDED_MODULE: importable only in the app's own
    # interpreter, never from a terminal.
    try:
        import rt  # noqa: F401
        return True
    except ImportError:
        return False


class RtIpc:
    def __init__(self, pipe_name=PIPE_NAME):
        # ★ Refuse to run in-process. The script would hold the main thread
        # while every IPC request waits FOR the main thread, so the first call
        # dies with "main thread dispatch timeout" - which reads as the app
        # being hung, not as the script being started from the wrong place.
        if _running_inside_the_app():
            raise RtIpcError(
                "this is an IPC client: run it from a separate terminal "
                "(python scripts\\test\\<script>.py) with the app open, not "
                "from the app's script workspace - in-process it deadlocks "
                "on the main thread")
        self._k32 = ctypes.windll.kernel32
        self._id = 0
        self.handle = self._k32.CreateFileW(
            pipe_name, _GENERIC_READ | _GENERIC_WRITE, 0, None,
            _OPEN_EXISTING, 0, None)
        if self.handle == -1 or (self.handle & 0xFFFFFFFFFFFFFFFF) == \
                (wintypes.HANDLE(-1).value & 0xFFFFFFFFFFFFFFFF):
            raise RtIpcError(
                "cannot open %s (error %d) - is the target application running?"
                % (pipe_name, self._k32.GetLastError()))
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

        Test-side parcel budget for flow_source.create / flow_source.update:
        `parcels=N` (and optionally `limit_parcels=True/False`). The engine's
        source budget is MASS since 2026-10-27 (use_mass_limit,
        max_emitted_mass_kg); a test that needs exactly N grains sizes it
        here from the source's own parcel_mass_kg, after the call. Without
        limit_parcels an update keeps the source's current on/off state.
        """
        if method in ("flow_source.create", "flow_source.update") and (
                "parcels" in params or "limit_parcels" in params):
            return self._call_with_parcel_budget(method, params)
        if self.__dict__.get("_parcel_budgets"):
            if method.startswith(self._SETTINGS_PREFIXES):
                self._parcel_budgets_dirty = True
            elif method in self._SIZING_TRIGGERS and self.__dict__.get("_parcel_budgets_dirty"):
                self._size_parcel_budgets()
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

    # A parcel's mass follows the domain's settings (grain radius, voxel,
    # PPC), which tests often change AFTER creating the source. The budgets
    # are remembered and re-sized from the live parcel_mass_kg before the
    # next reset/step/run that follows any settings change.
    _SIZING_TRIGGERS = ("fluid.reset", "fluid.step", "fluid.run", "fluid.settle",
                        "timeline.set_frame", "timeline.play")
    _SETTINGS_PREFIXES = ("fluid.set_", "flow_source.", "substance.", "fluid.create_domain")

    def _call_with_parcel_budget(self, method, params):
        params = dict(params)
        parcels = params.pop("parcels", None)
        limit = params.pop("limit_parcels", None)
        result = self.call(method, **params)
        name = params.get("name") or (result or {}).get("name")
        if limit is not None:
            self.call("flow_source.update", name=name, use_mass_limit=bool(limit))
        if parcels is not None:
            self.__dict__.setdefault("_parcel_budgets", {})[name] = int(parcels)
            self._size_parcel_budgets()
        return result

    def _size_parcel_budgets(self):
        budgets = self.__dict__.get("_parcel_budgets") or {}
        if not budgets:
            return
        for source in self.call("flow_source.list"):
            parcels = budgets.get(source["name"])
            if parcels is None:
                continue
            mass = float(source.get("parcel_mass_kg") or 0.0)
            if mass <= 0.0:
                continue  # pours no particles (yet): sized again after the next change
            # The engine stops once the poured mass reaches the budget, so
            # half a parcel short of N lands exactly on N parcels.
            budget = (parcels - 0.5) * mass
            if abs(float(source.get("max_emitted_mass_kg", 0.0)) - budget) > 1e-9 * budget:
                self.call("flow_source.update", name=source["name"],
                          max_emitted_mass_kg=budget)
        # Its own list/update calls must not mark the budgets stale again.
        self._parcel_budgets_dirty = False

    def try_call(self, method, /, **params):
        """(ok, result_or_message). For cases where the REFUSAL is the result."""
        try:
            return True, self.call(method, **params)
        except RtIpcError as exc:
            return False, str(exc)
