"""Player-owned camera worker. Preview frames live only in named shared memory."""
import argparse
import ctypes
import json
import math
import mmap
import os
from pathlib import Path
import struct
import time
import uuid
from types import SimpleNamespace

PREVIEW_BYTES = 16 + 640 * 360 * 4


def validate_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("device_id"), str):
        raise ValueError("Select a camera device")
    if not request["device_id"]:
        raise ValueError("Select a camera device")
    for key, low, high in (("width", 160, 8192), ("height", 120, 8192), ("people", 1, 10)):
        value = request.get(key)
        if type(value) is not int or not low <= value <= high:
            raise ValueError(f"Invalid {key}")
    fps = request.get("fps")
    if type(fps) not in (int, float) or not math.isfinite(fps) or not 1 <= fps <= 240:
        raise ValueError("Invalid camera frame rate")
    confidence = request.get("confidence", 0.5)
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0.1 <= confidence <= 1:
        raise ValueError("Invalid tracking confidence")
    for key in ("tracking", "mirror", "video_card"):
        if type(request.get(key)) is not bool:
            raise ValueError(f"Invalid {key}")
    if request["tracking"] and not Path(request.get("calibration", "")).is_file():
        raise ValueError("Tracking requires a saved room calibration")
    if request["tracking"] and not Path(request.get("model", "")).is_file():
        raise ValueError("Pose model is missing")
    return request


def camera_index(device_id):
    """Resolve a stable MF symbolic link afresh; never fall back to camera zero."""
    from ctypes import wintypes as wt
    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]
        def __init__(self, value):
            super().__init__((ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID(value).bytes_le))
    def method(pointer, index, restype, *arguments):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *arguments)(table[index])
    def check(hr):
        if hr < 0:
            raise OSError(f"Windows camera inventory failed (0x{hr & 0xffffffff:08x})")
    ole = ctypes.OleDLL("ole32")
    mf = ctypes.WinDLL("mfplat")
    sources = ctypes.WinDLL("mf")
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    attributes = ctypes.c_void_p()
    devices = ctypes.POINTER(ctypes.c_void_p)()
    count = wt.DWORD()
    check(ole.CoInitializeEx(None, 0))
    started = False
    try:
        check(mf.MFStartup(0x20070, 0))
        started = True
        check(mf.MFCreateAttributes(ctypes.byref(attributes), 1))
        key = GUID("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")
        kind = GUID("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")
        check(method(attributes, 24, ctypes.c_long, ctypes.POINTER(GUID),
                     ctypes.POINTER(GUID))(attributes, ctypes.byref(key), ctypes.byref(kind)))
        check(sources.MFEnumDeviceSources(attributes, ctypes.byref(devices), ctypes.byref(count)))
        link_key = GUID("58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f")
        for index in range(count.value):
            value = ctypes.c_void_p()
            length = wt.DWORD()
            check(method(devices[index], 13, ctypes.c_long, ctypes.POINTER(GUID),
                         ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(wt.DWORD))(
                             devices[index], ctypes.byref(link_key), ctypes.byref(value), ctypes.byref(length)))
            try:
                if ctypes.wstring_at(value, length.value) == device_id:
                    return index
            finally:
                ole.CoTaskMemFree(value)
        raise ValueError("Selected camera is disconnected; refresh devices")
    finally:
        for index in range(count.value):
            method(devices[index], 2, wt.ULONG)(devices[index])
        ole.CoTaskMemFree(devices)
        if attributes:
            method(attributes, 2, wt.ULONG)(attributes)
        if started:
            mf.MFShutdown()
        ole.CoUninitialize()


def write_status(path, value):
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value), encoding="utf-8")
    # A concurrent short C++ read may deny replacement on Windows. Retry next tick.
    try:
        os.replace(temporary, target)
    except PermissionError:
        pass


class Preview:
    def __init__(self, request):
        from ctypes import wintypes as wt
        self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        self.kernel.OpenMutexW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
        self.kernel.OpenMutexW.restype = wt.HANDLE
        self.kernel.OpenEventW.argtypes = [wt.DWORD, wt.BOOL, wt.LPCWSTR]
        self.kernel.OpenEventW.restype = wt.HANDLE
        self.kernel.WaitForSingleObject.argtypes = [wt.HANDLE, wt.DWORD]
        self.kernel.ReleaseMutex.argtypes = [wt.HANDLE]
        self.kernel.CloseHandle.argtypes = [wt.HANDLE]
        self.mutex = self.kernel.OpenMutexW(0x100001, False, request["mutex"])
        self.stop_event = self.kernel.OpenEventW(0x100000, False, request["stop_event"])
        if not self.mutex or not self.stop_event:
            raise OSError("Player camera lifecycle is unavailable")
        self.memory = mmap.mmap(-1, PREVIEW_BYTES, tagname=request["mapping"])
        self.request = request
        self.sequence = 0
        self.last_status = 0.0
        self.last_frame = 0.0
        self.actual = None
        self.rate_started = time.monotonic()
        self.rate_frames = 0
        self.rate = 0.0

    def stopped(self):
        return self.kernel.WaitForSingleObject(self.stop_event, 0) == 0

    def frame(self, image, cv2, people=None):
        now = time.monotonic()
        height, width = image.shape[:2]
        if (width, height) != (self.request["width"], self.request["height"]):
            raise ValueError("Camera delivered a different resolution; choose a supported mode")
        self.actual = (width, height)
        self.rate_frames += 1
        elapsed = now - self.rate_started
        if elapsed >= 1.0:
            self.rate = self.rate_frames / elapsed
            self.rate_started = now
            self.rate_frames = 0
        if now - self.last_frame >= 1 / 15:
            rgba = cv2.cvtColor(cv2.resize(image, (640, 360)), cv2.COLOR_BGR2RGBA)
            if self.request["mirror"]:
                rgba = cv2.flip(rgba, 1)
            if self.kernel.WaitForSingleObject(self.mutex, 0) == 0:
                try:
                    self.sequence += 1
                    self.memory.seek(0)
                    self.memory.write(struct.pack("<4I", 640, 360, self.sequence, 0))
                    self.memory.write(rgba.tobytes())
                finally:
                    self.kernel.ReleaseMutex(self.mutex)
            self.last_frame = now
        if now - self.last_status >= 0.25:
            write_status(self.request["status_path"], {
                "state": "tracking" if self.request["tracking"] else "preview",
                "width": width, "height": height, "frames": self.sequence,
                "processed_fps": self.rate, "requested_fps": self.request["fps"],
                "people": people if people is not None else 0,
                "mirror_scope": "operator_preview_only", "image_storage": "memory_only"})
            self.last_status = now

    def close(self):
        self.memory.close()
        self.kernel.CloseHandle(self.mutex)
        self.kernel.CloseHandle(self.stop_event)


def run(request):
    preview = None
    capture = None
    try:
        validate_request(request)
        import cv2
        index = camera_index(request["device_id"])
        preview = Preview(request)
        if request["tracking"]:
            try:
                from .bridge import run as run_bridge
            except ImportError:
                from bridge import run as run_bridge
            args = SimpleNamespace(
                camera=index, camera_backend=cv2.CAP_MSMF, camera_fps=request["fps"],
                video=None, offline=False, stop_stdin=False, video_card=request["video_card"],
                model=Path(request["model"]), calibration=Path(request["calibration"]),
                people=request["people"], confidence=request["confidence"], seconds=0,
                metrics=None, pipe=request["pipe"], stop_requested=preview.stopped,
                on_operator_frame=lambda frame, people: preview.frame(frame, cv2, people))
            report = run_bridge(args)
            if report["status"] != "STOPPED":
                raise RuntimeError("Tracking stopped: " + ", ".join(report.get("errors", [])))
        else:
            capture = cv2.VideoCapture(index, cv2.CAP_MSMF)
            if not capture.isOpened():
                raise RuntimeError("Camera is busy, disconnected or permission was denied")
            for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, request["width"]),
                                (cv2.CAP_PROP_FRAME_HEIGHT, request["height"]),
                                (cv2.CAP_PROP_FPS, request["fps"])):
                capture.set(prop, value)
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            while not preview.stopped():
                ok, image = capture.read()
                if not ok:
                    raise RuntimeError("Camera disconnected or frame unavailable")
                preview.frame(image, cv2)
        write_status(request["status_path"], {"state": "stopped"})
        return 0
    except Exception as error:
        write_status(request["status_path"], {"state": "error", "error": str(error)})
        return 1
    finally:
        if capture is not None:
            capture.release()
        if preview is not None:
            preview.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    options = parser.parse_args()
    raise SystemExit(run(json.loads(options.request.read_text(encoding="utf-8"))))
