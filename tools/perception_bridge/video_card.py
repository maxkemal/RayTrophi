"""Prepare a single-person cutout and publish the latest frame in Windows shared memory."""
import ctypes
import math
import mmap
import struct
import time


MAX_SIDE = 640
FRAME_MAP = r"Local\RayTrophiVideoCard.v1"
FRAME_LOCK = FRAME_MAP + ".lock"
WRITER_LOCK = FRAME_MAP + ".writer"
HEADER = struct.Struct("<QQfffIII")
MAGIC = 0x31565452  # RTV1


def mask_input(rgb, cv2):
    # Aligned rows let MediaPipe expose its float mask without repacking it.
    height, width = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(height, width))
    size = tuple(max(16, round(side * scale / 16) * 16) for side in (width, height))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def cutout(rgb, mask, image_points, person, confidence, cv2, np):
    """Return a cropped RGBA frame, a room floor anchor and card height."""
    if person["confidence"] < 0.5:
        return None
    source_height, source_width = rgb.shape[:2]
    scale = min(1.0, MAX_SIDE / max(source_height, source_width))
    width, height = max(1, round(source_width * scale)), max(1, round(source_height * scale))
    alpha = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    # Keep antialiased edges while suppressing low-confidence background pixels.
    alpha = np.uint8(np.clip((alpha - 0.35) / 0.45, 0.0, 1.0) * 255)
    x, y, w, h = cv2.boundingRect(alpha)
    if not w or not h:
        return None
    ankles = [j for j in (27, 28) if confidence(image_points[j]) >= 0.5
              and 0 <= image_points[j].x < 1 and 0 <= image_points[j].y < 1]
    if not ankles:
        return None
    foot_row = sum(image_points[j].y for j in ankles) / len(ankles) * height
    head_row = image_points[0].y * height
    joints = person["joints"]
    head_height = joints["head"][1]
    if confidence(image_points[0]) < 0.5 or foot_row - head_row < height * 0.05:
        return None
    card_height = head_height * h / (foot_row - head_row)
    if not math.isfinite(card_height) or not 0.3 <= card_height <= 3.2:
        return None

    small = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
    rgba = cv2.cvtColor(small[y:y+h, x:x+w], cv2.COLOR_RGB2RGBA)
    rgba[:, :, 3] = alpha[y:y+h, x:x+w]
    rgba[rgba[:, :, 3] == 0, :3] = 0
    feet = [joints["l_ankle" if j == 27 else "r_ankle"] for j in ankles]
    x = sum(foot[0] for foot in feet) / len(feet)
    z = sum(foot[2] for foot in feet) / len(feet)
    return rgba, x, z, card_height


class VideoCardPublisher:
    def __init__(self, *, synthetic_depth=None):
        if synthetic_depth is not None and (not math.isfinite(synthetic_depth)
                                             or abs(synthetic_depth) > 20):
            raise ValueError("synthetic depth must be finite and within +/-20 metres")
        self.synthetic_depth = synthetic_depth
        self.stride = 8 if synthetic_depth is not None else 4
        self.magic = 0x31445452 if synthetic_depth is not None else MAGIC
        frame_map = (r"Local\RayTrophiDepthCard.v1" if synthetic_depth is not None
                     else FRAME_MAP)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.CreateMutexW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_wchar_p]
        kernel.CreateMutexW.restype = ctypes.c_void_p
        kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel.WaitForSingleObject.restype = ctypes.c_uint32
        for fn in (kernel.ReleaseMutex, kernel.CloseHandle):
            fn.argtypes = [ctypes.c_void_p]
        kernel.GetTickCount64.restype = ctypes.c_uint64
        self.kernel = kernel
        ctypes.set_last_error(0)
        self.writer = kernel.CreateMutexW(None, False, frame_map + ".writer")
        if not self.writer:
            raise ctypes.WinError(ctypes.get_last_error())
        if ctypes.get_last_error() == 183:
            kernel.CloseHandle(self.writer)
            raise RuntimeError("a video card publisher is already running")
        self.lock = None
        self.mapping = None
        self.serial = 0
        try:
            self.lock = kernel.CreateMutexW(None, False, frame_map + ".lock")
            if not self.lock:
                raise ctypes.WinError(ctypes.get_last_error())
            self.mapping = mmap.mmap(-1, HEADER.size + MAX_SIDE * MAX_SIDE * self.stride,
                                     frame_map, access=mmap.ACCESS_WRITE)
            self.publish(None)
        except BaseException:
            self.close()
            raise

    def publish(self, frame, *, closing=False):
        # A missed slot is harmless. The next pose/video frame supersedes it.
        status = self.kernel.WaitForSingleObject(self.lock, 1000 if closing else 0)
        if status == 258:
            return False
        if status not in (0, 128):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            if frame is None:
                w = h = 0
                x = z = height = 0.0
                if closing:
                    self.mapping[HEADER.size:] = bytes(MAX_SIDE * MAX_SIDE * self.stride)
            else:
                rgba, x, z, height = frame[:4]
                h, w = rgba.shape[:2]
                if self.synthetic_depth is None:
                    payload = rgba.tobytes()
                else:
                    import numpy as np
                    samples = np.empty((h, w), dtype=[("rgba", "<u4"), ("depth", "<f4")])
                    samples["rgba"] = rgba.copy().view("<u4").reshape(h, w)
                    depth = self.synthetic_depth if len(frame) == 4 else np.asarray(frame[4])
                    if (not np.isfinite(depth).all() or (np.abs(depth) > 20).any()
                            or (np.ndim(depth) != 0 and np.shape(depth) != (h, w))):
                        raise ValueError("depth offsets must match the RGBA frame and be within +/-20m")
                    samples["depth"] = depth
                    payload = samples.tobytes()
                self.mapping[HEADER.size:HEADER.size + len(payload)] = payload
            self.serial = max(self.serial + 1, time.perf_counter_ns())
            HEADER.pack_into(self.mapping, 0, self.serial,
                             self.kernel.GetTickCount64(), x, z, height, w, h, self.magic)
            return True
        finally:
            self.kernel.ReleaseMutex(self.lock)

    def close(self):
        try:
            if self.mapping is not None:
                try:
                    if not self.publish(None, closing=True):
                        raise TimeoutError("video card frame mutex did not clear at shutdown")
                finally:
                    self.mapping.close()
                    self.mapping = None
        finally:
            if self.lock:
                self.kernel.CloseHandle(self.lock)
                self.lock = None
            if self.writer:
                self.kernel.CloseHandle(self.writer)
                self.writer = None
