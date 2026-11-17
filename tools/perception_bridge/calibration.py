"""Four ground correspondences -> projective pixel to room X/Z mapping."""
import json
import math
from pathlib import Path


def solve(matrix, rhs):
    rows = [list(map(float, row)) + [float(b)] for row, b in zip(matrix, rhs)]
    n = len(rhs)
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(rows[r][col]))
        if abs(rows[pivot][col]) < 1e-10:
            raise ValueError("degenerate calibration: use four distinct floor corners")
        rows[col], rows[pivot] = rows[pivot], rows[col]
        divisor = rows[col][col]
        rows[col] = [v / divisor for v in rows[col]]
        for r in range(n):
            if r != col:
                factor = rows[r][col]
                rows[r] = [a - factor * b for a, b in zip(rows[r], rows[col])]
    return [row[-1] for row in rows]


class GroundCalibration:
    def __init__(self, config):
        self.width, self.height = config["image_size"]
        if (type(self.width) is not int or type(self.height) is not int
                or min(self.width, self.height) <= 0):
            raise ValueError("image_size must contain positive integer pixels")
        self.yaw = float(config.get("camera_yaw_degrees", 0))
        self.pitch = float(config.get("camera_pitch_degrees", 0))
        self.roll = float(config.get("camera_roll_degrees", 0))
        if not all(math.isfinite(a) for a in (self.yaw, self.pitch, self.roll)):
            raise ValueError("camera rotation must be finite")
        pairs = config["points"]
        if len(pairs) != 4:
            raise ValueError("four pixel/room_xz correspondences required")
        matrix, rhs = [], []
        for pair in pairs:
            u, v = pair["pixel"]
            x, z = pair["room_xz"]
            if not all(math.isfinite(a) for a in (u, v, x, z)):
                raise ValueError("calibration coordinates must be finite")
            if not 0 <= u < self.width or not 0 <= v < self.height:
                raise ValueError("floor pixel outside calibrated resolution")
            # Normalize pixels to avoid unnecessarily ill-conditioned fitting.
            u, v = u / self.width, v / self.height
            matrix.extend(([u, v, 1, 0, 0, 0, -x*u, -x*v],
                           [0, 0, 0, u, v, 1, -z*u, -z*v]))
            rhs.extend((x, z))
        self.h = solve(matrix, rhs) + [1.0]
        a, b, c, d, e, f, g, h, i = self.h
        if abs(a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)) < 1e-10:
            raise ValueError("degenerate room coordinates")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def project(self, u, v):
        if not all(math.isfinite(a) for a in (u, v)):
            raise ValueError("nonfinite foot pixel")
        u, v = u / self.width, v / self.height
        h = self.h
        denominator = h[6]*u + h[7]*v + h[8]
        if abs(denominator) < 1e-8:
            raise ValueError("foot pixel is on calibration horizon")
        result = ((h[0]*u + h[1]*v + h[2]) / denominator,
                  (h[3]*u + h[4]*v + h[5]) / denominator)
        if not all(math.isfinite(a) for a in result):
            raise ValueError("nonfinite calibrated position")
        return result

    def rotate(self, x, y, z):
        # Camera Y-up -> room: pitch about X, yaw about Y, roll about Z.
        pitch, yaw, roll = map(math.radians, (self.pitch, self.yaw, self.roll))
        y, z = math.cos(pitch)*y - math.sin(pitch)*z, math.sin(pitch)*y + math.cos(pitch)*z
        x, z = math.cos(yaw)*x + math.sin(yaw)*z, -math.sin(yaw)*x + math.cos(yaw)*z
        x, y = math.cos(roll)*x - math.sin(roll)*y, math.sin(roll)*x + math.cos(roll)*y
        return x, y, z
