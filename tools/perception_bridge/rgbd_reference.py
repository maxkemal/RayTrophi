"""Offline RGB-D reprojection reference for the Player camera-card compositor."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Pinhole:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self):
        if (self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0
                or not np.isfinite((self.fx, self.fy, self.cx, self.cy)).all()):
            raise ValueError("invalid pinhole intrinsics")


def reproject_cutout(rgba, depth_m, sensor, view, sensor_to_view, *, near_m=0.05, far_m=20.0):
    """Project aligned color/depth pixels into the virtual view using nearest-Z.

    Both camera spaces use X right, Y up, and Z forward (metres). The transform
    maps sensor camera coordinates to Player view coordinates. Returned depth is
    linear view Z. Pixels without a valid sample remain transparent, at +inf.
    """
    shape = (sensor.height, sensor.width)
    if rgba.shape != (*shape, 4) or rgba.dtype != np.uint8 or depth_m.shape != shape:
        raise ValueError("RGBA8 and depth must match the sensor intrinsics")
    transform = np.asarray(sensor_to_view, dtype=np.float64)
    if (transform.shape != (4, 4) or not np.isfinite(transform).all()
            or not np.allclose(transform[3], (0, 0, 0, 1))):
        raise ValueError("sensor_to_view must be a finite affine transform")
    rotation = transform[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)):
        raise ValueError("sensor_to_view must preserve metric scale and handedness")
    if not (0 < near_m < far_m < np.inf):
        raise ValueError("invalid depth range")

    valid = ((rgba[:, :, 3] > 0) & np.isfinite(depth_m)
             & (depth_m > 0) & (depth_m <= far_m))
    rows, cols = np.nonzero(valid)
    depth = depth_m[rows, cols].astype(np.float64)
    xyz = np.column_stack(((cols - sensor.cx) * depth / sensor.fx,
                           (sensor.cy - rows) * depth / sensor.fy, depth))
    xyz = xyz @ rotation.T + transform[:3, 3]
    visible = (xyz[:, 2] >= near_m) & (xyz[:, 2] <= far_m)
    x, y, z = xyz[visible].T
    u = np.rint(view.cx + view.fx * x / z)
    v = np.rint(view.cy - view.fy * y / z)
    inside = (u >= 0) & (u < view.width) & (v >= 0) & (v < view.height)
    target = v[inside].astype(np.intp) * view.width + u[inside].astype(np.intp)
    source = np.flatnonzero(valid)[visible][inside]
    z = z[inside]

    pixels = view.width * view.height
    nearest = np.full(pixels, np.inf)
    np.minimum.at(nearest, target, z)
    wins = z == nearest[target]
    # Resolve equal-depth samples deterministically using source raster order.
    winner = np.full(pixels, -1, dtype=np.intp)
    np.maximum.at(winner, target[wins], source[wins])
    filled = winner >= 0
    color = np.zeros((pixels, 4), dtype=np.uint8)
    color[filled] = rgba.reshape(-1, 4)[winner[filled]]
    return color.reshape(view.height, view.width, 4), nearest.reshape(view.height, view.width)


def composite_pixel(scene_rgb, scene_z, person_rgb, person_alpha, person_z,
                    particle_rgb, particle_alpha, particle_z, *, additive=False):
    """Reference for one opaque scene sample, one cutout and one effect fragment.

    RGB values must be linear. Normal alpha and additive particles both respect
    opaque scene depth, and are applied before/after the cutout by view Z.
    """
    color = np.asarray(scene_rgb, dtype=np.float64)
    show_person = 0 < person_z < np.inf and person_z <= scene_z and person_alpha > 0
    show_particle = 0 < particle_z < np.inf and particle_z <= scene_z and particle_alpha > 0

    def over(background, foreground, alpha):
        return background * (1.0 - alpha) + np.asarray(foreground) * alpha

    def effect(background):
        if additive:
            return background + np.asarray(particle_rgb) * particle_alpha
        return over(background, particle_rgb, particle_alpha)

    if show_particle and (not show_person or particle_z >= person_z):
        color = effect(color)
    if show_person:
        color = over(color, person_rgb, person_alpha)
    if show_particle and show_person and particle_z < person_z:
        color = effect(color)
    return color
