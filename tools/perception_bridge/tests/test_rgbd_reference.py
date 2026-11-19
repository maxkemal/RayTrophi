import unittest

try:
    import numpy as np
except ModuleNotFoundError:
    np = None

if np is not None:
    from tools.perception_bridge.rgbd_reference import Pinhole, composite_pixel, reproject_cutout


@unittest.skipIf(np is None, "RGB-D reference uses the optional bridge NumPy environment")
class RgbdReferenceTests(unittest.TestCase):
    def test_invalid_depth_and_transparent_pixels_leave_holes(self):
        camera = Pinhole(3, 2, 2, 2, 1, 0.5)
        rgba = np.full((2, 3, 4), 255, dtype=np.uint8)
        rgba[1, 0, 3] = 0
        depth = np.array([[1, 0, np.nan], [1, 2, np.inf]])
        result, z = reproject_cutout(rgba, depth, camera, camera, np.eye(4))
        np.testing.assert_array_equal(result[:, :, 3], [[255, 0, 0], [0, 255, 0]])
        np.testing.assert_array_equal(z, [[1, np.inf, np.inf], [np.inf, 2, np.inf]])

    def test_extrinsics_shift_pixel_and_nearest_surface_wins(self):
        sensor = Pinhole(2, 1, 4, 4, 0, 0)
        view = Pinhole(4, 2, 1, 1, 1, 0)
        rgba = np.array([[[255, 0, 0, 255], [0, 255, 0, 255]]], dtype=np.uint8)
        depth = np.array([[2.0, 1.0]])
        transform = np.eye(4)
        transform[0, 3] = 0.125
        image, z = reproject_cutout(rgba, depth, sensor, view, transform)
        np.testing.assert_array_equal(image[0, 1], [0, 255, 0, 255])
        self.assertEqual(z[0, 1], 1)
        transform[0, 3] = 1.0
        image, z = reproject_cutout(rgba, depth, sensor, view, transform)
        np.testing.assert_array_equal(image[0, 2], [0, 255, 0, 255])

    def test_non_metric_extrinsics_rejected(self):
        camera = Pinhole(1, 1, 1, 1, 0, 0)
        rgba = np.full((1, 1, 4), 255, dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "metric scale"):
            reproject_cutout(rgba, np.ones((1, 1)), camera, camera, np.diag([2, 1, 1, 1]))

    def test_opaque_scene_and_character_occlude_effects(self):
        scene, person, particle = (0, 0, 1), (0, 1, 0), (1, 0, 0)
        hidden = composite_pixel(scene, 1, person, 1, 2, particle, 1, 3)
        behind = composite_pixel(scene, 4, person, 1, 2, particle, 1, 3)
        front = composite_pixel(scene, 4, person, 1, 2, particle, 1, 1)
        np.testing.assert_allclose(hidden, scene)
        np.testing.assert_allclose(behind, person)
        np.testing.assert_allclose(front, particle)

    def test_translucent_edges_preserve_behind_effect_and_additive_order(self):
        scene, person, particle = (0, 0, 1), (0, 1, 0), (1, 0, 0)
        behind = composite_pixel(scene, 4, person, 0.5, 2, particle, 0.5, 3)
        add_behind = composite_pixel(scene, 4, person, 0.5, 2, particle, 0.5, 3, additive=True)
        add_front = composite_pixel(scene, 4, person, 0.5, 2, particle, 0.5, 1, additive=True)
        np.testing.assert_allclose(behind, (0.25, 0.5, 0.25))
        np.testing.assert_allclose(add_behind, (0.25, 0.5, 0.5))
        np.testing.assert_allclose(add_front, (0.5, 0.5, 0.5))
