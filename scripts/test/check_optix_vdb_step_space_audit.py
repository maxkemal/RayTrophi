"""Read-only numerical reproduction of the OptiX scaled-VDB coverage defect.
No OptiX implementation is changed; this models the inspected source order.
"""
import math
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "RayTrophiStudio/source"


def coverage(length, native_voxel, scale, multiplier, cap):
    requested = native_voxel*multiplier
    required = math.ceil(length/max(requested, 1e-5))+2
    limit = min(required, max(1, min(2048, cap)))
    step = max(requested, length/cap) if required > cap else requested
    step = max(.0001, min(step, max(native_voxel*scale, length/16)))
    return limit*step, limit, step


class OptixStepSpaceAudit(unittest.TestCase):
    def test_scaled_volume_can_stop_after_short_prefix_at_every_authored_cap(self):
        for cap in (16, 256, 1024):
            distance, limit, step = coverage(8., 100., .001, .207, cap)
            self.assertEqual(limit, 3)
            self.assertEqual(step, .5)
            self.assertEqual(distance, 1.5)
            self.assertLess(distance, 8.)

    def test_direction_changes_which_density_is_in_the_traversed_prefix(self):
        # Nonzero density occupies only the lower 1.5 units of an 8-unit column.
        # Truncated walks from the top and bottom see different occupied spans.
        distance, _, _ = coverage(8., 100., .001, .207, 1024)
        bottom_visible = min(distance, 1.5)
        top_visible = max(0., distance-(8.-1.5))
        self.assertGreater(bottom_visible, 0.)
        self.assertEqual(top_visible, 0.)

    def test_source_still_has_local_world_step_mismatch_and_post_budget_clamp(self):
        producer = (ROOT / "src/Render/VolumetricRenderer.cpp").read_text(encoding="utf-8")
        self.assertIn("renderVoxelSize = vulkanRT ? worldVoxelSize : vdb->getVoxelSize()", producer)
        source = (ROOT / "src/Device/ray_color.cuh").read_text(encoding="utf-8")
        section = source.split("// Step size - ADAPTIVE PRECISION", 1)[1]
        section = section.split("// Raymarching State", 1)[0]
        self.assertLess(section.index("int required_steps"), section.index("// Quality clamp"))
        self.assertLess(section.index("int march_step_limit"), section.index("// Quality clamp"))
        self.assertIn("base_step = fmaxf(0.0001f, fminf", section)


if __name__ == "__main__":
    unittest.main()
