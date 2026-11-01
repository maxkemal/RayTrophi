"""Benchmark identity must detect changed fields even at the same timeline frame."""
import copy
import unittest
from gas_probe_snapshot import capture, require_same


class Client:
    def __init__(self):
        self.cap = 256
        self.camera = {"position": [0, 1, 2], "target": [0, 0, 0], "fov": 40}
        self.field = {"material_target": "matter:D:gas", "density_hash": "aaa",
                      "temperature_hash": "bbb", "active_cells": 1, "density_sum": 1.,
                      "optical_bounds": {"first_voxel": [0, 0, 0], "end_voxel": [8, 8, 8]},
                      "visible": True}
        self.id = 1

    def call(self, method, **kwargs):
        if method == "timeline.get_frame": return 50
        if method == "camera.get": return copy.deepcopy(self.camera)
        if method == "fluid.inspect_fog_fields":
            return {"state_version": self.id, "grid": [8, 8, 8],
                    "gas_snapshot": "host", "fields": [copy.deepcopy(self.field)]}
        if method == "volume.get_material": return {"max_steps": self.cap, "visible": True}
        if method == "render.volume_slots":
            return {"backends": [{"role": "render", "slots": [{"name": "gas", "vdb_id": self.id,
                "aabb_min": [0, 0, 0], "aabb_max": [1, 1, 1], "is_active": 1}]}]}
        raise AssertionError(method)


class SnapshotContracts(unittest.TestCase):
    def test_max_steps_and_resource_id_churn_are_not_a_scene_change(self):
        c = Client(); before = capture(c, ["matter:D:gas"])
        c.cap = 16; c.id = 99
        require_same(before, capture(c, ["matter:D:gas"]))

    def test_spatial_rearrangement_with_equal_sum_is_a_scene_change(self):
        c = Client(); before = capture(c, ["matter:D:gas"])
        c.field["density_hash"] = "different_voxel_order"
        with self.assertRaisesRegex(RuntimeError, "fields"):
            require_same(before, capture(c, ["matter:D:gas"]))

    def test_camera_change_is_a_scene_change(self):
        c = Client(); before = capture(c, ["matter:D:gas"])
        c.camera["target"] = [1, 0, 0]
        with self.assertRaisesRegex(RuntimeError, "camera"):
            require_same(before, capture(c, ["matter:D:gas"]))

    def test_old_build_summary_is_not_claimed_to_be_full_field_identity(self):
        c = Client(); del c.field["density_hash"]
        self.assertEqual(capture(c, ["matter:D:gas"])["fingerprint_strength"], "summary_only")


if __name__ == "__main__":
    unittest.main()
