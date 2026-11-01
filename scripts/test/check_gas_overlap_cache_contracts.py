"""Non-build checks for mixed cached media and retired live Fog resources."""
from pathlib import Path
import math
import unittest

ROOT = Path(__file__).resolve().parents[2] / "RayTrophiStudio/source"


class GasOverlapCacheContracts(unittest.TestCase):
    def test_extinction_weighted_source_is_additive(self):
        # Independent homogeneous emitters must integrate as one joint medium,
        # including zero extinction channels and three or more contributors.
        for count in (1, 2, 3, 16):
            for channel in range(3):
                sigma = [(i + channel) * .13 for i in range(count)]
                source = [(i + 1) * .7 for i in range(count)]
                total = sum(sigma)
                emitted = sum(s * j for s, j in zip(sigma, source))
                length = 1.7
                analytic = emitted * (-math.expm1(-total * length)) / total if total else 0
                transmittance, radiance = 1., 0.
                for _ in range(37):
                    segment_t = math.exp(-total * length / 37)
                    radiance += transmittance * (emitted / total if total else 0) * (1-segment_t)
                    transmittance *= segment_t
                self.assertAlmostEqual(radiance, analytic, places=12)
                self.assertAlmostEqual(transmittance, math.exp(-total * length), places=12)

    def test_single_heavy_call_path_and_safe_fallback(self):
        shader = ROOT / "shaders"
        route = (shader / "volume_cached_segment.glsl").read_text(encoding="utf-8")
        cache = (shader / "volume_radiance_cache.glsl").read_text(encoding="utf-8")
        march = cache[cache.index("bool volumeRadianceCacheMarch("):]
        self.assertEqual(route.count("volumeRadianceCacheMarch("), 1)
        self.assertEqual(march.count("volumeRadianceCacheFetch("), 1)
        self.assertNotIn("VrcMedium media[", march)
        self.assertIn("program != MATPROG_NONE || v.random_walk_enabled > 0.5", route)
        self.assertLess(march.index("if (!ok) return false;"), march.index("radiance = rad;"))

    def test_empty_mixture_sample_stops_before_thermal_work(self):
        shader = (ROOT / "shaders/volume_mixture_material_sample.glsl").read_text(encoding="utf-8")
        evaluate = shader.split("bool mfSampleMaterial(", 1)[1].split("MfTerms mfEvaluate(", 1)[0]
        guard = "if (density <= 0.0 && program == MATPROG_NONE)"
        self.assertLess(evaluate.index(guard), evaluate.index("sampleTemperature("))
        self.assertLess(evaluate.index("mfSamples++"), evaluate.index(guard))

    def test_camera_affine_reuse_keeps_walk_fallback(self):
        shader = (ROOT / "shaders/volume_mixture_adaptive.glsl").read_text(encoding="utf-8")
        for suffix in ("A", "B"):
            self.assertIn("if (!mfWalking && index == mfCacheIndex" + suffix + ")", shader)
            self.assertIn("ray = mfCameraRay" + suffix + ";", shader)
            self.assertIn("ray = volumeNanoRay(v, origin, direction, mfCacheBuffer" + suffix,
                          shader)
        self.assertGreaterEqual(shader.count("if (mfWalking) {"), 2)

    def test_retired_fog_is_removed_but_appearance_survives(self):
        publisher = (ROOT / "src/Physics/Fluid/MatterFogPublisher.cpp").read_text(encoding="utf-8")
        retirement = publisher[publisher.index("    // A representation change"):
                               publisher.index("    std::size_t new_streams")]
        for operation in ("field.key == it->first", "manager.unloadVDB(stream.vdb_id)",
                          "removeVDBVolume(stream.volume)", "world.objects.erase",
                          "streams.erase(it)", "dirtyFogGeometry()"):
            self.assertIn(operation, retirement)
        self.assertNotIn("desc.matter_fog_streams.erase", retirement)
        ui = (ROOT / "src/UI/scene_ui_vdb.cpp").read_text(encoding="utf-8")
        self.assertIn("selection.selected.vdb_volume) == scene.vdb_volumes.end()", ui)


if __name__ == "__main__":
    unittest.main()
