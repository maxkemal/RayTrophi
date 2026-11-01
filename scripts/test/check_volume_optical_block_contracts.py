"""Guard the rollback of the failed section-20 optical-block experiment.

Live cost rose 329 -> 766 ms, with more camera samples and majorant queries.
Keep the measured successful bounds/cache path while rejecting its rerouting.
"""
from pathlib import Path
from volume_shader_source import read_volume_shader
import unittest

SOURCE = Path(__file__).resolve().parents[2] / "RayTrophiStudio/source"


class OpticalBlockRollbackContracts(unittest.TestCase):
    def test_failed_block_path_is_not_reachable(self):
        for name in ("volume_closesthit.rchit", "volume_mixture.glsl",
                     "volume_mixture_adaptive.glsl"):
            source = read_volume_shader(SOURCE / "shaders" / name)
            for retired in ("matterFogBoundedSingle", "mfBoundedBlockSpan", "mfBlockTauLeft",
                            'include "volume_mixture_optical_blocks.glsl"'):
                self.assertNotIn(retired, source, name)

    def test_successful_bounds_and_cache_are_retained(self):
        publisher = (SOURCE / "src/Physics/Fluid/MatterFogPublisher.cpp").read_text(encoding="utf-8")
        self.assertIn("fogOpticalBounds(field.density", publisher)
        hit = read_volume_shader(SOURCE / "shaders/volume_closesthit.rchit")
        self.assertIn("bool segmentCached = volumeCachedSegment(", hit)
        mixture = (SOURCE / "shaders/volume_mixture.glsl").read_text(encoding="utf-8")
        self.assertIn("mfContinue(cachedRadiance, cachedTransmission", mixture)
        sample = (SOURCE / "shaders/volume_mixture_material_sample.glsl").read_text(encoding="utf-8")
        self.assertIn("if (density <= 0.0 && program == MATPROG_NONE)", sample)


if __name__ == "__main__":
    unittest.main()
