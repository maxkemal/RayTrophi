"""Non-build density-channel regressions; shader execution remains user-run."""
from pathlib import Path
import unittest

SHADERS = Path(__file__).resolve().parents[2] / "RayTrophiStudio/source/shaders"


def read(name):
    return (SHADERS / name).read_text(encoding="utf-8")


def curve(x, values):
    if x <= 0:
        return x
    if x >= 1:
        return x * values[-1]
    t = x * 7
    i = min(int(t), 6)
    return values[i] + (values[i+1] - values[i]) * (t-i)


def cutoff(density, threshold):
    if density <= threshold:
        return 0
    x = min(max((density-threshold)/threshold, 0), 1) if threshold else 1
    return density*x*x*(3-2*x)


def channels(raw, values, threshold, multiplier, independent):
    optical = cutoff(curve(raw, values), threshold)*multiplier
    source = cutoff(raw, threshold)*multiplier if independent else optical
    return optical, source


class DensityChannels(unittest.TestCase):
    def test_plateau_curve_changes_smoke_without_flattening_source(self):
        plateau = (0, 1, 1, 1, 1, 1, 1, 1)
        linear = tuple(i/7 for i in range(8))
        a = channels(.2, plateau, .007, 2.083, True)
        b = channels(.4, plateau, .007, 2.083, True)
        self.assertEqual(a[0], b[0])
        self.assertAlmostEqual(a[1]*2, b[1])
        for raw in (.01, .06, .2, .4, 1, 2):
            self.assertEqual(channels(raw, plateau, .007, 2.083, True)[1],
                             channels(raw, linear, .007, 2.083, True)[1])

    def test_mask_no_longer_rejects_raw_point_because_smoke_curve_is_steep(self):
        # Live mask center .52, width .20: .42..62 transition.
        plateau = (0, 1, 1, 1, 1, 1, 1, 1)
        optical, source = channels(.2, plateau, .007, 2.083, True)
        self.assertGreater(optical/2.083, .62)
        self.assertLess(source/2.083, .42)

    def test_legacy_and_zero_curve_source_support(self):
        plateau = (0, 1, 1, 1, 1, 1, 1, 1)
        for raw in (0, .002, .01, .08, .2, 1, 2):
            optical, source = channels(raw, plateau, .007, 3, False)
            self.assertEqual(optical, source)
        optical, source = channels(.2, (0,)*8, .007, 3, True)
        self.assertEqual(optical, 0)
        self.assertGreater(source, 0)

    def test_common_mapping_and_single_field_lookup(self):
        helper = read("volume_density_channels.glsl")
        self.assertIn("volumeDensityCutoff(normalized, cutoff) : optical", helper)
        self.assertIn("max(optical, normalized)", helper)
        for name in ("volume_hit_sampling.glsl", "volume_nano_optical.glsl",
                     "material_preview_volume_fields.glsl"):
            self.assertIn("volumeDensityChannels(", read(name))
        nano = read("volume_nano_optical.glsl")
        body = nano.split("float volumeNanoDensityChannels(", 1)[1].split(
            "float volumeNanoDensity(", 1)[0]
        self.assertEqual(body.count("volumeNanoCorners("), 1)
        self.assertIn("emissionDensity = channels.y", body)
        preview = read("material_preview_volume_fields.glsl")
        body = preview.split("vec2 sampleFieldChannels(", 1)[1].split(
            "float sampleFieldMultiplier(", 1)[0]
        self.assertEqual(body.count("sampleRawField("), 1)

    def test_emission_consumers_and_empty_sample_use_source_channel(self):
        self.assertIn("volumeEmissionAt(vol, samplePos, emissionDensity, temperature)",
                      read("volume_fog_shading.glsl"))
        self.assertIn("density <= 0.0 && emissionDensity <= 0.0",
                      read("volume_rt_empty_sample.glsl"))
        self.assertIn("density <= 0.0 && emissionDensity <= 0.0",
                      read("volume_mixture_material_sample.glsl"))
        self.assertIn("sampleCameraField(vi, volAddress(vi,232u), p, emissionDensity)",
                      read("material_preview_volume.frag"))


if __name__ == "__main__":
    unittest.main()
