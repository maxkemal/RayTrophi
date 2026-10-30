"""Mixture shadow-cache math/wiring contracts; never compile or launch Studio."""
import math
from pathlib import Path
import random
import unittest

ROOT = Path(__file__).resolve().parents[2]
SH = ROOT/'RayTrophiStudio/source/shaders'


def weights(color):
    positive = [max(c, 0) for c in color]
    peak = max(positive)
    return [c/peak for c in positive] if peak > 1e-5 else [1.0]*3


def old_shadow(fields):
    visibility = [1.0]*3
    for samples, dt, scatter, absorption, color, strength in fields:
        tau = [0.0]*3
        w = weights(color)
        for density in samples:
            rate = max(density, 0)*max(scatter, 0)
            for channel in range(3):
                tau[channel] += (rate + max(density, 0)*max(absorption, 0)*w[channel])*dt
            if min(tau) > 16:
                break
        strength = min(max(strength*.92, 0), 1)
        for channel in range(3):
            visibility[channel] *= 1-strength + strength*math.exp(-tau[channel])
    return visibility


def cached_shadow(fields):
    visibility = [1.0]*3
    for samples, dt, scatter, absorption, color, strength in fields:
        integral = sum(max(d, 0) for d in samples)*dt
        strength = min(max(strength*.92, 0), 1)
        for channel, weight in enumerate(weights(color)):
            sigma = max(scatter, 0) + max(absorption, 0)*weight
            visibility[channel] *= 1-strength + strength*math.exp(-integral*sigma)
    return visibility


class MixtureLightCache(unittest.TestCase):
    def test_independent_nonlinear_density_integrals_preserve_rgb_and_strength(self):
        rng = random.Random(20261030)
        for _ in range(1500):
            fields = []
            for index in range(2):
                n = rng.choice((1, 4, 8, 16, 32, 64))
                raw = [rng.uniform(-.1, 2) for _ in range(n)]
                # Each field keeps its own nonlinear remap/cutoff. Never sum raw
                # fields first or apply one material response to their sum.
                exponent = 1 + index*.7
                samples = [max(d-.2, 0)**exponent if d > .3 else 0 for d in raw]
                fields.append((samples, rng.uniform(.001, .2), rng.uniform(-1, 4),
                               rng.uniform(-1, 4), [rng.uniform(-1, 2) for _ in range(3)],
                               rng.uniform(-.2, 1.5)))
            reference, cache = old_shadow(fields), cached_shadow(fields)
            for a, b in zip(reference, cache):
                # The old march may stop once every channel is below exp(-16).
                # A full cached integral can only reduce that negligible tail.
                self.assertLessEqual(abs(a-b), 2*math.exp(-16) + 1e-12)

    def test_colored_absorption_is_not_collapsed_to_scalar(self):
        fields = [([1]*8, .125, 0, 2, [1, .1, 0], 1),
                  ([.5]*4, .25, 0, 3, [0, 1, .25], .6)]
        value = cached_shadow(fields)
        self.assertNotAlmostEqual(value[0], value[1])
        self.assertNotAlmostEqual(value[1], value[2])
        for a, b in zip(value, old_shadow(fields)):
            self.assertAlmostEqual(a, b, places=12)
        self.assertEqual(cached_shadow([([0]*8, .2, 2, 3, [0, 0, 0], 1)]), [1]*3)

    def test_filling_never_undersamples_authored_shadow_count(self):
        for authored in range(1, 65):
            for cells in (.01, 1, 4, 8, 12.5, 31, 63, 100):
                samples = min(max(max(math.ceil(cells), authored), 4), 64)
                self.assertGreaterEqual(samples, authored)
                self.assertLessEqual(samples, 64)
        fill = (SH/'volume_light_cache_fill.glsl').read_text(encoding='utf-8')
        self.assertIn('int authored = clamp(vol.shadow_steps, 1, 64);', fill)
        self.assertIn('max(int(ceil(chord / cellWorld)), authored), 4, 64)', fill)
        self.assertIn('VLC_FILLS_PER_INVOCATION = 4u', fill)

    def test_guarded_lookup_and_complete_aggregate_fallback(self):
        module = (SH/'volume_mixture_light_cache.glsl').read_text(encoding='utf-8')
        eligibility = module.split('void mfPrepareDirectLightCache()', 1)[1].split(
            'bool mfTryDirectLightCache(', 1)[0]
        for token in ('mfWalking', 'volumeLightCacheChannel(mfLightIndex',
                      'index == mfCacheIndexA', 'index == mfCacheIndexB',
                      'volumeNanoIndexEligible', 'volumeLightCacheSlot(index, slot)'):
            self.assertIn(token, eligibility)
        nano = (SH/'volume_nano_optical.glsl').read_text(encoding='utf-8')
        predicate = nano.split('bool volumeNanoIndexEligible(', 1)[1].split(
            'VolumeNanoRay volumeNanoRay(', 1)[0]
        for token in ('vol._ext_reserved[7] < 0.5', 'matProgramOffset', 'MATPROG_NONE'):
            self.assertIn(token, predicate)
        self.assertIn('!mfDirectLightPositional && hi > distance', module)
        self.assertIn('vec3 entry = position + direction * lo;', module)
        self.assertEqual(module.count('volumeLightCacheFetch(index, v, mfLightIndex'), 2)
        self.assertIn('if (!found) {\n            return false;', module)
        wrapper = module.split('vec3 mfDirectLightShadow(', 1)[1]
        self.assertIn('return mfShadow(position, direction, distance);', wrapper)
        self.assertIn('mfExtinctionTerms(index, mfReadOpticalResponse(index), 1.0)', module)
        code = module.split('bool mfTryDirectLightCache(', 1)[1]
        self.assertNotIn('dot(', code)

    def test_scene_light_integration_keeps_ambient_and_walk_only_solid_probe(self):
        shader = (SH/'volume_mixture.glsl').read_text(encoding='utf-8')
        direct = shader.split('vec3 mfDirect(', 1)[1].split('bool mfFlight(', 1)[0]
        self.assertIn('mfShadow(position, ambientDirection, distance)', direct)
        self.assertIn('mfGeometryBlocks(position, wi, distance)', direct)
        self.assertNotIn('rwSolidDistance', direct)
        probe = shader.split('bool mfGeometryBlocks(', 1)[1].split('vec3 mfDirect(', 1)[0]
        self.assertLess(probe.index('if (!mfWalking)'), probe.index('rwSolidDistance('))
        self.assertIn('opticalStep > 0.01', direct)
        self.assertIn('mfDirectLightShadow(position, wi, distance)', direct)
        self.assertLess(shader.index('mfWalking = randomWalk;'),
                        shader.index('mfPrepareDirectLightCache();'))
        self.assertLess(shader.index('mfPrepareDirectLightCache();'),
                        shader.index('if (!randomWalk) {'))
        self.assertIn('mfLightingStep - mfLightCacheStep >= mfShadowStride', direct)


if __name__ == '__main__':
    unittest.main()
