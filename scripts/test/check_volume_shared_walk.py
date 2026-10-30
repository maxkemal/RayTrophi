"""Non-build oracle for the aligned two-field RT traversal specialization.

Does not compile shaders or claim measured GPU parity/performance.
"""
import math
from pathlib import Path
import random
import unittest

from check_volume_mixture_adaptive_contracts import cell_exit, slope_bound, trilinear

SH = Path(__file__).resolve().parents[2] / 'RayTrophiStudio/source/shaders'


def mapped(raw, material):
    if not math.isfinite(raw):
        return 0.0
    low, high, multiplier, cutoff, curve_power = material
    value = max((raw-low)/max(high-low, 1e-6), 0.0)**curve_power
    if value <= cutoff or multiplier == 0:
        return 0.0
    x = min(max((value-cutoff)/cutoff, 0.0), 1.0) if cutoff else 1.0
    return value*multiplier*x*x*(3-2*x)


def reference_span(point, direction, t, remaining, base, fields):
    span, slope = min(remaining, base*4), 0.0
    for corners, curve_slope, coefficient, emissive, material in fields:
        span = min(span, cell_exit(point, direction))
        slope += slope_bound(corners, direction)*curve_slope*coefficient
        if emissive and mapped(max(corners), material) > 0:
            span = min(span, base)
    optical = max(math.sqrt(.04/slope) if slope > 1e-12 else base*4, base*.25)
    return min(remaining, max(min(span, optical), max(base*1e-5, abs(t)*2e-7)))


def shared_span(point, direction, t, remaining, base, fields):
    # One exit, while each field still contributes its own conservative bound.
    span = min(remaining, base*4, cell_exit(point, direction))
    slope = 0.0
    for corners, curve_slope, coefficient, emissive, material in fields:
        slope += slope_bound(corners, direction)*curve_slope*coefficient
        if emissive and mapped(max(corners), material) > 0:
            span = min(span, base)
    optical = max(math.sqrt(.04/slope) if slope > 1e-12 else base*4, base*.25)
    return min(remaining, max(min(span, optical), max(base*1e-5, abs(t)*2e-7)))


class SharedWalkContracts(unittest.TestCase):
    def test_partition_with_independent_materials_and_negative_rays(self):
        rng = random.Random(20261029)
        for _ in range(2000):
            point = [rng.uniform(-40, 40) for _ in range(3)]
            direction = [rng.uniform(-5, 5) for _ in range(3)]
            fields = [([rng.random()*4 for _ in range(8)], rng.uniform(.1, 50),
                       rng.uniform(0, 12), rng.choice([True, False]),
                       (rng.uniform(-.1, .2), rng.uniform(.5, 2),
                        rng.uniform(0, 15), rng.choice([0, .01, .1]),
                        rng.uniform(.5, 2))) for _ in range(2)]
            args = (point, direction, rng.uniform(0, 100), rng.uniform(.01, 10),
                    rng.uniform(.001, .3), fields)
            self.assertEqual(reference_span(*args), shared_span(*args))

    def test_full_ray_transmission_emission_and_independent_temperature(self):
        # Traverse a continuous lattice. Reference transforms/floors each field;
        # shared walk does so once. Keep different density curves, absorption
        # colors and nonlinear thermal responses to catch raw-density merging.
        materials = [(0, 1, 11.8, .01, 1), (.05, 1.2, 1.55, .02, 1.8)]
        def run(shared):
            t, transmission, radiance, trajectory = .125, [1.0]*3, [0.0]*3, []
            while t < 8.875:
                point = [t, .375, -.25]
                def corners(slot, cell):
                    return [.01 + .08*((cell[0]+(i & 1)+slot*3) % 7) +
                            .02*((cell[1]+((i >> 1) & 1)) % 4) +
                            .01*((cell[2]+((i >> 2) & 1)) % 5) for i in range(8)]
                cell = [math.floor(p) for p in point]
                fields = [(corners(i, cell), 20+i*5, 2+i, True, materials[i])
                          for i in range(2)]
                span = shared_span if shared else reference_span
                dt = span(point, [1, 0, 0], t, 8.875-t, .05, fields)
                midpoint = [t+dt*.5, .375, -.25]
                common_cell = [math.floor(p) for p in midpoint]
                common_fraction = [p-c for p, c in zip(midpoint, common_cell)]
                sigma, emission = [0.0]*3, [0.0]*3
                for i in range(2):
                    field_cell = common_cell if shared else [math.floor(p) for p in midpoint]
                    f = common_fraction if shared else [p-c for p, c in zip(midpoint, field_cell)]
                    density = mapped(trilinear(corners(i, field_cell), f), materials[i])
                    kelvin = 700 + 200*i + 20*midpoint[0]
                    for channel in range(3):
                        extinction = density*(1+i+channel*.15)
                        sigma[channel] += extinction
                        emission[channel] += extinction*(kelvin/1000)**(2+i+channel*.1)
                for channel in range(3):
                    attenuation = math.exp(-sigma[channel]*dt)
                    integral = (1-attenuation)/sigma[channel] if sigma[channel] > 1e-6 else dt
                    radiance[channel] += transmission[channel]*emission[channel]*integral
                    transmission[channel] *= attenuation
                trajectory.append((t, dt, tuple(sigma), tuple(emission)))
                t += dt
            return trajectory, transmission, radiance
        self.assertEqual(run(False), run(True))

    def test_source_gates_and_shared_density_lifetime(self):
        module = (SH/'volume_mixture_shared_walk.glsl').read_text()
        for guard in ('mfWalking', 'members != 2u', 'a.source_type != 6',
                      'b.source_type != 6', '!volumeNanoIndexEligible(a',
                      '!volumeNanoIndexEligible(b', 'a.density_curve[9] < 0.5',
                      'b.density_curve[9] < 0.5', 'a.aabb_min, b.aabb_min',
                      'a.aabb_max, b.aabb_max', 'mfCameraIntervalA, mfCameraIntervalB',
                      'mfCameraRayA.indexOrigin, mfCameraRayB.indexOrigin',
                      'mfCameraRayA.indexDirection, mfCameraRayB.indexDirection',
                      'mfCameraRayA.localOrigin, mfCameraRayB.localOrigin',
                      'mfCameraRayA.localDirection, mfCameraRayB.localDirection'):
            self.assertIn(guard, module)
        shader = (SH/'volume_mixture.glsl').read_text()
        self.assertIn('cameraFast && mfSharedDensityReady', shader)
        self.assertLess(shader.index('mfPrepareSharedDensity(mfCameraT);'),
                        shader.index('MfTerms sum = mfSum(position, direction, true);'))
        self.assertIn('mfSharedWalk = false;', shader)
        self.assertIn('mfSharedDensityReady = false;', shader)
        self.assertLess(shader.index('mfWalking = randomWalk;'),
                        shader.index('mfConfigureSharedWalk(origin, direction);'))
        helper = (SH/'volume_mixture_adaptive.glsl').read_text()
        self.assertIn('return mfSharedCameraSpan(t, remaining, baseStep);', helper)
        self.assertIn('return mfAdaptiveSpan(origin, direction, t, remaining, baseStep);', helper)
        self.assertIn('min(spanB, hi)', module)

    def test_unit_terms_scale_to_the_canonical_response(self):
        rng = __import__('random').Random(31)
        for _ in range(2000):
            scatter = rng.uniform(0, 3)
            absorb = rng.uniform(0, 3)
            weights = [rng.random() for _ in range(3)]
            peak = max(weights)
            weights = [w/peak for w in weights] if peak > 1e-5 else [1, 1, 1]
            density = rng.uniform(1e-4, 40)
            for channel in range(3):
                direct = density*scatter + density*absorb*weights[channel]
                unit = scatter + absorb*weights[channel]
                self.assertAlmostEqual(direct, unit*density, places=9)
        optics = (SH/'volume_mixture_optical_terms.glsl').read_text()
        shared = optics.split('MfTerms mfSharedEvaluate(', 1)[1]
        self.assertNotIn('mfReadOpticalResponse', shared)
        self.assertIn('density <= 0.0', shared)
        walk = (SH/'volume_mixture_shared_walk.glsl').read_text()
        self.assertIn('mfExtinctionTerms(mfCacheIndexA, mfReadOpticalResponse(mfCacheIndexA), 1.0)', walk)

    def test_occupied_leaf_only_repeats_the_answer_it_proved(self):
        nano = (SH/'volume_nano_optical.glsl').read_text()
        guard = nano.split('float rawMax', 1)[1].split('} else {', 1)[0]
        self.assertLess(guard.index('skipKind = 3u;'), guard.index('return baseStep;'))
        walk = (SH/'volume_mixture_shared_walk.glsl').read_text()
        empty = walk.split('float mfSharedEmptyStep(', 1)[1].split('void mfPrepareSharedDensity(', 1)[0]
        self.assertEqual(empty.count('kind == 3u'), 2)
        self.assertLess(empty.index('mfOccupiedLeafValid &&'), empty.index('nanoEmptyIndexStep('))
        self.assertIn('mfOccupiedLeafValid = false;', walk)

    def test_radiance_march_is_keyed_by_slot_and_skips_proven_empty(self):
        cache = (SH/'volume_radiance_cache.glsl').read_text()
        march = cache.split('bool volumeRadianceCacheMarch(', 1)[1]
        self.assertIn('if (mSlot == firstSlot) medium = firstMedium;', march)
        self.assertIn('else if (mSlot == secondSlot) medium = secondMedium;', march)
        self.assertNotIn('if (m == 0u) medium', march)
        self.assertLess(march.index('vrcProvenEmpty('), march.index('volumeRadianceCacheFetch('))
        empty = cache.split('bool vrcProvenEmpty(', 1)[1].split('struct VrcMedium', 1)[0]
        # An unpublished vertex must fall through to the fetch that fills it.
        self.assertIn('uint(word) != densityTag', empty)
        self.assertIn('> 0.0', empty)

    def test_radiance_march_obeys_the_authored_bounce_lod(self):
        cache = (SH/'volume_radiance_cache.glsl').read_text()
        march = cache.split('bool volumeRadianceCacheMarch(', 1)[1]
        self.assertIn('((marchLod >> 8) & 15u) + 1u', march)
        self.assertIn('(marchLod >> 12) & 511u', march)
        self.assertLess(march.index('float bounceStep'), march.index('float h = len'))

    def test_optical_response_and_reservoir_remain_shared_with_fallback(self):
        module = (SH/'volume_mixture_optical_terms.glsl').read_text()
        self.assertIn('mfSamples++;', module)
        self.assertIn('mfSharedTemperature(index, position)', module)
        walk = (SH/'volume_mixture_shared_walk.glsl').read_text()
        self.assertIn('return sampleTemperature(volumes.v[index], position);', walk)
        self.assertIn('prepared = v.emission_mode >= 2 && v.vdb_temp_address != uint64_t(0);', walk)
        self.assertIn('source = volumeEmissionAt(volumes.v[index], position, density, temperature)', module)
        self.assertIn('term.emission = source * term.extinction;', module)
        self.assertIn('term.scatterRate = max(density, 0.0) * scatterCoefficient;', module)
        shader = (SH/'volume_mixture.glsl').read_text()
        sample = (SH/'volume_mixture_material_sample.glsl').read_text()
        self.assertIn('response.scatterColor = material.volumeScatterColor;', sample)
        self.assertIn('response.absorptionCoefficient = max(material.volumeAbsorptionStrength, 0.0);', sample)
        self.assertIn('response.anisotropy = clamp(material.volumeAnisotropy, -0.99, 0.99);', sample)
        self.assertIn('response.multiScatter = clamp(material.volumeMultiScatter, 0.0, 1.0);', sample)
        self.assertIn('mfOpticalTerms(index, response, position, density, temperature, includeEmission)', sample)
        gather = shader.split('MfTerms mfSum(', 1)[1].split('#include', 1)[0]
        self.assertIn('mfSharedEvaluate(i, position, density, includeEmission)', gather)
        self.assertIn('mfEvaluate(i, position, direction, includeEmission)', gather)
        self.assertEqual(gather.count('rnd(payload.seed)'), 1)
        self.assertIn('sum.extinction += term.extinction;', gather)
        self.assertIn('sum.emission += term.emission;', gather)


if __name__ == '__main__':
    unittest.main()
