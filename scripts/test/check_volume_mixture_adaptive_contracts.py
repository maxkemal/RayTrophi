"""Non-build numerical bounds and source wiring; does not execute GPU shaders."""
from pathlib import Path
import math
import random
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(path):
    return (SOURCE / path).read_text(encoding='utf-8-sig')


def trilinear(corners, position):
    return sum(corners[i] * math.prod(position[a] if i & (1 << a)
                                     else 1-position[a] for a in range(3))
               for i in range(8))


def slope_bound(corners, direction):
    return sum(max(abs(corners[i | (1 << axis)] - corners[i])
                   for i in range(8) if not i & (1 << axis)) * abs(direction[axis])
               for axis in range(3))


def cell_exit(position, direction):
    cell = [math.floor(p) - int(d < 0 and p == math.floor(p))
            for p, d in zip(position, direction)]
    return min(((c + (1 if d > 0 else 0) - p) / d)
               for c, p, d in zip(cell, position, direction) if d)


class MixtureAdaptiveContracts(unittest.TestCase):
    def test_cell_cache_preserves_march_partition_and_reduces_corner_reads(self):
        # Independent continuous trilinear lattice. Compare the whole camera
        # integration trajectory, not merely a cache hit's returned value.
        def corners(cell):
            return [0.2 + 0.03 * ((cell[0] + (i & 1)) % 7) +
                    0.02 * (cell[1] + ((i >> 1) & 1)) +
                    0.01 * (cell[2] + ((i >> 2) & 1)) for i in range(8)]

        def march(cached):
            cache = None
            reads = 0
            t = 0.125
            trajectory = []
            transmission = 1.0
            while t < 12.875:
                point = [t, 0.25, 0.375]
                cell = tuple(math.floor(p) for p in point)
                if cached and cache is not None and cache[0] == cell:
                    bound = cache[1]
                else:
                    values = corners(cell)
                    bound = (slope_bound(values, [1, 0, 0]), max(values))
                    reads += 8
                    cache = (cell, bound)
                dt = min(12.875-t, cell_exit(point, [1, 0, 0]),
                         max(0.25*0.05, min(0.05, math.sqrt(0.04/max(bound[0], 1e-12)))))
                # Same midpoint emission/extinction integral as the reference.
                sample = [t + 0.5*dt - cell[0], 0.25, 0.375]
                density = trilinear(corners(cell), sample)
                transmission *= math.exp(-density*dt)
                trajectory.append((t, dt, density, transmission))
                t += dt
            return trajectory, reads

        reference, uncached_reads = march(False)
        optimized, cached_reads = march(True)
        self.assertEqual(optimized, reference)
        self.assertLess(cached_reads, uncached_reads/4)

    def test_camera_cell_cache_scope_and_walk_fallback(self):
        cache = read('shaders/volume_mixture_cell_cache.glsl')
        self.assertIn('if (mfWalking) return false;', cache)
        self.assertIn('all(equal(cell, mfBoundCellA))', cache)
        self.assertIn('all(equal(cell, mfBoundCellB))', cache)
        self.assertIn('mfBoundValidA = false;', cache)
        self.assertIn('mfBoundValidB = false;', cache)
        shader = read('shaders/volume_mixture.glsl')
        shade = shader.split('void shadeFogMixture(', 1)[1]
        self.assertLess(shade.index('mfResetCellBounds();'), shade.index('if (cached)'))
        self.assertNotIn('mfCurveSlope', shader)
        bounds = read('shaders/volume_mixture_adaptive.glsl').split(
            'bool mfAdaptiveCellBounds(', 1)[1].split('ivec3 mfForwardCell(', 1)[0]
        self.assertLess(bounds.index('mfLoadCellBounds('), bounds.index('volumeNanoCorners('))
        self.assertLess(bounds.index('any(isnan(low))'), bounds.index('mfStoreCellBounds('))

    def test_trilinear_directional_bound(self):
        rng = random.Random(20261026)
        for _ in range(3000):
            corners = [rng.uniform(0, 10) for _ in range(8)]
            a = [rng.random() for _ in range(3)]
            b = [rng.random() for _ in range(3)]
            direction = [y-x for x, y in zip(a, b)]
            limit = slope_bound(corners, direction)
            for t in (.13, .47, .91, 1):
                p = [x + t*d for x, d in zip(a, direction)]
                self.assertLessEqual(abs(trilinear(corners, p)-trilinear(corners, a)),
                                     limit*t + 1e-12)

    def test_cutoff_fade_derivative_bound(self):
        # h(d)=d*smoothstep(c,2c,d), with u=(d-c)/c.
        for i in range(10001):
            u = i/10000
            derivative = u*u*(3-2*u) + (1+u)*6*u*(1-u)
            self.assertGreaterEqual(derivative, 0)
            self.assertLessEqual(derivative, 4)

    def test_midpoint_optical_error_bound(self):
        rng = random.Random(260)
        for _ in range(300):
            corners = [rng.random()*50 for _ in range(8)]
            origin = [rng.random() for _ in range(3)]
            end = [rng.random() for _ in range(3)]
            direction = [b-a for a, b in zip(origin, end)]
            sample = lambda t: trilinear(corners, [a+t*d for a, d in zip(origin, direction)])
            # A trilinear field restricted to a straight chord is cubic;
            # Simpson's integral is exact inside this interpolation cell.
            exact = (sample(0) + 4*sample(.5) + sample(1))/6
            self.assertLessEqual(abs(exact-sample(.5)), slope_bound(corners, direction)/4+1e-12)

    def test_constant_medium_integral_independent_of_partition(self):
        rng = random.Random(9)
        for _ in range(500):
            sigma = [rng.uniform(.01, 4) for _ in range(3)]
            source = [rng.uniform(0, 3) for _ in range(3)]
            length = rng.uniform(.01, 10)
            parts = [rng.random() for _ in range(30)]
            parts = [p/sum(parts)*length for p in parts]
            transmission = [1., 1., 1.]
            radiance = [0., 0., 0.]
            for dt in parts:
                for channel in range(3):
                    step_t = math.exp(-sigma[channel]*dt)
                    radiance[channel] += transmission[channel]*source[channel]*(1-step_t)/sigma[channel]
                    transmission[channel] *= step_t
            for channel in range(3):
                exact_t = math.exp(-sigma[channel]*length)
                self.assertAlmostEqual(transmission[channel], exact_t, places=12)
                self.assertAlmostEqual(radiance[channel], source[channel]*(1-exact_t)/sigma[channel], places=11)

    def test_exponential_flight_survives_variable_segments(self):
        # One optical target survives every adaptive segment: never redraw it.
        target = 1.3
        intervals = [(0.2, .5), (.1, 2.), (.4, 4.)]
        distance = 0.
        for dt, sigma in intervals:
            if target < dt*sigma:
                distance += target/sigma
                break
            target -= dt*sigma
            distance += dt
        self.assertAlmostEqual(distance, .55)
        shader = read('shaders/volume_mixture.glsl')
        flight = shader.split('bool mfFlight(', 1)[1].split('void mfContinue(', 1)[0]
        self.assertLess(flight.index('float opticalTarget'), flight.index('for (int i'))
        self.assertIn('opticalTarget -= proposal * dt', flight)
        self.assertIn('hitDistance = start + length', flight)

    def test_forward_boundaries_negative_and_transformed_directions(self):
        self.assertEqual(cell_exit([0, 0, 0], [-1, -2, -3]), 1/3)
        self.assertEqual(cell_exit([-1, -2, -3], [-1, -2, -3]), 1/3)
        self.assertAlmostEqual(cell_exit([.3, .4, .7], [2, 0, -4]), .175)
        self.assertAlmostEqual(cell_exit([.3, .4, .7], [-2, 0, 4]), .075)

    def test_third_contributor_cannot_be_cancelled(self):
        # Opposite density gradients cannot cancel the safety bound; include
        # a third fine source even when the two broad fields are almost flat.
        components = [([0, 1]*4, [1, 0, 0]), ([1, 0]*4, [1, 0, 0]),
                      ([0, 100]*4, [10, 0, 0])]
        bound = sum(slope_bound(c, d) for c, d in components)
        self.assertEqual(bound, 1002)
        shader = read('shaders/volume_mixture_adaptive.glsl')
        self.assertIn('index < mfCount', shader)
        self.assertIn('slope += bound.x * (scatter', shader)
        self.assertIn('bound.x = mfRawSlope(low, high, direction) * mfCellCurveSlope(', shader)
        self.assertIn('span = min(span, mfCellExit', shader)
        self.assertIn('span = min(span, lo - t)', shader)

    def test_density_low_high_low_has_no_history(self):
        base = .05
        span = lambda scale: min(4*base, max(.25*base, math.sqrt(.04/(scale*100))))
        before = span(.1)
        self.assertLess(span(100), before)
        self.assertEqual(span(.1), before)
        shader = read('shaders/volume_mixture_adaptive.glsl')
        self.assertIn('abs(volumes.v[index].density_multiplier)', shader)
        self.assertIn('float slope = 0.0', shader)
        self.assertNotIn('densityThreshold', shader)

    def test_cell_curve_slope_bounds_the_range_it_is_given(self):
        def curve(x, knots):
            if x <= 0:
                return x
            if x >= 1:
                return x*knots[7]
            segment = min(int(x*7), 6)
            return knots[segment] + (knots[segment+1]-knots[segment])*(x*7-segment)

        def bound(low, high, knots, cutoff):
            top, bottom = curve(high, knots), curve(low, knots)
            if top <= cutoff:
                return 0
            slope = max(knots[7], 0) if high >= 1 else 0
            if low < 1:
                for segment in range(min(int(low*7), 6), min(int(high*7), 6)+1):
                    slope = max(slope, 7*abs(knots[segment+1]-knots[segment]))
            return slope*(4 if cutoff > 0 and bottom < cutoff*2 else 1)

        def mapped(x, knots, cutoff):
            d = curve(x, knots)
            if d <= cutoff:
                return 0
            u = min((d-cutoff)/cutoff, 1) if cutoff > 0 else 1
            return d*u*u*(3-2*u)

        rng = random.Random(20261030)
        for _ in range(4000):
            knots = [0, *sorted(rng.random() for _ in range(7))]
            cutoff = rng.choice([0, 0, .012, .2])
            low = rng.uniform(0, 1.2)
            high = low + rng.uniform(0, .4)
            limit = bound(low, high, knots, cutoff)
            points = [low + (high-low)*i/32 for i in range(33)]
            for a, b in zip(points, points[1:]):
                self.assertLessEqual(abs(mapped(b, knots, cutoff)-mapped(a, knots, cutoff)),
                                     limit*(b-a) + 1e-9)
        # The live saturated curve: flat above its first segment, steep below.
        saturated = [0, 1, 1, 1, 1, 1, 1, 1]
        self.assertEqual(bound(.3, .5, saturated, .012), 0)
        self.assertEqual(bound(.05, .1, saturated, .012), 7)

    def test_only_the_optical_bound_refines_past_the_authored_lattice(self):
        def refined(cell, optical, base):
            return max(max(min(cell, optical), min(base, optical)), 1e-6)
        rng = random.Random(30)
        for _ in range(3000):
            base = rng.uniform(.01, .1)
            cell = rng.uniform(1e-4, .3)
            optical = rng.uniform(.25*base, 4*base)
            span = refined(cell, optical, base)
            self.assertGreaterEqual(span, min(base, optical))
            self.assertLessEqual(span, max(cell, base))
            self.assertLessEqual(span, optical)
        helper = read('shaders/volume_mixture_adaptive.glsl')
        walk = read('shaders/volume_mixture_shared_walk.glsl')
        self.assertIn('mfRefinedSpan(span, opticalSpan, baseStep, progress)', helper)
        self.assertIn('mfRefinedSpan(span, opticalSpan, baseStep, progress)', walk)

    def test_camera_cutoff_matches_the_ordinary_march(self):
        shader = read('shaders/volume_mixture.glsl')
        self.assertIn('if (mfPeak(throughput) < 0.001) break;', shader)
        self.assertNotIn('mfPeak(throughput) < 1e-6', shader)
        self.assertIn('< 0.001) {', read('shaders/volume_fog_shading.glsl'))

    def test_mixture_shadow_keeps_the_authored_sample_count(self):
        shader = read('shaders/volume_mixture.glsl')
        shadow = shader.split('vec3 mfShadow(', 1)[1].split('vec3 mfDirect(', 1)[0]
        self.assertIn('mfShadowT = lo + (float(step) + 0.5) * dt;', shadow)
        self.assertNotIn('tauHint', shadow)

    def test_packing_and_existing_authoring_surfaces(self):
        shader = read('include/VolumeShader.h')
        self.assertIn('curve[9] = adaptive_stepping ? 1.0f : 0.0f', shader)
        producer = read('src/Render/VolumetricRenderer.cpp')
        self.assertEqual(producer.count('quality.packSamplingPolicy(gv.density_curve)'), 3)
        self.assertIn('float density_curve[12]', read('include/Backend/vulkan_volume_types.h'))
        self.assertIn('k == "adaptive_stepping"', read('src/Scene/VolumeMaterialAuthoring.cpp'))
        self.assertIn('drawVolumeAdaptiveStepping(*shader)', read('src/UI/scene_ui.cpp'))
        ui = read('src/UI/VolumeMaterialUI.cpp')
        self.assertIn('volume_material::configure(shader, {{"adaptive_stepping", adaptive}}', ui)
        binding = read('src/Api/RtVolumeMaterialBindings.cpp')
        self.assertIn('volume.set_material', binding)
        self.assertIn('volume.def("set_material"', binding)

    def test_both_integrators_and_conservative_fallbacks(self):
        shader = read('shaders/volume_mixture.glsl')
        self.assertEqual(shader.count('mfBudgetedSpan('), 1)
        self.assertEqual(shader.count('mfCameraBudgetedSpan('), 1)
        routing = shader.split('bool matterFogNeedsGeneralMixture(', 1)[1].split('bool mfContains(', 1)[0]
        self.assertIn('a.density_curve[9] > 0.5 || b.density_curve[9] > 0.5', routing)
        helper = read('shaders/volume_mixture_adaptive.glsl')
        for guard in ('v.density_curve[9] < 0.5', 'v.volume_type != 2',
                      'v._ext_reserved[7] >= 0.5', 'matProgramOffset',
                      'v.emission_mode != 0 && densityPresent'):
            self.assertIn(guard, helper)
        self.assertIn('mfStepBudgetHit ? VOLUME_MARCH_STEP_BUDGET', shader)
        self.assertIn('return remaining;', helper)
        self.assertIn('marchT += dt', shader)
        self.assertIn('start += dt', shader)


if __name__ == '__main__':
    unittest.main()
