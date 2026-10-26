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
        self.assertIn('slope += mfRawSlope', shader)
        self.assertIn('span = min(span, mfCellExit', shader)
        self.assertIn('span = min(span, lo - t)', shader)

    def test_density_low_high_low_has_no_history(self):
        base = .05
        span = lambda scale: min(4*base, max(.25*base, math.sqrt(.04/(scale*100))))
        before = span(.1)
        self.assertLess(span(100), before)
        self.assertEqual(span(.1), before)
        shader = read('shaders/volume_mixture_adaptive.glsl')
        self.assertIn('abs(v.density_multiplier)', shader)
        self.assertIn('float slope = 0.0', shader)
        self.assertNotIn('densityThreshold', shader)

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
        self.assertEqual(shader.count('mfBudgetedSpan('), 2)
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
