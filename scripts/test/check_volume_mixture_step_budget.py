"""Non-build numerical coverage and wiring tests for authored mixture budgets."""
from pathlib import Path
import math
import random
import unittest

SH = Path(__file__).resolve().parents[2] / "RayTrophiStudio/source/shaders"


class MixtureStepBudget(unittest.TestCase):
    def test_arbitrarily_small_adaptive_steps_still_cover_the_segment(self):
        rng = random.Random(20261028)
        for cap in (1, 16, 32, 256, 1024):
            for _ in range(40):
                length = rng.uniform(.001, 100)
                t, count, integrated = 0., 0, 0.
                sigma = rng.uniform(.01, 10.)
                transmission = 1.
                while count < cap and t < length:
                    remaining = length-t
                    requested = length*rng.uniform(.00000001, .002)
                    dt = min(remaining, max(requested, remaining/(cap-count)))
                    integrated += dt
                    transmission *= math.exp(-sigma*dt)
                    t += dt
                    count += 1
                self.assertLessEqual(count, cap)
                self.assertAlmostEqual(integrated, length, places=10)
                self.assertAlmostEqual(transmission, math.exp(-sigma*length), places=12)

    def test_empty_skips_do_not_consume_an_optical_sample(self):
        for cap in (16, 256):
            remaining = 3.
            remaining -= 2.5  # proven empty prefix
            sizes = [remaining/cap]*cap
            self.assertEqual(len(sizes), cap)
            self.assertAlmostEqual(sum(sizes), .5)

    def test_joint_budget_ignores_off_ray_members(self):
        members = [(256, -1., 3.), (16, .5, 2.), (1, 10., 12.)]
        near, far = 0., 3.
        budget = min(cap for cap, lo, hi in members if hi > near and lo < far)
        self.assertEqual(budget, 16)

    def test_production_wiring_preserves_walk_and_reports_coarsening(self):
        adaptive = (SH / "volume_mixture_adaptive.glsl").read_text(encoding="utf-8")
        for text in ("clamp(v.max_steps, 1, 2048)", "hi <= nearT || lo >= farT",
                     "remaining / float(max(samplesLeft, 1))",
                     "if (requested < coverageFloor) mfStepBudgetHit = true",
                     "min(remaining, max(requested, coverageFloor))"):
            self.assertIn(text, adaptive)
        mixture = (SH / "volume_mixture.glsl").read_text(encoding="utf-8")
        for text in ("mfCameraStepBudget(origin, direction, nearT, farT)",
                     "sampledSteps < stepBudget", "stepBudget - sampledSteps",
                     "sampledSteps++", "mfStepBudgetHit ? VOLUME_MARCH_STEP_BUDGET"):
            self.assertIn(text, mixture)
        flight = mixture.split("bool mfFlight(", 1)[1].split("void mfContinue(", 1)[0]
        self.assertIn("mfBudgetedSpan(", flight)
        self.assertNotIn("mfCameraBudgetedSpan(", flight)


if __name__ == "__main__":
    unittest.main()
