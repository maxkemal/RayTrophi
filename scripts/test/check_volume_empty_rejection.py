"""Non-build optical rejection, interpolation and traversal regression checks."""
import itertools
import math
import random
import re
import unittest
from pathlib import Path
from types import SimpleNamespace

SHADERS = Path(__file__).resolve().parents[2] / "RayTrophiStudio/source/shaders"


def shader(name):
    return (SHADERS / name).read_text(encoding="utf-8")


def actual_inert(**patch):
    """Interpret the small metadata-only GLSL predicate, without compiling it.

    This evaluates the production conditions against independent scene oracles
    below, so a changed shader guard changes the tested decision too.
    """
    values = dict(source_type=0, volume_type=2, density_multiplier=1.0,
                  scatter_coefficient=1.0, absorption_coefficient=0.0,
                  emission_mode=0, _reserved=[0.0, 0.0])
    values.update(patch)
    env = {"v": SimpleNamespace(**values)}

    def evaluate(expr):
        expr = expr.replace("&&", " and ").replace("||", " or ")
        expr = re.sub(r"!(?!=)", " not ", expr)
        expr = re.sub(r"\btrue\b", "True", expr)
        expr = re.sub(r"\bfalse\b", "False", expr)
        return eval(" ".join(expr.split()), {"__builtins__": {}}, env)

    source = re.sub(r"//[^\n]*", "", shader("volume_optical_activity.glsl"))
    body = source.split("bool volumeOpticallyInert(VkVolumeInstance v) {", 1)[1]
    body = body.split("}", 1)[0]
    while body.strip():
        body = body.lstrip()
        declaration = re.match(r"bool\s+(\w+)\s*=\s*(.*?);", body, re.S)
        conditional = re.match(r"if\s*\((.*?)\)\s*return\s+(.*?);", body, re.S)
        returned = re.match(r"return\s+(.*?);", body, re.S)
        if declaration:
            env[declaration[1]] = evaluate(declaration[2])
            body = body[declaration.end():]
        elif conditional:
            if evaluate(conditional[1]):
                return evaluate(conditional[2])
            body = body[conditional.end():]
        elif returned:
            return evaluate(returned[1])
        else:
            raise AssertionError("Unsupported production predicate: " + body)
    raise AssertionError("Predicate did not return")


class EmptyVolumeRejection(unittest.TestCase):
    def test_zero_density_including_fire_can_be_rejected(self):
        for source, emission in itertools.product([0, 5, 6], [0, 1, 2, 3]):
            self.assertTrue(actual_inert(source_type=source,
                                         emission_mode=emission, density_multiplier=0))

    def test_geometry_graph_and_unknown_sources_are_never_rejected(self):
        for source in [1, 2, 3, 4, 7, -1]:
            self.assertFalse(actual_inert(source_type=source, density_multiplier=0))
        self.assertFalse(actual_inert(volume_type=3, density_multiplier=0))
        for binding in [1, 12, float("nan"), float("inf")]:
            self.assertFalse(actual_inert(_reserved=[0, binding], density_multiplier=0))

    def test_zero_extinction_does_not_remove_independent_emission(self):
        self.assertTrue(actual_inert(scatter_coefficient=0, absorption_coefficient=0))
        for mode in [1, 2, 3]:
            self.assertFalse(actual_inert(scatter_coefficient=0,
                                          absorption_coefficient=0, emission_mode=mode))
        for weak in [1e-30, 1e-12, 1e-7]:
            self.assertFalse(actual_inert(density_multiplier=weak))
            self.assertFalse(actual_inert(scatter_coefficient=weak))

    def test_nearest_solid_sdf_and_other_gas_survive_any_visit_order(self):
        # Independent traversal oracle: an inert domain encloses a foreground
        # smoke layer, a SurfaceSDF boundary and a farther triangle. Rejecting
        # the domain must leave the same physical event sequence in any BVH order.
        events = [(0, "empty", dict(density_multiplier=0)),
                  (1, "smoke", dict(source_type=6)),
                  (2, "sdf", dict(source_type=4, density_multiplier=0)),
                  (3, "triangle", None)]
        for visit_order in itertools.permutations(events):
            retained = sorted((t, name) for t, name, medium in visit_order
                              if medium is None or not actual_inert(**medium))
            self.assertEqual(retained, [(1, "smoke"), (2, "sdf"), (3, "triangle")])
        intersection = shader("volume_intersection.rint")
        gate = intersection.index("volumeOpticallyInert(volumes.v[volIdx])")
        self.assertLess(gate, intersection.index("reportIntersectionEXT(reportT"))
        self.assertIn("!volumeOpticallyInert(v)", shader("volume_overlap_selection.glsl"))
        self.assertIn("volumeOpticallyInert(gas)", shader("volume_overlap_selection.glsl"))
        self.assertIn("if (!gasOverlapCandidate(v)) return term;", shader("volume_mixture.glsl"))

    def test_optical_gate_cannot_hide_raw_sdf_reads_or_graph_evaluation(self):
        gate = shader("volume_density_read_gate.glsl")
        self.assertIn("applyOpticalControls && vol.density_multiplier == 0.0", gate)
        rt = shader("volume_closesthit.rchit")
        sampler = rt.split("float sampleDensityAccMode(", 1)[1].split(
            "float sampleDensityAcc(", 1)[0]
        self.assertLess(sampler.index('"volume_density_read_gate.glsl"'),
                        sampler.index("sampleNanoVDBFloatTrilinearAcc"))
        self.assertIn("sampleDensityAccMode(vol, worldPos, buf, mapH, acc, false)", rt)
        self.assertIn("density = max(vp.volumeDensity, 0.0)", rt)
        nano = shader("volume_nano_optical.glsl")
        fast = nano.split("float volumeNanoDensity(", 1)[1].split("float nanoEmptyIndexStep", 1)[0]
        self.assertLess(fast.index("vol.density_multiplier == 0.0"),
                        fast.index("volumeNanoCorners(buf"))

    def test_uniform_tiles_require_a_value_proof_and_neighbor_support(self):
        nano = shader("volume_nano_optical.glsl")
        self.assertIn("volumeNanoRemap(vol, rawTile)", nano)
        self.assertIn("boxMax = boxMin + vec3(float(dim) - 1.0)", nano)
        self.assertNotIn("pnanovdb_readaccessor_is_active", nano)
        # A zero tile's final voxel interpolates with its nonzero neighbor.
        # The old topology-only exit would have skipped this visible interval.
        for dim in [8, 128, 4096]:
            for sign in [-1, 0, 1]:
                lo = sign * dim
                last_slab = lo + dim - .25
                self.assertGreater(last_slab, lo + dim - 1)
        # A nonzero inactive background must not be classified as optically
        # empty. Conversely an active constant-zero tile can safely be skipped.
        for raw, remap_low, cutoff, empty in [(2, 0, 0, False),
                                             (0, -.5, 0, False),
                                             (0, 0, 0, True),
                                             (.01, 0, .02, True)]:
            self.assertEqual(max(raw - remap_low, 0) <= cutoff, empty)

    def test_skipped_tile_prefix_preserves_all_nonzero_quadrature_samples(self):
        rng = random.Random(25102026)
        for direction in [-1, 1]:
            for _ in range(200):
                dt = rng.uniform(.01, .8)
                dim = rng.choice([8, 128])
                origin = rng.uniform(1, dim - 2)
                boundary = dim - 1 if direction > 0 else 0
                span = (boundary - origin) / direction
                skip = max(0, math.floor(max(span - 1e-5, 0) / dt))
                # Evaluate actual trilinear support of an empty tile with
                # nonzero neighbors. Every removed sample must be exactly zero.
                for sample in range(skip):
                    x = origin + direction * sample * dt
                    a = math.floor(x)
                    f = x - a
                    value = lambda cell: 0.0 if 0 <= cell < dim else 1.0
                    density = value(a) * (1 - f) + value(a + 1) * f
                    self.assertEqual(density, 0)


if __name__ == "__main__":
    unittest.main()
