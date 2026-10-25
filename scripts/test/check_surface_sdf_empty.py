"""Non-build SDF skip regression oracles and cross-language counter ABI checks."""
import math
import random
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(name):
    return (SOURCE / name).read_text(encoding="utf-8")


def search(values, origin, direction, step, accelerated, modifier, main):
    """Independent trilinear field/crossing oracle on a 1D slice of 8^3 leaves."""
    def raw(x):
        i = math.floor(x)
        f = x - i
        cell = lambda k: values[k] if 0 <= k < len(values) else 0
        return max(cell(i) * (1 - f) + cell(i + 1) * f, 0)

    def field(t):
        x = origin + direction * t
        return raw(x) - modifier(x)

    limit = (len(values) - 1 - origin) if direction > 0 else origin
    count = min(16384, math.ceil(limit / step) + 1)
    t = 0.0
    d = field(t)
    inside = d > (.55 if main else .5)
    if main and not inside:
        d = min(d, .5 - 1e-4)
    samples, skipped = 1, 0
    i = 0
    while i < count:
        if accelerated and not inside and d < .5:
            x = origin + direction * t
            base = math.floor(x / 8) * 8
            bound = max(values[k] if 0 <= k < len(values) else 0
                        for k in range(base, base + 8))
            boundary = base + 7 if direction > 0 else base
            span = min(limit - t, (boundary - x) / direction)
            span -= max(1e-6, abs(span) * 1e-5)
            skip = min(count - i, max(0, math.floor(span / step)))
            if bound < .5 - 1e-5 and x < base + 7 and skip >= 2:
                t += skip * step
                d = field(t)
                samples += 1
                skipped += skip - 1
                i += skip
                if t >= limit:
                    break
                continue
        nxt = min(t + step, limit)
        other = field(nxt)
        samples += 1
        crossed = d >= .5 and other < .5 if inside else d < .5 and other >= .5
        if crossed:
            # Compare the bracket after the same four original refinements.
            a, b = t, nxt
            for _ in range(4):
                mid = (a + b) / 2
                if (field(mid) > .5) == inside:
                    a = mid
                else:
                    b = mid
            return (a + b) / 2, samples, skipped
        t, d = nxt, other
        i += 1
        if t >= limit:
            break
    return None, samples, skipped


class SdfEmptyChecks(unittest.TestCase):
    def test_sparse_entry_keeps_crossing_and_reduces_reads(self):
        values = [0] * 40 + [1] * 16 + [0] * 8
        for main in [False, True]:
            baseline = search(values, .1, 1, .1, False, lambda x: 0, main)
            fast = search(values, .1, 1, .1, True, lambda x: 0, main)
            self.assertAlmostEqual(baseline[0], fast[0], places=8)
            self.assertLess(fast[1], baseline[1] / 3)
            self.assertGreater(fast[2], 0)

    def test_randomized_directions_and_modifiers(self):
        rng = random.Random(25102026)
        for _ in range(400):
            values = [rng.uniform(0, .2) if i // 8 < 4 else rng.uniform(.7, 1)
                      for i in range(64)]
            direction = rng.choice([-1, 1])
            origin = rng.uniform(.01, 63)
            step = rng.uniform(.025, .4)
            # Nonnegative subtractive pores/masks may also cut interior holes.
            modifier = lambda x: .8 if 43 < x < 46 else .08 * (1 + math.sin(x))
            for main in [False, True]:
                a = search(values, origin, direction, step, False, modifier, main)
                b = search(values, origin, direction, step, True, modifier, main)
                if a[0] is None:
                    self.assertIsNone(b[0])
                else:
                    self.assertIsNotNone(b[0])
                    self.assertAlmostEqual(a[0], b[0], places=7)
        helper = read("shaders/surface_sdf_empty.glsl")
        self.assertIn("floor(index / float(cache.width))", helper)
        self.assertIn("vec3(float(cache.width) - 1.0)", helper)

    def test_neighbor_slab_and_inside_exit_are_not_skipped(self):
        for direction, origin in [(1, 7.1), (-1, 55.9), (1, 40.01)]:
            values = [0] * 8 + [1] * 48 + [0] * 8
            modifier = lambda x: 1 if 41 < x < 42 else 0
            for main in [False, True]:
                a = search(values, origin, direction, .05, False, modifier, main)
                b = search(values, origin, direction, .05, True, modifier, main)
                self.assertAlmostEqual(a[0], b[0], places=8)
                self.assertEqual(b[2], 0)

    def test_bounds_are_raw_conservative_and_cached_per_search(self):
        helper = read("shaders/surface_sdf_empty.glsl")
        for forbidden in ["density_multiplier", "density_remap", "volumeNanoRemap",
                          "pnanovdb_readaccessor_is_active"]:
            self.assertNotIn(forbidden, helper)
        for guard in ["startedInside", "!(previous < 0.5)", "v._uvw_pad[0] < 0.5",
                      "PNANOVDB_GRID_FLAGS_HAS_MIN_MAX", "isnan(bound)",
                      "isinf(bound)", "cache.upper < 0.5 - 1e-5"]:
            self.assertIn(guard, helper)
        self.assertLess(helper.index("if (!cached)"), helper.index("queries++"))
        for name in ["main", "arbiter"]:
            loop = read(f"shaders/surface_sdf_{name}_search.glsl")
            self.assertIn("sdfRegion.width = 0u", loop)
            self.assertIn("sdfSkipped += uint(skip - 1)", loop)
            self.assertIn("sampleIsoField", loop)
            self.assertIn("s += skip - 1", loop)
            self.assertIn("sdfSamples++", loop)

    def test_shader_cpp_counter_abi_and_all_public_surfaces(self):
        shader = read("shaders/volume_instrumentation.glsl")
        body = shader.split("buffer VolumeInstrumentationBuffer {", 1)[1].split(
            "} volumeInstrumentation", 1)[0]
        glsl_fields = re.findall(r"^\s+(?:uint|float)\s+(\w+);", body, re.M)
        cpp = read("include/Backend/VulkanVolumeInstrumentation.h")
        body = cpp.split("struct VolumePerformanceStats {", 1)[1].split("};", 1)[0]
        cpp_fields = re.findall(r"^\s+(?:uint32_t|float)\s+(\w+)\s*=", body, re.M)
        self.assertEqual(glsl_fields, cpp_fields)
        self.assertEqual(len(cpp_fields), 52)
        self.assertIn("sizeof(VolumePerformanceStats) == 208", cpp)
        self.assertEqual(cpp_fields[15], "enabled")
        self.assertEqual(cpp_fields[25:29], ["regionMinX", "regionMinY", "regionMaxX", "regionMaxY"])
        for part in ["main", "arbiter"]:
            for kind in ["samples", "queries", "skipped"]:
                field = f"sdf_{part}_{kind}"
                for name in ["include/Api/RtApi.h", "src/Api/RtApi.cpp",
                             "src/Api/RtIpc.cpp", "src/Api/RtPython.cpp"]:
                    self.assertIn(field, read(name))
        record = read("shaders/surface_sdf_instrumentation.glsl")
        self.assertLess(record.index("if (!volumeInstrumentationEnabled()) return"),
                        record.index("atomicAdd"))

    def test_large_shader_only_wires_modules_and_keeps_handoff(self):
        shader = read("shaders/volume_closesthit.rchit")
        for part in ["main", "arbiter"]:
            self.assertIn(f'#include "surface_sdf_{part}_search.glsl"', shader)
        self.assertIn("plSolidHandoffWord(payload.volumeTraversalMask, solidT)", shader)
        self.assertIn("const uint PROBE_MASK  = 0xF5", shader)
        self.assertIn("surfaceSdfMarchStep", shader)


if __name__ == "__main__":
    unittest.main()
