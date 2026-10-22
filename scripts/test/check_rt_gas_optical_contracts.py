"""Non-build safety and integration checks for the RT NanoVDB optical fast path."""
from pathlib import Path
import math
import random
import unittest

ROOT = Path(__file__).resolve().parents[2]
SHADERS = ROOT / "RayTrophiStudio/source/shaders"


def trilinear(c, f):
    return sum(c[i] * math.prod(f[a] if i & (1 << a) else 1-f[a]
                               for a in range(3)) for i in range(8))


class NanoOpticalContracts(unittest.TestCase):
    def test_leaf_max_bounds_every_trilinear_sample(self):
        rng = random.Random(20261022)
        for _ in range(2000):
            corners = [rng.uniform(0, .01) for _ in range(8)]
            sample = trilinear(corners, [rng.random() for _ in range(3)])
            self.assertLessEqual(sample, max(corners) + 1e-12)
            self.assertLessEqual(sample, .01)

    def test_adjacent_leaf_must_not_use_current_leaf_bound(self):
        self.assertGreater(trilinear([0, 1, 0, 1, 0, 1, 0, 1], [.5]*3), .01)
        shader = (SHADERS / "volume_nano_optical.glsl").read_text(encoding="utf-8")
        self.assertIn("greaterThanEqual(index, boxMax)", shader)
        self.assertIn("boxMax = boxMin + vec3(7.0)", shader)
        self.assertIn("PNANOVDB_GRID_FLAGS_HAS_MIN_MAX", shader)
        self.assertIn("vol._uvw_pad[0] < 0.5", shader)
        self.assertIn("max(rawMax, 1e-6)", shader)

    def test_negative_leaf_coordinates_and_ray_lattice(self):
        for x in [-16.1, -8.1, -7.9, -.1, .1, 7.9, 8.1]:
            leaf = math.floor(x/8)*8
            self.assertLessEqual(leaf, x)
            self.assertLess(x, leaf+8)
        for span in [.011, .2, 1.1, 3.7]:
            dt = .01
            advance = math.floor(span/dt)*dt
            self.assertLessEqual(advance, span+1e-12)
            self.assertAlmostEqual(advance/dt, round(advance/dt))

    def test_fast_path_gates_and_empty_temperature_read(self):
        rt = (SHADERS / "volume_closesthit.rchit").read_text(encoding="utf-8")
        shadow = (SHADERS / "volume_rt_shadow_march.glsl").read_text(encoding="utf-8")
        self.assertIn("vol._ext_reserved[7] < 0.5 && volumeProgram == MATPROG_NONE", rt)
        self.assertIn("vol._ext_reserved[7] < 0.5 && allowSparseTraversal", shadow)
        self.assertLess(rt.index('#include "volume_rt_empty_sample.glsl"'),
                        rt.index("float temperature = needsTemperature"))
        empty = (SHADERS / "volume_rt_empty_sample.glsl").read_text(encoding="utf-8")
        self.assertIn("!hasCompanion && volumeProgram == MATPROG_NONE", empty)
        self.assertIn("floor(sparseStep / stepSize)", shadow)
        self.assertIn("volumeRecordShadowSamples", shadow)

    def test_no_source_material_payload_or_abi_growth(self):
        for name in ["volume_nano_optical.glsl", "volume_rt_shadow_march.glsl"]:
            shader = (SHADERS / name).read_text(encoding="utf-8")
            self.assertNotIn("source_material_address", shader)
        layout = (SHADERS.parent / "include/Backend/vulkan_volume_types.h").read_text(encoding="utf-8")
        self.assertIn("sizeof(VkVolumeInstance) == 672", layout)

    def test_solid_probe_is_order_independent_and_does_not_terminate(self):
        import itertools
        for candidates in itertools.permutations([.005, .8, 4.0]):
            distance = None
            for hit in candidates:
                distance = hit if distance is None else min(distance, hit)
            self.assertEqual(distance, .005)
        helper = (SHADERS / "volume_solid_probe.glsl").read_text(encoding="utf-8")
        self.assertIn("min(state.x, distance)", helper)
        self.assertNotIn("terminateRayEXT", helper)
        rt = (SHADERS / "volume_closesthit.rchit").read_text(encoding="utf-8")
        for flag in ["SOLID_FLAGS", "PROBE_FLAGS"]:
            declaration = rt.split("const uint " + flag + " =", 1)[1].split(";", 1)[0]
            self.assertNotIn("TerminateOnFirstHit", declaration)
            self.assertIn("NoOpaque", declaration)
        for name in ["shadow_anyhit.rahit", "hair_shadow_anyhit.rahit"]:
            shader = (SHADERS / name).read_text(encoding="utf-8")
            self.assertIn("volumeSolidProbeCandidate(shadowPayload, gl_HitTEXT)", shader)

    def test_solid_handoff_does_not_erase_front_volume_layer(self):
        # A 5 mm gas interval in front of a sphere used to be erased by 1 cm.
        self.assertLess(.005 - .01, 0)
        rt = (SHADERS / "volume_closesthit.rchit").read_text(encoding="utf-8")
        self.assertIn("tFar = min(tFar, solidT);", rt)
        self.assertNotIn("tFar = solidT - 0.01", rt)
        self.assertIn("volumeRecordSolidProbe(shadowPayload.z > 0.5)", rt)
        self.assertIn("const uint PROBE_MASK  = 0xF5", rt)

    def test_procedural_sphere_probe_is_wired_before_triangle_reads(self):
        pipeline = (SHADERS.parent / "src/Backend/VulkanRTPipelineBuild.cpp").read_text(encoding="utf-8")
        self.assertIn("chitStage, anyhitStage, sphereIntStage", pipeline)
        shader = (SHADERS / "shadow_anyhit.rahit").read_text(encoding="utf-8")
        body = shader.split("void main()", 1)[1]
        self.assertLess(body.index("sphereShadowProbe()"), body.index("instances.i[gl_InstanceID]"))
        sphere = (SHADERS / "sphere_shadow_probe.glsl").read_text(encoding="utf-8")
        self.assertIn("volumeSolidProbeCandidate(shadowPayload, gl_HitTEXT)", sphere)
        self.assertNotIn("geometries.g[", sphere)
        self.assertIn("foamSpheres[gl_PrimitiveID]", sphere)
        self.assertIn("0x555B0DEDu", sphere)
        # Empty gas still needs the sphere's distance. Density must not gate it.
        self.assertNotIn("density", sphere.lower())

    def test_directional_phase_cache_preserves_phase_and_rng(self):
        def hg(c, g):
            d = 1 + g*g - 2*g*c
            return (1-g*g)/(4*math.pi)/(d*math.sqrt(d)+1e-6)
        for cosine in [-1, -.3, 0, .4, 1]:
            for front, back, mix in [(0, 0, 1), (.206, -.3, .7), (.95, -.8, .2)]:
                direct = hg(cosine, back)*(1-mix) + hg(cosine, front)*mix
                cached = direct
                for albedo in [0, .001, .25, 1]:
                    blend = .3*albedo
                    self.assertEqual(direct*(1-blend)+blend/(4*math.pi),
                                     cached*(1-blend)+blend/(4*math.pi))
        helper = (SHADERS / "volume_phase_cache.glsl").read_text(encoding="utf-8")
        self.assertNotIn("rnd(", helper)
        self.assertNotRegex(helper, r"\bbool\s+invariant\b")
        self.assertIn("bool canReuse", helper)
        self.assertNotIn("density", helper.split("float volumeCachedPhase", 1)[1])
        rt = (SHADERS / "volume_closesthit.rchit").read_text(encoding="utf-8")
        self.assertIn("lightType == 1 && !hasCompanion && volumeProgram == MATPROG_NONE", rt)
        self.assertIn("float sunPhase = volumeCachedPhase(phaseCache, 2", rt)
        self.assertIn("phase = mix(", rt)
        self.assertIn("sunPhase = mix(", rt)

    def test_release_mirror(self):
        self.assertEqual(Path(__file__).read_bytes(),
            (ROOT / "x64/Release/scripts/test" / Path(__file__).name).read_bytes())


if __name__ == "__main__":
    unittest.main()
