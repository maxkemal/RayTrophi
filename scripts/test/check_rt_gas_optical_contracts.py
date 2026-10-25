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
    def test_prepared_shadow_extinction_matches_full_material_terms(self):
        rng = random.Random(24102026)
        for _ in range(2000):
            scatter, absorb = rng.uniform(-1, 30), rng.uniform(-1, 30)
            color = [rng.uniform(-1, 2) for _ in range(3)]
            front, back, mix = rng.uniform(-.9, .9), rng.uniform(-.9, .9), rng.random()
            walking = rng.choice([False, True])
            ordinal, exact = rng.randrange(520), rng.randrange(-2, 520)
            density = rng.uniform(-.1, 10)
            mean = back * (1 - mix) + front * mix
            coefficient = max(scatter, 0)
            if walking and mean > .05 and ordinal >= min(max(exact, 1), 512):
                coefficient *= 1 - mean
            weights = [max(c, 0) for c in color]
            peak = max(weights)
            weights = [c / peak for c in weights] if peak > 1e-5 else [1]*3
            prepared = [max(density, 0) * coefficient +
                        max(density, 0) * max(absorb, 0) * c for c in weights]
            reference_scatter = max(density, 0) * max(scatter, 0)
            if walking and mean > .05 and ordinal >= min(max(exact, 1), 512):
                reference_scatter = max(density, 0) * (max(scatter, 0) * (1 - mean))
            reference = [reference_scatter + max(density, 0) * max(absorb, 0) * c
                         for c in weights]
            self.assertEqual(prepared, reference)
        shader = (SHADERS / "volume_mixture.glsl").read_text(encoding="utf-8")
        shadow = shader.split("vec3 mfShadow(", 1)[1].split("vec3 mfDirect(", 1)[0]
        self.assertLess(shadow.index("shadowScatter = scatterCoefficient"),
                        shadow.index("for (int step = 0; step < steps; ++step)"))
        self.assertIn("if (mfShadowIndexPath)", shadow)
        self.assertIn("mfShadowSamples++;", shadow)
        self.assertIn("mfEvaluate(index, samplePosition, direction, false).extinction", shadow)

    def test_mixture_shadow_skip_keeps_midpoints_and_optical_depth(self):
        # Independent reference integration: a proven zero prefix followed by
        # nonzero density. Check distinct authored budgets, including 1 and 2.
        rng = random.Random(20261024)
        for steps in [1, 2, 3, 8, 16, 32, 64]:
            for _ in range(100):
                length = rng.uniform(.01, 50.0)
                boundary = rng.uniform(0.0, length)
                dt = length / steps
                samples = [(i + .5) * dt for i in range(steps)]
                density = lambda t: 0.0 if t < boundary else .2 + t * .03
                reference = sum(density(t) * dt for t in samples)
                optical = 0.0
                retained = []
                step = 0
                while step < steps:
                    t = (step + .5) * dt
                    span = min(boundary - t, length - t) if t < boundary else dt
                    if span > dt * 1.01:
                        step += max(1, int(math.floor(span / dt)))
                        continue
                    retained.append(t)
                    optical += density(t) * dt
                    step += 1
                self.assertAlmostEqual(optical, reference, places=10)
                self.assertTrue(all(t in retained for t in samples if density(t) > 0))

    def test_mixture_shadow_proof_guards_and_interval_reuse(self):
        shader = (SHADERS / "volume_mixture.glsl").read_text(encoding="utf-8")
        helper = shader.split("float mfShadowEmptyStep(", 1)[1].split(
            "// Each material's authored shadow budget", 1)[0]
        for guard in ["!mfShadowIndexPath", "v._uvw_pad[0] < 0.5",
                      "v._ext_reserved[7] >= 0.5", "volumeNanoRemap(v, 1e-6)",
                      "!= MATPROG_NONE"]:
            self.assertIn(guard, helper)
        self.assertEqual(helper.count("return nanoEmptyIndexStep("), 2)
        shadow = shader.split("vec3 mfShadow(", 1)[1].split("vec3 mfDirect(", 1)[0]
        self.assertLess(shadow.index("mfShadowEmptyStep(index, v, dt)"),
                        shadow.index("mfEvaluate(index, samplePosition"))
        self.assertIn("int(floor(emptySpan / dt))", shadow)
        self.assertIn("step += skip - 1;", shadow)
        self.assertIn("!cameraFast && !shadowFast && !mfContains", shader)

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

    def test_joint_empty_prefix_keeps_visible_samples_and_lattice(self):
        # Independent occupied supports: never jump over a vapor sample merely
        # because the Snow support starts later. Compare complete optical sums
        # on the same jittered lattice, including negative world coordinates.
        rng = random.Random(20261024)
        for _ in range(1000):
            dt = rng.uniform(.001, .1)
            origin = rng.uniform(-10, 10)
            jitter = rng.random()
            occupied = [sorted(rng.sample(range(96), 12)) for _ in range(2)]
            values = [[rng.random() if i in support else 0.0
                       for i in range(96)] for support in occupied]
            reference = sum((values[0][i] + values[1][i]) * dt for i in range(96))
            measured = 0.0
            i = 0
            while i < 96:
                if values[0][i] == 0.0 and values[1][i] == 0.0:
                    next_live = min(next((k for k in support if k >= i), 96)
                                    for support in occupied)
                    if next_live > i:
                        before = origin + (i + jitter) * dt
                        i = next_live
                        after = origin + (i + jitter) * dt
                        self.assertGreater(after, before)
                        continue
                measured += (values[0][i] + values[1][i]) * dt
                i += 1
            self.assertEqual(reference, measured)
        shader = (SHADERS / "volume_companion_empty.glsl").read_text(encoding="utf-8")
        for guard in ["nanoIndexPath && compNanoIndexPath", "insideBoth",
                      "vol._uvw_pad[0] > 0.5 && comp._uvw_pad[0] > 0.5",
                      "min(primarySpan, companionSpan)", "tFar - t",
                      "floor(jointSpan / baseStep)"]:
            self.assertIn(guard, shader)
        self.assertNotIn("rnd(", shader)

    def test_sun_shadow_reuse_requires_exact_chord(self):
        shader = (SHADERS / "volume_sun_shadow_cache.glsl").read_text(encoding="utf-8")
        # Nearby directions cannot share jittered quadrature, even if a dot
        # product rounds to one. A finite directional chord must not truncate
        # either the primary or companion's unlimited sky chord.
        for guard in ["all(equal(direction, normalize(worldData.w.sunDir)))",
                      "primaryDistance != fullPrimary", "lightLimit < fullCompanion",
                      "volumeSunQueryStep != step", "volumeSunQueryDistance != distance",
                      "all(equal(volumeSunQueryPosition, position))",
                      "all(equal(volumeSunQueryDirection, direction))"]:
            self.assertIn(guard, shader)
        self.assertNotIn("rnd(", shader)
        rt = (SHADERS / "volume_closesthit.rchit").read_text(encoding="utf-8")
        self.assertLess(rt.index("volumeRememberSunShadow(lightType"),
                        rt.index("volumeReuseSunShadow(samplePos"))
        self.assertIn("inscatter += sunLi * sunPhase * sunShadowTr", rt)

    def test_leaf_direct_read_never_crosses_a_leaf_boundary(self):
        import itertools
        for cell in itertools.product(range(-10, 11), repeat=3):
            same_leaf = all((axis & 7) < 7 for axis in cell)
            corner_leaves = {tuple((cell[a] + ((i >> a) & 1)) // 8
                                   for a in range(3)) for i in range(8)}
            self.assertEqual(same_leaf, len(corner_leaves) == 1)
        shader = (SHADERS / "volume_nano_optical.glsl").read_text(encoding="utf-8")
        for guard in ["cell & ivec3(7)", "sameLeaf && level != 0u",
                      "pnanovdb_readaccessor_get_value_address_and_level(",
                      "pnanovdb_leaf_get_value_address("]:
            self.assertIn(guard, shader)

    def test_solid_handoff_bound_stays_before_surface_and_preserves_mask(self):
        import struct
        def bits(f):
            return struct.unpack('<I', struct.pack('<f', f))[0]
        def number(u):
            return struct.unpack('<f', struct.pack('<I', u))[0]
        rng = random.Random(20261025)
        distances = [10**rng.uniform(-6, 12) for _ in range(5000)]
        # Include exact bin boundaries: rounding must still leave room BEFORE
        # the surface instead of placing tMin exactly on the hit.
        distances += [number(u) for u in [0x3F800000, 0x40000000, 0x4B000000]]
        for value in distances:
            distance = number(bits(value))
            bound = bits(distance) & 0xFFFFFF00
            bound = bound - 0x100 if bound >= 0x100 else 0
            for mask in [0xFF, 0xFD, 0xF5]:
                word = bound | mask
                self.assertEqual(word & 0xFF, mask)
                self.assertLess(number(word & 0xFFFFFF00), distance)
        raygen = (SHADERS / "raygen.rgen").read_text(encoding="utf-8")
        self.assertIn("solidHandoffTMin = useSkipVolumeAABBs", raygen)
        self.assertLess(raygen.index("plSolidHandoffTMin(payload.volumeTraversalMask"),
                        raygen.index("payload.volumeTraversalMask = cullMask"))
        rt = (SHADERS / "volume_closesthit.rchit").read_text(encoding="utf-8")
        self.assertEqual(rt.count("plSolidHandoffWord(payload.volumeTraversalMask, solidT)"), 1)
        sdf = rt.split("// ── Solid geometry inside the domain", 1)[1]
        self.assertIn("plSolidHandoffWord(payload.volumeTraversalMask, solidT)",
                      sdf.split("if (hitT < 0.0)", 1)[0])

    def test_distance_probe_precedes_triangle_buffer_reads(self):
        shader = (SHADERS / "shadow_anyhit.rahit").read_text(encoding="utf-8")
        main = shader.split("void main()", 1)[1]
        probe = main.index("volumeSolidProbeCandidate(shadowPayload, gl_HitTEXT)")
        self.assertLess(probe, main.index("instances.i[gl_InstanceID]"))
        self.assertIn("gl_RayFlagsSkipClosestHitShaderEXT) != 0u &&", main[:probe])
        self.assertNotIn("terminateRayEXT", main[:probe])

    def test_surface_bounces_stay_charged_after_free_handoff(self):
        shader = (SHADERS / "raygen.rgen").read_text(encoding="utf-8")
        self.assertLess(shader.index("payload.skipGasVolumes = false;", shader.index("bool useSkipGasVolumes")),
                        shader.index("traceRayEXT(", shader.index("bool useSkipGasVolumes")))
        charged = shader.split("if (!isTransparentPass && !mediumPass &&", 1)[1]
        charged = charged.split("rayOrigin = payload.scatterOrigin", 1)[0]
        self.assertIn("!payload.skipGasVolumes", charged)
        self.assertIn("diffuseBounceCount++", charged)
        self.assertIn("transmissionBounceCount++", charged)
        self.assertIn("bounce++", charged)
        self.assertIn("volumeRecordPass(4u)", charged)

    def test_retired_inline_probe_cannot_load_from_stale_spv(self):
        backend = (SHADERS.parent / "src/Backend/VulkanBackend.cpp").read_text(encoding="utf-8")
        self.assertIn('volChitSPV = loadSPV(shaderDir + "/volume_closesthit.spv")', backend)
        self.assertNotIn("volumeSolidProbeShaderPath", backend)
        shader = (SHADERS / "volume_closesthit.rchit").read_text(encoding="utf-8")
        self.assertNotIn("VOLUME_TRY_INLINE_SOLID", shader)
        self.assertNotIn("VOLUME_INLINE_SOLID_QUERY", shader)
        plan = (ROOT / "RayTrophiStudio/compile_shaders.ps1").read_text(encoding="utf-8")
        self.assertNotIn("-DVOLUME_INLINE_SOLID_QUERY", plan)
        self.assertIn("'volume_closesthit_rayquery.spv'", plan.split('$staleNames =', 1)[1])

    def test_surface_shadow_membership_matches_fog_not_sdf(self):
        shader = (SHADERS / "volume_surface_shadow.glsl").read_text(encoding="utf-8")
        first_lookup = shader.index("pnanovdb_grid_get_tree")
        for reject in ["if (vol.source_type == 4) continue",
                       "vol.shadow_steps <= 0 || vol.shadow_strength <= 0.0",
                       "if (tFW <= tNW) continue"]:
            self.assertLess(shader.index(reject), first_lookup)
        # RayFusion excludes source_type 4 from participating-medium shadows.
        raster = (SHADERS / "material_preview_volume_shadow.comp").read_text(encoding="utf-8")
        self.assertIn("volInt(vi,428u)==4", raster)
        surface = (SHADERS / "closesthit.rchit").read_text(encoding="utf-8")
        self.assertIn('#include "volume_surface_shadow.glsl"', surface)
        self.assertNotIn("float computeVolumeShadowTransmittance(", surface)

    def test_surface_shadow_small_budgets_and_counters(self):
        for requested in range(1, 65):
            low = min(3, requested)
            high = min(requested, 16)
            self.assertLessEqual(low, high)
            for scale in [.25, .5, 1]:
                steps = max(low, min(math.ceil(requested * scale), high))
                self.assertLessEqual(steps, requested)
                self.assertGreaterEqual(steps, 1)
        shader = (SHADERS / "volume_surface_shadow.glsl").read_text(encoding="utf-8")
        self.assertIn("clamp(steps, min(3, reqSteps), min(reqSteps, 16))", shader)
        self.assertEqual(shader.count("measuredShadowSamples++"), 2)
        self.assertIn("volumeRecordShadowSamples(measuredShadowSamples)", shader)
        rt = (SHADERS / "closesthit.rchit").read_text(encoding="utf-8")
        self.assertLess(rt.index('#include "volume_surface_shadow.glsl"'),
                        rt.index('#include "bsdf_scatter.glsl"'))


if __name__ == "__main__":
    unittest.main()
