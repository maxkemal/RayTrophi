"""Non-build math/contracts; GPU execution and C++ compilation remain user-run."""
import itertools
import math
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio/source"


def read(name):
    return (SRC / name).read_text(encoding="utf-8")


def integral(sigma, length):
    tau = sigma * length
    return length * (1 - tau/2 + tau*tau/6) if tau < .001 else -math.expm1(-tau)/sigma


class TransportMath(unittest.TestCase):
    def test_emission_integral_vacuum_and_opaque_limits(self):
        previous = math.inf
        for sigma in (0, 1e-12, 1e-8, .1, 1, 100, 10000):
            result = 3 * integral(sigma, 2)
            expected = 6 if sigma == 0 else 3 * -math.expm1(-sigma*2)/sigma
            self.assertAlmostEqual(result, expected, places=10)
            self.assertLessEqual(result, previous)
            previous = result

    def test_homogeneous_emission_invariant_to_partition_count(self):
        for sigma, count in itertools.product((0, 1e-7, .1, 1, 20), (1, 30, 32, 128, 256)):
            length = 2
            dt = length/count
            measured = sum(math.exp(-sigma*k*dt)*integral(sigma, dt) for k in range(count))
            self.assertAlmostEqual(measured, integral(sigma, length), places=8)

    def test_j_invariant_to_smoke_multiplier_and_sigma_is_not(self):
        for power in (0, 1, 1.861, 4):
            reference = .4 ** power
            for multiplier in (0, .001, 1, 30, 60):
                sampled = .4 * (multiplier if multiplier > 0 else 1)
                base = sampled/(multiplier if multiplier > 0 else 1)
                self.assertAlmostEqual(base ** power, reference)
                optical = sampled if multiplier > 0 else 0
                self.assertAlmostEqual(optical, .4*multiplier)

    def test_guided_ris_and_full_local_sparse_box_pdf_keep_energy(self):
        box = (.2, .3, .5)
        sparse = (0, 0, 1)
        local = (.75, .25, 0)
        q = tuple(.75*l + .25*(.875*s+.125*b) for l, s, b in zip(local, sparse, box))
        self.assertAlmostEqual(sum(q), 1)
        self.assertTrue(all(p > 0 for p in q))
        j = (0, 2, 9)
        guide = (1, .7, 1e-6)
        visibility_kernel = (.4, .3, .01)
        expected = sum(a*b for a, b in zip(j, visibility_kernel))
        for count in (1, 2, 3):
            total = 0
            for draws in itertools.product(range(3), repeat=count):
                probability = math.prod(q[i] for i in draws)
                weights = [j[i]/q[i]*guide[i] for i in draws]
                summed = sum(weights)
                if not summed:
                    continue
                for i, weight in zip(draws, weights):
                    if weight:
                        value = (j[i]/q[i])*(summed/(count*weight))*visibility_kernel[i]
                        total += probability*(weight/summed)*value
            self.assertAlmostEqual(total, expected, places=10)


class TransportContracts(unittest.TestCase):
    def test_preview_fields_owns_transport_dependency_for_shadow_compute(self):
        fields = read("shaders/material_preview_volume_fields.glsl")
        dependency = '#include "../include/VolumeEmissionTransport.h"'
        self.assertIn(dependency, fields)
        self.assertLess(fields.index(dependency), fields.index("float sampleEmissionField("))
        shadow = read("shaders/material_preview_volume_shadow.comp")
        self.assertIn('#include "material_preview_volume_fields.glsl"', shadow)

    def test_shared_core_and_all_camera_consumers(self):
        core = read("include/VolumeEmissionTransport.h")
        for fn in ("volumeEmissionInputDensity", "volumeEmissionOpticalDensity",
                   "volumeEmissionIntegral", "volumeEmissionChannel"):
            self.assertIn(fn, core)
        self.assertIn("tau < 0.001f", core)
        self.assertIn("volumeEmissionChannel", read("shaders/volume_material_response.glsl"))
        for file in ("src/Device/ray_color.cuh", "src/Render/Renderer.cpp",
                     "shaders/material_preview_volume.frag", "shaders/volume_fog_shading.glsl"):
            text = read(file)
            self.assertIn("volumeEmissionIntegrate", text)
            self.assertIn("volumeEmissionOpticalDensity", text)
        self.assertIn("volumeEmissionCoefficient", read("shaders/volume_mixture_optical_terms.glsl"))
        self.assertNotIn("volumeEmissionIndependent", read("shaders/volume_cached_segment.glsl"))
        for shader in ("shaders/volume_internal_rt.glsl", "shaders/material_preview_volume.frag"):
            self.assertIn("volumeEmissionLimitScale", read(shader))
        fog = read("shaders/volume_fog_shading.glsl")
        self.assertIn("(vol.emission_mode != 0 || matterFogJointWalk(volIdx, volCount))", fog)
        cache = read("shaders/volume_radiance_cache.glsl")
        self.assertIn("source = sigmaT * inscatter + emis;", cache)
        self.assertIn("density *= volumeSootScale(vol, p, emissionDensity, temperature);", cache)
        self.assertIn("density = max(density, 1e-12);", cache)

    def test_opt_in_default_validation_and_common_bindings(self):
        response = read("include/VolumeEmissionResponse.h")
        self.assertIn("bool independent_emission = false", response)
        self.assertIn('!j.at("independent_emission").is_boolean()', response)
        author = read("src/Scene/VolumeMaterialAuthoring.cpp")
        self.assertIn('next.emission_response.independent_emission = boolean(v, k)', author)
        self.assertIn('{"independent_emission", s.emission_response.independent_emission}', author)
        self.assertIn('patch["independent_emission"]', read("src/UI/VolumeMaterialUI.cpp"))
        self.assertIn("independent_emission", read("src/Api/RtVolumeMaterialBindings.cpp"))
        self.assertIn("previousResponse.independent_emission", author)

    def test_zero_density_geometry_and_cache_policy(self):
        self.assertIn("volumeEmissionIndependent(v._retired_cloud[0])",
                      read("shaders/volume_optical_activity.glsl"))
        self.assertIn("independent_emission", read("include/Fluid/FogOpticalBounds.h"))
        self.assertIn("independent emission coefficients must retain each source density",
                      read("src/Physics/Fluid/MatterFogPublisher.cpp"))
        integral = read("shaders/volume_internal_integral.glsl")
        self.assertIn("volumeEmissionIntegrate(source, source, sigma, dt, flags)", integral)
        self.assertIn("volumeEmissionSegmentActive", integral)
        # A retired fire-scatter gate on this slot silently killed internal light.
        for shader in ("volume_internal_rt.glsl", "volume_internal_preview.glsl"):
            source = read("shaders/" + shader)
            self.assertNotIn("density_curve[11] > 0.0 ||", source)
            self.assertNotIn("668u) > 0.0", source)

    def test_walk_bounds_indirect_flights_only(self):
        mixture = read("shaders/volume_mixture.glsl")
        self.assertIn("if (mfEventOrdinal == 0u && volumeCameraSegment) return gathered;", mixture)
        self.assertEqual(mixture.count("radiance += mfBoundFlightEmission(gathered);"), 2)

    def test_shared_header_registration_and_float_literals(self):
        core = re.sub(r"//[^\n]*", "", read("include/VolumeEmissionTransport.h"))
        literals = re.findall(r"\b\d+\.\d+(?:[eE][+-]?\d+)?[fF]?", core)
        self.assertTrue(literals)
        self.assertTrue(all(x.endswith(("f", "F")) for x in literals))
        for file in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
            tree = ET.parse(ROOT / "RayTrophiStudio" / file)
            self.assertEqual(sum(n.attrib.get("Include") ==
                r"source\include\VolumeEmissionTransport.h" for n in tree.iter()), 1)

    def test_shader_emission_source_call_arity(self):
        for file in ("volume_material_response.glsl", "volume_internal_rt.glsl",
                     "volume_fog_shading.glsl", "volume_mixture_optical_terms.glsl",
                     "volume_radiance_cache.glsl", "material_preview_volume.frag"):
            text = read("shaders/" + file)
            for match in re.finditer(r"\bvolumeEmissionSource\s*\(", text):
                depth, arguments = 1, 1
                for char in text[match.end():]:
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    elif char == "," and depth == 1:
                        arguments += 1
                self.assertEqual((depth, arguments), (0, 8), file)


if __name__ == "__main__":
    unittest.main()
