"""Non-build wiring/ABI checks; scalar runtime cases are in volume_emission_mask_test.cpp."""
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SRC / path).read_text(encoding="utf-8")


class EmissionMaskContracts(unittest.TestCase):
    def test_packet_tail_offset_and_every_consumer_stride(self):
        layout = read("include/Backend/vulkan_volume_types.h")
        body = layout.split("VkVolumeInstance {", 1)[1].split("};", 1)[0]
        body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
        offset = 0
        offsets = {}
        for statement in body.split(";"):
            if not statement.strip():
                continue
            m = re.fullmatch(r"\s*(float|int|uint64_t)\s+(\w+)(?:\[(\d+)\])?\s*", statement)
            self.assertIsNotNone(m, statement)
            size = 8 if m[1] == "uint64_t" else 4
            offset = (offset + size - 1) // size * size
            offsets[m[2]] = offset
            offset += size * int(m[3] or 1)
        self.assertEqual(offsets["emission_mask"], 672)
        self.assertEqual((offset + 15) // 16 * 16, 688)
        for shader in ("volume_closesthit.rchit", "closesthit.rchit",
                       "raygen.rgen", "volume_intersection.rint"):
            self.assertIn("float emission_mask[4];", read("shaders/" + shader))
        for raw in ("material_preview_volume_fields.glsl", "material_preview_sdf_surface.frag",
                    "surface_sdf_shadow_query.glsl"):
            self.assertIn("STRIDE_WORDS = 172u", read("shaders/" + raw))
        self.assertIn("672u + uint(k) * 4u", read("shaders/material_preview_volume.frag"))

    def test_publisher_and_both_backend_packets_copy_mask(self):
        self.assertEqual(read("src/Render/VolumetricRenderer.cpp").count(
            "packMask(gv.emission_mask)"), 3)
        self.assertEqual(read("src/Backend/VulkanBackend_Volumes.cpp").count(
            "src.emission_mask + 4, dst.emission_mask"), 2)
        self.assertEqual(read("include/params.h").count("float emission_mask[4] = {}"), 2)

    def test_mask_precedes_limit_and_reaches_all_emission_paths(self):
        rt = read("shaders/volume_internal_rt.glsl")
        self.assertLess(rt.index("emission *= mask;"), rt.index("emission * volumeEmissionLimitScale("))
        self.assertLess(rt.index("volumeFlameRadiance("), rt.index("emission * volumeEmissionLimitScale("))
        self.assertIn("volumeEmissionMaskFactor(density, volumeEmissionSamplingMultiplier", rt)
        self.assertIn("volumeFlameRadiance(flame,flameGain", read("shaders/material_preview_volume.frag"))
        cuda = read("src/Device/ray_color.cuh")
        self.assertEqual(cuda.count("volumeEmissionMaskFactor("), 4)
        self.assertIn("emission_response.finish(step_emission, d,",
                      read("src/Render/Renderer.cpp"))
        for file in ("volume_fog_shading.glsl", "volume_mixture_optical_terms.glsl",
                     "volume_radiance_cache.glsl"):
            self.assertIn("volumeEmissionMaskFactor(", read("shaders/" + file))

    def test_mask_is_not_applied_to_extinction(self):
        terms = read("shaders/volume_mixture_optical_terms.glsl")
        extinction = terms.split("MfTerms mfExtinctionTerms", 1)[1]
        self.assertNotIn("volumeEmissionMaskFactor", extinction)
        march = read("shaders/volume_fog_shading.glsl")
        for line in march.splitlines():
            if "sigma_t_rgb =" in line or "sampleTransmittance =" in line:
                self.assertNotIn("mask", line)

    def test_controls_serialization_and_atomic_service(self):
        header = read("include/VolumeEmissionResponse.h")
        authoring = read("src/Scene/VolumeMaterialAuthoring.cpp")
        ui = read("src/UI/VolumeMaterialUI.cpp")
        for field in ("enabled", "center", "width", "strength"):
            self.assertIn('"mask_' + field + '"', header)
            self.assertIn('k == "emission_mask_' + field + '"', authoring)
            self.assertIn('patch["emission_mask_' + field + '"]', ui)
            self.assertIn("mask_" + field + " = previousResponse.mask_" + field, authoring)
        self.assertIn('number(v, 1e-6f, 100000, k)', authoring)
        self.assertIn('number(v, 0, 1, k)', authoring)
        self.assertIn("volume_material::configure(shader, patch, error)", ui)
        self.assertIn("volume_material::configure(next, material_patch, error)",
                      read("src/Api/RtVolumeMaterialBindings.cpp"))

    def test_cache_and_merge_cannot_reuse_unmasked_source(self):
        self.assertIn("floatBitsToUint(vol.emission_mask[k])",
                      read("shaders/volume_radiance_cache.glsl"))
        merge = read("src/Physics/Fluid/MatterFogPublisher.cpp")
        self.assertIn("shader->emission_response.mask_enabled", merge)
        self.assertIn("density emission masks must be evaluated per source", merge)
        for field in ("enabled", "center", "width", "strength"):
            self.assertIn('response.erase("mask_' + field + '")', merge)

    def test_shared_gpu_math_does_not_promote_to_fp64(self):
        for name in ("VolumeEmissionMask.h", "VolumeTemperatureMapping.h"):
            source = re.sub(r"//[^\n]*", "", read("include/" + name))
            literals = re.findall(r"\b\d+\.\d+(?:[eE][+-]?\d+)?[fF]?", source)
            self.assertTrue(literals)
            self.assertTrue(all(x.endswith(("f", "F")) for x in literals), literals)

    def test_project_registration(self):
        for name in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
            tree = ET.parse(ROOT / "RayTrophiStudio" / name)
            self.assertEqual(sum(n.attrib.get("Include") ==
                r"source\include\VolumeEmissionMask.h" for n in tree.iter()), 1)


if __name__ == "__main__":
    unittest.main()
