"""Non-build wiring/ABI checks. These do not validate compiled CPU/GPU execution."""

from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8")


class VolumeMaterialContracts(unittest.TestCase):
    def test_response_raw_offset_and_stride(self):
        text = read("include/Backend/vulkan_volume_types.h")
        body = text.split("VkVolumeInstance {", 1)[1].split("};", 1)[0]
        body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
        offset = 0
        offsets = {}
        for statement in body.split(";"):
            if not statement.strip():
                continue
            match = re.fullmatch(r"\s*(float|int|uint64_t)\s+(\w+)(?:\[(\d+)\])?\s*",
                                 statement)
            self.assertIsNotNone(match, statement)
            kind, name, count = match.groups()
            size = 8 if kind == "uint64_t" else 4
            offset = (offset + size - 1) // size * size
            offsets[name] = offset
            offset += size * int(count or 1)
        self.assertEqual(offsets["_retired_cloud"], 444)
        self.assertEqual((offset + 15) // 16 * 16, 672)
        preview = read("shaders/material_preview_volume.frag")
        self.assertIn("volFloat(vi, 444u + uint(k) * 4u)", preview)

    def test_density_curve_consumers_and_majorants(self):
        layout = read("include/Backend/vulkan_volume_types.h")
        self.assertIn("float density_curve[12]", layout)
        for file in ("volume_closesthit.rchit", "closesthit.rchit",
                     "raygen.rgen", "volume_intersection.rint"):
            self.assertIn("float density_curve[12]", read("shaders/" + file))
        rt = read("shaders/volume_closesthit.rchit")
        self.assertEqual(rt.count("volumeDensityCurve("), 5)
        self.assertIn("volumeDensityCurve(max((leafMax", rt)
        self.assertIn("volumeDensityCurve(max((blockMax", rt)
        self.assertIn("volumeDensityCurve(max((density",
                      read("shaders/closesthit.rchit"))
        for file in ("material_preview_volume_fields.glsl",
                     "surface_sdf_shadow_query.glsl", "material_preview_sdf_surface.frag"):
            self.assertIn("STRIDE_WORDS = 168u", read("shaders/" + file))
        preview = read("shaders/material_preview_volume_fields.glsl")
        self.assertIn("624u + uint(i)*4u", preview)
        self.assertIn("previewDensityCurve(vi, max((blockMax", preview)
        self.assertEqual(read("src/Render/VolumetricRenderer.cpp").count(
            "density.curve.pack(gv.density_curve)"), 3)
        self.assertEqual(read("src/Backend/VulkanBackend_Volumes.cpp").count(
            "src.density_curve + 12, dst.density_curve"), 2)
        self.assertIn('k == "density_curve"', read("src/Scene/VolumeMaterialAuthoring.cpp"))

    def test_common_emission_shader_consumers(self):
        rt = read("shaders/volume_closesthit.rchit")
        preview = read("shaders/material_preview_volume.frag")
        for text in (rt, preview):
            self.assertIn('#include "volume_material_response.glsl"', text)
        self.assertIn("volumeEmissionAt(vol, samplePos, density, temperature)", rt)
        self.assertIn("volumeEmissionSource(", read("shaders/volume_internal_rt.glsl"))
        self.assertEqual(preview.count("volumeEmissionSource("), 2)
        helper = read("shaders/volume_material_response.glsl")
        self.assertIn("response[0] < 0.5", helper)
        self.assertIn("density <= 0.0", helper)

    def test_internal_light_transport_and_controls(self):
        rt = read("shaders/volume_internal_rt.glsl")
        preview = read("shaders/volume_internal_preview.glsl")
        for text in (rt, preview):
            self.assertIn('#include "volume_internal_integral.glsl"', text)
        self.assertIn("rwSolidDistance(", rt)
        self.assertIn("sampleHG(", rt)
        self.assertIn("phaseDual(", preview)
        integral = read("shaders/volume_internal_integral.glsl")
        self.assertIn("INTERNAL_EMISSION(volume, p, density)", integral)
        self.assertIn("throughput *= segmentT", integral)
        self.assertNotIn("emissive_list_address", integral)
        bindings = read("src/Scene/VolumeMaterialAuthoring.cpp")
        self.assertIn('k == "internal_light_steps"', bindings)
        self.assertIn('k == "internal_light_stride"', bindings)
        layout = read("include/Backend/vulkan_volume_types.h")
        body = layout.split("VkVolumeInstance {", 1)[1].split("};", 1)[0]
        body = re.sub(r"//[^\n]*|/\*.*?\*/", "", body, flags=re.S)
        offset = 0
        offsets = {}
        for kind, name, count in re.findall(
                r"\b(float|int|uint64_t)\s+(\w+)(?:\[(\d+)\])?\s*;", body):
            size = 8 if kind == "uint64_t" else 4
            offset = (offset + size - 1) // size * size
            offsets[name] = offset
            offset += size * int(count or 1)
        self.assertEqual(offsets["_ramp_pad"], 280)

    def test_scalar_midpoint_forward_reverse_and_limiter(self):
        forward = read("shaders/sim_grid_advect_scalar.glsl")
        correction = read("shaders/sim_grid_maccormack_scalar.glsl")
        self.assertIn("gridScalarTrace(vec3(lx, ly, lz), pc.dt * inv_h)", forward)
        self.assertIn("gridScalarTrace(vec3(lx, ly, lz), -s)", correction)
        self.assertIn("gridScalarTrace(vec3(lx, ly, lz), s)", correction)
        self.assertIn("clamp(phi_fwd[id]", correction)
        cpu = read("src/Physics/GridFluidSolver.cpp")
        self.assertIn("traceScalarMidpoint(", cpu)
        self.assertIn("const Vec3 ahead = trace(p, -dt)", cpu)

    def test_no_step_dependent_sky_suppression(self):
        rt = read("shaders/volume_closesthit.rchit")
        self.assertNotIn("scatter_keep", rt)
        self.assertNotIn("higherOrderProbability", rt)
        self.assertIn("volumeHigherOrderSkyVisibility(", rt)
        self.assertIn("volumeHigherOrderSkyVisibility(",
                      read("shaders/material_preview_volume.frag"))

    def test_material_transport_wiring(self):
        renderer = read("src/Render/VolumetricRenderer.cpp")
        self.assertEqual(renderer.count("emission_response.pack(gv.emission_response)"), 3)
        self.assertIn("dst._retired_cloud[k] = src.emission_response[k]",
                      read("src/Backend/VulkanBackend_Volumes.cpp"))

    def test_ui_python_ipc_use_core(self):
        ui = read("src/UI/VolumeMaterialUI.cpp")
        binding = read("src/Api/RtVolumeMaterialBindings.cpp")
        self.assertIn("volume_material::configure(", ui)
        self.assertIn("volume_material::configure(", binding)
        self.assertIn("registerVolumeMaterialPython(module)", read("src/Api/RtPython.cpp"))
        self.assertIn("dispatchVolumeMaterialIpc(", read("src/Api/RtIpc.cpp"))
        for method in ("list_materials", "get_material", "set_material"):
            self.assertIn('"volume.' + method + '"', binding)

    def test_vs_project_sources_registered(self):
        project = ET.parse(ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj")
        ns = {"m": "http://schemas.microsoft.com/developer/msbuild/2003"}
        sources = {node.attrib.get("Include") for node in project.findall(".//m:ClCompile", ns)}
        for path in (r"source\src\Api\RtVolumeMaterialBindings.cpp",
                     r"source\src\Scene\VolumeMaterialAuthoring.cpp",
                     r"source\src\UI\VolumeMaterialUI.cpp"):
            self.assertIn(path, sources)

    def test_manual_project_serializers_keep_response(self):
        project = read("src/Core/ProjectManager.cpp")
        self.assertEqual(project.count('s["emission_response"] ='), 2)
        self.assertEqual(project.count('emission_response.fromJson('), 2)
        shader = read("include/VolumeShader.h")
        self.assertIn('j["emission_response"] = emission_response.toJson()', shader)


if __name__ == "__main__":
    unittest.main()
