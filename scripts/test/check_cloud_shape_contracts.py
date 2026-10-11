"""Non-build checks for shared cloud shape and reference cirrus integration.

These checks catch missing consumer wiring and accidental texture-budget changes.
They do not compile GLSL or establish visual quality or GPU performance.
"""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
SHADERS = ROOT / "RayTrophiStudio/source/shaders"


def source(name):
    text = (SHADERS / name).read_text(encoding="utf-8")
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", text, flags=re.S)


def function_body(text, name):
    match = re.search(r"\b" + re.escape(name) + r"\s*\([^;{}]*\)\s*\{", text)
    if match is None:
        raise AssertionError(f"Missing function definition: {name}")
    start = match.end()
    depth = 1
    for index in range(start, len(text)):
        depth += (text[index] == "{") - (text[index] == "}")
        if depth == 0:
            return text[start:index]
    raise AssertionError(f"Unclosed function: {name}")


class CloudShapeContracts(unittest.TestCase):
    def test_shape_lookup_budget(self):
        body = function_body(source("cloud_common.glsl"), "cloudLayerDensity")
        for texture, count in (("cloudBaseNoise", 2), ("cloudDetailNoise", 1),
                               ("cloudCurlNoise", 1)):
            with self.subTest(texture=texture):
                reads = re.findall(r"textureLod\s*\(\s*" + texture + r"\b", body)
                self.assertEqual(len(reads), count)

    def test_cirrus_lookup_budget(self):
        body = function_body(source("cloud_rt.glsl"), "cloudCirrus")
        self.assertEqual(len(re.findall(r"\btextureLod\s*\(", body)), 2)
        feathers = source("cloud_cirrus_shape.glsl")
        self.assertNotIn("textureLod", feathers)
        self.assertIn("cloudCirrusWisps(n, n2)", body)

    def test_both_transport_branches_include_cirrus(self):
        text = source("cloud_rt.glsl")
        production = function_body(text, "cloudMarchLimited")
        render = function_body(text, "cloudRenderLimited")
        # Production exits through the early branch. The remaining reference
        # continuation must composite the same authored cirrus layer too.
        branch = re.search(r"if\s*\(\s*!cloudRtReference\(\)\s*\)\s*\{", render)
        self.assertIsNotNone(branch)
        start = branch.end()
        depth = 1
        index = start
        while depth:
            depth += (render[index] == "{") - (render[index] == "}")
            index += 1
        self.assertIn("cloudMarchLimited(", render[start:index])
        self.assertIn("return;", render[start:index])
        self.assertIn("cloudCirrus(", production)
        self.assertIn("cloudCirrus(", render[index:])

    def test_shared_gpu_query_and_renderer_field(self):
        self.assertIn('#include "cloud_common.glsl"', source("cloud_sample.comp"))
        self.assertIn('#include "cloud_common.glsl"', source("cloud_rt.glsl"))
        self.assertIn('#include "cloud_rt.glsl"', source("cloud_raster.comp"))

    def test_material_consumed_by_both_transport_paths(self):
        text = source("cloud_rt.glsl")
        march = function_body(text, "cloudMarchLimited")
        reference = function_body(text, "cloudRenderLimited")
        cirrus = function_body(text, "cloudCirrus")
        self.assertIn("L.e.rgb", march)
        self.assertIn("L.f.rgb", march)
        self.assertIn("cloudParams.layers[li].e.rgb", reference)
        self.assertIn("cloudParams.layers[li].f.rgb", reference)
        self.assertIn("cloudParams.scatterAlbedo.rgb", cirrus)
        self.assertIn("cloudParams.emissionSource.rgb", cirrus)
        self.assertIn("cloudParams.layers[li].e.w",
                      function_body(text, "cloudSamplePhase"))
        self.assertIn("cloudParams.scatterAlbedo.w",
                      function_body(source("cloud_common.glsl"), "cloudPhase"))

    def test_vdb_edits_cloud_authority(self):
        ui = (ROOT / "RayTrophiStudio/source/src/UI/CloudMaterialUI.cpp").read_text(encoding="utf-8")
        vdb = (ROOT / "RayTrophiStudio/source/src/UI/scene_ui_vdb.cpp").read_text(encoding="utf-8")
        self.assertIn("rtapi::setCloudsJson(", ui)
        self.assertIn("cloud_ui::drawWorldCloudProperties(ctx);", vdb)
        python = (ROOT / "RayTrophiStudio/source/src/Api/RtPython.cpp").read_text(encoding="utf-8")
        ipc = (ROOT / "RayTrophiStudio/source/src/Api/RtIpc.cpp").read_text(encoding="utf-8")
        self.assertIn("setCloudsJson(", python)
        self.assertIn("setCloudsJson(", ipc)

    def test_rotated_tracking_matches_weather(self):
        common = source("cloud_common.glsl")
        rt = source("cloud_rt.glsl")
        self.assertIn("cloudWeatherUV(L, p)", function_body(common, "cloudWeatherAt"))
        self.assertIn("cloudWeatherUV(L, p)", function_body(rt, "cloudLayerMajorant"))
        self.assertIn("cloudRotateXZ(", function_body(common, "cloudWeatherUV"))

    def test_cpp_shader_packet_layout(self):
        cpp = (ROOT / "RayTrophiStudio/source/include/Backend/CloudParams.h").read_text(
            encoding="utf-8")
        glsl = source("cloud_common.glsl")
        layer_cpp = cpp.split("struct CloudLayerGPU {", 1)[1].split("};", 1)[0]
        layer_glsl = glsl.split("struct CloudLayerG {", 1)[1].split("};", 1)[0]
        self.assertEqual(re.findall(r"float\s+(\w+)\[4\]", layer_cpp),
                         re.findall(r"vec4\s+(\w+)\s*;", layer_glsl))
        packet_cpp = cpp.split("struct CloudParamsGPU {", 1)[1].split("};", 1)[0]
        packet_glsl = glsl.split("buffer CloudParamsBlock {", 1)[1].split("}", 1)[0]
        fields = re.findall(r"float\s+(\w+)\[4\]", packet_cpp)
        self.assertEqual(fields, re.findall(r"vec4\s+(\w+)\s*;", packet_glsl))
        self.assertEqual(3 * 112 + len(fields) * 16 + 4 * 64, 736)

    def test_production_empty_skip_is_bounded(self):
        march = function_body(source("cloud_rt.glsl"), "cloudMarchLimited")
        self.assertIn("min(cloudSegmentLen(p, d), tEnd - t)", march)
        self.assertNotIn("max(cloudSegmentLen(p, d) * 0.5, dt)", march)
        self.assertIn("budget > reserve", march)
        self.assertIn("completionStep", march)

    def test_finite_surface_segments(self):
        text = source("cloud_rt.glsl")
        march = function_body(text, "cloudMarchLimited")
        reference = function_body(text, "cloudRenderLimited")
        self.assertIn("cloudIntervals(o, d, tLimit, iv)", march)
        self.assertIn("cloudTransmittance(o, d, tLimit, rng)", reference)
        self.assertIn("b == 0 ? tLimit : CLOUD_RT_MAX_DIST", reference)
        self.assertIn("t > tLimit", function_body(text, "cloudCirrus"))
        self.assertIn("min(kPrecipMaxDist, tLimit)",
                      function_body(text, "cloudPrecipMarch"))
        for shader in ("closesthit.rchit", "hair_closesthit.rchit", "volume_closesthit.rchit"):
            body = function_body(source(shader), "main")
            with self.subTest(shader=shader):
                self.assertIn("cloudApply", body)
                self.assertIn("PL_CAMERA_SEGMENT | PL_NO_SKY_RADIANCE", body)
                self.assertIn("payload.primaryMeta |= pathFlags", body)

    def test_raster_surface_depth(self):
        scene = source("cloud_scene.comp")
        self.assertIn("depth >= 1.0", scene)
        self.assertIn("dot(direction, worldData.camFwd.xyz)", scene)
        self.assertIn("cloudMarchLimited(", scene)
        viewport = (ROOT / "RayTrophiStudio/source/src/Backend/VulkanViewportBackend.cpp").read_text(
            encoding="utf-8")
        post = function_body(viewport, "VulkanViewportBackend::recordRasterPostPass")
        self.assertLess(post.index("recordCloudScenePass("), post.index("vkCmdDispatch("))

    def test_surface_descriptor_lifetime(self):
        cpp = (ROOT / "RayTrophiStudio/source/src/Backend/VulkanDeviceClouds.cpp").read_text(
            encoding="utf-8")
        scene = function_body(cpp, "VulkanDevice::recordCloudScene")
        self.assertIn("&m_cloudSceneSet", scene)
        self.assertNotIn("&m_cloudRasterSets", scene)
        self.assertIn("hdr != m_cloudSceneHdr", scene)
        self.assertIn("VK_ACCESS_SHADER_WRITE_BIT", scene)
        viewport = (ROOT / "RayTrophiStudio/source/src/Backend/VulkanViewportBackend.cpp").read_text(
            encoding="utf-8")
        self.assertIn("invalidateCloudSceneDescriptors();", viewport)

    def test_straight_pass_keeps_bounce_budget(self):
        helper = source("cloud_hit_segment.glsl")
        self.assertIn("plIsTransparentPass(payload, direction)", helper)
        self.assertIn("if (straightPass) payload.bounceType = BOUNCE_MEDIUM_PASS", helper)
        self.assertIn("plIsTransparentPass(payload, rayDir)", source("raygen.rgen"))

    def test_linked_density_keeps_tracking_bound(self):
        common = source("cloud_common.glsl")
        blend = function_body(common, "cloudStormLayerAt")
        density = function_body(common, "cloudLayerDensity")
        majorant = function_body(source("cloud_rt.glsl"), "cloudLayerMajorant")
        self.assertIn("cloudStormLayerAt(li, alt)", density)
        for field in ("a.z", "b.x", "b.z", "c.xy"):
            self.assertIn(f"mix(lower.{field}, upper.{field}, t)", blend)
        self.assertIn("max(L.a.z, member.a.z)", majorant)
        self.assertIn("max(L.b.x, member.b.x)", majorant)
        self.assertIn("member.d.w != L.d.w", blend)
        self.assertIn("member.d.w != L.d.w", majorant)

    def test_storm_support_fits_neighbourhood_and_top(self):
        # Worst possible [-.26,+.26] billow displacement, smallest ellipse
        # factor, largest outflow radius and centre offset. If support reaches
        # an omitted cell, moving across a lattice boundary creates a cut.
        self.assertLess(0.72 * 1.26 / 0.85 + 0.15, 1.25)
        for index in range(3):
            self.assertLess(0.14 + index * 0.25 + (0.22 + index * 0.025) * 1.26, 1.0)
        self.assertLess(0.77 + 0.16 * 1.26, 1.0)
        envelope = function_body(source("cloud_common.glsl"), "cloudStormEnvelope")
        self.assertIn("lobe < 4", envelope)
        self.assertNotIn("float stem", envelope)
        self.assertIn("0.22 + k * 0.025", envelope)
        self.assertIn("mix(0.58, 0.72, r.y)", envelope)
        self.assertIn("billow.x - 0.5", envelope)


if __name__ == "__main__":
    unittest.main()
