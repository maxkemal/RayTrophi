"""Non-build integration checks; compiled scalar cases are in the companion C++ test."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "RayTrophiStudio/source"


def read(path):
    return (SOURCE / path).read_text(encoding="utf-8")


class VolumeTemperatureContracts(unittest.TestCase):
    def test_render_consumers_use_shared_temperature_contract(self):
        for path in ("shaders/volume_internal_rt.glsl",
                     "shaders/material_preview_volume.frag",
                     "src/Device/ray_color.cuh", "src/Render/Renderer.cpp"):
            self.assertIn("volumeMapTemperature(", read(path), path)
        self.assertIn('../include/VolumeTemperatureMapping.h',
                      read("shaders/volume_material_response.glsl"))
        self.assertEqual(read("src/Device/ray_color.cuh").count(
            "const auto thermal = volumeMapTemperature("), 2)

    def test_live_source_units_and_no_synthetic_temperature(self):
        publisher = read("src/Render/VolumetricRenderer.cpp")
        self.assertEqual(publisher.count("packTemperaturePolicy(gv.density_curve"), 3)
        self.assertIn('live_data->filepath == "[LIVE SIMULATION]"', publisher)
        self.assertEqual(publisher.count("gv.density_curve[10] == 0.0f"), 2)
        rt = read("shaders/volume_internal_rt.glsl")
        self.assertIn("volumeResolveTemperatureUnits(", rt)
        self.assertIn("vol.vdb_temp_address != uint64_t(0)", rt)
        self.assertNotIn("if (temperature <= 0.0) temperature = density", rt)

    def test_ui_python_and_ipc_share_atomic_service(self):
        authoring = read("src/Scene/VolumeMaterialAuthoring.cpp")
        self.assertIn('integer(v, 0, 2, k)', authoring)
        self.assertIn('{"temperature_units", s.emission_response.temperature_units}', authoring)
        ui = read("src/UI/VolumeMaterialUI.cpp")
        self.assertIn('patch["temperature_units"] = temperatureUnits', ui)
        self.assertIn("volume_material::configure(shader, patch, error)", ui)
        binding = read("src/Api/RtVolumeMaterialBindings.cpp")
        self.assertIn("volume_material::configure(next, material_patch, error)", binding)
        self.assertRegex(read("src/Api/RtPythonVolumeMaterialBindings.cpp"),
                         r'volume\.def\(\s*"set_material"')
        self.assertIn('method != "volume.set_material"', binding)

    def test_serialization_and_presets_preserve_source_policy(self):
        header = read("include/VolumeEmissionResponse.h")
        self.assertIn('{"temperature_units", temperature_units}', header)
        self.assertIn('j.value("temperature_units", 0)', header)
        self.assertIn('!j.at("temperature_units").is_number_integer()', header)
        self.assertIn('j.at("temperature_units") > 2', header)
        authoring = read("src/Scene/VolumeMaterialAuthoring.cpp")
        self.assertIn("shader.emission_response.temperature_units = temperatureUnits", authoring)
        project = read("src/Core/ProjectManager.cpp")
        self.assertEqual(project.count('emission_response.fromJson('), 2)

    def test_nee_internal_and_camera_share_emission_evaluator(self):
        march = read("shaders/volume_fog_shading.glsl")
        self.assertNotIn("bool eKelvinUnits", march)
        self.assertIn("volumeEmissionAt(vol, samplePos, emissionDensity, temperature)", march)
        self.assertIn("#define INTERNAL_EMISSION(v, p, d) volumeEmissionAt(v, p, d)",
                      read("shaders/volume_internal_rt.glsl"))

    def test_unit_policy_invalidates_radiance_and_merged_material(self):
        cache = read("shaders/volume_radiance_cache.glsl")
        stamp = cache.split("uint volumeRadianceMaterialStamp", 1)[1].split("\n}", 1)[0]
        for dependency in ("vol.density_curve[10]", "vol.temperature_scale",
                           "vol.max_temperature", "vol._ext_reserved[6]"):
            self.assertIn(dependency, stamp)
        self.assertIn("key += firstIdentity.dump()", read("src/Physics/Fluid/MatterFogPublisher.cpp"))

    def test_packet_offset_and_project_registration(self):
        policy = read("include/VolumeEmissionResponse.h")
        self.assertIn("curve[10] =", policy)
        self.assertIn("volFloat(vi, 664u)", read("shaders/material_preview_volume.frag"))
        self.assertIn('sizeof(VkVolumeInstance) == 688',
                      read("include/Backend/vulkan_volume_types.h"))
        for name in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
            project = ET.parse(ROOT / "RayTrophiStudio" / name)
            self.assertEqual(sum(n.attrib.get("Include") ==
                r"source\include\VolumeTemperatureMapping.h" for n in project.iter()), 1)


if __name__ == "__main__":
    unittest.main()
