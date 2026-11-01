"""Non-build guards for NanoVDB GPU sampling residency and diagnostics."""
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio/source"


class VolumeGridResidencyContracts(unittest.TestCase):
    def test_observed_rtx3060_memory_order_reproduces_host_heap_selection(self):
        # Read from vkGetPhysicalDeviceMemoryProperties on the user's device.
        # CPU_TO_GPU asks for flags 6, so first-match picks type 3 (system heap),
        # even though type 5 has flags 7 and is CPU-visible VRAM.
        flags = [0, 1, 1, 6, 14, 7]
        first = lambda required: next(i for i, f in enumerate(flags) if f & required == required)
        self.assertEqual(first(6), 3)
        self.assertFalse(flags[first(6)] & 1)
        self.assertEqual(first(1), 1)
        self.assertTrue(flags[first(1)] & 1)

    def test_both_hot_nanovdb_fields_require_device_local_allocation(self):
        source = (SRC / "src/Backend/VulkanBackend_Volumes.cpp").read_text(encoding="utf-8")
        density = source[source.index("const size_t allocSize = gridSize"):
                         source.index("// Upload temperature NanoVDB grid")]
        temperature = source[source.index("const size_t allocTempSize = tempGridSize"):
                             source.index("const size_t allocTempSize = tempGridSize")+2300]
        self.assertIn("ci.location = VulkanRT::MemoryLocation::GPU_ONLY", density)
        self.assertIn("ci2.location = VulkanRT::MemoryLocation::GPU_ONLY", temperature)
        self.assertIn("m_device->uploadBuffer(it->second, hostGrid, gridSize)", density)
        self.assertIn("hostTempGrid, tempGridSize", temperature)
        self.assertIn("synchronizeGridUpload()", density)
        self.assertIn("synchronizeGridUpload()", temperature)

    def test_residency_reports_actual_tracked_heap_and_unknown_explicitly(self):
        source = (SRC / "src/Backend/VulkanVolumeResidency.cpp").read_text(encoding="utf-8")
        self.assertIn("m_vramTracked.find(buffer.memory)", source)
        self.assertIn("m_vramTrackMutex", source)
        self.assertIn("found->second.deviceLocal ? 1 : 0", source)
        self.assertEqual(source.count("return -1;"), 2)
        for filename in ("RtApi.cpp", "RtIpc.cpp", "RtPython.cpp"):
            binding = (SRC / "src/Api" / filename).read_text(encoding="utf-8")
            for field in ("density_device_local", "temperature_device_local"):
                self.assertIn(field, binding)
        backend = (SRC / "src/Backend/VulkanBackend_Volumes.cpp").read_text(encoding="utf-8")
        self.assertIn("entry.second.deviceAddress == address", backend)
        self.assertIn("residency(volume.vdb_grid_address)", backend)
        self.assertIn("residency(volume.vdb_temp_address)", backend)

    def test_new_module_is_registered_once(self):
        for name in ("RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"):
            tree = ET.parse(ROOT / "RayTrophiStudio" / name)
            includes = [n.attrib.get("Include", "") for n in tree.iter()]
            self.assertEqual(includes.count(r"source\src\Backend\VulkanVolumeResidency.cpp"), 1)


if __name__ == "__main__":
    unittest.main()
