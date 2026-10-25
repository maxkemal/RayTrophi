"""Non-build contracts for opt-in Vulkan RT GPU timing. Run from a terminal."""
from pathlib import Path
import re
import unittest
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(name):
    return (SOURCE / name).read_text(encoding='utf-8')


class RayGpuTimingContracts(unittest.TestCase):
    def test_disabled_guard_precedes_gpu_commands(self):
        text = read('src/Backend/VulkanRayGpuTiming.cpp')
        body = text.split('uint32_t VulkanRayGpuTiming::begin(', 1)[1]
        body = body.split('void VulkanRayGpuTiming::end(', 1)[0]
        self.assertLess(body.index('!m_enabled || m_recorded >= capacity'),
                        body.index('vkCmdWriteTimestamp'))
        self.assertNotIn('collect()', body)
        self.assertNotIn('vkCreateQueryPool', body)
        off = text.split('if (!enabled) {', 1)[1].split('}', 1)[0]
        self.assertIn('m_enabled = false', off)
        self.assertNotIn('vk', off)
        self.assertIn('if (!m_rayGpuTiming || !m_rayGpuTiming->enabled())', text)

    def test_result_refresh_is_nonblocking(self):
        body = read('src/Backend/VulkanRayGpuTiming.cpp').split(
            'void VulkanRayGpuTiming::collect()', 1)[1].split(
            'RayGpuTimingSnapshot VulkanRayGpuTiming::snapshot()', 1)[0]
        for forbidden in ['VK_QUERY_RESULT_WAIT_BIT', 'vkWaitForFences',
                          'vkDeviceWaitIdle', 'vkQueueWaitIdle']:
            self.assertNotIn(forbidden, body)
        self.assertIn('VK_QUERY_RESULT_WITH_AVAILABILITY_BIT', body)
        self.assertIn('result == VK_NOT_READY', body)
        self.assertIn('result != VK_SUCCESS && result != VK_NOT_READY', body)
        self.assertIn('++m_failed', body)

    def test_timestamp_wrap_and_units(self):
        # Vulkan counters may expose fewer than 64 significant bits and wrap.
        for bits in [8, 32, 36, 48, 64]:
            mask = (1 << bits) - 1
            for ticks in [1, 17, 63]:
                begin = mask - 5
                end = (begin + ticks) & mask
                delta = (end - begin) & mask
                self.assertEqual(delta, ticks)
                self.assertAlmostEqual(delta * 2.5 / 1e6, ticks * .0000025)

    def test_all_camera_dispatch_paths_are_bracketed(self):
        text = read('src/Backend/VulkanBackend.cpp')
        calls = list(re.finditer(r'fpCmdTraceRaysKHR\(cmd,\s*&m_sbtRaygenRegion', text))
        self.assertEqual(len(calls), 4)
        for call in calls:
            self.assertIn('beginRayGpuTiming(cmd, w, h)', text[call.start()-180:call.start()])
            self.assertIn('endRayGpuTiming(cmd, timingToken)',
                          text[call.end():call.end()+300])
        self.assertEqual(text.count('beginRayGpuTiming(cmd, w, h)'), 4)
        self.assertLess(text.index('m_rayGpuTiming->destroy()'),
                        text.index('vkDestroyDevice(m_device'))

    def test_ui_script_ipc_share_the_core_and_manual_refresh(self):
        api = read('src/Api/RtRayGpuTiming.cpp')
        ipc = read('src/Api/RtIpcRayGpuTiming.cpp')
        self.assertIn('rtapi::setRayGpuTimingCapture(enabled)', api)
        self.assertIn('rtapi::setRayGpuTimingCapture(enabled)', ipc)
        self.assertIn('rtapi::rayGpuTimingCaptureStats()', ipc)
        self.assertIn('enabled = requireBool(params, "enabled")', ipc)
        self.assertIn('GPU timing stats accepts no parameters', ipc)
        ui = api.split('void drawRayGpuTimingCapture()', 1)[1]
        self.assertEqual(ui.count('rayGpuTimingCaptureStats()'), 1)
        self.assertIn('ImGui::Button("Refresh RT GPU timings")', ui)
        self.assertIn('gpu_mean_ms', api)
        self.assertIn('nlohmann::json(nullptr)', api)

    def test_project_registrations_and_capture_bound(self):
        self.assertIn('capacity = 64', read('include/Backend/VulkanRayGpuTiming.h'))
        paths = ['source\\include\\Backend\\VulkanRayGpuTiming.h',
                 'source\\include\\Api\\RayGpuTimingApi.h',
                 'source\\src\\Backend\\VulkanRayGpuTiming.cpp',
                 'source\\src\\Api\\RtRayGpuTiming.cpp',
                 'source\\src\\Api\\RtIpcRayGpuTiming.cpp',
                 'source\\src\\Api\\RtRayGpuTimingBindings.h']
        for name in ['RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters']:
            tree = ET.parse(ROOT / 'RayTrophiStudio' / name)
            entries = [n.get('Include') for n in tree.iter() if n.get('Include')]
            for path in paths:
                self.assertEqual(entries.count(path), 1, (name, path))


if __name__ == '__main__':
    unittest.main()
