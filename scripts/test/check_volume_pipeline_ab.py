"""Non-build tests; never launch Studio or compile shaders/C++."""
from pathlib import Path
import unittest

from rt_probe_volume_pipeline_ab import capture, compare, VOLUME_STAGES

ROOT = Path(__file__).resolve().parents[2]


class FakeClient:
    def __init__(self, variant='with-volume'):
        self.stages = ['raygen', 'miss', 'closesthit', 'cloud_callable']
        if variant == 'with-volume':
            self.stages += sorted(VOLUME_STAGES)
        self.timing = False
        self.counters = True

    def call(self, method, **params):
        if method == 'render.rt_pipeline_status':
            return {'backend': 'vulkan', 'rt_pipeline_state': 'ready',
                    'stage_seconds': [{'stage': s} for s in self.stages]}
        if method == 'sim.control_state': return {'playing': False}
        if method == 'viewport.status': return {'shading': 'rendered'}
        if method == 'volume.list_materials': return {'targets': []}
        if method == 'camera.get': return {'position': [0, 0, 4], 'fov': 45}
        if method == 'project.path': return 'same_scene.rtp'
        if method == 'timeline.get_frame': return 0
        if method == 'render.get_settings': return {'max_samples': 128}
        if method == 'render.volume_stats': return {'enabled': self.counters, 'region': [0, 0, 1, 1]}
        if method == 'render.set_gpu_timing': self.timing = params['enabled']; return {}
        if method == 'render.volume_counters': self.counters = params['enabled']; return {}
        if method == 'reset_accumulation': return {}
        if method == 'render.gpu_timing_stats':
            return {'enabled': self.timing, 'samples': [
                {'sample_index': i, 'gpu_ms': 1000 if i < 8 else 10,
                 'width': 800, 'height': 600} for i in range(40)]}
        raise AssertionError(method)


class VolumePipelineAB(unittest.TestCase):
    def test_fixed_warm_samples_and_restoration(self):
        client = FakeClient()
        row = capture(client, 'with-volume')
        self.assertEqual(row['gpu_warm_median_ms'], 10)
        self.assertEqual(row['gpu_captured_peak_ms'], 1000)
        self.assertEqual([r['sample_index'] for r in row['samples']], list(range(8, 40)))
        self.assertFalse(client.timing)
        self.assertTrue(client.counters)

    def test_wrong_installed_variant_rejects(self):
        with self.assertRaisesRegex(RuntimeError, 'Installed stages'):
            capture(FakeClient(), 'without-volume')

    def test_capture_failure_restores_instrumentation(self):
        client = FakeClient()
        original_call = client.call
        def fail(method, **params):
            if method == 'reset_accumulation': raise RuntimeError('mock capture failure')
            return original_call(method, **params)
        client.call = fail
        with self.assertRaisesRegex(RuntimeError, 'mock capture failure'):
            capture(client, 'with-volume')
        self.assertFalse(client.timing)
        self.assertTrue(client.counters)

    def test_volume_scene_rejected_before_measurement(self):
        client = FakeClient()
        original_call = client.call
        def with_volume(method, **params):
            if method == 'volume.list_materials': return {'targets': [{'target': 'vdb:1'}]}
            return original_call(method, **params)
        client.call = with_volume
        with self.assertRaisesRegex(RuntimeError, 'without authored volume'):
            capture(client, 'with-volume')
        self.assertFalse(client.timing)

    def test_comparison_rejects_changed_context_or_stages(self):
        a = capture(FakeClient(), 'with-volume')
        b = capture(FakeClient('without-volume'), 'without-volume')
        b['gpu_warm_median_ms'] = 6
        self.assertEqual(compare(a, b)['reduction_percent'], 40)
        for key in ('context_signature', 'dimensions', 'sample_range', 'common_stages'):
            changed = dict(b, **{key: 'changed'})
            with self.assertRaisesRegex(RuntimeError, key): compare(a, changed)

    def test_policy_precedes_hash_and_guard_precedes_trace(self):
        module = (ROOT/'RayTrophiStudio/source/src/Backend/VulkanRTPipelineBuild.cpp').read_text(encoding='utf-8-sig')
        request = module.split('VulkanDevice::requestRTPipelineBuild(', 1)[1]
        self.assertLess(request.index('shaders.volumeClosestHit.clear();'),
                        request.index('hashShaderSet(shaders)'))
        self.assertLess(request.index('shaders.volumeIntersection.clear();'),
                        request.index('hashShaderSet(shaders)'))
        self.assertIn('mixVec(s.volumeClosestHit); mixVec(s.volumeIntersection);', module)
        backend = (ROOT/'RayTrophiStudio/source/src/Backend/VulkanBackend.cpp').read_text(encoding='utf-8-sig')
        render = backend.split('void VulkanBackendAdapter::renderProgressiveImpl(', 1)[1]
        self.assertLess(render.index('volumePipelineProbeAllowsScene('),
                        render.index('m_device->traceRaysAndReadback('))
        policy = (ROOT/'RayTrophiStudio/source/include/Backend/VulkanVolumePipelineProbe.h').read_text(encoding='utf-8-sig')
        self.assertIn('static const bool enabled', policy)
        self.assertIn('std::strcmp(value, "1") == 0', policy)
        self.assertIn('publishedVolumes == 0 && tlasSlots == 0', policy)


if __name__ == '__main__':
    unittest.main()
