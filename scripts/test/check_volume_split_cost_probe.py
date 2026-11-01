"""Collector mocks: independent timestamps, fixed samples and state restoration."""
import copy
import unittest

from rt_probe_volume_split_cost import capture, historical_comparison


TARGET = 'matter:Domain:gas'


class FakeClient:
    def __init__(self, change_context=False):
        self.timing = False
        self.counters = False
        self.region = [.1, .2, .8, .9]
        self.arm = 0
        self.settings_reads = 0
        self.change_context = change_context

    def call(self, method, **params):
        if method == 'sim.control_state':
            return dict(playing=False)
        if method == 'viewport.status':
            return dict(shading='rendered', samples=128)
        if method == 'render.rt_pipeline_status':
            return dict(rt_pipeline_state='ready')
        if method == 'timeline.get_frame':
            return 99
        if method == 'camera.get':
            return dict(position=[1, 2, 3])
        if method == 'volume.get_material':
            return dict(visible=True, max_steps=128)
        if method == 'fluid.inspect_fog_fields':
            return dict(grid=[1, 1, 1], fields=[dict(material_target=TARGET,
                density_hash='a', temperature_hash='b', active_cells=1, density_sum=1.)])
        if method == 'render.get_settings':
            self.settings_reads += 1
            return dict(max_bounces=2 if self.change_context and self.settings_reads > 1 else 10)
        if method == 'render.gpu_timing_stats':
            return dict(enabled=self.timing, pending=0, samples=[
                dict(sample_index=i, width=800, height=600, gpu_ms=self.arm*100+i)
                for i in range(32)] if self.timing else [])
        if method == 'render.volume_stats':
            return dict(enabled=self.counters, region=list(self.region), paths_traced=10,
                        volume_rays=3)
        if method == 'render.set_gpu_timing':
            if params['enabled']:
                self.arm += 1
            self.timing = params['enabled']
            return {}
        if method == 'render.volume_counters':
            self.counters = params['enabled']
            self.region = list(params.get('region', self.region))
            return {}
        if method == 'reset_accumulation':
            return {}
        raise AssertionError(method)


class SplitCostProbe(unittest.TestCase):
    def test_repeats_rearm_timing_and_restore_controls(self):
        client = FakeClient()
        report = capture(client, [TARGET])
        self.assertEqual([r['gpu_median_ms'] for r in report['timing_cases']], [115.5, 215.5])
        self.assertEqual(client.arm, 2)
        self.assertEqual([s['sample_index'] for s in report['timing_cases'][0]['samples']],
                         list(range(8, 24)))
        self.assertFalse(client.timing)
        self.assertFalse(client.counters)
        self.assertEqual(client.region, [.1, .2, .8, .9])
        self.assertTrue(report['context_unchanged'])

    def test_context_change_rejects_but_restores_instrumentation(self):
        client = FakeClient(change_context=True)
        with self.assertRaisesRegex(RuntimeError, 'Camera, frame'):
            capture(client, [TARGET])
        self.assertFalse(client.timing)
        self.assertFalse(client.counters)
        self.assertEqual(client.region, [.1, .2, .8, .9])

    def test_changed_fields_invalidate_historical_comparison(self):
        row = dict(frame=99, camera={}, fields={'hash': 'a'}, render_settings={},
                   materials={TARGET: dict(max_steps=128)})
        altered = copy.deepcopy(row)
        altered['fields']['hash'] = 'b'
        self.assertEqual(historical_comparison(row, altered), ['fields'])


if __name__ == '__main__':
    unittest.main()
