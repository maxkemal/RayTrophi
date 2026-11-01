"""RT cost of overlapping live gas streams, one visible at a time vs together (2026-10-28).

For the first two gas targets of the open scene (volume.list_materials, kind
matter:...:gas*), renders the Rendered viewport with only the first, only the
second, then both visible, and prints per configuration: GPU ms per camera RT
dispatch (render.gpu_timing_stats, mean of the captured samples) and the volume
counters per path. Each configuration waits and waits for natural GPU warm-up samples
first: a visibility change rebuilds, and the first frames after it are not the
steady cost. Visibility is restored at the end.

GPU timing uses natural progression with counters off; counters are a separate pass.
Probe, not a gate. Keep the camera still. Before the companion radiance cache
(RT_GAS_SECONDARY_MARCH.md section 16) the 4-source scene read
366 / 86 / 964 ms and 0 / 0 / 3.33 secondary density samples per path.
"""
import argparse
import time

from rt_ipc import RtIpc
from gas_probe_sampling import measure

KEYS = ('volume_rays', 'camera_density_samples', 'secondary_density_samples',
        'shadow_density_samples', 'gas_handoffs', 'paths_bounce_capped',
        'radiance_cache_marches', 'radiance_cache_fallbacks', 'majorant_queries',
        'empty_segments_skipped')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=6)
    args = parser.parse_args()
    client = RtIpc()
    call = client.call
    targets = [t['target'] for t in call('volume.list_materials')['targets']
               if t['target'].startswith('matter:') and ':gas' in t['target']][:2]
    assert len(targets) == 2, 'needs two live gas streams, found %s' % targets
    first, second = targets
    original_visible = {t: call('volume.get_material', target=t)['visible'] for t in targets}
    prior_counters = call('render.volume_stats')
    prior_timing = call('render.gpu_timing_stats')

    def run(a, b):
        call('volume.set_material', target=first, settings={'visible': a})
        call('volume.set_material', target=second, settings={'visible': b})
        timing, counters = measure(client, samples=args.frames)
        samples = [s.get('gpu_ms', 0.0) for s in timing.get('samples', [])]
        paths = max(1, counters.get('paths_traced', 0))
        return (sum(samples) / len(samples) if samples else -1.0,
                {k: counters.get(k, 0) / paths for k in KEYS})

    try:
        rows = {'first': run(True, False), 'second': run(False, True), 'both': run(True, True)}
    finally:
        for target, visible in original_visible.items():
            call('volume.set_material', target=target, settings={'visible': visible})
        call('render.set_gpu_timing', enabled=prior_timing.get('enabled', False))
        call('render.volume_counters', enabled=prior_counters['enabled'],
             region=prior_counters['region'])
        client.close()
    print('timing = natural Vulkan progression, counters disabled; counter pass is separate')
    print('first = %s\nsecond = %s' % (first, second))
    print('%-26s %9s %9s %9s %9s' % ('per path', 'first', 'second', 'sum', 'both'))
    a, b, c = rows['first'][0], rows['second'][0], rows['both'][0]
    print('%-26s %9.0f %9.0f %9.0f %9.0f' % ('GPU ms / RT sample', a, b, a + b, c))
    for key in KEYS:
        a, b, c = rows['first'][1][key], rows['second'][1][key], rows['both'][1][key]
        print('%-26s %9.3f %9.3f %9.3f %9.3f' % (key, a, b, a + b, c))


if __name__ == '__main__':
    main()
