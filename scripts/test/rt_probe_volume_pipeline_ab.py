"""External IPC capture for a volume-free RT pipeline A/B, without app launch.

User starts the same binary/project with RT_VK_AB_NO_VOLUME unset/0 or 1.
This client verifies installed stages, samples fixed GPU sample indices with
volume counters off, and restores instrumentation. It never edits the scene.
"""
import argparse
import json
from pathlib import Path
import statistics
import time

from rt_ipc import RtIpc
from gas_probe_snapshot import RAY_CAMERA_KEYS, signature


VOLUME_STAGES = {'volume_closesthit', 'volume_intersection'}


def context(client):
    camera = client.call('camera.get')
    return {'project': client.call('project.path'),
            'frame': client.call('timeline.get_frame'),
            'camera': {key: camera.get(key) for key in RAY_CAMERA_KEYS},
            'settings': client.call('render.get_settings')}


def capture(client, variant, warmup=8, samples=32, timeout=120):
    pipeline = client.call('render.rt_pipeline_status')
    if pipeline.get('backend') != 'vulkan' or pipeline.get('rt_pipeline_state') != 'ready':
        raise RuntimeError('Wait for the Vulkan RT pipeline to be ready')
    stages = {x['stage'] for x in pipeline.get('stage_seconds', [])}
    installed = stages & VOLUME_STAGES
    expected = VOLUME_STAGES if variant == 'with-volume' else set()
    if installed != expected:
        raise RuntimeError('Installed stages disagree with requested variant: %s' % sorted(installed))
    if client.call('sim.control_state')['playing']:
        raise RuntimeError('Pause the timeline before capture')
    if client.call('viewport.status').get('shading') != 'rendered':
        raise RuntimeError('Use the Rendered viewport')
    if client.call('volume.list_materials')['targets']:
        raise RuntimeError('Use a project without authored volume/SDF/gas fields')
    before = context(client)
    prior_timing = client.call('render.gpu_timing_stats')
    prior_counters = client.call('render.volume_stats')
    try:
        client.call('render.volume_counters', enabled=False)
        client.call('render.set_gpu_timing', enabled=True)
        client.call('reset_accumulation')
        deadline = time.monotonic() + timeout
        requested = set(range(warmup, warmup+samples))
        while True:
            if client.call('timeline.get_frame') != before['frame']:
                raise RuntimeError('Timeline changed during capture')
            stats = client.call('render.gpu_timing_stats')
            rows = {x['sample_index']: x for x in stats.get('samples', [])}
            if requested <= rows.keys():
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Capture timed out; allow at least %d viewport samples' %
                                   (warmup+samples))
            time.sleep(.1)
        selected = [rows[i] for i in sorted(requested)]
        dimensions = {(x['width'], x['height']) for x in selected}
        if len(dimensions) != 1:
            raise RuntimeError('Viewport resolution changed during capture')
        if context(client) != before:
            raise RuntimeError('Camera, project, frame or render settings changed during capture')
        values = [x['gpu_ms'] for x in selected]
        return {'variant': variant, 'context': before,
                'context_signature': signature(before),
                'identity_limit': 'Context only; user must keep scene contents and binary/shaders identical',
                'dimensions': list(next(iter(dimensions))),
                'common_stages': sorted(stages-VOLUME_STAGES),
                'pipeline': pipeline, 'sample_range': [warmup, warmup+samples-1],
                'gpu_warm_median_ms': statistics.median(values),
                'gpu_warm_mean_ms': statistics.mean(values),
                'gpu_captured_peak_ms': max(x['gpu_ms'] for x in rows.values()),
                'samples': selected}
    finally:
        try:
            client.call('render.set_gpu_timing', enabled=prior_timing.get('enabled', False))
        finally:
            client.call('render.volume_counters', enabled=prior_counters['enabled'],
                        region=prior_counters['region'])


def compare(a, b):
    for key in ('context_signature', 'dimensions', 'sample_range', 'common_stages'):
        if a[key] != b[key]:
            raise RuntimeError('Reports are not comparable: %s differs' % key)
    if a['variant'] == b['variant']:
        return {'repeat_median_delta_ms': b['gpu_warm_median_ms']-a['gpu_warm_median_ms']}
    by_variant = {row['variant']: row for row in (a, b)}
    full = by_variant['with-volume']['gpu_warm_median_ms']
    omitted = by_variant['without-volume']['gpu_warm_median_ms']
    return {'with_volume_ms': full, 'without_volume_ms': omitted,
            'reduction_percent': 100*(full-omitted)/full if full > 0 else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['with-volume', 'without-volume'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    client = RtIpc()
    try:
        report = capture(client, args.variant)
    finally:
        client.close()
    if args.compare:
        report['comparison'] = compare(json.loads(args.compare.read_text(encoding='utf-8')), report)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({key: report[key] for key in
        ('variant', 'gpu_warm_median_ms', 'gpu_captured_peak_ms')}, indent=2))
    if 'comparison' in report:
        print(json.dumps(report['comparison'], indent=2))


if __name__ == '__main__':
    main()
