"""External fixed-index timing of the current paused scene; no visibility edits."""
import argparse
import json
from pathlib import Path
import statistics
import time

from gas_probe_snapshot import RAY_CAMERA_KEYS
from rt_ipc import RtIpc


def context(client, targets):
    camera = client.call('camera.get')
    materials = {target: client.call('volume.get_material', target=target) for target in targets}
    fields = {}
    for domain in sorted({t[len('matter:'):].split(':gas', 1)[0] for t in targets}):
        info = client.call('fluid.inspect_fog_fields', domain=domain)
        fields[domain] = dict(grid=info['grid'], fields=[
            {key: row[key] for key in ('material_target', 'density_hash', 'temperature_hash',
                                     'active_cells', 'density_sum')}
            for row in info['fields'] if row['material_target'] in targets])
    return dict(frame=client.call('timeline.get_frame'),
                camera={key: camera.get(key) for key in RAY_CAMERA_KEYS},
                fields=fields, materials=materials,
                render_settings=client.call('render.get_settings'))


def historical_comparison(old, current):
    differences = []
    for key in ('frame', 'camera', 'fields', 'render_settings'):
        if old[key] != current[key]:
            differences.append(key)
    for target, material in old['materials'].items():
        for key, value in material.items():
            if current['materials'].get(target, {}).get(key) != value:
                differences.append('materials:' + target + ':' + key)
    return differences


def capture(client, targets, timeout=55):
    if client.call('sim.control_state')['playing']:
        raise RuntimeError('Timeline must already be paused')
    if client.call('viewport.status').get('shading') != 'rendered':
        raise RuntimeError('Rendered viewport required')
    pipeline = client.call('render.rt_pipeline_status')
    if pipeline.get('rt_pipeline_state') != 'ready':
        raise RuntimeError('RT pipeline is not ready')
    before = context(client, targets)
    prior_timing = client.call('render.gpu_timing_stats')
    prior_counters = client.call('render.volume_stats')
    report = dict(pipeline=pipeline, context=before, timing_cases=[], complete=False)

    def wait_for(read, ready):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if client.call('timeline.get_frame') != before['frame']:
                raise RuntimeError('Timeline changed')
            value = read()
            if ready(value):
                return value
            time.sleep(.05)
        raise RuntimeError('Natural accumulation timed out')

    try:
        client.call('render.volume_counters', enabled=False)
        requested = set(range(8, 24))
        for repeat in range(2):
            # Accumulation reset does not clear the timestamp capture. Drain
            # pending queries and explicitly rearm, or repeat 2 reuses run 1.
            client.call('render.set_gpu_timing', enabled=False)
            wait_for(lambda: client.call('render.gpu_timing_stats'),
                     lambda t: t.get('pending', 0) == 0)
            client.call('render.set_gpu_timing', enabled=True)
            client.call('reset_accumulation')
            timing = wait_for(lambda: client.call('render.gpu_timing_stats'),
                              lambda t: requested <= {r['sample_index'] for r in t.get('samples', [])})
            rows = {r['sample_index']: r for r in timing['samples']}
            samples = [rows[i] for i in sorted(requested)]
            dimensions = {(r['width'], r['height']) for r in samples}
            if len(dimensions) != 1:
                raise RuntimeError('Viewport dimensions changed')
            row = dict(repeat=repeat + 1, dimensions=list(next(iter(dimensions))),
                       gpu_median_ms=statistics.median(r['gpu_ms'] for r in samples),
                       gpu_mean_ms=statistics.mean(r['gpu_ms'] for r in samples), samples=samples)
            report['timing_cases'].append(row)
            print(json.dumps({k: v for k, v in row.items() if k != 'samples'}), flush=True)
        client.call('render.set_gpu_timing', enabled=False)
        client.call('render.volume_counters', enabled=True, region=[0, 0, 1, 1])
        client.call('reset_accumulation')
        wait_for(lambda: client.call('viewport.status'), lambda s: s.get('samples', 0) >= 8)
        counters = client.call('render.volume_stats')
        report['counters'] = counters
        paths = counters.get('paths_traced', 0)
        if paths <= 0:
            raise RuntimeError('Counter pass returned no paths')
        report['per_path'] = {k: v / paths for k, v in counters.items()
                              if isinstance(v, (int, float)) and not isinstance(v, bool)}
        after = context(client, targets)
        report['context_unchanged'] = after == before
        if after != before:
            raise RuntimeError('Camera, frame, gas fields, materials or render settings changed')
        report['complete'] = True
        return report
    finally:
        try:
            client.call('render.set_gpu_timing', enabled=prior_timing.get('enabled', False))
        finally:
            client.call('render.volume_counters', enabled=prior_counters['enabled'],
                        region=prior_counters['region'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old = json.loads(args.baseline.read_text(encoding='utf-8'))
    reference_context = old.get('baseline', old.get('context'))
    if reference_context is None:
        raise RuntimeError('Baseline has no measurement context')
    targets = old.get('targets', list(reference_context['materials']))
    client = RtIpc()
    try:
        report = capture(client, targets)
    finally:
        client.close()
    report['targets'] = targets
    report['historical_context_differences'] = historical_comparison(reference_context, report['context'])
    report['historical_visibility'] = old.get('original_visibility')
    report['comparison_limit'] = 'Historical run, not same-binary A/B; geometry/SDF/light identity not established.'
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('Report:', args.output)
    print('Historical context differences:', report['historical_context_differences'])


if __name__ == '__main__':
    main()
