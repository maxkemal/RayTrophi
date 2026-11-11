"""External DEM cost measurement in an empty, paused scene; cleans up its domain.

Uses the recorded user-scene recipe. No build, shader compilation or app launch.
Reports host stage timings, not GPU timestamps. Intact and all-broken states
have different dynamics: their timing ratio is not an isolated kernel speedup.
"""
import argparse
import copy
import json
import statistics
import time
from pathlib import Path
from rt_ipc import RtIpc

DOMAIN = 'T_CodexDemCost'


def summary(rows):
    result = {'samples': len(rows)}
    for name, values in {
            'fluid_total_ms': [r['step_stats']['total_ms'] for r in rows],
            'dem_total_ms': [r['runtime']['host_ms']['step_total'] for r in rows],
            'rpc_step_ms': [r['rpc_step_ms'] for r in rows]}.items():
        ordered = sorted(values)
        result[name] = {'mean': statistics.mean(values), 'median': statistics.median(values),
                        'min': ordered[0], 'max': ordered[-1]}
    for key in ('substeps', 'dispatches', 'upload_bytes', 'download_bytes', 'working_set_bytes'):
        values = [r['runtime'][key] for r in rows]
        result[key] = {'min': min(values), 'max': max(values), 'mean': statistics.mean(values)}
    result['host_ms_mean'] = {key: statistics.mean(r['runtime']['host_ms'][key] for r in rows)
                              for key in rows[0]['runtime']['host_ms']}
    result['dispatches_per_substep'] = sorted({r['runtime']['dispatches'] /
                                               r['runtime']['substeps'] for r in rows})
    result['work_modes'] = sorted({r['runtime']['sleep_work_mode'] for r in rows})
    return result


def main(report_path):
    client = RtIpc()
    call = client.call
    baseline = json.loads(Path('docs/dev/DEM_BOND_USER_AB_BEFORE.json').read_text('utf-8'))
    source = baseline['recipe']['sources'][0]
    fields = ('name', 'enabled', 'substance', 'bounds_min', 'bounds_max', 'mesh_object',
              'velocity', 'temperature_mode', 'temperature_kelvin', 'grain_saturation', 'bond_body')
    made = False
    owns_scene = False
    report = {'domain': DOMAIN, 'frame_dt': 1 / 24, 'fixture': baseline,
              'timing_kind': 'engine host timings; rpc includes queue/render overhead', 'cases': []}
    try:
        assert not call('sim.control_state')['playing'], 'Pause first'
        assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
        for label, strength in (('intact', 1e6), ('fracturing', 1e4)):
            if made:
                call('fluid.remove_domain', domain=DOMAIN)
                made = False
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-2.5, 0, -2.5], domain_max=[2.5, 5, 2.5], voxel_size=.1)
            made = True
            owns_scene = True
            call('fluid.set_param', domain=DOMAIN, backend='vulkan', boundary='closed',
                 default_substance='Ice', thermal_liquid_enabled=False, max_particles=100000)
            call('fluid.set_grain_settings', domain=DOMAIN, **baseline['settings'])
            call('fluid.set_seed_source_settings', domain=DOMAIN,
                 settings=baseline['recipe']['settings'])
            settings = {key: copy.deepcopy(source[key]) for key in fields}
            settings['bond_body'].update(tensile_strength_pa=strength, shear_strength_pa=strength)
            call('fluid.add_seed_source', domain=DOMAIN, settings=settings)
            call('fluid.apply_seed_sources', domain=DOMAIN)
            case = {'label': label, 'strength_pa': strength,
                    'initial_digest': call('fluid.state_digest', domain=DOMAIN), 'samples': []}
            report['cases'].append(case)
            assert case['initial_digest']['particles'] == 59130
            print(json.dumps({'starting': label, 'particles': 59130}), flush=True)
            all_broken_samples = []
            for frame in range(1, 61):
                before = call('fluid.dem_bodies', domain=DOMAIN)
                start = time.perf_counter()
                call('fluid.step', dt=report['frame_dt'])
                rpc_ms = (time.perf_counter() - start) * 1000
                models = call('fluid.matter_models', domain=DOMAIN)
                assert not models['grain_readiness']['step_held'], models['grain_readiness']
                runtime = models['grain_diagnostics']['runtime']
                stats = call('fluid.step_stats', domain=DOMAIN)
                after = call('fluid.dem_bodies', domain=DOMAIN)
                row = {'frame': frame, 'before': before, 'after': after,
                       'runtime': runtime, 'step_stats': stats, 'rpc_step_ms': rpc_ms}
                case['samples'].append(row)
                if frame % 4 == 0:
                    print(json.dumps({'case': label, 'frame': frame, 'broken': after['broken'],
                                      'dem_ms': runtime['host_ms']['step_total'],
                                      'substeps': runtime['substeps'],
                                      'dispatches': runtime['dispatches']}), flush=True)
                if label == 'intact':
                    assert after['broken'] == 0, 'Strong fixture fractured; not an intact cost sample'
                    if frame == 24:
                        case['steady_summary'] = summary(case['samples'][2:])
                        break
                elif before['bonds'] > 0 and before['broken'] == before['bonds']:
                    assert after['broken'] == before['broken']
                    all_broken_samples.append(row)
                    if len(all_broken_samples) == 10:
                        case['all_broken_summary'] = summary(all_broken_samples[2:])
                        break
            if label == 'fracturing':
                assert len(all_broken_samples) == 10, 'Could not collect an all-broken window'
            case['final_digest'] = call('fluid.state_digest', domain=DOMAIN)
            assert case['final_digest']['particles'] == 59130
            Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                        encoding='utf-8')
        assert report['cases'][0]['initial_digest']['position_hash'] == \
            report['cases'][1]['initial_digest']['position_hash']
    finally:
        if made:
            call('fluid.remove_domain', domain=DOMAIN)
        if owns_scene:
            call('timeline.pause')
            call('timeline.set_frame', frame=0)
        report['cleanup'] = {'domains': call('fluid.list_domains'),
                             'control': call('sim.control_state')}
        Path(report_path).write_text(json.dumps(report, indent=2, ensure_ascii=False),
                                    encoding='utf-8')
        client.close()
    for case in report['cases']:
        print(json.dumps({'case': case['label'], 'summary': case.get('steady_summary',
                         case.get('all_broken_summary'))}), flush=True)
    print('SAVED', report_path, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--report', default='docs/dev/DEM_BOND_COST_LIVE.json')
    main(parser.parse_args().report)
