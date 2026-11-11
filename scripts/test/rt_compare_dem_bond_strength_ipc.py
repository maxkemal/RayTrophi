"""External, mutating A/B on an explicitly selected paused single-layer domain.

Re-applies the current recipe. Keeps E, geometry and solver settings unchanged;
only compares tensile/shear strength. Leaves the stronger recipe at frame zero.
This is not an ice material calibration or a test of an unbuilt optimization.
"""
import argparse
import copy
import json
from pathlib import Path
from rt_ipc import RtIpc


def run(domain, steps, dt, strength, report_path, cold_runtime=False, baseline_strength=None):
    client = RtIpc()
    call = client.call
    report = {'domain': domain, 'frame_dt': dt, 'steps': steps, 'cases': []}
    baseline = None
    source_id = None
    try:
        assert not call('sim.control_state')['playing'], 'Pause the scene first'
        recipe = call('fluid.get_seed_sources', domain=domain)
        sources = [s for s in recipe['sources'] if s['enabled']]
        assert recipe['enabled'] and len(sources) == 1, 'Requires one enabled seed layer'
        source = sources[0]
        baseline = copy.deepcopy(source['bond_body'])
        assert baseline['enabled'], 'The source must already create a bonded body'
        assert not recipe['settings']['settle'], 'Turn Seed Settle off'
        source_id = source['id']
        report['before'] = {'source': source, 'digest': call('fluid.state_digest', domain=domain),
                            'bonds': call('fluid.dem_bodies', domain=domain)}
        report['solver'] = call('fluid.grain_settings', domain=domain)
        domain_config = next(row for row in call('fluid.list_domains')['domains']
                             if row['name'] == domain)
        source_fields = ('name', 'enabled', 'substance', 'bounds_min', 'bounds_max',
                         'mesh_object', 'velocity', 'temperature_mode', 'temperature_kelvin',
                         'grain_saturation', 'bond_body')
        report['cold_runtime'] = cold_runtime

        def recreate(tensile, shear):
            # Explicit opt-in for a dedicated recreated test domain. A fresh
            # runtime removes adaptive CFL/contact hints from the previous run.
            nonlocal source_id
            assert len(call('fluid.list_domains')['domains']) == 1
            call('fluid.remove_domain', domain=domain)
            call('fluid.create_domain', name=domain, type=domain_config['type'],
                 domain_min=domain_config['domain_min'], domain_max=domain_config['domain_max'],
                 voxel_size=domain_config['voxel_size'])
            call('fluid.set_param', domain=domain, backend=domain_config['backend'],
                 boundary=domain_config['boundary'],
                 default_substance=domain_config['default_substance'],
                 thermal_liquid_enabled=domain_config['thermal_liquid_enabled'],
                 max_particles=domain_config['max_particles'])
            call('fluid.set_grain_settings', domain=domain, **report['solver'])
            call('fluid.set_seed_source_settings', domain=domain, settings=recipe['settings'])
            authored = {key: copy.deepcopy(source[key]) for key in source_fields}
            authored['bond_body']['tensile_strength_pa'] = tensile
            authored['bond_body']['shear_strength_pa'] = shear
            call('fluid.add_seed_source', domain=domain, settings=authored)
            source_id = call('fluid.get_seed_sources', domain=domain)['sources'][0]['id']

        baseline_tensile = baseline['tensile_strength_pa'] if baseline_strength is None \
            else baseline_strength
        baseline_shear = baseline['shear_strength_pa'] if baseline_strength is None \
            else baseline_strength
        for label, tensile, shear in (
                ('baseline', baseline_tensile, baseline_shear),
                ('stronger', strength, strength)):
            if cold_runtime:
                recreate(tensile, shear)
            else:
                call('fluid.update_seed_source', domain=domain, id=source_id,
                     settings={'bond_body': {'tensile_strength_pa': tensile,
                                             'shear_strength_pa': shear}})
            call('fluid.apply_seed_sources', domain=domain)
            source_now = next(s for s in call('fluid.get_seed_sources', domain=domain)['sources']
                              if s['id'] == source_id)
            assert source_now['bond_body']['young_pa'] == baseline['young_pa']
            case = {'label': label, 'tensile_strength_pa': tensile, 'shear_strength_pa': shear,
                    'initial_digest': call('fluid.state_digest', domain=domain), 'samples': []}
            print(json.dumps({'starting': label, 'particles': case['initial_digest']['particles']}),
                  flush=True)
            for step in range(1, steps + 1):
                call('fluid.step', dt=dt)
                models = call('fluid.matter_models', domain=domain)
                readiness = models['grain_readiness']
                if readiness['step_held']:
                    raise AssertionError(readiness['status'])
                grain = models['grain_diagnostics']
                state = call('fluid.dem_bodies', domain=domain)
                row = {'step': step, 'bonds': state, 'cfl': grain.get('cfl_terms'),
                       'runtime': grain.get('runtime')}
                case['samples'].append(row)
                if step % 4 == 0 or step == steps:
                    print(json.dumps({'case': label, 'step': step, 'broken': state['broken'],
                                      'bonds': state['bonds']}), flush=True)
            case['final_digest'] = call('fluid.state_digest', domain=domain)
            assert case['final_digest']['particles'] == case['initial_digest']['particles']
            report['cases'].append(case)
        assert report['cases'][0]['initial_digest']['position_hash'] == \
            report['cases'][1]['initial_digest']['position_hash'], 'A/B initial geometry differs'
        if cold_runtime:
            first = [case['samples'][0]['cfl'] for case in report['cases']]
            assert first[0]['contacts_assumed'] == first[1]['contacts_assumed']
            assert first[0]['substeps'] == first[1]['substeps']
        # The user requested the stronger comparison. Leave its persistent recipe
        # ready for visual inspection, not a moving/partly destroyed state.
        if cold_runtime:
            recreate(strength, strength)
        call('fluid.apply_seed_sources', domain=domain)
        report['left_at_start'] = call('fluid.dem_bodies', domain=domain)
        report['scope'] = 'bounded strength A/B; E unchanged; not calibrated ice'
        Path(report_path).write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                    encoding='utf-8')
        print('SAVED', report_path, flush=True)
    except Exception:
        if baseline is not None and source_id is not None:
            call('fluid.update_seed_source', domain=domain, id=source_id,
                 settings={'bond_body': baseline})
            call('fluid.apply_seed_sources', domain=domain)
        raise
    finally:
        client.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domain')
    parser.add_argument('--steps', type=int, default=24)
    parser.add_argument('--dt', type=float, default=1.0 / 24.0)
    parser.add_argument('--strength', type=float, default=1e6)
    parser.add_argument('--report', default='docs/dev/DEM_BOND_USER_AB.json')
    parser.add_argument('--cold-runtime', action='store_true',
                        help='Recreates the selected test domain; does not preserve other domain settings')
    parser.add_argument('--baseline-strength', type=float)
    args = parser.parse_args()
    run(args.domain, args.steps, args.dt, args.strength, args.report,
        args.cold_runtime, args.baseline_strength)
