"""Finite dry / water-only / wet / dry-return comparison through external IPC."""
import argparse
import json
from pathlib import Path
from rt_ipc import RtIpc
from rt_domain_material import domain_material  # noqa: E402

DOMAIN = 'G2_DryWet_Comparison'
LOG = Path('docs/dev/matter_g2_dry_wet_compare_2026-10-05.json')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-empty-fluid-skipped', action='store_true')
    args = parser.parse_args()
    c = RtIpc()
    data = {'dt': 1 / 60, 'arms': []}
    def call(method, **params):
        result = c.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result
    def save():
        LOG.write_text(json.dumps(data, indent=2), encoding='utf-8')
    def sample(step):
        assert call('sim.control_state') == control, 'Control changed: discard run'
        inventory = call('fluid.matter_models', domain=DOMAIN)
        assert not inventory['mixed_execution']['step_held']
        assert not inventory['pore_exchange']['held']
        if args.expect_empty_fluid_skipped and not water:
            assert inventory['models'][0]['particles'] == 0
            assert not inventory['mixed_execution']['pressure_on_gpu']
            assert inventory['mixed_execution']['contact_pairs'] == 0
            assert 'granular-only' in inventory['mixed_execution']['status']
        return {'step': step, 'inventory': inventory}
    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'colliders': call('collider.list')['colliders']}
    assert not original['control']['playing'], 'Pause before comparison'
    data['original'] = original
    save()
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for collider in original['colliders']:
            call('collider.update', name=collider['name'], enabled=False)
        if DOMAIN not in [d['name'] for d in original['domains']]:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1.6, 0, -1.6], domain_max=[1.6, 2.4, 1.6], voxel_size=.1)
        material = domain_material(call, 'T: g2_dry_wet_compare', 'Sand',
            granular_young_modulus=2000000., granular_friction_angle=49., granular_cohesion=0., granular_tensile_cutoff=0., granular_rebonding=False, granular_damage_rate=0.)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
             backend='vulkan', boundary='closed', default_substance=material,
             solid_phase=False, thermal_liquid_enabled=False)
        names = {s['name'] for s in call('flow_source.list')}
        for name, substance, model, position, rate, limit, start, end in [
            ('G2_Sand_Finite', 'Sand', 'granular', [0, .6, 0], 1200, 360, 0., .3),
            ('G2_Water_Finite', 'Water', 'fluid', [0, .85, 0], 1200, 360, .5, .8)]:
            call('flow_source.update' if name in names else 'flow_source.create',
                 name=name, domain=DOMAIN, enabled=False, phase='liquid', source_mode='point',
                 fluid_substance=substance, position=position,
                 radius=.26, velocity=[0, 0, 0], fluid_velocity_spread=0.,
                 fluid_particles_per_second=rate, fluid_temperature_override=True,
                 fluid_temperature_kelvin=293.15, use_time_limit=True,
                 start_time=start, end_time=end, use_particle_limit=True,
                 max_emitted_particles=limit)
        for label, water, wet in [('dry', False, False), ('water_only', True, False),
                                  ('wet_response', True, True),
                                  ('dry_after_water', False, False)]:
            call('fluid.set_pore_exchange', domain=DOMAIN, enabled=True, porosity=.35,
                 permeability_m2=1e-8, viscosity_pa_s=.001, gravity_m_s2=9.81,
                 drainage_scale=1., wet_response_enabled=wet, wet_appearance_enabled=wet)
            call('flow_source.update', name='G2_Sand_Finite', enabled=True)
            call('flow_source.update', name='G2_Water_Finite', enabled=water)
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            control = call('sim.control_state')
            arm = {'name': label, 'samples': []}
            data['arms'].append(arm)
            print('START', label, flush=True)
            for step in range(1, 151):
                call('fluid.step', dt=data['dt'])
                if step in (25, 60, 90, 120, 150):
                    s = sample(step)
                    arm['samples'].append(s)
                    save()
                    g = s['inventory']['acceptance_metrics']['granular']
                    print(label, step, json.dumps(g), flush=True)
            baseline = arm['samples'][1]['inventory']
            def masses(inv):
                pore = inv['pore_exchange']['pore_water_kg']
                return inv['models'][0]['mass_kg'] + pore, inv['acceptance_metrics']['granular']['dry_mass_kg']
            water0, dry0 = masses(baseline)
            arm['max_water_drift_kg'] = max(abs(masses(s['inventory'])[0] - water0)
                for s in arm['samples'][1:])
            arm['max_dry_drift_kg'] = max(abs(masses(s['inventory'])[1] - dry0)
                for s in arm['samples'][1:])
            assert arm['max_water_drift_kg'] <= max(1e-5, water0 * 1e-6)
            assert arm['max_dry_drift_kg'] <= max(1e-6, dry0 * 1e-7)
            save()
        initial = data['arms'][0]['samples'][0]['inventory']['acceptance_metrics']['granular']
        data['dry_baseline_comparable'] = all(
            abs(arm['samples'][0]['inventory']['acceptance_metrics']['granular']
                ['dry_center_of_mass'][1] - initial['dry_center_of_mass'][1]) <= .001
            and abs(arm['samples'][0]['inventory']['acceptance_metrics']['granular']
                ['horizontal_rms_radius_m'] - initial['horizontal_rms_radius_m']) <= .001
            for arm in data['arms'])
        data['completed'] = data['dry_baseline_comparable']
        save()
        assert data['dry_baseline_comparable'], 'Dry pre-water baseline differs; reject causal A/B'
    finally:
        call('flow_source.update', name='G2_Sand_Finite', enabled=False)
        call('flow_source.update', name='G2_Water_Finite', enabled=False)
        call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for collider in original['colliders']:
            call('collider.update', name=collider['name'], enabled=collider['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'], visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        data['restored_authoring'] = True
        data['final_control'] = call('sim.control_state')
        save()
        c.close()

if __name__ == '__main__':
    main()
