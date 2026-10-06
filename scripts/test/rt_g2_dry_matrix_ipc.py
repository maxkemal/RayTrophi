"""External IPC dry, unbonded Sand matrix. Edits/reset runtime; never saves/builds.

Authoring enabled/visible states are restored in finally; timeline stays paused at 0.
Settling and convergence are reported separately from conservation/publication gates.
"""
import argparse
import json
import math
from pathlib import Path

from rt_ipc import RtIpc


DOMAIN = 'G2_Dry_Matrix'
SOURCE = 'G2_Dry_Matrix_Sand'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=float, default=8.0)
    parser.add_argument('--expect-empty-fluid-skipped', action='store_true')
    parser.add_argument('--log', type=Path,
                        default=Path(__file__).resolve().parents[2] /
                        'docs/dev/matter_g2_dry_matrix_2026-10-05.json')
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds < 4:
        parser.error('--seconds must be finite and at least 4')
    client = RtIpc()
    data = {'arms': [], 'settling_thresholds': {
        'last_two_seconds_com_y_delta_m': .001,
        'last_two_seconds_rms_relative_delta': .01,
        'maximum_sampled_specific_kinetic_energy_j_kg': .001},
        'convergence_thresholds': {'com_y_relative_delta': .05,
                                   'rms_relative_delta': .05,
                                   'dry_mass_relative_delta': .001}}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    def save():
        args.log.write_text(json.dumps(data, indent=2), encoding='utf-8')

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'colliders': call('collider.list')['colliders']}
    assert not original['control']['playing'], 'Pause first'
    data['original'] = original
    save()
    source_ready = False
    domain_ready = False
    try:
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=False)
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=False, visible=False)
        for collider in original['colliders']:
            call('collider.update', name=collider['name'], enabled=False)
        if DOMAIN not in {d['name'] for d in original['domains']}:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1.6, 0, -1.6], domain_max=[1.6, 2.4, 1.6], voxel_size=.1)
        domain_ready = True
        existing_sources = {s['name'] for s in call('flow_source.list')}
        call('flow_source.update' if SOURCE in existing_sources else 'flow_source.create',
             name=SOURCE, domain=DOMAIN, enabled=False, phase='liquid', source_mode='point',
             fluid_substance='Sand', initial_constitutive_model='granular',
             position=[0, .6, 0], radius=.26, velocity=[0, 0, 0], fluid_velocity_spread=0.,
             fluid_temperature_override=True, fluid_temperature_kelvin=293.15,
             use_time_limit=True, start_time=0., end_time=.35, use_particle_limit=True,
             max_emitted_particles=360, fluid_particles_per_second=1200)
        source_ready = True
        for label, voxel, dt, load in [('reference', .1, 1 / 60, 1),
                                       ('dt_half', .1, 1 / 120, 1),
                                       ('grid_finer', .08, 1 / 60, 1),
                                       ('load_double', .1, 1 / 60, 2)]:
            count = round(360 * load * (.1 / voxel) ** 3)
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 voxel_size=voxel, backend='vulkan', boundary='closed', preset='sand',
                 granular_enabled=True, granular_young_modulus=2000000.,
                 granular_poisson_ratio=.3, granular_friction_angle=35.,
                 granular_dilatancy=0., granular_hardening=0., granular_cohesion=0.,
                 granular_tensile_cutoff=0., granular_rebonding=False,
                 granular_damage_rate=0., solid_phase=False, thermal_liquid_enabled=False)
            call('fluid.set_pore_exchange', domain=DOMAIN, enabled=True, porosity=.35,
                 wet_response_enabled=False, wet_appearance_enabled=False)
            call('flow_source.update', name=SOURCE, enabled=True,
                 max_emitted_particles=count, fluid_particles_per_second=count / .3)
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            control = call('sim.control_state')
            arm = {'name': label, 'voxel_m': voxel, 'dt_s': dt, 'load_multiplier': load,
                   'expected_count': count, 'samples': []}
            data['arms'].append(arm)
            print('START', label, flush=True)
            steps = round(args.seconds / dt)
            interval = round(1 / dt)
            for step in range(1, steps + 1):
                call('fluid.step', dt=dt)
                if step % interval != 0 and step != steps:
                    continue
                assert call('sim.control_state') == control, 'Control changed: discard run'
                inv = call('fluid.matter_models', domain=DOMAIN)
                info = call('fluid.get', domain=DOMAIN)
                granular = inv['acceptance_metrics']['granular']
                assert granular['particles'] == count, 'Finite source count mismatch'
                assert inv['acceptance_metrics']['exactly_dry_particles'] == count
                assert inv['pore_exchange']['pore_water_kg'] == 0
                assert inv['models'][0]['particles'] == 0
                assert not inv['mixed_execution']['step_held']
                assert not inv['pore_exchange']['held']
                if args.expect_empty_fluid_skipped:
                    execution = inv['mixed_execution']
                    assert not execution['pressure_on_gpu']
                    assert execution['contact_pairs'] == 0
                    assert 'granular-only' in execution['status']
                assert info['granular_invalid'] == 0
                assert not info['granular_stiffness_capped']
                assert info['granular_solver_substeps'] >= info['granular_required_substeps']
                assert math.isclose(info['granular_effective_young_modulus'], 2000000.,
                                    rel_tol=1e-6)
                baseline_mass = arm['samples'][0]['granular']['dry_mass_kg'] \
                    if arm['samples'] else granular['dry_mass_kg']
                assert abs(granular['dry_mass_kg'] - baseline_mass) <= baseline_mass * 1e-7
                arm['samples'].append({'seconds': step * dt, 'granular': granular,
                                       'mechanics': inv['granular_mechanics'],
                                       'step_stats': call('fluid.step_stats', domain=DOMAIN)})
                save()
                print(label, step * dt, json.dumps(granular), flush=True)
            last = arm['samples'][-1]['granular']
            window = [s['granular'] for s in arm['samples']
                      if s['seconds'] >= args.seconds - 2 - 1e-6]
            com_delta = max(g['dry_center_of_mass'][1] for g in window) \
                - min(g['dry_center_of_mass'][1] for g in window)
            rms_delta = (max(g['horizontal_rms_radius_m'] for g in window)
                         - min(g['horizontal_rms_radius_m'] for g in window)) \
                / last['horizontal_rms_radius_m']
            energy = max(g['kinetic_energy_j'] / g['dry_mass_kg'] for g in window)
            arm['settling'] = {'com_y_range_m': com_delta, 'rms_relative_range': rms_delta,
                               'max_specific_energy_j_kg': energy,
                               'passed': com_delta <= .001 and rms_delta <= .01
                               and energy <= .001}
            save()
        ref = data['arms'][0]['samples'][-1]['granular']
        data['comparisons'] = []
        for arm in data['arms'][1:3]:
            last = arm['samples'][-1]['granular']
            differences = {key: abs(last[key] - ref[key]) / abs(ref[key]) for key in
                           ['dry_mass_kg', 'horizontal_rms_radius_m']}
            differences['com_y'] = abs(last['dry_center_of_mass'][1]
                                      - ref['dry_center_of_mass'][1]) \
                / abs(ref['dry_center_of_mass'][1])
            data['comparisons'].append({'name': arm['name'], 'relative_differences': differences,
                'within_shape_threshold': differences['com_y'] <= .05 and
                differences['horizontal_rms_radius_m'] <= .05 and
                differences['dry_mass_kg'] <= .001,
                'settled_on_both_sides': arm['settling']['passed'] and
                data['arms'][0]['settling']['passed']})
        data['measurement_completed'] = True
        data['g2_complete'] = False  # Repose/support and additional acceptance remain separate.
    finally:
        if source_ready:
            call('flow_source.update', name=SOURCE, enabled=False)
        if domain_ready:
            call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for collider in original['colliders']:
            call('collider.update', name=collider['name'], enabled=collider['enabled'])
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=domain['enabled'],
                 visible=domain['visible'])
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=source['enabled'])
        data['restored_authoring'] = True
        data['final_control'] = call('sim.control_state')
        save()
        client.close()


if __name__ == '__main__':
    main()
