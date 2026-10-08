"""External IPC free-flight gravity probe, separate from pile/contact acceptance.

Tests Sand carrier COM at 60/120 Hz before ground contact. Restores authoring,
resets runtime, leaves frame0 paused. No project save or application build.
"""
import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc
from rt_domain_material import domain_material  # noqa: E402


DOMAIN = 'G2_Free_Fall'
SOURCE = 'G2_Free_Fall_Sand'
LOG = Path(__file__).resolve().parents[2] / 'docs/dev/matter_g2_free_fall_2026-10-05.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expect-empty-fluid-skipped', action='store_true')
    args = parser.parse_args()
    client = RtIpc()
    data = {'arms': [], 'expected_gravity_m_s2': 9.81, 'gravity_tolerance_relative': .05}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    def save():
        LOG.write_text(json.dumps(data, indent=2), encoding='utf-8')

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'colliders': call('collider.list')['colliders']}
    assert not original['control']['playing'], 'Pause first'
    assert not call('forcefield.list'), 'This reference requires no external force fields'
    data['original'] = original
    save()
    domain_ready = False
    source_ready = False
    try:
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=False)
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=False, visible=False)
        for collider in original['colliders']:
            call('collider.update', name=collider['name'], enabled=False)
        if DOMAIN not in {d['name'] for d in original['domains']}:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-.6, -2, -.6], domain_max=[.6, 4, .6], voxel_size=.1)
        domain_ready = True
        material = domain_material(call, 'T: g2_free_fall', 'Sand',
            granular_young_modulus=200000., granular_friction_angle=35., granular_cohesion=0., granular_tensile_cutoff=0., granular_rebonding=False)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
             backend='vulkan', boundary='closed', default_substance=material,
             solid_phase=False, thermal_liquid_enabled=False)
        call('fluid.set_pore_exchange', domain=DOMAIN, enabled=True,
             wet_response_enabled=False, wet_appearance_enabled=False)
        existing = {s['name'] for s in call('flow_source.list')}
        call('flow_source.update' if SOURCE in existing else 'flow_source.create',
             name=SOURCE, domain=DOMAIN, enabled=False, phase='liquid', source_mode='point',
             fluid_substance='Sand', position=[0, 2, 0],
             radius=.0001, velocity=[0, 0, 0], fluid_velocity_spread=0.,
             fluid_temperature_override=True, fluid_temperature_kelvin=293.15,
             use_time_limit=True, use_particle_limit=True, max_emitted_particles=1,
             start_time=0., end_time=.04, fluid_particles_per_second=61.)
        source_ready = True
        for count in (1, 64):
            for dt in (1 / 60, 1 / 120):
                call('flow_source.update', name=SOURCE, enabled=True,
                     radius=.0001 if count == 1 else .16, max_emitted_particles=count,
                     fluid_particles_per_second=count * 1.01 / dt, end_time=dt * 2)
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                control = call('sim.control_state')
                arm = {'count': count, 'dt_s': dt, 'samples': []}
                data['arms'].append(arm)
                steps = round(.2 / dt)
                for step in range(1, steps + 1):
                    call('fluid.step', dt=dt)
                    assert call('sim.control_state') == control, 'Control changed'
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    granular = inventory['acceptance_metrics']['granular']
                    assert granular['particles'] == count
                    assert granular['pore_water_kg'] == 0
                    assert inventory['models'][0]['particles'] == 0
                    assert not inventory['mixed_execution']['step_held']
                    if args.expect_empty_fluid_skipped:
                        execution = inventory['mixed_execution']
                        assert not execution['pressure_on_gpu']
                        assert execution['contact_pairs'] == 0
                        assert 'granular-only' in execution['status']
                    assert granular['bounds_min'][1] > 1, 'No-contact free flight required'
                    arm['samples'].append({'step': step, 'granular': granular})
                first = arm['samples'][0]
                middle = arm['samples'][steps // 2 - 1]
                last = arm['samples'][-1]
                y0 = first['granular']['dry_center_of_mass'][1]
                ym = middle['granular']['dry_center_of_mass'][1]
                yf = last['granular']['dry_center_of_mass'][1]
                tm = (middle['step'] - first['step']) * dt
                tf = (last['step'] - first['step']) * dt
                acceleration = -2 * ((yf - y0) / tf - (ym - y0) / tm) / (tf - tm)
                arm['fitted_downward_acceleration_m_s2'] = acceleration
                arm['gravity_gate_passed'] = abs(acceleration - 9.81) <= 9.81 * .05
                arm['drop_after_first_step_m'] = y0 - yf
                arm['step_stats'] = call('fluid.step_stats', domain=DOMAIN)
                save()
                print(json.dumps({k: v for k, v in arm.items() if k != 'samples'}), flush=True)
        data['measurement_completed'] = True
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
