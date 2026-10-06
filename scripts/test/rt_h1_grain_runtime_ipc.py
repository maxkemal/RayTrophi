"""Real dry grain runtime after user C++/shader build; external IPC only.

Creates isolated disabled-after-test sources/domain, restores existing enabled/visible
authoring. Leaves frame0 paused, does not save scene. Numerical success remains pending
until this script runs against the new binary. No reference replay.
"""
import argparse
import math
import time
import json
from pathlib import Path
from rt_ipc import RtIpc

DOMAIN = 'H1_Grain_Runtime'
SOURCE = 'H1_Grain_Runtime_Source'
LOG = Path(__file__).resolve().parents[2] / 'docs/dev/matter_h1_grain_runtime_live.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pile-only", action="store_true")
    parser.add_argument("--extended-only", action="store_true")
    parser.add_argument("--corner-only", action="store_true")
    parser.add_argument("--settle-only", action="store_true")
    parser.add_argument("--settle-normal-damping", type=float, default=4.)
    parser.add_argument("--static-only", action="store_true",
                        help="slope hold: static friction + rolling spring vs kinetic-only")
    parser.add_argument("--convergence-only", action="store_true",
                        help="pile at contact_resolution 24 vs 48 (real substep-dt halving)")
    parser.add_argument("--repose-only", action="store_true",
                        help="poured free-standing pile: repose angle, mu_r sensitivity, grain-size convergence")
    args = parser.parse_args()
    args.extended_only = args.extended_only or args.corner_only
    client = RtIpc()
    data = {'arms': [], 'completed': False}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    def save():
        temporary = LOG.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        for attempt in range(10):
            try:
                temporary.replace(LOG)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.1)

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'colliders': call('collider.list')['colliders']}
    assert not original['control']['playing'], 'Pause application first'
    data['original'] = original
    domain_ready = source_ready = repose_ready = False
    ramp_collider = None
    extra_colliders = []
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=False)
        if DOMAIN not in {d['name'] for d in original['domains']}:
            call('fluid.create_domain', name=DOMAIN, type='matter', domain_min=[-.6, 0, -.6],
                 domain_max=[.6, 3, .6], voxel_size=.1)
        domain_ready = True
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', preset='sand', granular_enabled=True, solid_phase=False,
             thermal_liquid_enabled=False)
        call('fluid.set_pore_exchange', domain=DOMAIN, enabled=False,
             wet_response_enabled=False, wet_appearance_enabled=False)
        call('fluid.reset')
        call('fluid.set_grain_settings', domain=DOMAIN, enabled=True, radius_m=.025,
             stiffness_n_m=20000., normal_damping_n_s_m=4., sliding_damping_n_s_m=4.,
             friction=.5, rolling_friction=.02, twisting_friction=0., max_substeps=1024,
             tangential_stiffness_ratio=2/7, contact_resolution=24, packing_fraction=.6)
        settings = call('fluid.grain_settings', domain=DOMAIN)
        assert settings['enabled'] and abs(settings['radius_m']-.025) < 1e-6
        before = settings.copy()
        for patch in ({'radius_m': 0}, {'enabled': 1}, {'max_substeps': 1.5}, {'bogus': True},
                      {'contact_resolution': 4}, {'tangential_stiffness_ratio': 2.},
                      {'packing_fraction': .9}):
            try:
                call('fluid.set_grain_settings', domain=DOMAIN, **patch)
            except Exception:
                pass
            else:
                raise AssertionError(('invalid patch accepted', patch))
            assert call('fluid.grain_settings', domain=DOMAIN) == before
        exists = SOURCE in {s['name'] for s in call('flow_source.list')}
        call('flow_source.update' if exists else 'flow_source.create', name=SOURCE,
             domain=DOMAIN, enabled=False, phase='liquid', source_mode='point',
             fluid_substance='Sand', initial_constitutive_model='granular',
             position=[0, 1.5, 0], radius=.0001, velocity=[0, 0, 0],
             fluid_velocity_spread=0., use_particle_limit=True, max_emitted_particles=1,
             use_time_limit=True, start_time=0., end_time=.05,
             fluid_particles_per_second=200., fluid_temperature_override=True,
             fluid_temperature_kelvin=293.15)
        source_ready = True

        def runtime_of(inventory):
            runtime = inventory['grain_diagnostics']['runtime']
            assert runtime and runtime['dispatches'] == runtime['substeps'] + 2, runtime
            assert runtime['substeps'] % 2 == 0, runtime
            return runtime

        # Grain mass is born from the sphere, not from voxel/PPC parcel size:
        # Sand 1600 kg/m3 bulk / .6 packing * 4/3 pi .025^3.
        expected_grain_mass = 1600/.6*4/3*math.pi*.025**3

        def make_ramp(name, rotation):
            obj = call('scene.add_primitive', type='plane', name=name, size=1.)
            material = call('material.create', type='principled', name=name+'_Material')
            call('material.assign', object_name=obj, material_name=material)
            call('scene.set_transform', name=obj, translation=[0, 1., 0], rotation=rotation)
            collider = name+'_Collider'
            exists = collider in {c['name'] for c in call('collider.list')['colliders']}
            call('collider.update' if exists else 'collider.create', name=collider,
                 source_mode='mesh_bvh', source_object=obj, enabled=True)
            extra_colliders.append(collider)
            return obj, collider

        if args.static_only:
            # A single grain on a 20 degree flat mesh slope; mu=.5 and mu_r=1
            # both exceed tan(20)=.364, so a static contact must hold it.
            # Kinetic-only sliding (ratio 0) reaches v = m g sin / c_s instead:
            # the arm proves the instrument can tell the two apart.
            _, collider = make_ramp('H1_Grain_Static_Ramp', [0, 0, 20])
            angle = math.radians(20)
            data['static_hold'] = {}
            for ratio in (2/7, 0.):
                call('fluid.reset')
                call('fluid.set_grain_settings', domain=DOMAIN, tangential_stiffness_ratio=ratio,
                     rolling_friction=1., friction=.5)
                # Born resting on the slope (overlap 50 um, below its weight's
                # 80 um), at rest: a drop at Cn=4 (restitution ~.9) bounces for
                # the whole window and drifts downhill in flight, which measures
                # the fixture, not static friction.
                rest = .025-5e-5
                call('flow_source.update', name=SOURCE,
                     position=[-math.sin(angle)*rest, 1.+math.cos(angle)*rest, 0], radius=.0001,
                     velocity=[0, 0, 0], max_emitted_particles=1, end_time=.05, enabled=True)
                control = call('sim.control_state')
                samples = []
                data['static_hold'][f'{ratio:.4f}'] = {'samples': samples}
                for step in range(1, 241):
                    call('fluid.step', dt=1/120)
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    g = inventory['acceptance_metrics']['granular']
                    assert g['particles'] == 1
                    x, y, _ = g['dry_center_of_mass']
                    along = math.cos(angle)*x+math.sin(angle)*(y-1.)
                    normal = -math.sin(angle)*x+math.cos(angle)*(y-1.)
                    samples.append({'step': step, 'along_m': along, 'normal_m': normal,
                                    'mass_kg': g['dry_mass_kg'], 'runtime': runtime_of(inventory),
                                    'max_spin': inventory['grain_diagnostics']['max_spin_rad_s']})
                    if step % 20 == 0:
                        save()
                    assert normal >= .025*.7, ('ramp penetration', step, normal, along)
                creep = abs(samples[-1]['along_m']-samples[119]['along_m'])
                data['static_hold'][f'{ratio:.4f}']['creep_last_1s_m'] = creep
                save()
                print('static hold ratio', ratio, 'creep over last 1 s', creep,
                      'runtime', samples[-1]['runtime'], flush=True)
                if ratio > 0:
                    assert abs(samples[-1]['mass_kg']-expected_grain_mass) <= 1e-4*expected_grain_mass, \
                        ('grain mass not derived from radius', samples[-1]['mass_kg'], expected_grain_mass)
                    assert creep <= 5e-4, ('static friction did not hold the grain', creep)
                    assert samples[-1]['runtime']['sticking_contacts_last_substep'] >= 1
                else:
                    assert creep >= 1e-2, ('kinetic-only arm did not creep; test cannot discriminate', creep)
            call('collider.update', name=collider, enabled=False)
            data['completed'] = True
            return
        if args.repose_only:
            # A free-standing pile in a wide domain (walls >= .5 m from the
            # toe). Grain size arms keep total mass, scale k with r (same
            # relative overlap) and damping with r^2 (same restitution).
            REPOSE = 'H1_Grain_Repose'
            if REPOSE not in {d['name'] for d in call('fluid.list_domains')['domains']}:
                call('fluid.create_domain', name=REPOSE, type='matter', domain_min=[-1.2, 0, -1.2],
                     domain_max=[1.2, 1.6, 1.2], voxel_size=.1)
            repose_ready = True
            call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
            call('fluid.set_param', domain=REPOSE, enabled=True, visible=False, backend='vulkan',
                 boundary='closed', preset='sand', granular_enabled=True, solid_phase=False,
                 thermal_liquid_enabled=False, max_particles=40000)
            call('fluid.set_pore_exchange', domain=REPOSE, enabled=False,
                 wet_response_enabled=False, wet_appearance_enabled=False)
            call('flow_source.update', name=SOURCE, domain=REPOSE, enabled=False)
            data['repose'] = []
            for label, radius, mu_r, count in [('mu_r_.05', .025, .05, 1500), ('mu_r_.3', .025, .3, 1500),
                                               ('mu_r_.1', .025, .1, 1500),
                                               ('mu_r_.1_r_.0175', .0175, .1, 4373)]:
                scale = radius/.025
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                call('fluid.set_grain_settings', domain=REPOSE, enabled=True, radius_m=radius,
                     stiffness_n_m=100000.*scale, normal_damping_n_s_m=8.*scale**2,
                     sliding_damping_n_s_m=4.*scale**2, friction=.5, rolling_friction=mu_r,
                     twisting_friction=.1, tangential_stiffness_ratio=2/7, contact_resolution=24,
                     packing_fraction=.6, max_substeps=4096)
                call('flow_source.update', name=SOURCE, domain=REPOSE, position=[0, .6, 0], radius=.08,
                     velocity=[0, -.5, 0], max_emitted_particles=count, use_particle_limit=True,
                     fluid_particles_per_second=count/2.5, start_time=0., end_time=30., enabled=True)
                control = call('sim.control_state')
                arm = {'label': label, 'radius': radius, 'mu_r': mu_r, 'count': count, 'samples': []}
                data['repose'].append(arm)
                born_at = None
                for step in range(1, 1201):
                    call('fluid.step', dt=1/60)
                    if step % 10:
                        continue
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=REPOSE)
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    g = inventory['acceptance_metrics']['granular']
                    pile = inventory['grain_diagnostics']['pile']
                    arm['samples'].append({'step': step, 'particles': g['particles'],
                        'dry_mass_kg': g['dry_mass_kg'], 'kinetic_energy_j': g['kinetic_energy_j'],
                        'bounds_min_y': g['bounds_min'][1], 'pile': pile,
                        'runtime': runtime_of(inventory)})
                    assert g['bounds_min'][1] >= radius*.7, ('floor penetration', label, step)
                    if born_at is None and g['particles'] == count:
                        born_at = step
                    if born_at is not None and step >= born_at + 240:
                        break
                save()
                assert born_at is not None, ('pour did not finish', label, arm['samples'][-1]['particles'])
                tail = [x for x in arm['samples'] if x['step'] >= arm['samples'][-1]['step'] - 60]
                last = arm['samples'][-1]
                pile = last['pile']
                angles = [x['pile']['repose_angle_deg'] for x in tail]
                arm['summary'] = {'angle_deg': pile['repose_angle_deg'], 'peak_m': pile['peak_height_m'],
                    'base_radius_m': pile['base_radius_m'], 'rings_fit': pile['rings_fit'],
                    'tail_angle_range_deg': (max(angles)-min(angles)) if None not in angles else None,
                    'tail_peak_range_m': max(x['pile']['peak_height_m'] for x in tail) -
                        min(x['pile']['peak_height_m'] for x in tail),
                    'kinetic_energy_per_grain_j': last['kinetic_energy_j']/count,
                    'mass_drift_after_birth_kg': max(x['dry_mass_kg'] for x in tail) -
                        min(x['dry_mass_kg'] for x in tail),
                    'runtime': last['runtime']}
                save()
                print('repose', label, json.dumps(arm['summary']), flush=True)
                assert pile['measured'], ('pile profile not measurable', label, pile)
                assert pile['base_radius_m'] <= 1.2 - .3, ('pile reaches the walls; not free-standing', label)
                assert (arm['summary']['tail_angle_range_deg'] is not None and
                        arm['summary']['tail_angle_range_deg'] <= 1.5), ('pile still moving', label)
                assert arm['summary']['kinetic_energy_per_grain_j'] <= 1e-5, ('pile not at rest', label)
                assert arm['summary']['mass_drift_after_birth_kg'] <= 1e-5
            angle = {arm['label']: arm['summary']['angle_deg'] for arm in data['repose']}
            data['repose_gates'] = angle
            save()
            # Sensitivity proves the measurement sees the rolling spring at all.
            assert angle['mu_r_.3'] >= angle['mu_r_.05'] + 3, ('repose insensitive to mu_r', angle)
            assert 15 <= angle['mu_r_.1'] <= 45, ('implausible repose angle', angle)
            assert abs(angle['mu_r_.1'] - angle['mu_r_.1_r_.0175']) <= 4, ('grain-size dependent repose', angle)
            call('flow_source.update', name=SOURCE, domain=DOMAIN, enabled=False)
            data['completed'] = True
            return
        if args.convergence_only:
            # The old 60/120 Hz pile compared runs with the SAME substep dt.
            # Halving the substep itself is the convergence measurement.
            arms = []
            for resolution in (24, 48):
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                # Cn=Cs=1 loosens the damping bound (4.0e-4 s) so the accuracy
                # bound (2.7e-4 / 1.4e-4 s) is the one being halved. With the
                # default profile damping binds and both arms run identical
                # substeps: a 0.0 difference that measures nothing.
                call('fluid.set_grain_settings', domain=DOMAIN, contact_resolution=resolution,
                     max_substeps=4096, normal_damping_n_s_m=1., sliding_damping_n_s_m=1.)
                call('flow_source.update', name=SOURCE, position=[0, .25, 0], radius=.16,
                     velocity=[0, -.2, 0], max_emitted_particles=64,
                     fluid_particles_per_second=64*1.01*60, enabled=True, end_time=2/60)
                control = call('sim.control_state')
                arm = {'contact_resolution': resolution, 'samples': []}
                arms.append(arm)
                data['convergence'] = arms
                for step in range(1, 37):
                    call('fluid.step', dt=1/60)
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    g = inventory['acceptance_metrics']['granular']
                    assert g['particles'] == 64
                    arm['samples'].append({'step': step, 'granular': g,
                                           'runtime': runtime_of(inventory)})
                save()
                print('convergence', resolution, arm['samples'][-1]['runtime'], flush=True)
            a, b = [arm['samples'][-1] for arm in arms]
            assert a['runtime']['substep_limit'] == b['runtime']['substep_limit'] == 'accuracy',                 ('convergence arm is not halving the accuracy bound', a['runtime'], b['runtime'])
            assert b['runtime']['substeps'] >= 1.8*a['runtime']['substeps'], (a['runtime'], b['runtime'])
            ga, gb = a['granular'], b['granular']
            data['convergence_com_relative'] = abs(ga['dry_center_of_mass'][1]-gb['dry_center_of_mass'][1]) / max(
                gb['dry_center_of_mass'][1], .025)
            data['convergence_rms_relative'] = abs(ga['horizontal_rms_radius_m']-gb['horizontal_rms_radius_m']) / max(
                gb['horizontal_rms_radius_m'], .025)
            save()
            print('convergence COM/RMS relative', data['convergence_com_relative'],
                  data['convergence_rms_relative'], flush=True)
            assert data['convergence_com_relative'] <= .05, 'pile COM changes >5% when substep halves'
            assert data['convergence_rms_relative'] <= .05, 'pile spread changes >5% when substep halves'
            data['completed'] = True
            return
        if args.settle_only:
            call('fluid.reset')
            call('fluid.set_grain_settings', domain=DOMAIN, stiffness_n_m=100000.,
                 twisting_friction=.1, normal_damping_n_s_m=args.settle_normal_damping)
            call('flow_source.update', name=SOURCE, position=[0, .65, 0], radius=.43,
                 max_emitted_particles=256, fluid_particles_per_second=256*1.01*60,
                 start_time=0., end_time=1/30, enabled=True)
            control = call('sim.control_state')
            data['settle_profile'] = call('fluid.grain_settings', domain=DOMAIN)
            data['settle'] = []
            for step in range(1, 481):
                call('fluid.step', dt=1/60)
                if step == 1 or step % 30 == 0:
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    sample = {'step': step, 'inventory': inventory,
                              'stats': call('fluid.step_stats', domain=DOMAIN)}
                    data['settle'].append(sample)
                    save()
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    g = inventory['acceptance_metrics']['granular']
                    assert g['particles'] == 256
                    assert g['bounds_min'][1] >= .025*.7
                    print('settle', step, 'KE', g['kinetic_energy_j'],
                          'spin_energy', inventory['grain_diagnostics']['spin_energy_j'],
                          'runtime', runtime_of(inventory), flush=True)
            samples = data['settle']
            masses = [s['inventory']['acceptance_metrics']['granular']['dry_mass_kg'] for s in samples]
            tail = [s for s in samples if s['step'] >= 360]
            com = [s['inventory']['acceptance_metrics']['granular']['dry_center_of_mass'][1] for s in tail]
            rms = [s['inventory']['acceptance_metrics']['granular']['horizontal_rms_radius_m'] for s in tail]
            energy = [s['inventory']['acceptance_metrics']['granular']['kinetic_energy_j'] +
                      s['inventory']['grain_diagnostics']['spin_energy_j'] for s in tail]
            data['settle_gates'] = {'mass_drift_kg': max(masses)-min(masses),
                'tail_com_range_m': max(com)-min(com), 'tail_rms_range_m': max(rms)-min(rms),
                'tail_energy_per_grain_j': max(energy)/256}
            data['completed'] = (max(masses)-min(masses) <= 1e-5 and
                                 max(com)-min(com) <= .001 and max(rms)-min(rms) <= .001 and
                                 max(energy)/256 <= 1e-5)
            save()
            assert data['completed'], data['settle_gates']
            return
        if args.extended_only:
            # A real transformed flat TriangleMesh, not an analytic test plane.
            ramp = call('scene.add_primitive', type='plane', name='H1_Grain_Flat_Ramp', size=1.)
            material = call('material.create', type='principled', name='H1_Grain_Ramp_Material')
            call('material.assign', object_name=ramp, material_name=material)
            call('material.set', object_name=ramp, param='base_color', value=[.18, .22, .28])
            call('scene.set_transform', name=ramp, translation=[0, 1., 0], rotation=[0, 0, 20])
            ramp_collider = 'H1_Grain_Flat_Ramp_Collider'
            exists = ramp_collider in {c['name'] for c in call('collider.list')['colliders']}
            call('collider.update' if exists else 'collider.create', name=ramp_collider,
                 source_mode='mesh_bvh', source_object=ramp, enabled=True)
            call('fluid.reset')
            call('flow_source.update', name=SOURCE, position=[0, 1.08, 0],
                 radius=.0001, velocity=[0, 0, 0], enabled=True)
            control = call('sim.control_state')
            samples = []
            data['flat_ramp'] = {'object': ramp, 'collider': ramp_collider, 'samples': samples}
            angle = math.radians(20)
            for step in range(1, 43):
                call('fluid.step', dt=1/120)
                assert call('sim.control_state') == control
                inventory = call('fluid.matter_models', domain=DOMAIN)
                samples.append(inventory)
                save()
                assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                g = inventory['acceptance_metrics']['granular']
                assert g['particles'] == 1
                x, y, _ = g['dry_center_of_mass']
                distance = -math.sin(angle)*x+math.cos(angle)*(y-1.)
                assert distance >= .025*.7, ('ramp penetration', step, distance)
            g = samples[-1]['acceptance_metrics']['granular']
            assert g['dry_center_of_mass'][0] < -.005, 'grain did not travel down flat ramp'
            assert max(s['grain_diagnostics']['max_spin_rad_s'] for s in samples) > .1
            data['flat_ramp']['passed'] = True
            call('collider.update', name=ramp_collider, enabled=False)
            print('PASS flat mesh ramp', g['dry_center_of_mass'], flush=True)
            # Two independent flat supports must react in the same microstep.
            # Zero sliding isolates normal contact stiffness from friction.
            for name, rotation in [('Floor', [0, 0, 0]), ('Wall', [0, 0, 90])]:
                obj = call('scene.add_primitive', type='plane',
                           name='H1_Grain_Corner_'+name, size=1.)
                call('material.assign', object_name=obj, material_name=material)
                call('scene.set_transform', name=obj, translation=[0, 1., 0], rotation=rotation)
                collider = 'H1_Grain_Corner_'+name+'_Collider'
                exists = collider in {c['name'] for c in call('collider.list')['colliders']}
                call('collider.update' if exists else 'collider.create', name=collider,
                     source_mode='mesh_bvh', source_object=obj, enabled=True)
                extra_colliders.append(collider)
            call('fluid.reset')
            call('fluid.set_grain_settings', domain=DOMAIN, sliding_damping_n_s_m=0.,
                 friction=0., rolling_friction=0.)
            call('flow_source.update', name=SOURCE, position=[.1, 1.1, 0],
                 velocity=[-2., -2., 0], enabled=True)
            control = call('sim.control_state')
            data['flat_corner'] = []
            for step in range(1, 13):
                call('fluid.step', dt=1/120)
                assert call('sim.control_state') == control
                inventory = call('fluid.matter_models', domain=DOMAIN)
                data['flat_corner'].append(inventory)
                save()
                assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                center = inventory['acceptance_metrics']['granular']['dry_center_of_mass']
                assert min(center[0], center[1]-1.) >= .025*.7, 'flat corner penetration >30% radius'
            for collider in extra_colliders:
                call('collider.update', name=collider, enabled=False)
            call('fluid.reset')
            call('fluid.set_grain_settings', domain=DOMAIN, sliding_damping_n_s_m=4.,
                 friction=.5, rolling_friction=.02)
            print('PASS flat corner', flush=True)
            if args.corner_only:
                data['completed'] = True
                return
            data['density'] = []
            for count in (256, 1024):
                call('fluid.reset')
                call('fluid.set_grain_settings', domain=DOMAIN, stiffness_n_m=100000.)
                call('flow_source.update', name=SOURCE, position=[0, .65, 0], radius=.43,
                     max_emitted_particles=count, fluid_particles_per_second=count*1.01*60,
                     start_time=0., end_time=1/30, velocity=[0, 0, 0], enabled=True)
                control = call('sim.control_state')
                arm = {'count': count, 'samples': []}
                data['density'].append(arm)
                for step in range(1, 37):
                    call('fluid.step', dt=1/60)
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    arm['samples'].append(inventory)
                    save()
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    g = inventory['acceptance_metrics']['granular']
                    assert g['particles'] == count, 'birth budget did not fill requested volume'
                    assert g['bounds_min'][1] >= .025*.7, 'dense pile floor penetration'
                masses = [s['acceptance_metrics']['granular']['dry_mass_kg'] for s in arm['samples']]
                assert max(masses)-min(masses) <= 1e-5
                arm['step_stats'] = call('fluid.step_stats', domain=DOMAIN)
                arm['runtime'] = runtime_of(arm['samples'][-1])
                arm['passed'] = True
                print('PASS grain density', count, arm['step_stats'], flush=True)
            data['completed'] = True
            return
        if not args.pile_only:
            for dt in (1/60, 1/120):
                call('flow_source.update', name=SOURCE, enabled=True)
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                control = call('sim.control_state')
                samples = []
                for step in range(1, round(.2/dt)+1):
                    call('fluid.step', dt=dt)
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    assert 'Dry grain Vulkan' in inventory['mixed_execution']['status']
                    assert not inventory['mixed_execution']['pressure_on_gpu']
                    assert inventory['grain_diagnostics']['transport_owner'] == 'grain'
                    g = inventory['acceptance_metrics']['granular']
                    assert g['particles'] == 1 and g['pore_water_kg'] == 0
                    samples.append({'time': step*dt, 'granular': g})
                first, mid, last = samples[0], samples[len(samples)//2-1], samples[-1]
                y0 = first['granular']['dry_center_of_mass'][1]
                tm, tf = mid['time']-first['time'], last['time']-first['time']
                acceleration = -2*((last['granular']['dry_center_of_mass'][1]-y0)/tf-
                    (mid['granular']['dry_center_of_mass'][1]-y0)/tm)/(tf-tm)
                assert abs(acceleration-9.81) < .05*9.81, acceleration
                masses = [s['granular']['dry_mass_kg'] for s in samples]
                assert max(masses)-min(masses) <= 1e-7
                data['arms'].append({'dt': dt, 'acceleration': acceleration, 'samples': samples,
                                     'step_stats': call('fluid.step_stats', domain=DOMAIN)})
                save()
                print('PASS grain free-fall', dt, acceleration, flush=True)
            # Closed-floor collision: one actual carrier, not a rendered replay.
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('flow_source.update', name=SOURCE, position=[0, .12, 0],
                 velocity=[.3, -.5, 0], enabled=True)
            control = call('sim.control_state')
            samples = []
            for step in range(1, 91):
                call('fluid.step', dt=1/120)
                assert call('sim.control_state') == control
                inventory = call('fluid.matter_models', domain=DOMAIN)
                assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                g = inventory['acceptance_metrics']['granular']
                assert g['particles'] == 1
                assert g['bounds_min'][1] >= .025*.8, 'excess floor penetration'
                samples.append({'step': step, 'granular': g, 'grain': inventory['grain_diagnostics']})
            assert max(s['grain']['max_spin_rad_s'] for s in samples) > .01, 'no contact spin'
            data['floor'] = samples
        # Real multi-grain contacts, two outer timesteps with identical source.
        pile_arms = []
        for dt in (1/60, 1/120):
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('flow_source.update', name=SOURCE, position=[0, .25, 0], radius=.16,
                 velocity=[0, -.2, 0], max_emitted_particles=64,
                 fluid_particles_per_second=64*1.01/dt, enabled=True, end_time=dt*2)
            control = call('sim.control_state')
            pile = []
            pile_arms.append({'dt': dt, 'samples': pile})
            data['pile'] = pile_arms
            for step in range(1, round(.6/dt)+1):
                call('fluid.step', dt=dt)
                assert call('sim.control_state') == control
                inventory = call('fluid.matter_models', domain=DOMAIN)
                assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                g = inventory['acceptance_metrics']['granular']
                pile.append({'step': step, 'granular': g, 'grain': inventory['grain_diagnostics']})
                save()
                assert g['particles'] == 64
                assert g['bounds_min'][1] >= .025*.7, 'pile penetration exceeds 30% radius'
            assert max(x['granular']['dry_mass_kg'] for x in pile)-min(
                x['granular']['dry_mass_kg'] for x in pile) <= 1e-6
            assert max(x['grain']['max_spin_rad_s'] for x in pile) > .01
            pile_arms[-1]['step_stats'] = call('fluid.step_stats', domain=DOMAIN)
            data['pile'] = pile_arms
            save()
        a, b = [arm['samples'][-1]['granular'] for arm in pile_arms]
        data['pile_dt_com_relative'] = abs(a['dry_center_of_mass'][1]-b['dry_center_of_mass'][1]) / max(
            b['dry_center_of_mass'][1], .025)
        data['pile_dt_rms_relative'] = abs(a['horizontal_rms_radius_m']-b['horizontal_rms_radius_m']) / max(
            b['horizontal_rms_radius_m'], .025)
        assert data['pile_dt_com_relative'] <= .1, 'pile COM dt sensitivity >10%'
        assert data['pile_dt_rms_relative'] <= .1, 'pile spread dt sensitivity >10%'
        data['completed'] = True
    finally:
        for collider in extra_colliders:
            call('collider.update', name=collider, enabled=False)
        if ramp_collider:
            call('collider.update', name=ramp_collider, enabled=False)
        if source_ready:
            call('flow_source.update', name=SOURCE, domain=DOMAIN, enabled=False)
        if repose_ready:
            call('fluid.set_param', domain='H1_Grain_Repose', enabled=False, visible=False)
        if domain_ready:
            call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=c['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'], visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        data['final_control'] = call('sim.control_state')
        data['restored_authoring'] = True
        save()
        client.close()


if __name__ == '__main__':
    main()
