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
from grain_units import restitution_of

DOMAIN = 'H1_Grain_Runtime'
SOURCE = 'H1_Grain_Runtime_Source'
LOG = Path(__file__).resolve().parents[2] / 'docs/dev/matter_h1_grain_runtime_live.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pile-only", action="store_true")
    parser.add_argument("--extended-only", action="store_true")
    parser.add_argument("--corner-only", action="store_true")
    parser.add_argument("--settle-only", action="store_true")
    # Old normal damping (N s/m) at k = 1e5, converted to restitution.
    parser.add_argument("--settle-normal-damping", type=float, default=4.)
    parser.add_argument("--static-only", action="store_true",
                        help="slope hold: static friction + rolling spring vs kinetic-only")
    parser.add_argument("--convergence-only", action="store_true",
                        help="pile at contact_resolution 24 vs 48 (real substep-dt halving)")
    parser.add_argument("--repose-only", action="store_true",
                        help="poured free-standing pile: repose angle, mu_r sensitivity, grain-size convergence")
    parser.add_argument("--history-only", action="store_true",
                        help="device contact history: carried while the state is unchanged, reset after fluid.reset")
    parser.add_argument("--porous-only", action="store_true",
                        help="B5 volume exclusion A/B: grains poured into a pool raise the water by N V / A")
    parser.add_argument("--wet-only", action="store_true",
                        help="B6 wet grains: cohesion A/B (column collapse) and water absorption from a pool")
    parser.add_argument("--coexist-only", action="store_true",
                        help="water + grains in one domain: buoyancy/drag A/B and a water pour onto a pile")
    parser.add_argument("--readiness-only", action="store_true",
                        help="grain blockers: readiness report, API edits that would break a grain domain are "
                             "rejected unchanged, old grain keys name their replacement")
    parser.add_argument("--coexist-hydrostatic", action="store_true",
                        help="with --coexist-only: volume exclusion off, so the immersed arm uses "
                             "hydrostatic buoyancy instead of the projection's pressure gradient")
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
    domain_ready = source_ready = repose_ready = coexist_ready = False
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
             stiffness_n_m=20000., restitution=restitution_of(4., 20000.), sliding_damping_n_s_m=4.,
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

        if args.readiness_only:
            # One rule (matterGrainBlockers) behind validation, the panel locks and
            # matter_models: a grain domain cannot be moved off what the solver needs.
            call('fluid.reset')
            readiness = call('fluid.matter_models', domain=DOMAIN)['grain_readiness']
            data['readiness'] = {'before': readiness}
            assert readiness['enabled'] and readiness['ready'], readiness
            rejected = {}
            for method, params in [('fluid.set_param', dict(domain=DOMAIN, backend='cpu')),
                                   ('fluid.set_param', dict(domain=DOMAIN, boundary='open')),
                                   ('fluid.set_pore_exchange', dict(domain=DOMAIN, enabled=True)),
                                   ('fluid.set_grain_settings', dict(domain=DOMAIN, normal_damping_n_s_m=4.)),
                                   ('fluid.set_grain_settings', dict(domain=DOMAIN, drag_viscosity_pa_s=1e-3)),
                                   ('fluid.set_grain_settings', dict(domain=DOMAIN, solver_kind='xpbd'))]:
                try:
                    call(method, **params)
                    rejected[f'{method} {sorted(params)}'] = None
                except RuntimeError as error:
                    rejected[f'{method} {sorted(params)}'] = str(error)
            data['readiness']['rejected'] = rejected
            after = call('fluid.matter_models', domain=DOMAIN)['grain_readiness']
            data['readiness']['after'] = after
            save()
            for name, message in rejected.items():
                print('rejected' if message else 'ACCEPTED', name, (message or '')[:120], flush=True)
            assert all(rejected.values()), ('an edit that breaks the grain domain was accepted', rejected)
            assert after['ready'] and not after['blockers'], ('a rejected edit still changed the domain', after)
            assert 'restitution' in rejected[next(k for k in rejected if 'normal_damping' in k)]
            print('PASS grain readiness', flush=True)
            data['completed'] = True
            return
        if args.history_only:
            # Contact springs live on the device and are valid only for the
            # state they were published with. Before B9a a fluid.reset (ids
            # restart at 1) let new grains inherit the old grains' springs.
            data['history'] = []
            for attempt in range(2):
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                call('flow_source.update', name=SOURCE, position=[0, .25, 0], radius=.16,
                     velocity=[0, -.2, 0], max_emitted_particles=64,
                     fluid_particles_per_second=64*1.01*60, enabled=True, end_time=2/60)
                control = call('sim.control_state')
                runs = []
                data['history'].append(runs)
                for step in range(1, 31):
                    call('fluid.step', dt=1/60)
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    runs.append({'step': step, 'runtime': runtime_of(inventory)})
                save()
                first = runs[0]['runtime']
                assert first['history_reset_this_step'], ('fresh run reused old springs', attempt, first)
                print('history attempt', attempt, 'first step reason', first['history_reset_reason'],
                      flush=True)
                if attempt == 1:
                    assert first['history_reset_reason'] in ('host_state_changed', 'allocation'), first
                carried = [r for r in runs[3:] if r['runtime']['history_reset_this_step']]
                assert not carried, ('history reset while the state was unchanged', carried[:2])
                # B8: grains are re-ordered by cell every frame; falling grains
                # change cells, so the device gather must have run.
                remapped = [r for r in runs[1:] if r['runtime'].get('history_remapped')]
                assert remapped, 'cell order never changed: remap path untested'
                data['history_remapped_steps'] = len(remapped)
            data['completed'] = True
            return
        if args.coexist_only or args.porous_only or args.wet_only:
            # One Matter domain, two transport owners: Sand grains (DEM) and
            # Water parcels (liquid lane). Before this batch any liquid
            # carrier held the whole grain step: parcels stayed at the emitter.
            COEXIST = 'H1_Grain_Coexist'
            WATER = 'H1_Grain_Coexist_Water'
            if COEXIST not in {d['name'] for d in call('fluid.list_domains')['domains']}:
                call('fluid.create_domain', name=COEXIST, type='matter', domain_min=[-.4, 0, -.4],
                     domain_max=[.4, .8, .4], voxel_size=.05)
            call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
            call('fluid.set_param', domain=COEXIST, enabled=True, visible=False, backend='vulkan',
                 boundary='closed', preset='sand', granular_enabled=True, solid_phase=False,
                 thermal_liquid_enabled=False, max_particles=40000)
            call('fluid.set_pore_exchange', domain=COEXIST, enabled=False,
                 wet_response_enabled=False, wet_appearance_enabled=False)
            names = {x['name'] for x in call('flow_source.list')}
            call('flow_source.update' if WATER in names else 'flow_source.create', name=WATER,
                 domain=COEXIST, enabled=False, phase='liquid', source_mode='point',
                 fluid_substance='Water', initial_constitutive_model='fluid', position=[0, .2, 0],
                 radius=.2, velocity=[0, 0, 0], fluid_velocity_spread=0.,
                 fluid_particles_per_second=24000., fluid_temperature_override=True,
                 fluid_temperature_kelvin=293.15, use_time_limit=True, start_time=0., end_time=.5,
                 use_particle_limit=True, max_emitted_particles=12000)
            coexist_ready = True
            call('flow_source.update', name=SOURCE, domain=COEXIST, enabled=False)
            data['coexist'] = {}

            def model_totals(inventory):
                return {m['model']: m for m in inventory['models']}

            def liquid_of(inventory):
                return inventory['grain_diagnostics']['liquid']

            if args.wet_only:
                # (1) Cohesion A/B without liquid: grains born half saturated,
                # standing for .5 mm sand (Bond-number scale 2500): bridges
                # must hold the collapsed column together; dry it spreads.
                arms = {}
                for wet in (False, True):
                    call('timeline.set_frame', frame=0)
                    call('fluid.reset')
                    call('fluid.set_grain_settings', domain=COEXIST, enabled=True, radius_m=.025,
                         stiffness_n_m=100000., restitution=restitution_of(8., 100000.), sliding_damping_n_s_m=4.,
                         friction=.5, rolling_friction=.1, twisting_friction=.1, max_substeps=2048,
                         tangential_stiffness_ratio=2/7, contact_resolution=24, packing_fraction=.6,
                         fluid_coupling=True, volume_exclusion=True, wet_grains=wet,
                         water_capacity_fraction=.05, absorption_rate_per_s=4., drying_rate_per_s=0.,
                         represented_grain_radius_m=.0005 if wet else 0., birth_saturation=.5 if wet else 0.)
                    call('flow_source.update', name=WATER, enabled=False)
                    # Births never overlap (the birth filter rejects them): a
                    # .25 m ball holds ~380 grains at once, so 512 are born over
                    # 1 s as the first ones fall away (live: 512 in 1/30 s never
                    # reached the count).
                    call('flow_source.update', name=SOURCE, domain=COEXIST, position=[0, .45, 0],
                         radius=.25, velocity=[0, 0, 0], max_emitted_particles=512,
                         use_particle_limit=True, use_time_limit=True,
                         fluid_particles_per_second=1024., start_time=0., end_time=1.,
                         enabled=True)
                    control = call('sim.control_state')
                    arm = {'wet': wet, 'samples': []}
                    arms[str(wet)] = arm
                    for step in range(1, 241):
                        call('fluid.step', dt=1/60)
                        if step % 15:
                            continue
                        assert call('sim.control_state') == control
                        inventory = call('fluid.matter_models', domain=COEXIST)
                        assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                        g = inventory['acceptance_metrics']['granular']
                        if step < 75:
                            continue  # still being born
                        arm['samples'].append({'step': step, 'granular': g,
                            'wet': liquid_of(inventory)['wet'], 'runtime': runtime_of(inventory)})
                        assert g['particles'] == 512, ('wet' if wet else 'dry', step, g['particles'])
                        assert g['bounds_min'][1] >= .025*.7, ('floor penetration', wet, step,
                                                                g['bounds_min'])
                    data['coexist']['cohesion'] = arms
                    save()
                    last = arm['samples'][-1]
                    print('cohesion wet' if wet else 'cohesion dry', 'rms', last['granular']['horizontal_rms_radius_m'],
                          'bridges', last['wet']['liquid_bridges_last_substep'], flush=True)
                    water = [x['granular']['pore_water_kg'] for x in arm['samples']]
                    assert max(water)-min(water) <= 1e-6, ('held water changed without liquid/drying', water)
                    if wet:
                        assert last['wet']['liquid_bridges_last_substep'] > 0, last['wet']
                        assert water[-1] > 0
                    else:
                        assert last['wet']['liquid_bridges_last_substep'] == 0, last['wet']
                rms = {k: v['samples'][-1]['granular']['horizontal_rms_radius_m'] for k, v in arms.items()}
                data['coexist']['cohesion_rms'] = rms
                save()
                assert rms['True'] <= .8*rms['False'], ('liquid bridges did not hold the column', rms)
                # (2) Absorption: grains poured into a pool take water; liquid
                # mass + grain water stays constant (no drying).
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                call('fluid.set_grain_settings', domain=COEXIST, represented_grain_radius_m=0.,
                     birth_saturation=0., wet_grains=True)
                call('flow_source.update', name=WATER, position=[0, .2, 0], radius=.2,
                     fluid_particles_per_second=24000., start_time=0., end_time=.5,
                     max_emitted_particles=12000, enabled=True)
                call('flow_source.update', name=SOURCE, domain=COEXIST, position=[0, .6, 0], radius=.2,
                     max_emitted_particles=256, fluid_particles_per_second=512., start_time=1.,
                     end_time=1.6, enabled=True)
                control = call('sim.control_state')
                absorb = []
                data['coexist']['absorption'] = absorb
                for step in range(1, 241):
                    call('fluid.step', dt=1/60)
                    if step % 20:
                        continue
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=COEXIST)
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    g = inventory['acceptance_metrics']['granular']
                    absorb.append({'step': step, 'grain_water_kg': g['pore_water_kg'],
                        'liquid_kg': model_totals(inventory)['fluid']['mass_kg'],
                        'wet': liquid_of(inventory)['wet']})
                    save()
                    assert abs(absorb[-1]['wet']['water_balance_error_kg']) <= 1e-5, absorb[-1]['wet']
                after = [x for x in absorb if x['step'] >= 40]  # water emission ended at .5 s
                totals = [x['liquid_kg'] + x['grain_water_kg'] for x in after]
                data['coexist']['absorption_gates'] = {'grain_water_kg': after[-1]['grain_water_kg'],
                    'total_water_drift_kg': max(totals)-min(totals)}
                save()
                print('absorption', json.dumps(data['coexist']['absorption_gates']), flush=True)
                assert after[-1]['grain_water_kg'] > 0, 'submerged grains did not absorb water'
                assert max(totals)-min(totals) <= 1e-4*max(totals), ('water not conserved', totals)
                call('flow_source.update', name=WATER, enabled=False)
                call('flow_source.update', name=SOURCE, domain=DOMAIN, enabled=False)
                data['completed'] = True
                return
            if args.porous_only:
                # Archimedes by displacement: N grains sunk in a closed pool of
                # floor area A raise the free surface by N V / A only if the
                # liquid's projection sees their volume. Without it the pile is
                # a ghost to continuity and the surface stays put.
                count, area = 1000, .8*.8
                rise = count*4/3*math.pi*.025**3/area
                data['coexist']['porous'] = {'expected_rise_m': rise}
                for exclude in (True, False):
                    call('timeline.set_frame', frame=0)
                    call('fluid.reset')
                    call('fluid.set_grain_settings', domain=COEXIST, enabled=True, radius_m=.025,
                         stiffness_n_m=100000., restitution=restitution_of(8., 100000.), sliding_damping_n_s_m=4.,
                         friction=.5, rolling_friction=.1, twisting_friction=.1, max_substeps=2048,
                         tangential_stiffness_ratio=2/7, contact_resolution=24, packing_fraction=.6,
                         fluid_coupling=True, volume_exclusion=exclude)
                    call('flow_source.update', name=WATER, position=[0, .2, 0], radius=.2,
                         fluid_particles_per_second=24000., start_time=0., end_time=.5,
                         max_emitted_particles=12000, enabled=True)
                    call('flow_source.update', name=SOURCE, domain=COEXIST, position=[0, .6, 0],
                         radius=.25, velocity=[0, 0, 0], max_emitted_particles=count,
                         use_particle_limit=True, use_time_limit=True,
                         fluid_particles_per_second=count/.5, start_time=1.5, end_time=2.5,
                         enabled=True)
                    control = call('sim.control_state')
                    arm = {'volume_exclusion': exclude, 'samples': []}
                    data['coexist']['porous'][str(exclude)] = arm
                    for step in range(1, 301):
                        call('fluid.step', dt=1/60)
                        if step % 10:
                            continue
                        assert call('sim.control_state') == control
                        inventory = call('fluid.matter_models', domain=COEXIST)
                        assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                        liquid = inventory['grain_diagnostics']['liquid']
                        g = inventory['acceptance_metrics']['granular']
                        arm['samples'].append({'step': step, 'liquid': liquid, 'granular': g,
                            'fluid': model_totals(inventory)['fluid'],
                            'status': inventory['mixed_execution']['status']})
                        save()
                    before = [x for x in arm['samples'] if x['step'] == 80][0]
                    after = arm['samples'][-1]
                    assert after['granular']['particles'] == count, after['granular']
                    surface = (after['liquid']['shape']['surface_p95_m'] -
                               before['liquid']['shape']['surface_p95_m'])
                    arm['surface_rise_m'] = surface
                    arm['grain_floor_m'] = after['granular']['bounds_min'][1]
                    save()
                    print('porous', exclude, 'surface rise', surface, 'expected', rise, flush=True)
                    assert after['granular']['bounds_min'][1] >= .025*.7, 'pile floor penetration'
                    assert after['liquid']['volume_exclusion'] == exclude, after['liquid']
                    if exclude:
                        assert after['liquid']['pressure_force'], after['liquid']
                        assert after['liquid']['porous_cells'] > 0, after['liquid']
                # The pool's own settling drift (seen live: +.041 m with the
                # exclusion off) is common to both arms; only the difference is
                # the grains' displacement.
                porous = data['coexist']['porous']
                displaced = porous['True']['surface_rise_m'] - porous['False']['surface_rise_m']
                porous['displacement_m'] = displaced
                save()
                print('porous displacement (on - off)', displaced, 'expected', rise, flush=True)
                assert .5*rise <= displaced <= 1.5*rise, ('displacement not seen', displaced, rise)
                call('flow_source.update', name=WATER, enabled=False)
                call('flow_source.update', name=SOURCE, domain=DOMAIN, enabled=False)
                data['completed'] = True
                return

            # 1. Immersed release, coupling on vs off. A grain born at rest in
            # still water starts with the buoyancy-reduced acceleration
            # g' = g (1 - rho_w V / m) (drag ~0 at rest); without coupling, g.
            # This is the analytic gate for the liquid sample, the submerged
            # fraction and the force path; the A/B arm proves the instrument.
            water_density = 1000.
            volume = 4/3*math.pi*.025**3
            expected = {True: 9.81*(1-water_density*volume/expected_grain_mass), False: 9.81}
            for coupled in (True, False):
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                call('fluid.set_grain_settings', domain=COEXIST, enabled=True, radius_m=.025,
                     stiffness_n_m=20000., restitution=restitution_of(4., 20000.), sliding_damping_n_s_m=4.,
                     friction=.5, rolling_friction=.02, twisting_friction=0., max_substeps=1024,
                     tangential_stiffness_ratio=2/7, contact_resolution=24, packing_fraction=.6,
                     fluid_coupling=coupled,
                     volume_exclusion=not args.coexist_hydrostatic)
                call('flow_source.update', name=WATER, enabled=True)
                call('flow_source.update', name=SOURCE, domain=COEXIST, position=[0, .15, 0],
                     radius=.0001, velocity=[0, 0, 0], max_emitted_particles=1,
                     use_time_limit=True, start_time=1.5, end_time=1.55, enabled=True)
                control = call('sim.control_state')
                arm = {'coupled': coupled, 'samples': []}
                data['coexist'][f'immersed_{coupled}'] = arm
                for step in range(1, 241):
                    call('fluid.step', dt=1/120)
                    if step < 170 and step % 30:
                        continue
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=COEXIST)
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    g = inventory['acceptance_metrics']['granular']
                    totals = model_totals(inventory)
                    arm['samples'].append({'step': step, 'time': step/120, 'granular': g,
                        'fluid': totals['fluid'], 'liquid': liquid_of(inventory),
                        'status': inventory['mixed_execution']['status']})
                    if g['particles'] == 1 and coupled:
                        # Per-frame upward impulse the grain got from the liquid,
                        # against Archimedes rho V g dt (same for both models).
                        liquid = liquid_of(inventory)
                        print('  t', round(step/120, 4), 'y', round(g['dry_center_of_mass'][1], 5),
                              'pressure_y', liquid['pressure_impulse_n_s'][1],
                              'buoyancy_y', liquid['buoyancy_impulse_n_s'][1],
                              'drag_y', liquid['drag_impulse_n_s'][1],
                              'archimedes', round(water_density*volume*9.81/120, 6),
                              'submerged', liquid['max_submerged_fraction'],
                              'porous_cells', liquid['porous_cells'],
                              'status', inventory['mixed_execution']['status'], flush=True)
                    if g['particles'] == 1 and len([x for x in arm['samples']
                                                    if x['granular']['particles'] == 1]) >= 10:
                        break
                save()
                born = [x for x in arm['samples'] if x['granular']['particles'] == 1]
                assert len(born) >= 6, ('grain was not born under water', arm['samples'][-1])
                assert born[0]['fluid']['particles'] >= 6000, ('water pool missing', born[0]['fluid'])
                # Quadratic fit y = y0 + v0 t - a t^2 / 2 over the first 6 samples.
                pts = [(x['time']-born[0]['time'], x['granular']['dry_center_of_mass'][1])
                       for x in born[:6]]
                n = len(pts)
                sums = [[sum(t**(i+j) for t, _ in pts) for j in range(3)] for i in range(3)]
                rhs = [sum(y*t**i for t, y in pts) for i in range(3)]
                a00, a01, a02 = sums[0]; a10, a11, a12 = sums[1]; a20, a21, a22 = sums[2]
                det = (a00*(a11*a22-a12*a21)-a01*(a10*a22-a12*a20)+a02*(a10*a21-a11*a20))
                c2 = (a00*(a11*rhs[2]-rhs[1]*a21)-a01*(a10*rhs[2]-rhs[1]*a20)+
                      rhs[0]*(a10*a21-a11*a20))/det
                acceleration = -2*c2
                arm['acceleration'] = acceleration
                arm['expected'] = expected[coupled]
                save()
                print('coexist immersed coupled' if coupled else 'coexist immersed uncoupled',
                      'a', acceleration, 'expected', expected[coupled], flush=True)
                tolerance = .15 if coupled else .05
                assert abs(acceleration-expected[coupled]) <= tolerance*expected[coupled], \
                    ('immersed grain acceleration', coupled, acceleration, expected[coupled])
                if coupled:
                    liquid = born[2]['liquid']
                    assert liquid['coupled_grains'] == 1 and liquid['max_submerged_fraction'] >= .8, liquid
                    assert 'Grain + liquid' in born[2]['status'], born[2]['status']
            call('flow_source.update', name=WATER, enabled=False)
            call('flow_source.update', name=SOURCE, enabled=False)

            # 2. The reported case: water poured onto a resting grain pile in
            # the same domain. The step must not hold, the water must fall,
            # both owners keep their mass, and every exchanged impulse lands.
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('fluid.set_grain_settings', domain=COEXIST, stiffness_n_m=100000.,
                 restitution=restitution_of(8., 100000.), twisting_friction=.1, fluid_coupling=True)
            call('flow_source.update', name=SOURCE, domain=COEXIST, position=[0, .35, 0],
                 radius=.28, velocity=[0, 0, 0], max_emitted_particles=256,
                 fluid_particles_per_second=256*1.01*60, use_time_limit=True,
                 start_time=0., end_time=1/30, enabled=True)
            call('flow_source.update', name=WATER, position=[0, .6, 0], radius=.12,
                 fluid_particles_per_second=12000., start_time=1., end_time=1.5,
                 max_emitted_particles=6000, enabled=True)
            control = call('sim.control_state')
            pour = []
            data['coexist']['pour'] = pour
            for step in range(1, 181):
                call('fluid.step', dt=1/60)
                if step % 10:
                    continue
                assert call('sim.control_state') == control
                inventory = call('fluid.matter_models', domain=COEXIST)
                assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                g = inventory['acceptance_metrics']['granular']
                totals = model_totals(inventory)
                liquid = liquid_of(inventory)
                pour.append({'step': step, 'granular': g, 'fluid': totals['fluid'],
                             'liquid': liquid, 'runtime': runtime_of(inventory),
                             'status': inventory['mixed_execution']['status']})
                save()
                assert g['particles'] == 256, g['particles']
                assert g['bounds_min'][1] >= .025*.7, ('pile floor penetration under water', step)
                exchanged = sum(abs(v) for v in liquid['drag_impulse_n_s']) + \
                    sum(abs(v) for v in liquid['buoyancy_impulse_n_s'])
                assert liquid['momentum_residual_n_s'] <= 1e-3*exchanged + 1e-6, liquid
                assert liquid['unmatched_impulse_n_s'] <= 1e-9, liquid
            after = [x for x in pour if x['step'] >= 100]
            fluid_mass = [x['fluid']['mass_kg'] for x in after]
            grain_mass = [x['granular']['dry_mass_kg'] for x in pour]
            data['coexist']['pour_gates'] = {
                'water_particles': after[-1]['fluid']['particles'],
                'water_mass_drift_kg': max(fluid_mass)-min(fluid_mass),
                'grain_mass_drift_kg': max(grain_mass)-min(grain_mass),
                'max_coupled_grains': max(x['liquid']['coupled_grains'] for x in pour),
                'water_fell': min(x['fluid']['momentum_kg_m_s'][1] for x in pour
                                  if x['fluid']['particles'] > 0)}
            save()
            print('coexist pour', json.dumps(data['coexist']['pour_gates']), flush=True)
            gates = data['coexist']['pour_gates']
            assert gates['water_particles'] >= 3000, gates
            assert gates['water_mass_drift_kg'] <= 1e-6*max(fluid_mass), gates
            assert gates['grain_mass_drift_kg'] <= 1e-6, gates
            assert gates['water_fell'] < -.01, ('water did not move: parcels stuck at the emitter', gates)
            assert gates['max_coupled_grains'] > 0, ('water never reached the pile', gates)
            call('flow_source.update', name=WATER, enabled=False)
            call('flow_source.update', name=SOURCE, domain=DOMAIN, enabled=False)
            data['completed'] = True
            return
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
                     stiffness_n_m=100000.*scale, restitution=restitution_of(8.*scale**2, 100000.*scale, radius),
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
                     max_substeps=4096, restitution=restitution_of(1., 20000.), sliding_damping_n_s_m=1.)
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
                 twisting_friction=.1,
                 restitution=restitution_of(args.settle_normal_damping, 100000.))
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
            # Settled = the centres rest. A grain left spinning about its contact
            # normal decays only through the twisting torque mu_t Fn a on a patch
            # of a = sqrt(R overlap) ~ .65 mm (2.5 rad/s^2 for this grain): a top
            # on a table spins for seconds, which is physics, not an unsettled
            # pile. So spin is gated as dissipative (never grows), not as zero.
            energy = [s['inventory']['acceptance_metrics']['granular']['kinetic_energy_j'] for s in tail]
            spin = [s['inventory']['grain_diagnostics']['spin_energy_j'] for s in tail]
            data['settle_gates'] = {'mass_drift_kg': max(masses)-min(masses),
                'tail_com_range_m': max(com)-min(com), 'tail_rms_range_m': max(rms)-min(rms),
                'tail_energy_per_grain_j': max(energy)/256, 'tail_spin_j': spin,
                'spin_grows': any(b > a*(1+1e-3)+1e-12 for a, b in zip(spin, spin[1:]))}
            # B4: a settled pile with no births and no liquid is exactly what
            # the runtime published, so bank 0 must be reused (no state upload).
            # Never resident = some host stage touches grains every frame.
            resident = [s['inventory']['grain_diagnostics']['runtime'].get('state_resident')
                        for s in tail]
            data['settle_gates']['resident_tail_samples'] = sum(1 for r in resident if r)
            data['settle_gates']['tail_upload_bytes'] = [
                s['inventory']['grain_diagnostics']['runtime'].get('upload_bytes') for s in tail]
            data['completed'] = (max(masses)-min(masses) <= 1e-5 and
                                 max(com)-min(com) <= .001 and max(rms)-min(rms) <= .001 and
                                 max(energy)/256 <= 1e-5 and
                                 not data['settle_gates']['spin_grows'])
            save()
            assert data['completed'], data['settle_gates']
            assert data['settle_gates']['resident_tail_samples'] > 0, \
                ('grain state re-uploaded every frame of a settled pile', data['settle_gates'])
            print('PASS settle', {k: v for k, v in data['settle_gates'].items()
                                  if k not in ('tail_upload_bytes', 'tail_spin_j')}, flush=True)
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
        if coexist_ready:
            call('flow_source.update', name='H1_Grain_Coexist_Water', enabled=False)
            call('fluid.set_param', domain='H1_Grain_Coexist', enabled=False, visible=False)
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
