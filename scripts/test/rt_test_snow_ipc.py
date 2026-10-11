"""Snow acceptance (docs/dev/MADDE_UI_TEK_OTORITE.md section 6, U5).

Three arms of the same drop: a block of snow at 263 K falls 0.6 m onto the floor.

  snow           built-in Snow: compaction hardening + cohesion + rebonding
  no_compaction  Snow with granular_compaction_hardening = 0
  loose          Snow without compaction, cohesion, tensile strength or rebonding

Gates (provisional, the material values are not calibrated yet):
  compaction  snow compacts on impact (plastic volume max > 1.005),
              no_compaction does not (max <= 1.005)
  cohesion    snow spreads less than loose (horizontal RMS radius < 0.8x)
  rest        every arm comes to rest (kinetic energy per particle small)

MPM reads its material from the domain's DEFAULT SUBSTANCE, so each arm points
both the domain and the source at the arm's substance. Restores authoring,
leaves frame 0 paused. No project save, no build.
"""
import json
from pathlib import Path

from rt_ipc import RtIpc
from rt_domain_material import remove_material

DOMAIN = 'Snow_Accept'
SOURCE = 'Snow_Accept_Block'
LOG = Path(__file__).resolve().parents[2] / 'docs/dev/snow_acceptance_live.json'
COUNT = 2000
STEPS = 180


def main():
    client = RtIpc()
    data = {'arms': []}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    def save():
        LOG.write_text(json.dumps(data, indent=2), encoding='utf-8')

    def derive(name, **fields):
        rows = {row['name'] for row in call('substance.list')['substances']}
        if name not in rows:
            call('substance.derive', name=name, based_on='Snow')
        call('substance.set', name=name, fields=fields)
        return name

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    domain_ready = source_ready = False
    heat_field = None
    derived = []
    try:
        # Scene fields off: a Thermal field now heats matter parcels too.
        for field in original['fields']:
            call('forcefield.set_param', field=field['name'], enabled=False)
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=False)
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=False, visible=False)
        if DOMAIN not in {d['name'] for d in original['domains']}:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=.05)
        domain_ready = True
        arms = [('snow', 'Snow'),
                ('no_compaction', derive('T: snow no compaction',
                                         granular_compaction_hardening=0.)),
                ('loose', derive('T: snow loose', granular_compaction_hardening=0.,
                                 granular_cohesion=0., granular_tensile_cutoff=0.,
                                 granular_rebonding=False))]
        derived = [name for _, name in arms[1:]]
        existing = {s['name'] for s in call('flow_source.list')}
        for label, substance in arms:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
                 backend='vulkan', boundary='closed', default_substance=substance,
                 solid_phase=False, thermal_liquid_enabled=False)
            call('flow_source.update' if SOURCE in existing else 'flow_source.create',
                 name=SOURCE, domain=DOMAIN, enabled=True, phase='liquid',
                 source_mode='point', fluid_substance=substance, position=[0, .8, 0],
                 radius=.18, velocity=[0, 0, 0], fluid_velocity_spread=0.,
                 fluid_temperature_override=True, fluid_temperature_kelvin=263.15,
                 use_time_limit=True, use_particle_limit=True,
                 max_emitted_particles=COUNT, start_time=0., end_time=.05,
                 fluid_particles_per_second=COUNT * 30.)
            existing.add(SOURCE)
            source_ready = True
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            arm = {'label': label, 'substance': substance, 'samples': []}
            data['arms'].append(arm)
            for step in range(1, STEPS + 1):
                call('fluid.step', dt=1 / 60)
                if step % 30:
                    continue
                inventory = call('fluid.matter_models', domain=DOMAIN)
                assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                g = inventory['acceptance_metrics']['granular']
                pv = call('attr.stats', scope='domain', id=DOMAIN, name='granular_plastic_volume')
                arm['samples'].append({
                    'step': step, 'particles': g['particles'],
                    'height_m': (g['bounds_max'][1] - g['bounds_min'][1]) if g['bounds_max'] else None,
                    'rms_radius_m': g['horizontal_rms_radius_m'],
                    'kinetic_energy_j': g['kinetic_energy_j'],
                    'plastic_volume': pv})
            last = arm['samples'][-1]
            arm['final'] = last
            save()
            print(label, json.dumps({k: last[k] for k in
                  ('particles', 'height_m', 'rms_radius_m', 'kinetic_energy_j')}),
                  'pv', json.dumps(last['plastic_volume']), flush=True)

        # Born at the domain ambient (no custom temperature): Snow must still be
        # poured as snow. Before 2026-10-09 a warm ambient birthed it above its
        # melt point and every parcel ran as water (the fixed 263 K above hid it).
        call('fluid.set_param', domain=DOMAIN, default_substance='Snow')
        call('flow_source.update', name=SOURCE, enabled=True, fluid_substance='Snow',
             fluid_temperature_override=False)
        listed = {row['name']: row for row in call('flow_source.list')}[SOURCE]
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        for _ in range(20):
            call('fluid.step', dt=1 / 60)
        domain_row = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
        ambient = {'birth_kelvin': listed['fluid_birth_kelvin'],
                   'born_solid': listed['fluid_born_solid'],
                   'granular_particles': domain_row['granular_model_particles'],
                   'fluid_particles': domain_row['fluid_model_particles']}
        data['ambient_birth'] = ambient
        print('ambient_birth', json.dumps(ambient), flush=True)

        # A Thermal force field over the pile melts it (2026-10-09: fields
        # heated only meshes, snow under one stayed snow).
        heat_field = call('forcefield.create', type='thermal', name='T: snow heat')['name']
        call('forcefield.set_param', field=heat_field, enabled=True, shape='sphere',
             position=[0, 0, 0], inner_radius=1.5, falloff_radius=2.0,
             thermal_delta_kelvin=200.0)
        for _ in range(120):
            call('fluid.step', dt=1 / 60)
        domain_row = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
        heat = {'heated': domain_row.get('thermal_field_heated_particles'),
                'max_kelvin': domain_row['particle_max_kelvin'],
                'granular_particles': domain_row['granular_model_particles'],
                'fluid_particles': domain_row['fluid_model_particles']}
        data['heat_field'] = heat
        print('heat_field', json.dumps(heat), flush=True)

        # Seed Fluid Now and Fill Domain pour the Default Substance by the same
        # birth rule as a source (2026-10-09: both birthed Snow at ambient and
        # it ran as water).
        call('flow_source.update', name=SOURCE, enabled=False)
        call('forcefield.remove', field=heat_field)  # it would melt the seeds
        heat_field = None
        seeded = {}
        for label, extra in (('seed_box', {'seed_min': [-.3, .2, -.3], 'seed_max': [.3, .5, .3]}),
                             ('fill', {'fill_level': .15})):
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True, **extra)
            for _ in range(5):
                call('fluid.step', dt=1 / 60)
            row = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
            seeded[label] = {'granular_particles': row['granular_model_particles'],
                             'fluid_particles': row['fluid_model_particles'],
                             'min_kelvin': row['particle_min_kelvin']}
        call('fluid.clear', domain=DOMAIN, clear_seed=True)
        data['seeded'] = seeded
        print('seeded', json.dumps(seeded), flush=True)

        final = {a['label']: a['final'] for a in data['arms']}
        pv_max = {k: (v['plastic_volume'] or {}).get('max_value') for k, v in final.items()}
        verdicts = {
            'plastic_volume_readable': all(
                (v['plastic_volume'] or {}).get('available') for v in final.values()),
            'compaction': pv_max['snow'] is not None and pv_max['snow'] > 1.005 and
                          pv_max['no_compaction'] is not None and pv_max['no_compaction'] <= 1.005,
            'cohesion': final['snow']['rms_radius_m'] < .8 * final['loose']['rms_radius_m'],
            'ambient_birth_is_snow': ambient['birth_kelvin'] < 273.15 and
                                     ambient['granular_particles'] > 0 and
                                     ambient['fluid_particles'] == 0,
            'seed_and_fill_are_snow': all(v['granular_particles'] > 0 and v['fluid_particles'] == 0
                                          for v in seeded.values()),
            'heat_field_melts': (heat['heated'] or 0) > 0 and heat['max_kelvin'] > 273.15 and
                                heat['fluid_particles'] > 0,
            'rest': all(v['kinetic_energy_j'] / max(v['particles'], 1) < 1e-4
                        for v in final.values()),
        }
        data['verdicts'] = verdicts
        save()
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        if heat_field:
            call('forcefield.remove', field=heat_field)
        for field in original['fields']:
            call('forcefield.set_param', field=field['name'], enabled=field['enabled'])
        if source_ready:
            # Release the derived substances so they can be removed.
            call('flow_source.update', name=SOURCE, enabled=False, fluid_substance='Snow')
        if domain_ready:
            call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False,
                 default_substance='Water')
        for name in derived:
            remove_material(call, name)
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=domain['enabled'],
                 visible=domain['visible'])
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=source['enabled'])
        call('timeline.set_frame', frame=0)
        save()
        client.close()


if __name__ == '__main__':
    main()
