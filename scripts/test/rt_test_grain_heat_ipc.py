"""Grain heat acceptance (2026-10-19): grains join the thermal chain.

Until this batch a grain domain refused Thermal liquid outright. Now grains
keep their temperature and trade heat (MatterGrainThermal): a film with the
liquid of their cell, a film with the domain's air (the gas lane), contact
conduction with the grains beside them, and cell-to-cell conduction.

Scene: one closed Matter domain, a Water pool at 293 K, and hot Gravel (DEM)
born at 360 K that falls through the air into it. Expected scale: a 1 cm
grain settles with tau ~36 s, so the mean drops ~10 K over the run - not to
293 K. Tens of kelvin in a second would mean the film or the area is wrong.
Expect, from fluid.list_domains (thermal_grain_*) and fluid.matter_models:
  * no 'thermal_liquid' grain blocker, and grains are measured;
  * the grains cool (mean drops), the water takes heat (> 0 J), and the air
    takes heat while they fall (> 0 J);
  * the pass conserves: |residual| is rounding against the heat moved;
Scene domains, sources and fields are switched off and restored; everything
created is removed.
"""

import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_GrainHeat'
WATER = 'T: grain heat water'
GRAVEL = 'T: grain heat gravel'
DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=420)
    parser.add_argument('--hot', type=float, default=360.0, help='gravel birth K')
    args = parser.parse_args()

    client = RtIpc()
    call = client.call
    data = {'samples': []}
    out = ROOT / 'docs/dev/grain_heat_live.json'
    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list'),
                'colliders': call('collider.list')['colliders']}
    assert not original['control']['playing'], 'Pause first'
    created_domain = DOMAIN not in {d['name'] for d in original['domains']}
    created_sources = []
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=False)
        if created_domain:
            call('fluid.create_domain', name=DOMAIN, type='matter', domain_min=[-.4, 0, -.4],
                 domain_max=[.4, .8, .4], voxel_size=.05)
        # ★ thermal_liquid_enabled with grains: rejected before 2026-10-19.
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', solid_phase=False,
             thermal_liquid_enabled=True, max_particles=40000)
        call('fluid.set_pore_exchange', domain=DOMAIN, enabled=False,
             wet_response_enabled=False, wet_appearance_enabled=False)
        call('flow_source.create', name=WATER, domain=DOMAIN, phase='liquid',
             source_mode='point', fluid_substance='Water', position=[0, .15, 0], radius=.2,
             velocity=[0, 0, 0], fluid_velocity_spread=0., fluid_particles_per_second=24000.,
             fluid_temperature_override=True, fluid_temperature_kelvin=293.15,
             use_time_limit=True, start_time=0., end_time=.5,
             use_particle_limit=True, max_emitted_particles=12000)
        created_sources.append(WATER)
        call('flow_source.create', name=GRAVEL, domain=DOMAIN, phase='liquid',
             source_mode='point', fluid_substance='Gravel', position=[0, .6, 0], radius=.1,
             velocity=[0, 0, 0], fluid_velocity_spread=0., fluid_particles_per_second=1000.,
             fluid_temperature_override=True, fluid_temperature_kelvin=args.hot,
             use_time_limit=True, start_time=.6, end_time=1.,
             use_particle_limit=True, max_emitted_particles=400)
        created_sources.append(GRAVEL)
        # 1 cm: tau = C / (h_eff A) ~ 36 s in water (h_eff 130 W/(m^2 K) with the
        # grain's own r/(5k) resistance), so ~10 K of the 67 K go in 6 s. At 2 cm
        # tau is ~125 s and the drop (~3 K) sits too close to the gate.
        call('fluid.set_grain_settings', domain=DOMAIN, radius_m=.01, fluid_coupling=True,
             max_substeps=2048)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')

        totals = {'to_liquid_j': 0.0, 'to_gas_j': 0.0, 'to_walls_j': 0.0, 'max_residual_j': 0.0,
                  'grains_max': 0}
        first_mean = None
        readiness = None
        for step in range(1, args.steps + 1):
            call('fluid.step', dt=DT)
            row = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
            if row.get('thermal_grain_measured'):
                totals['to_liquid_j'] += row.get('thermal_grain_heat_to_liquid_j', 0.0)
                totals['to_gas_j'] += row.get('thermal_grain_heat_to_gas_j', 0.0)
                totals['to_walls_j'] += row.get('thermal_grain_heat_to_walls_j', 0.0)
                totals['max_residual_j'] = max(totals['max_residual_j'],
                                               abs(row.get('thermal_grain_energy_residual_j', 0.0)))
                totals['grains_max'] = max(totals['grains_max'], row.get('thermal_grain_count', 0))
                if first_mean is None and row.get('thermal_grain_count', 0) > 0:
                    first_mean = row.get('thermal_grain_mean_kelvin')
            if readiness is None and step >= 40:
                models = call('fluid.matter_models', domain=DOMAIN)
                readiness = models.get('grain_readiness') or {}
            if step % 30 == 0:
                sample = {'step': step,
                          'grains': row.get('thermal_grain_count'),
                          'wetted': row.get('thermal_grain_wetted'),
                          'grain_mean_k': row.get('thermal_grain_mean_kelvin'),
                          'grain_max_k': row.get('thermal_grain_max_kelvin'),
                          'to_liquid_j': row.get('thermal_grain_heat_to_liquid_j'),
                          'to_gas_j': row.get('thermal_grain_heat_to_gas_j'),
                          'to_walls_j': row.get('thermal_grain_heat_to_walls_j'),
                          'residual_j': row.get('thermal_grain_energy_residual_j'),
                          'liquid_max_k': row.get('particle_max_kelvin'),
                          'ms': row.get('thermal_grain_ms')}
                data['samples'].append(sample)
                print(json.dumps(sample), flush=True)
        last = data['samples'][-1] if data['samples'] else {}
        moved = abs(totals['to_liquid_j']) + abs(totals['to_gas_j']) + abs(totals['to_walls_j'])
        blockers = [b.get('code') for b in (readiness or {}).get('blockers', [])]
        verdicts = {
            'thermal_not_a_blocker': 'thermal_liquid' not in blockers,
            'grains_measured': totals['grains_max'] > 0,
            'grains_cool': first_mean is not None and
                           (last.get('grain_mean_k') or first_mean) < first_mean - 1.0,
            'water_takes_heat': totals['to_liquid_j'] > 0.0,
            # Needs the gas lane to step (a Matter domain's air).
            'air_takes_heat': totals['to_gas_j'] > 0.0,
            'energy_closes': totals['max_residual_j'] <= 1e-6 * max(moved, 1.0) + 1e-6,
            # ★ 2026-10-19 live: 360 -> 299 K in 1.5 s and ending at 293.0 K,
            # below the water - an infinite wall sink, not physics. tau ~36 s
            # leaves most of the excess after the run; losing over half of it
            # means some path is not paying with capacity.
            'cools_at_physical_rate': first_mean is not None and
                (last.get('grain_mean_k') or 0.0) - 293.15 > 0.5 * (first_mean - 293.15),
            # The water is what takes it, not the walls.
            'water_takes_most': totals['to_liquid_j'] > totals['to_walls_j'],
        }
        data.update({'totals': totals, 'first_grain_mean_k': first_mean,
                     'blockers': blockers, 'verdicts': verdicts})
        print(json.dumps({'totals': totals, 'first_grain_mean_k': first_mean}), flush=True)
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for name in created_sources:
            try:
                call('flow_source.remove', name=name)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        if created_domain:
            try:
                call('fluid.remove_domain', domain=DOMAIN)
            except Exception:  # noqa: BLE001 - leave it disabled instead
                call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=c['enabled'])
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=f['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'],
                 visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        call('timeline.set_frame', frame=0)
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
