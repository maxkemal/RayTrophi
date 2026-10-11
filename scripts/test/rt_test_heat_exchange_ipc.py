"""Liquid heat transport acceptance (2026-10-10).

A Water tank in a closed matter domain, thermal liquid ON, is heated from
below by a Thermal field, then the field is removed. Expect, from the
substance's own c, k, L, M and Tb:
  * conduction runs (thermal_conducting_cells > 0);
  * the surface trades heat with the gas grid above it
    (thermal_gas_exchange_particles > 0, heat goes liquid -> gas once the
    water is warmer than the air);
  * below-boiling evaporation happens and is ledgered as balanced
    `vaporization` rows for Water;
Scene domains, sources and force fields are switched off and restored.
"""

import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_HeatXfer'
FIELD = 'T: heat xfer'
DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]


def row(call):
    return {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]


def main():
    client = RtIpc()
    call = client.call
    data = {'samples': []}
    out = ROOT / 'docs/dev/heat_exchange_acceptance_live.json'
    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    created = DOMAIN not in {d['name'] for d in original['domains']}
    field = None
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        if created:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
             backend='vulkan', boundary='closed', default_substance='Water',
             thermal_liquid_enabled=True)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True, fill_level=.1)
        field = call('forcefield.create', type='thermal', name=FIELD)['name']
        call('forcefield.set_param', field=field, enabled=True, shape='sphere',
             position=[0, -1.0, 0], inner_radius=1.0, falloff_radius=1.6,
             thermal_delta_kelvin=60.0)   # warms, does not boil

        evaporated_rows = []
        def sample(step, phase):
            r = row(call)
            s = {'step': step, 'phase': phase,
                 'mean_kelvin': r.get('thermal_mean_kelvin'),
                 'max_kelvin': r['particle_max_kelvin'],
                 'conducting_cells': r.get('thermal_conducting_cells', 0),
                 'gas_exchange': r.get('thermal_gas_exchange_particles', 0),
                 'heat_to_gas_j': r.get('thermal_heat_to_gas_j', 0.0),
                 'evaporated_kg': r.get('thermal_evaporated_kg', 0.0),
                 'gas_phase_mass_kg': r['gas_phase_mass_kg']}
            data['samples'].append(s)
            print(json.dumps(s), flush=True)
            return s

        def step_and_ledger():
            call('fluid.step', dt=DT)
            ledger = call('matter.exchanges')
            evaporated_rows.extend(r for r in ledger['exchanges']
                                   if r['kind'] == 'vaporization' and r['substance'] == 'Water'
                                   and r['source'] == DOMAIN)

        for step in range(1, 91):
            step_and_ledger()
            if step % 30 == 0:
                sample(step, 'heating')
        hot = data['samples'][-1]
        # Turn the heater off MID-RUN. Since 2026-10-10 a field edit keeps the
        # run (invalidateSimCacheAfterCurrentFrame); before, it rewound and
        # reseeded the tank (mean 310 K -> fresh 293 K).
        call('forcefield.remove', field=field)
        field = None
        off = sample(90, 'source_off')
        for step in range(91, 271):
            step_and_ledger()
            if step % 60 == 0:
                last = sample(step, 'cooling')
        # Freeze from the top: a cold field over the surface (a pond in frost).
        field = call('forcefield.create', type='thermal', name=FIELD)['name']
        call('forcefield.set_param', field=field, enabled=True, shape='box',
             position=[0, 0.3, 0], inner_radius=0.3, falloff_radius=0.6,
             thermal_delta_kelvin=-80.0)
        frozen = 0
        for step in range(271, 451):
            call('fluid.step', dt=DT)
            if step % 60 == 0:
                r = row(call)
                frozen = max(frozen, r.get('thermal_frozen_particles', 0))
                print(json.dumps({'step': step, 'phase': 'freezing',
                                  'frozen': r.get('thermal_frozen_particles', 0),
                                  'cold_unsupported': r.get('thermal_cold_unsupported', 0),
                                  'min_kelvin': r['particle_min_kelvin']}), flush=True)
        data['frozen'] = frozen
        worst_mass = max((abs(r['target_mass_kg'] - r['source_mass_kg'])
                          for r in evaporated_rows), default=0.0)
        verdicts = {
            'conducts': any(s['conducting_cells'] > 0 for s in data['samples']),
            'surface_exchanges_with_gas': any(s['gas_exchange'] > 0 for s in data['samples']),
            # Water warmer than the air above it must push heat INTO the gas.
            'heat_flows_to_air_when_warmer': any(
                s['heat_to_gas_j'] > 0 for s in data['samples']
                if s['phase'] == 'heating' and (s['mean_kelvin'] or 0) > 293.5),
            'evaporates_below_boiling': len(evaporated_rows) > 0 and hot['max_kelvin'] < 373.15,
            'evaporation_balanced': worst_mass < 1e-9,
            'field_edit_keeps_run': off['mean_kelvin'] > 300.0,
            'cools_after_source_off': last['mean_kelvin'] < hot['mean_kelvin'],
            'freezes_from_surface': frozen > 0,
        }
        data['verdicts'] = verdicts
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        if field:
            call('forcefield.remove', field=field)
        try:
            if created:
                call('fluid.remove_domain', domain=DOMAIN)
            else:
                call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        except Exception:  # noqa: BLE001
            pass
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=f['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'], visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        call('timeline.set_frame', frame=0)
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
