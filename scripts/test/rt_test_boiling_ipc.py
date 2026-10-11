"""Boiling acceptance (2026-10-10): water heated past 373.15 K turns to steam
in the matter domain's own gas phase.

A Water tank is seeded (Fill Domain) in a closed matter domain and a Thermal
force field heats it. Expect:
  * `vaporization` exchanges for Water from the domain to itself, each one
    balanced (mass in = mass out, latent paid = latent required);
  * the domain's gas-phase inventory holds what boiled off (closed box);
  * the gas density tracer is live (steam is drawn as fog);
  * the liquid lost particles or mass.
Scene domains, sources and force fields are switched off for the run and
restored afterwards.
"""

import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_Boil'
FIELD = 'T: boil heat'
DT = 1 / 60
STEPS = 120
ROOT = Path(__file__).resolve().parents[2]


def main():
    client = RtIpc()
    call = client.call
    data = {'samples': []}
    out = ROOT / 'docs/dev/boiling_acceptance_live.json'

    def save():
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    created_domain = DOMAIN not in {d['name'] for d in original['domains']}
    field = None
    try:
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=False)
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        if created_domain:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
             backend='vulkan', boundary='closed', default_substance='Water')
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True,
             fill_level=.1)

        field = call('forcefield.create', type='thermal', name=FIELD)['name']
        call('forcefield.set_param', field=field, enabled=True, shape='sphere',
             position=[0, -1.0, 0], inner_radius=1.0, falloff_radius=1.6,
             thermal_delta_kelvin=400.0)
        # Heated from BELOW: a uniform field heats every parcel alike and
        # leaves nothing to convect (that is physics, not a failure).

        first = None
        boiled_kg = 0.0
        worst_mass_error = 0.0
        worst_energy_error = 0.0
        for step in range(1, STEPS + 1):
            call('fluid.step', dt=DT)
            ledger = call('matter.exchanges')
            rows = [r for r in ledger['exchanges']
                    if r['kind'] == 'vaporization' and r['substance'] == 'Water'
                    and r['source'] == DOMAIN and r['target'] == DOMAIN]
            for r in rows:
                boiled_kg += r['source_mass_kg']
                worst_mass_error = max(worst_mass_error,
                                       abs(r['target_mass_kg'] - r['source_mass_kg']))
                worst_energy_error = max(worst_energy_error, abs(
                    r['target_energy_j'] + r['latent_required_j'] -
                    r['source_energy_j'] - r['latent_accounted_j']))
            mist_kg = sum(r['source_mass_kg'] for r in ledger['exchanges']
                          if r['kind'] == 'mist_to_gas' and r['source'] == DOMAIN)
            # Water that came back out of the air: onto the liquid surface or as
            # rain (2026-10-11 fog-water batch).
            data.setdefault('returned_kg', 0.0)
            data['returned_kg'] += sum(
                r['target_mass_kg'] for r in ledger['exchanges']
                if (r['kind'] == 'condensation' and r['source'] == DOMAIN) or
                   r['kind'] == 'precipitation')
            data.setdefault('mist_kg', 0.0)
            data['mist_kg'] += mist_kg
            if step % 20 == 0 or step == 1:
                row = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
                sample = {'step': step, 'particles': row['particle_count'],
                          'max_kelvin': row['particle_max_kelvin'],
                          'boiled_kg': boiled_kg,
                          'gas_phase_mass_kg': row['gas_phase_mass_kg'],
                          'gas_condensate_kg': row.get('gas_condensate_kg', 0.0),
                          'gas_rain_kg': row.get('gas_rain_kg', 0.0),
                          'gas_active_cells': row.get('active_density_cells', 0),
                          'gas_phase_active_cells': row['gas_phase_active_cells'],
                          'convecting': row.get('thermal_convecting_particles', 0)}
                if first is None:
                    first = sample
                data['samples'].append(sample)
                print(json.dumps(sample), flush=True)
        last = data['samples'][-1]
        transferred = boiled_kg + data['mist_kg']
        verdicts = {
            'boils': boiled_kg > 0.0,
            'exchange_balanced': worst_mass_error < 1e-9 and worst_energy_error < 1e-3,
            # Closed box: the air keeps what left the liquid and did not come
            # back (vapour + fog water + rain water; float sums).
            'inventory_holds_steam': transferred > 0.0 and
                last['gas_phase_mass_kg'] + last['gas_condensate_kg'] + last['gas_rain_kg'] >=
                .95 * (transferred - data.get('returned_kg', 0.0)),
            'steam_visible': last['gas_phase_active_cells'] > 0,
            # Natural convection: heated water rises (thermal_expansion_per_k).
            'convects': any(x['convecting'] > 0 for x in data['samples']),
            'liquid_lost_mass': last['particles'] < first['particles'] or boiled_kg > 0.0,
        }
        data.update({'boiled_kg': boiled_kg, 'transferred_kg': transferred,
                     'worst_mass_error': worst_mass_error,
                     'worst_energy_error': worst_energy_error, 'verdicts': verdicts})
        save()
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        if field:
            call('forcefield.remove', field=field)
        if created_domain:
            try:
                call('fluid.remove_domain', domain=DOMAIN)
            except Exception:  # noqa: BLE001 - leave it disabled instead
                call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        else:
            call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=f['enabled'])
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
