"""FG1 acceptance (2026-10-18): water puts a fire out inside ONE Matter domain.

There is no quench switch any more. A fire over boiling water must lose to the
physics already on the books: the surface exchange and boiling take its heat,
and the steam they make displaces the oxygen it burns in
(GridFluid::CombustionAir). Combustion also reads the AIR temperature: the gas
field holds the virtual temperature, and pure steam at 373 K reads ~600 K there.

Two arms in one closed Matter domain, same pilot fire (a fuel + heat flow
source just above the tank):
  A  dry  - no water. The fire burns, nothing is oxygen-limited, and the lowest
            oxygen fraction is the air's (0.2095 * oxygen availability).
  B  wet  - a Water tank under the fire, boiled from below by a Thermal field.
            Steam must dilute the oxygen (lowest fraction well below 0.2095)
            and throttle the burn (oxygen-limited cells > 0); the ledger stays
            balanced.
Reported, not gated: burning-cell totals per arm and the steam-masked count
(cells whose Tv is above ignition while the air is not).

Scene domains, sources and fields are switched off for the run and restored;
everything created is removed.
"""

import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_FireSteam'
SOURCE = 'T: pilot fire'
FIELD = 'T: tank heat'
DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]
AIR_OXYGEN = 0.2095


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=150)
    args = parser.parse_args()

    client = RtIpc()
    call = client.call
    data = {'arms': {}}
    out = ROOT / 'docs/dev/fire_steam_quench_live.json'

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    created_domain = DOMAIN not in {d['name'] for d in original['domains']}
    source_created = False
    field = None

    def run_arm(name, wet):
        nonlocal field
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        if wet:
            call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True,
                 fill_level=.15)
            field = call('forcefield.create', type='thermal', name=FIELD)['name']
            # From below, as in the boiling acceptance: steam rises into the fire.
            call('forcefield.set_param', field=field, enabled=True, shape='sphere',
                 position=[0, -1.0, 0], inner_radius=1.0, falloff_radius=1.6,
                 thermal_delta_kelvin=400.0)
        arm = {'samples': [], 'burning_total': 0, 'inerted_total': 0, 'fuel_total': 0.0,
               'masked_total': 0, 'min_oxygen': None, 'measured_steps': 0}
        for step in range(1, args.steps + 1):
            call('fluid.step', dt=DT)
            s = call('gas.step_stats', domain=DOMAIN)
            if not s.get('combustion_measured'):
                continue
            arm['measured_steps'] += 1
            arm['burning_total'] += s['combustion_burning_cells']
            arm['fuel_total'] += s.get('combustion_burned_fuel', 0.0)
            arm['inerted_total'] += s['combustion_inerted_cells']
            arm['masked_total'] += s['combustion_steam_masked_cells']
            o2 = s['combustion_min_oxygen_fraction']
            if o2 >= 0 and (arm['min_oxygen'] is None or o2 < arm['min_oxygen']):
                arm['min_oxygen'] = o2
            if step % 25 == 0:
                row = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
                sample = {'step': step,
                          'burning': s['combustion_burning_cells'],
                          'inerted': s['combustion_inerted_cells'],
                          'masked': s['combustion_steam_masked_cells'],
                          'min_oxygen': o2,
                          'gas_phase_mass_kg': row.get('gas_phase_mass_kg', 0.0),
                          'particles': row.get('particle_count', 0)}
                arm['samples'].append(sample)
                print(name, json.dumps(sample), flush=True)
        ledger = call('matter.exchanges')
        arm['ledger_mass_error_kg'] = ledger['mass_error_kg']
        arm['ledger_energy_error_j'] = ledger['energy_error_j']
        if field:
            call('forcefield.remove', field=field)
            field = None
        data['arms'][name] = arm
        return arm

    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        if created_domain:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
             backend='vulkan', boundary='closed', default_substance='Water',
             thermal_liquid_enabled=True)
        call('gas.set_settings', domain=DOMAIN, fire_enabled=True,
             ignition_temperature=0.05, burn_rate=1.0, heat_release=1.0,
             smoke_generation=0.2)
        # Just above the tank surface (fill .15 of 2 m -> 0.3 m): the steam
        # column rises through the pilot instead of beside it.
        call('flow_source.create', name=SOURCE, domain=DOMAIN, source_mode='point',
             position=[0.0, 0.42, 0.0], radius=0.12, density=0.0,
             temperature=1.0, fuel=2.0, velocity=[0.0, 0.0, 0.0])
        source_created = True

        dry = run_arm('dry', wet=False)
        wet = run_arm('wet', wet=True)

        verdicts = {
            'stats_measured': dry['measured_steps'] > 0 and wet['measured_steps'] > 0,
            'dry_fire_burns': dry['burning_total'] > 0,
            # Dry air: nothing throttled, and the instrument reads the air's O2.
            'dry_not_oxygen_limited': dry['inerted_total'] == 0,
            'dry_oxygen_is_air': dry['min_oxygen'] is not None and
                                 abs(dry['min_oxygen'] - AIR_OXYGEN) < 1e-3,
            'steam_dilutes_oxygen': wet['min_oxygen'] is not None and
                                    wet['min_oxygen'] < 0.18,
            'steam_throttles_burn': wet['inerted_total'] > 0,
            # The FG1 claim, by AMOUNT: cells alone cannot tell a smothered
            # fire from one that spread thinly (2026-10-18 first run).
            'steam_reduces_burned_fuel': dry['fuel_total'] > 0 and
                                         wet['fuel_total'] < 0.9 * dry['fuel_total'],
            'ledger_balanced': abs(wet['ledger_mass_error_kg']) < 1e-6 and
                               abs(wet['ledger_energy_error_j']) < 1e-2,
        }
        data['verdicts'] = verdicts
        data['reported'] = {
            'burning_dry_over_wet': (dry['burning_total'] / wet['burning_total']
                                     if wet['burning_total'] else None),
            'steam_masked_cells_wet': wet['masked_total'],
            'fuel_dry': dry['fuel_total'], 'fuel_wet': wet['fuel_total'],
        }
        print(json.dumps(data['reported']), flush=True)
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        if field:
            try:
                call('forcefield.remove', field=field)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        if source_created:
            try:
                call('flow_source.remove', name=SOURCE)
            except Exception:  # noqa: BLE001
                pass
        if created_domain:
            try:
                call('fluid.remove_domain', domain=DOMAIN)
            except Exception:  # noqa: BLE001 - leave it disabled instead
                call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
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
