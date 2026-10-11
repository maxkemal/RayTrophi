"""Non-condensable gas pool acceptance (2026-10-11, part 4).

Fuel vapour from a burning liquid used to enter the gas phase's VAPOUR pool,
where the saturation adjustment treated it as the domain's water: burning
fuel could condense into fog and rain as water. It now has its own pool.

Run in a DISPOSABLE, paused scene: adds the burning_fuel_spill preset, steps
it, reads the gas domains' inventory, removes the domains it added.
Expect, on the gas domain(s) the preset created:
  * gas_foreign_kg > 0 (the combustion exchange deposited fuel vapour, and
    oil spray rides the air as non-condensable aerosol, not fog water);
  * the ledger's combustion target mass is all in the foreign pool, none in
    gas_phase_mass_kg (vapour stays what boiled water left there: here 0);
  * fog_condensed_kg stays 0 on every step (nothing condensable to condense).
"""

import json
from pathlib import Path

from rt_ipc import RtIpc

DT = 1 / 60
STEPS = 120
ROOT = Path(__file__).resolve().parents[2]


def main():
    client = RtIpc()
    call = client.call
    out = ROOT / 'docs/dev/gas_species_live.json'
    data = {'samples': []}
    assert not call('sim.control_state')['playing'], 'Pause first'
    before = {d['name'] for d in call('fluid.list_domains')['domains']}
    added = []
    try:
        call('particle.add_preset', preset='burning_fuel_spill')
        added = [d['name'] for d in call('fluid.list_domains')['domains']
                 if d['name'] not in before]
        assert added, 'burning_fuel_spill added no fluid domain'
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        # The preset leaves its seed pending for the FRAME LOOP; fluid.step
        # never runs it (2026-10-11: 0 particles, nothing burned). Seed here.
        for name in added:
            call('fluid.seed', domain=name, replace=True)
        seeded = sum(d['particle_count'] for d in call('fluid.list_domains')['domains']
                     if d['name'] in added)
        assert seeded > 0, 'fluid.seed produced no fuel parcels'
        data['seeded_particles'] = seeded
        combusted = 0.0
        fog_condensed = 0.0
        for step in range(1, STEPS + 1):
            call('fluid.step', dt=DT)
            # The combustion exchange records fuel vapour as 'vaporization' of
            # the fuel substance; its spray as 'mist_to_gas'.
            for r in call('matter.exchanges')['exchanges']:
                if (r['kind'] in ('vaporization', 'mist_to_gas') and r['target'] in added and
                        r['substance'] != 'Water'):
                    combusted += r['target_mass_kg']
            rows = {d['name']: d for d in call('fluid.list_domains')['domains']
                    if d['name'] in added}
            fog_condensed += sum(d.get('fog_condensed_kg', 0.0) for d in rows.values())
            if step % 30 == 0:
                sample = {'step': step, 'combusted_kg': combusted,
                          'fog_condensed_kg': fog_condensed,
                          'domains': {n: {'vapour_kg': d.get('gas_phase_mass_kg'),
                                          'foreign_kg': d.get('gas_foreign_kg'),
                                          'fog_kg': d.get('gas_condensate_kg')}
                                      for n, d in rows.items()}}
                data['samples'].append(sample)
                print(json.dumps(sample), flush=True)
        last = data['samples'][-1]['domains']
        foreign = sum((v['foreign_kg'] or 0.0) for v in last.values())
        vapour = sum((v['vapour_kg'] or 0.0) for v in last.values())
        verdicts = {
            'combustion_ran': combusted > 0.0,
            'fuel_in_foreign_pool': foreign > 0.0,
            # Open domains lose some through the walls; none may be in vapour.
            'fuel_not_vapour': vapour <= 1e-6 * max(combusted, 1e-9),
            'no_fuel_fog': fog_condensed == 0.0,
        }
        data.update({'combusted_kg': combusted, 'foreign_kg': foreign, 'vapour_kg': vapour,
                     'fog_condensed_kg': fog_condensed, 'verdicts': verdicts})
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for name in added:
            try:
                call('fluid.remove_domain', domain=name)
            except Exception:  # noqa: BLE001 - leave it disabled instead
                call('fluid.set_param', domain=name, enabled=False, visible=False)
        call('timeline.set_frame', frame=0)
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
