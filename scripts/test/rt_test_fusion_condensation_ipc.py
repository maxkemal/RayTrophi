"""Fusion latent heat + condensation acceptance (2026-10-11, T2).

Part A - freezing pays its latent heat. A Water tank in a closed matter
domain sits in a cold Thermal field (-80 K). Expect:
  * parcels pass through a mushy state (part solid) before they freeze;
  * `freezing` exchanges for Water, each balanced, whose latent heat equals
    the latent heat the fusion pass released into the parcels
    (thermal_fusion_released_j) - the ledger no longer claims heat that was
    never taken from anything;
  * the tank freezes (thermal_frozen_particles > 0) within the run.

Part B - steam condenses on cold water. One corner of the tank is boiled
(+400 K field), the opposite side is held cold (-60 K field). Expect:
  * `vaporization` from the hot corner;
  * `condensation` exchanges gas -> liquid, each balanced;
  * the ledger's condensed mass matches thermal_condensed_kg.

Scene domains, sources and force fields are switched off for the run and
restored afterwards.
"""

import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_FusionCondense'
DT = 1 / 60
FREEZE_STEPS = 150
CONDENSE_STEPS = 240
ROOT = Path(__file__).resolve().parents[2]


def balanced(row):
    return (abs(row['target_mass_kg'] - row['source_mass_kg']) < 1e-9 and
            abs(row['target_energy_j'] + row['latent_required_j'] -
                row['source_energy_j'] - row['latent_accounted_j']) < 1e-3)


def main():
    client = RtIpc()
    call = client.call
    data = {'freeze': [], 'condense': []}
    out = ROOT / 'docs/dev/fusion_condensation_live.json'

    def save():
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')

    def row():
        return {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]

    def field(name, position, inner, falloff, delta):
        created = call('forcefield.create', type='thermal', name=name)['name']
        call('forcefield.set_param', field=created, enabled=True, shape='sphere',
             position=position, inner_radius=inner, falloff_radius=falloff,
             thermal_delta_kelvin=delta)
        fields.append(created)
        return created

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    created_domain = DOMAIN not in {d['name'] for d in original['domains']}
    fields = []
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
             backend='vulkan', boundary='closed', default_substance='Water',
             thermal_liquid_enabled=True)

        def reseed():
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True,
                 fill_level=.1)

        # ---- Part A: freezing ------------------------------------------------
        reseed()
        field('T: freeze cold', [0, 0, 0], 2.0, 2.5, -80.0)
        released = 0.0
        ledger_latent = 0.0
        freezing_rows = 0
        unbalanced = []
        max_mushy = 0
        for step in range(1, FREEZE_STEPS + 1):
            call('fluid.step', dt=DT)
            for r in call('matter.exchanges')['exchanges']:
                if r['kind'] == 'freezing' and r['substance'] == 'Water':
                    freezing_rows += 1
                    ledger_latent += r['latent_accounted_j']
                    if not balanced(r):
                        unbalanced.append(r)
            d = row()
            released += d.get('thermal_fusion_released_j', 0.0)
            max_mushy = max(max_mushy, d.get('thermal_mushy_particles', 0))
            if step % 15 == 0 or step == 1:
                sample = {'step': step, 'min_kelvin': d['particle_min_kelvin'],
                          'mean_kelvin': d.get('thermal_mean_kelvin'),
                          'mushy': d.get('thermal_mushy_particles'),
                          'frozen': d.get('thermal_frozen_particles'),
                          'released_j': released, 'ledger_latent_j': ledger_latent}
                data['freeze'].append(sample)
                print('freeze', json.dumps(sample), flush=True)
        last_freeze = data['freeze'][-1]
        for f in list(fields):
            call('forcefield.remove', field=f)
            fields.remove(f)

        # ---- Part B: condensation -------------------------------------------
        reseed()
        field('T: boil corner', [-.8, -.6, -.8], .6, .9, 400.0)
        field('T: cold side', [.8, -.5, .8], 1.0, 1.3, -60.0)
        vaporized = 0.0
        condensed_ledger = 0.0
        condensed_stats = 0.0
        splits = 0
        condensation_rows = 0
        for step in range(1, CONDENSE_STEPS + 1):
            call('fluid.step', dt=DT)
            for r in call('matter.exchanges')['exchanges']:
                if r['kind'] == 'vaporization' and r['source'] == DOMAIN:
                    vaporized += r['source_mass_kg']
                if r['kind'] == 'condensation' and r['target'] == DOMAIN:
                    condensation_rows += 1
                    condensed_ledger += r['target_mass_kg']
                    if not balanced(r):
                        unbalanced.append(r)
            d = row()
            condensed_stats += d.get('thermal_condensed_kg', 0.0)
            splits += d.get('thermal_condensation_splits', 0)
            if step % 20 == 0 or step == 1:
                sample = {'step': step, 'particles': d['particle_count'],
                          'max_kelvin': d['particle_max_kelvin'],
                          'min_kelvin': d['particle_min_kelvin'],
                          'vaporized_kg': vaporized, 'condensed_ledger_kg': condensed_ledger,
                          'condensed_stats_kg': condensed_stats, 'splits': splits,
                          'gas_phase_mass_kg': d['gas_phase_mass_kg']}
                data['condense'].append(sample)
                print('condense', json.dumps(sample), flush=True)

        verdicts = {
            'mushy_seen': max_mushy > 0,
            'latent_released': released > 0.0,
            'water_freezes': (last_freeze['frozen'] or 0) > 0,
            # The ledger records exactly the latent heat the fusion pass moved.
            'freeze_ledger_matches_latent': freezing_rows > 0 and
                abs(ledger_latent - released) <= .01 * max(released, 1e-12),
            'boils': vaporized > 0.0,
            'condenses': condensation_rows > 0 and condensed_ledger > 0.0,
            'condense_stats_match_ledger':
                abs(condensed_stats - condensed_ledger) <= .01 * max(condensed_ledger, 1e-12),
            'exchanges_balanced': not unbalanced,
        }
        data.update({'released_j': released, 'ledger_latent_j': ledger_latent,
                     'max_mushy': max_mushy, 'vaporized_kg': vaporized,
                     'condensed_ledger_kg': condensed_ledger,
                     'condensed_stats_kg': condensed_stats, 'splits': splits,
                     'unbalanced': unbalanced[:5], 'verdicts': verdicts})
        save()
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for f in fields:
            call('forcefield.remove', field=f)
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
