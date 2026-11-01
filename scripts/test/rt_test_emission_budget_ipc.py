"""Flow-source budget in kg and the Max Particles ceiling (2026-10-27).

Two arms in one empty Matter domain (Water, 5 cm voxels):

  budget  A source limited to 2.0 kg pours exactly that much: it stops within
          one parcel of 2.0 kg, its particle count is mass / parcel_mass_kg,
          and the retired count keys are refused with a hint.
  ceiling Max Particles 400 under a 6000 /s source: the domain fills to the
          ceiling and the source is then held back. withheld_mass_kg_total
          (source) and emission_withheld_kg_total (domain) are > 0 and agree,
          and matter.exchanges lists an emission_withheld record.

Empty scene, Pause first. Numbers go to %TEMP%/rt_test_emission_budget.json.
"""
import json
import os
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_EmissionBudget'
SOURCE = 'T_EmissionBudget_src'
DT = 1 / 60


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data, verdicts = {}, {}

    def source_row():
        return next(s for s in call('flow_source.list') if s['name'] == SOURCE)

    def domain_row():
        return next(d for d in call('fluid.list_domains')['domains'] if d['name'] == DOMAIN)

    try:
        call('fluid.create_domain', name=DOMAIN, type='matter', domain_min=[-.4, 0, -.4],
             domain_max=[.4, .8, .4], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', max_particles=60000)
        call('flow_source.create', name=SOURCE, domain=DOMAIN, phase='liquid',
             source_mode='point', fluid_substance='Water', position=[0, .5, 0], radius=.12,
             velocity=[0, 0, 0], fluid_velocity_spread=0., fluid_particles_per_second=6000.)

        # ── budget ─────────────────────────────────────────────────────
        ok, message = client.try_call('flow_source.update', name=SOURCE,
                                      max_emitted_particles=100)
        verdicts['retired_count_key_refused'] = not ok and 'max_emitted_mass_kg' in message
        call('flow_source.update', name=SOURCE, use_mass_limit=True, max_emitted_mass_kg=2.0)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        call('fluid.run', steps=90, dt=DT, sample_every=0, domains=[DOMAIN])
        row = source_row()
        mass, parcel = row['total_emitted_mass_kg'], row['parcel_mass_kg']
        data['budget'] = {k: row[k] for k in ('total_emitted_mass_kg', 'total_emitted_particles',
                                              'parcel_mass_kg', 'max_emitted_mass_kg')}
        print('budget', json.dumps(data['budget']), flush=True)
        verdicts['budget_stops_at_mass'] = parcel > 0 and 2.0 <= mass < 2.0 + parcel + 1e-9
        verdicts['budget_counts_parcels'] = parcel > 0 and abs(
            row['total_emitted_particles'] - mass / parcel) < 0.5

        # ── ceiling ────────────────────────────────────────────────────
        call('flow_source.update', name=SOURCE, use_mass_limit=False)
        call('fluid.set_param', domain=DOMAIN, max_particles=400)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        call('fluid.run', steps=60, dt=DT, sample_every=0, domains=[DOMAIN])
        exchanges = call('matter.exchanges')
        src, dom = source_row(), domain_row()
        data['ceiling'] = {'source_withheld_total_kg': src['withheld_mass_kg_total'],
                           'domain_withheld_total_kg': dom.get('emission_withheld_kg_total'),
                           'particles': dom.get('particle_count'),
                           'kinds': sorted({r['kind'] for r in exchanges.get('exchanges', [])})}
        print('ceiling', json.dumps(data['ceiling']), flush=True)
        verdicts['ceiling_withholds'] = src['withheld_mass_kg_total'] > 0.0
        verdicts['domain_matches_source'] = abs(
            (dom.get('emission_withheld_kg_total') or 0.0) - src['withheld_mass_kg_total']) <= \
            1e-9 + 1e-6 * src['withheld_mass_kg_total']
        verdicts['ledger_lists_withheld'] = 'emission_withheld' in data['ceiling']['kinds']
    finally:
        call('timeline.set_frame', frame=0)
        try:
            call('flow_source.remove', name=SOURCE)
            call('fluid.remove_domain', domain=DOMAIN)
        finally:
            data['verdicts'] = verdicts
            path = Path(os.environ.get('TEMP', '.')) / 'rt_test_emission_budget.json'
            path.write_text(json.dumps(data, indent=2), encoding='utf-8')
            client.close()
    for key, passed in verdicts.items():
        print('PASS' if passed else 'FAIL', key, flush=True)
    if not all(verdicts.values()):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
