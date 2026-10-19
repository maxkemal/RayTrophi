"""Seed settle acceptance (2026-10-19): a fill seed starts at rest.

Short on purpose (MATTER_ROADMAP_2026_10 test rule). Four arms, 30 frames each:
Sand (DEM grains, Matter domain) and Water (plain Fluid domain), each seeded
by Fill Domain with the settle on and off. Gates:
  * settle_ran: the settled arms report settle_ran, steps >= 1 and the same
    particle count before and after;
  * grains_at_rest: settled Sand never moves faster than REST_M_S in the 30
    frames and its centre of mass drifts < 2 mm;
  * grains_control_moves: unsettled Sand moves faster than settled Sand (the
    test can see the effect at all);
  * liquid_at_rest: settled Water's mean parcel speed stays under REST_M_S;
  * liquid_control_moves: unsettled Water's peak mean speed is higher.
Thresholds are a first guess; the first live run records the numbers.
"""
import json
from pathlib import Path

from rt_ipc import RtIpc

DT = 1 / 60
FRAMES = 30
REST_M_S = 0.05
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/dev/seed_settle_live.json'


def main():
    client = RtIpc()
    call = client.call
    data = {}
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'

    def row(name):
        return {d['name']: d for d in call('fluid.list_domains')['domains']}[name]

    def arm(label, substance, kind, settle):
        name = 'T_SeedSettle_' + label
        call('fluid.create_domain', name=name, type=kind, domain_min=[-.3, 0, -.3],
             domain_max=[.3, .5, .3], voxel_size=.05)
        try:
            call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
                 boundary='closed', default_substance=substance, thermal_liquid_enabled=False)
            call('fluid.settle', domain=name, on_seed=settle, request=False)
            call('timeline.set_frame', frame=0)
            call('fluid.seed', domain=name, particles_per_cell=4, replace=True, fill_level=.3)
            seeded = row(name)['particle_count']
            samples = []
            for frame in range(FRAMES):
                call('fluid.step', dt=DT)
                r = row(name)
                s = {'max_speed': r['live_max_speed_m_s'], 'mean_speed': r['live_mean_speed_m_s'],
                     'particles': r['particle_count']}
                if kind == 'matter':
                    g = call('fluid.matter_models', domain=name)['acceptance_metrics']['granular']
                    s['com_y'] = (g.get('dry_center_of_mass') or [0, 0, 0])[1]
                    s['energy_j'] = g['kinetic_energy_j']
                if frame == 0:
                    s['settle'] = {k: r[k] for k in r if k.startswith('settle_')}
                samples.append(s)
            data[label] = {'seeded': seeded, 'samples': samples}
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return data[label]
        finally:
            call('fluid.remove_domain', domain=name)

    try:
        sand_on = arm('sand_on', 'Sand', 'matter', True)
        sand_off = arm('sand_off', 'Sand', 'matter', False)
        water_on = arm('water_on', 'Water', 'fluid', True)
        water_off = arm('water_off', 'Water', 'fluid', False)

        def peak(a, key):
            return max(s[key] for s in a['samples'])

        def ran(a):
            st = a['samples'][0]['settle']
            return (st['settle_ran'] and st['settle_steps'] >= 1 and
                    st['settle_particles'] == a['seeded'] and
                    a['samples'][-1]['particles'] == a['seeded'])

        coms = [s['com_y'] for s in sand_on['samples']]
        summary = {
            'sand_on_peak_m_s': peak(sand_on, 'max_speed'),
            'sand_off_peak_m_s': peak(sand_off, 'max_speed'),
            'sand_on_com_drift_m': max(coms) - min(coms),
            'water_on_peak_mean_m_s': peak(water_on, 'mean_speed'),
            'water_off_peak_mean_m_s': peak(water_off, 'mean_speed'),
            'sand_settle': sand_on['samples'][0]['settle'],
            'water_settle': water_on['samples'][0]['settle'],
        }
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        verdicts = {
            'settle_ran': ran(sand_on) and ran(water_on),
            'grains_at_rest': summary['sand_on_peak_m_s'] < REST_M_S and
                              summary['sand_on_com_drift_m'] < .002,
            'grains_control_moves': summary['sand_off_peak_m_s'] > summary['sand_on_peak_m_s'],
            'liquid_at_rest': summary['water_on_peak_mean_m_s'] < REST_M_S,
            'liquid_control_moves': summary['water_off_peak_mean_m_s'] >
                                    summary['water_on_peak_mean_m_s'],
        }
        data['verdicts'] = verdicts
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
