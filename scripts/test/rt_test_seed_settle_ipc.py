"""Seed settle acceptance (2026-10-19): a fill seed starts at rest.

Short on purpose (MATTER_ROADMAP_2026_10 test rule). Four arms, 30 frames each:
Sand (DEM grains, Matter domain) and Water (plain Fluid domain), each seeded
by Fill Domain with the settle on and off. Motion is measured by POSITION
(live_motion_p95_m: 95th percentile of a carrier sample's displacement per
frame), not by the velocity column. Gates:
  * settle_ran: the settled arms report settle_ran, the same particle count,
    and a stop before the step limit (at rest or at the solver's floor);
  * grains_at_rest: settled Sand's per-frame p95 displacement stays under
    REST_M and its centre of mass drifts < 2 mm;
  * grains_control_moves: unsettled Sand moves more than settled Sand;
  * liquid_at_rest: settled Water's per-frame p95 displacement stays under
    REST_M. ★ Expected to FAIL while Water runs FLIP 0.90: a resting tank
    keeps ~0.16 m/s of FLIP transfer noise (2026-10-19 probe: FLIP 0 gave
    0.004 m/s). That is a solver decision, not a settle failure;
  * liquid_control_moves: unsettled Water moves more than settled Water.
"""
import json
from pathlib import Path

from rt_ipc import RtIpc

DT = 1 / 60
FRAMES = 30
REST_M = 0.0005  # 0.5 mm per frame = 3 cm/s
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
                s = {'motion_p95_m': r['live_motion_p95_m'], 'motion_mean_m': r['live_motion_mean_m'],
                     'motion_samples': r['live_motion_samples'],
                     'mean_speed': r['live_mean_speed_m_s'], 'particles': r['particle_count']}
                if kind == 'matter':
                    g = call('fluid.matter_models', domain=name)['acceptance_metrics']['granular']
                    s['com_y'] = (g.get('dry_center_of_mass') or [0, 0, 0])[1]
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

        def peak(a):
            # Frame 0 has no previous sample to compare against.
            return max(s['motion_p95_m'] for s in a['samples'][1:])

        def ran(a):
            st = a['samples'][0]['settle']
            return (st['settle_ran'] and st['settle_particles'] == a['seeded'] and
                    a['samples'][-1]['particles'] == a['seeded'] and
                    st['settle_steps'] < st['settle_max_steps'])

        coms = [s['com_y'] for s in sand_on['samples']]
        summary = {
            'sand_on_peak_p95_m': peak(sand_on),
            'sand_off_peak_p95_m': peak(sand_off),
            'sand_on_com_drift_m': max(coms) - min(coms),
            'water_on_peak_p95_m': peak(water_on),
            'water_off_peak_p95_m': peak(water_off),
            'sand_settle': sand_on['samples'][0]['settle'],
            'water_settle': water_on['samples'][0]['settle'],
        }
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        verdicts = {
            'settle_ran': ran(sand_on) and ran(water_on),
            'grains_at_rest': summary['sand_on_peak_p95_m'] < REST_M and
                              summary['sand_on_com_drift_m'] < .002,
            'grains_control_moves': summary['sand_off_peak_p95_m'] > summary['sand_on_peak_p95_m'],
            'liquid_at_rest': summary['water_on_peak_p95_m'] < REST_M,
            'liquid_control_moves': summary['water_off_peak_p95_m'] >
                                    summary['water_on_peak_p95_m'],
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
