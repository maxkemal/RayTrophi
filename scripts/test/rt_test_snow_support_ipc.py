"""Snow settle sleeps only what is SUPPORTED (GranularConstitutive.h settleParticle).

2026-11-03 live: a snow block seeded in mid air with settle-on-seed hung from
its top for 24 frames while its bottom tore away. Settle damped and slept any
parcel under pressure; gravity's ~0.025 m/s a substep stayed under the sleep
speed and was zeroed every substep. Now a parcel must show it does not
accelerate (mean acceleration along gravity < 0.25 g over 0.2 s) first.

Arms (Snow, h 5 cm, closed domain, no collider):
  air    a 0.5 x 0.3 x 0.5 m block at y 0.6-0.9, settle on seed, 6 frames;
  floor  the same block on the floor (y 0-0.3), settle on seed, 48 frames.

Gates:
  * air_falls: the top drops >= 70 % of the free fall 0.5 g t^2 in 0.25 s
    (before the bottom lands at ~0.35 s);
  * air_intact: parcels with damage > 0.5 stay <= 2 % (a free-falling block
    carries no stress; the hanging top tore it);
  * floor_rests: mean speed <= 0.05 m/s at the end;
  * floor_sleeps: >= 30 % of the parcels asleep at the end (support works
    for a real pile, sleep is not just switched off);
  * floor_no_ratchet: max plastic J grows <= 0.03 over the last 24 frames
    (the rest-jitter ratchet SettleParams::damping_max_speed was added for).
Output: %TEMP%/rt_snow_support.json
"""
import json
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

DT = 1 / 24
DOMAIN = 'T_SnowSupport'
OUT = Path(tempfile.gettempdir()) / 'rt_snow_support.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}

    def arm(label, y0, frames):
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-.5, 0, -.5],
             domain_max=[.5, 1.5, .5], voxel_size=0.05)
        try:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 backend='vulkan', boundary='closed', default_substance='Snow',
                 thermal_liquid_enabled=False)
            call('timeline.set_frame', frame=0)
            call('fluid.settle', domain=DOMAIN, on_seed=True, request=False)
            call('fluid.seed', domain=DOMAIN, seed_min=[-.25, y0, -.25],
                 seed_max=[.25, y0 + .3, .25], particles_per_cell=8, replace=True)
            seeded = call('fluid.state_digest', domain=DOMAIN)
            rows = []
            for frame in range(1, frames + 1):
                call('fluid.step', dt=DT)
                d = call('fluid.state_digest', domain=DOMAIN)
                g = call('fluid.get', domain=DOMAIN)
                rows.append({'frame': frame, 'particles': d['particles'],
                             'max_y': d['bounds_max'][1], 'p95_y': d['bounds_p95'][1],
                             'min_y': d['bounds_min'][1], 'speed': d['mean_speed'],
                             'sleeping': g.get('granular_sleeping'),
                             'damage_over_50': g.get('granular_damage_over_50'),
                             'max_jp': g.get('granular_max_plastic_jacobian')})
            data[label] = {'seeded': {k: seeded[k] for k in ('particles', 'bounds_max')},
                           'rows': rows}
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return seeded, rows
        finally:
            call('fluid.remove_domain', domain=DOMAIN)

    try:
        a_seed, air = arm('air', 0.6, 6)
        f_seed, floor = arm('floor', 0.0, 48)
        n_air = max(a_seed['particles'], 1)
        n_floor = max(f_seed['particles'], 1)
        t = 6 * DT
        free = 0.5 * 9.81 * t * t
        summary = {
            'air_top_drop_m': a_seed['bounds_max'][1] - air[-1]['max_y'],
            'air_free_fall_m': free,
            'air_damage_fraction': (air[-1]['damage_over_50'] or 0) / n_air,
            'floor_speed': floor[-1]['speed'],
            'floor_sleeping_fraction': (floor[-1]['sleeping'] or 0) / n_floor,
            'floor_jp_growth': (floor[-1]['max_jp'] or 1.0) - (floor[23]['max_jp'] or 1.0),
        }
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        verdicts = {
            'air_falls': summary['air_top_drop_m'] >= 0.7 * free,
            'air_intact': summary['air_damage_fraction'] <= 0.02,
            'floor_rests': summary['floor_speed'] <= 0.05,
            'floor_sleeps': summary['floor_sleeping_fraction'] >= 0.30,
            'floor_no_ratchet': summary['floor_jp_growth'] <= 0.03,
        }
        data['verdicts'] = verdicts
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
