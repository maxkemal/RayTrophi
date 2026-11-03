"""MPM domain sleep (docs/dev/MPM_UYUYAN_KAROLAR.md, stage B; MatterMpmSleep.h).

When every MPM parcel of a domain rests (supported, slow, in calm cells resting
on solid or on resting cells), the Matter step runs no substep. A partly
resting domain runs in full: partial freezing was built, measured 5-7 % slower
(the resting frame is bound by its serial ~1800-dispatch substep chain, not by
parcel work) and removed, 2026-11-04.

Arms (Snow, closed 5 m domain, h 0.1 m: the user's scene scale):
  pile_on / pile_off    a 2.25 x 1.75 x 2.25 m block on the floor; it keeps
                        creeping under its weight, so it never rests whole;
  layer_on / layer_off  a 2.25 x 0.25 x 2.25 m layer on the floor: rests;
  air                   the pile seeded 1 m up, 8 frames.

Gates:
  * pile_same: pile_on and pile_off end within 1 mm (centroid, p95 height):
    a domain that does not sleep runs exactly the awake path;
  * layer_sleeps: layer_on reports gpu_status matter_asleep on its last frame;
  * layer_faster: layer_on frames 48-60 cost <= 0.3 x layer_off (total_ms);
  * layer_same: the sleeping layer ends within 2 mm of the awake one. It may
    fall asleep during the settle-on-seed relaxation (frame 1), while the
    awake one keeps settling ~1 mm (2026-11-04: 1.02 mm, h is 100 mm);
  * off_never: layer_off never sleeps (the switch is honoured);
  * air_never: the falling block never sleeps.
Output: %TEMP%/rt_mpm_sleep.json
"""
import json
import math
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

DT = 1 / 24
DOMAIN = 'T_MpmSleep'
OUT = Path(tempfile.gettempdir()) / 'rt_mpm_sleep.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}

    def arm(label, sleep, y0, height, frames):
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-2.5, -2.5, -2.5],
             domain_max=[2.5, 2.5, 2.5], voxel_size=0.1)
        try:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 backend='vulkan', boundary='closed', default_substance='Snow',
                 thermal_liquid_enabled=False)
            call('fluid.set_grain_settings', domain=DOMAIN, sleep=sleep)
            call('timeline.set_frame', frame=0)
            call('fluid.settle', domain=DOMAIN, on_seed=True, request=False)
            call('fluid.seed', domain=DOMAIN, seed_min=[-1.125, y0, -1.125],
                 seed_max=[1.125, y0 + height, 1.125], particles_per_cell=8, replace=True)
            rows = []
            for frame in range(1, frames + 1):
                call('fluid.step', dt=DT)
                st = call('fluid.step_stats', domain=DOMAIN)
                rows.append({'frame': frame, 'total_ms': st.get('total_ms') or 0.0,
                             'resting': st.get('mpm_resting_particles') or 0,
                             'asleep': st.get('gpu_status') == 'matter_asleep'})
            d = call('fluid.state_digest', domain=DOMAIN)
            data[label] = {'rows': rows, 'particles': d['particles'],
                           'centroid': d['centroid'], 'p95': d['bounds_p95']}
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return data[label]
        finally:
            call('fluid.remove_domain', domain=DOMAIN)

    def tail_ms(a):
        tail = [r['total_ms'] for r in a['rows'] if r['frame'] >= 48]
        return sum(tail) / max(len(tail), 1)

    def delta(a, b):
        return max(math.dist(a['centroid'], b['centroid']), abs(a['p95'][1] - b['p95'][1]))

    try:
        pile_on = arm('pile_on', True, -2.5, 1.75, 60)
        pile_off = arm('pile_off', False, -2.5, 1.75, 60)
        layer_on = arm('layer_on', True, -2.5, 0.25, 60)
        layer_off = arm('layer_off', False, -2.5, 0.25, 60)
        air = arm('air', True, -1.5, 1.75, 8)
        summary = {
            'pile_delta_m': delta(pile_on, pile_off),
            'pile_resting_share': pile_on['rows'][-1]['resting'] / max(pile_on['particles'], 1),
            'pile_ms_on': tail_ms(pile_on), 'pile_ms_off': tail_ms(pile_off),
            'layer_particles': layer_on['particles'],
            'layer_asleep_last': layer_on['rows'][-1]['asleep'],
            'layer_first_asleep_frame': next((r['frame'] for r in layer_on['rows']
                                              if r['asleep']), None),
            'layer_ms_on': tail_ms(layer_on), 'layer_ms_off': tail_ms(layer_off),
            'layer_delta_m': delta(layer_on, layer_off),
            'off_ever_asleep': any(r['asleep'] for r in layer_off['rows']),
            'air_ever_asleep': any(r['asleep'] for r in air['rows']),
        }
        summary['layer_speedup'] = summary['layer_ms_off'] / max(summary['layer_ms_on'], 1e-6)
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        verdicts = {
            'pile_same': summary['pile_delta_m'] <= 0.001,
            'layer_sleeps': summary['layer_asleep_last'],
            'layer_faster': summary['layer_ms_on'] <= 0.3 * summary['layer_ms_off'],
            'layer_same': summary['layer_delta_m'] <= 0.002,
            'off_never': not summary['off_ever_asleep'],
            'air_never': not summary['air_ever_asleep'],
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
