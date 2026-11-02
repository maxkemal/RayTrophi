"""Elastic MPM solid acceptance (docs/dev/ELASTIK_MPM.md, E1 + E2 2026-11-02).

Short on purpose (MATTER_ROADMAP_2026_10 test rule). Same drop for every arm: a
15 cm cube seeded above the floor of an empty domain falls for 2 s. The body's
shape is read from fluid.state_digest PERCENTILE bounds (bounds_p05/p95): the
min/max bounds follow a few strays and read 0.35 vs 0.64 m width from run to
run on the same jelly (2026-11-02 live).

  Gelatin (elastic, damped) must land, keep its shape and calm down. Dough
  (elastoplastic) keeps a dent, so it ends lower than Gelatin, and once it
  stops its stored velocity must stop too (E1 live: a static Dough reported
  0.11 m/s -- an explicit-viscosity flip per substep). Water is the control:
  the same drop spreads it flat. Tear (E2): a Gelatin derivative with a low
  fracture strain dropped from higher tears; the stock Gelatin from 20 cm
  does not.

Gates:
  * lane_runs_elastic: list_domains mpm_law == "elastic" and every particle is
    an elastic model particle (a wrong lane or law fails here, before shape);
  * gelatin_lands: its lowest particle reaches the floor;
  * gelatin_keeps_shape: final p05-p95 height >= 70 % and width <= 140 % of the
    seed's;
  * gelatin_calms: final mean speed <= 25 % of the landing peak;
  * dough_keeps_dent / dough_holds_body: lower than Gelatin, >= 35 % of seed;
  * dough_stops: final mean speed < 0.03 m/s (live 0.17: a once-per-frame
    gravity kick rang the resting body, not a substep flip);
  * water_spreads: Water's final height < 50 % of the seed (control);
  * gelatin_intact: stock Gelatin detaches <= 0.5 % of its parcels (a corner
    parcel may crack; 2026-11-02 live the old threshold cracked ~3 %);
  * tear_tears: the low-fracture derivative reaches detached parcels;
  * finite: particle count constant, bounds finite in every arm.
"""
import json
import math
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

DT = 1 / 24
FRAMES = 48
SIDE = 0.15
LOW = 0.20
TEAR_LOW = 0.45
TEAR_SUBSTANCE = 'T_Elastic_TearGel'
OUT = Path(tempfile.gettempdir()) / 'rt_elastic_mpm.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}

    def row(name):
        return {d['name']: d for d in call('fluid.list_domains')['domains']}[name]

    def arm(label, substance, low=LOW):
        name = 'T_Elastic_' + label
        call('fluid.create_domain', name=name, type='fluid', domain_min=[-.4, 0, -.4],
             domain_max=[.4, .7, .4], voxel_size=.025)
        try:
            call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
                 boundary='closed', default_substance=substance,
                 thermal_liquid_enabled=False)
            call('fluid.settle', domain=name, on_seed=False, request=False)
            call('timeline.set_frame', frame=0)
            h = SIDE / 2
            call('fluid.seed', domain=name, seed_min=[-h, low, -h],
                 seed_max=[h, low + SIDE, h], particles_per_cell=8, replace=True)
            seeded = call('fluid.state_digest', domain=name)
            samples = []
            for frame in range(FRAMES):
                call('fluid.step', dt=DT)
                if frame % 4 == 3 or frame == FRAMES - 1:
                    d = call('fluid.state_digest', domain=name)
                    r = row(name)
                    samples.append({
                        'frame': frame + 1, 'particles': d['particles'],
                        'min': d['bounds_min'], 'max': d['bounds_max'],
                        'p05': d['bounds_p05'], 'p95': d['bounds_p95'],
                        'mean_speed': d['mean_speed'], 'mpm_law': r.get('mpm_law'),
                        'elastic_particles': r.get('elastic_model_particles'),
                        'max_damage': r.get('granular_max_damage'),
                        'detached': r.get('granular_detached'),
                        'substeps': r.get('granular_solver_substeps')})
            data[label] = {'seeded': seeded, 'samples': samples}
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return data[label]
        finally:
            call('fluid.remove_domain', domain=name)

    def height(s):
        return s['p95'][1] - s['p05'][1]

    def width(s):
        return max(s['p95'][0] - s['p05'][0], s['p95'][2] - s['p05'][2])

    def seed_shape(a):
        return {'p05': a['seeded']['bounds_p05'], 'p95': a['seeded']['bounds_p95']}

    def finite(a):
        n = a['seeded']['particles']
        return n > 0 and all(s['particles'] == n and all(
            math.isfinite(v) for v in s['min'] + s['max'] + s['p05'] + s['p95'])
            for s in a['samples'])

    created = False
    try:
        call('substance.derive', name=TEAR_SUBSTANCE, based_on='Gelatin')
        created = True
        call('substance.set', name=TEAR_SUBSTANCE,
             fields={'elastic_fracture_strain': 0.12, 'elastic_damage_rate': 60.0})
        gel = arm('gelatin', 'Gelatin')
        dough = arm('dough', 'Dough')
        water = arm('water', 'Water')
        tear = arm('tear', TEAR_SUBSTANCE, TEAR_LOW)
        seed_h, seed_w = height(seed_shape(gel)), width(seed_shape(gel))
        g_end, d_end, w_end = gel['samples'][-1], dough['samples'][-1], water['samples'][-1]
        g_peak = max(s['mean_speed'] for s in gel['samples'])
        summary = {
            'seed_height_m': seed_h, 'seed_width_m': seed_w,
            'gelatin_height_m': height(g_end), 'gelatin_width_m': width(g_end),
            'gelatin_min_y_m': min(s['min'][1] for s in gel['samples']),
            'gelatin_peak_speed': g_peak, 'gelatin_end_speed': g_end['mean_speed'],
            'gelatin_max_damage': max((s['max_damage'] or 0.0) for s in gel['samples']),
            'gelatin_detached': max((s['detached'] or 0) for s in gel['samples']),
            'dough_height_m': height(d_end), 'dough_end_speed': d_end['mean_speed'],
            'water_height_m': height(w_end),
            'tear_detached': max((s['detached'] or 0) for s in tear['samples']),
            'tear_max_damage': max((s['max_damage'] or 0.0) for s in tear['samples']),
            'mpm_law': g_end['mpm_law'],
            'substeps': {'gelatin': g_end['substeps'], 'dough': d_end['substeps']},
        }
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        verdicts = {
            'lane_runs_elastic': g_end['mpm_law'] == 'elastic' and
                                 g_end['elastic_particles'] == g_end['particles'],
            'gelatin_lands': summary['gelatin_min_y_m'] < 0.03,
            'gelatin_keeps_shape': summary['gelatin_height_m'] >= 0.7 * seed_h and
                                   summary['gelatin_width_m'] <= 1.4 * seed_w,
            'gelatin_calms': summary['gelatin_end_speed'] <= 0.25 * g_peak,
            'dough_keeps_dent': summary['dough_height_m'] < summary['gelatin_height_m'],
            'dough_holds_body': summary['dough_height_m'] >= 0.35 * seed_h,
            'dough_stops': summary['dough_end_speed'] < 0.03,
            'water_spreads': summary['water_height_m'] < 0.5 * seed_h,
            'gelatin_intact': summary['gelatin_detached'] <=
                              0.005 * gel['seeded']['particles'],
            'tear_tears': summary['tear_detached'] > 0,
            'finite': finite(gel) and finite(dough) and finite(water) and finite(tear),
        }
        data['verdicts'] = verdicts
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        if created:
            call('substance.remove', name=TEAR_SUBSTANCE)
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
