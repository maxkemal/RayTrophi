"""Liquid column (2026-10-20): does a 1 cm water slab keep its volume?

Measured 2026-10-19: a tank-wide 20 cm slab settles 15% low at 1.25 cm and
26-32% low at 1 cm voxels, resting or dropped; 5 and 2.5 cm hold. Lead
suspect: the liquid pressure solve's iteration cap (24) cannot build the
hydrostatic column over 20+ cells. This test sweeps the cap and reads the
solver's own report. Gates:
  * truncation_reported: a seed cut by a small Max Particles says so
    (seed_truncated, emitted < requested, top below the region top);
  * pressure_reported: the solve reports its cap and iterations used;
  * level_holds_at_400: with 400 iterations the settled level is within 5%
    of the level the particle count implies.
The per-cap numbers go to docs/dev/liquid_column_live.json; the decision
(raise / scale the cap, multigrid) is read from them, not from a gate.
"""
import json
from pathlib import Path

from rt_ipc import RtIpc

DT = 1 / 60
VOXEL = .01
PPC = 8
THICK = .2
AREA = .6 * .6
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/dev/liquid_column_live.json'


def main():
    client = RtIpc()
    call = client.call
    data = {}
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'

    def row(name):
        return {d['name']: d for d in call('fluid.list_domains')['domains']}[name]

    def domain(name, voxel, budget):
        call('fluid.create_domain', name=name, type='fluid', domain_min=[-.3, 0, -.3],
             domain_max=[.3, .5, .3], voxel_size=voxel)
        call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', thermal_liquid_enabled=False,
             max_particles=budget)
        call('fluid.settle', domain=name, on_seed=False, request=False)

    try:
        # Truncation: a 5 cm slab that needs 12*12*5*8 = 5760 parcels, budget 2000.
        name = 'T_Column_Cut'
        domain(name, .05, 2000)
        try:
            call('timeline.set_frame', frame=0)
            call('fluid.seed', domain=name, seed_min=[-.3, 0, -.3], seed_max=[.3, THICK, .3],
                 particles_per_cell=PPC, replace=True)
            r = row(name)
            data['cut'] = {k: r[k] for k in r if k.startswith('seed_')}
        finally:
            call('fluid.remove_domain', domain=name)

        for cap in (24, 100, 400):
            name = 'T_Column_%d' % cap
            domain(name, VOXEL, 4000000)
            try:
                call('fluid.set_param', domain=name, pressure_iterations=cap)
                call('timeline.set_frame', frame=0)
                call('fluid.seed', domain=name, seed_min=[-.3, 0, -.3], seed_max=[.3, THICK, .3],
                     particles_per_cell=PPC, replace=True)
                seeded = row(name)
                ys, iters, res = [], [], []
                for f in range(150):
                    call('fluid.step', dt=DT)
                    if f >= 120:
                        ys.append(call('fluid.state_digest', domain=name)['centroid'][1])
                        r = row(name)
                        iters.append(r['pressure_cg_iterations'])
                        res.append(r['pressure_cg_final_relative_residual'])
                expected = seeded['particle_count'] / PPC * VOXEL ** 3 / AREA
                level = 2 * sum(ys) / len(ys)
                data['cap%d' % cap] = {
                    'particles': seeded['particle_count'],
                    'seed_truncated': seeded['seed_truncated'],
                    'pressure_iterations_set': r['pressure_iterations'],
                    'cg_max_iterations': r['pressure_cg_max_iterations'],
                    'cg_iterations_mean': sum(iters) / len(iters),
                    'cg_residual_mean': sum(res) / len(res),
                    'multigrid': r['pressure_cg_multigrid'],
                    'expected_level_m': expected, 'settled_level_m': level,
                    'level_ratio': level / expected,
                }
                print('cap', cap, json.dumps(data['cap%d' % cap]), flush=True)
                OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            finally:
                call('fluid.remove_domain', domain=name)

        cut = data['cut']
        verdicts = {
            'truncation_reported': cut['seed_truncated'] and
                                   cut['seed_emitted_particles'] < cut['seed_requested_particles'] and
                                   cut['seed_top_y'] < cut['seed_region_top_y'],
            # Reported, and the cap the solve used follows the setting (some
            # paths scale it, so not necessarily equal).
            'pressure_reported': all(data['cap%d' % c]['cg_iterations_mean'] > 0 and
                                     data['cap%d' % c]['pressure_iterations_set'] == c
                                     for c in (24, 100, 400)) and
                                 data['cap24']['cg_max_iterations'] <
                                 data['cap100']['cg_max_iterations'] <
                                 data['cap400']['cg_max_iterations'],
            'level_holds_at_400': abs(data['cap400']['level_ratio'] - 1.0) < .05,
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
