"""Resting slab keeps its volume at any particles-per-cell (D1, 2026-10-20).

D0 probe (one-sided density relaxation with a one-standard-error dead band),
2 cm voxels, converged pressure, 600 frames, level / expected:
  ppc 8   0.937 -> 0.889 and still falling (the band let the column sink)
  ppc 27  0.937 -> 1.044 -> 1.032 (noise rectified into a swell)
D1 made the term two-sided with no band. Gates, against each run's own seed
level (2 x centroid reads a slab ~3 % low, so the seed is the reference):
  * ppc8_holds / ppc27_holds: the level after 600 frames within 2 %;
  * no_drift: the last 300 frames move the level less than 1 %.
Numbers go to docs/dev/liquid_density_live.json. Empty scene, Pause first.
"""
import json
from pathlib import Path

from rt_ipc import RtIpc

DT = 1 / 60
VOXEL = .02
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/dev/liquid_density_live.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}
    try:
        for ppc in (8, 27):
            name = 'T_Density_%d' % ppc
            call('fluid.create_domain', name=name, type='fluid', domain_min=[-.3, 0, -.3],
                 domain_max=[.3, .5, .3], voxel_size=VOXEL)
            call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
                 boundary='closed', default_substance='Water', thermal_liquid_enabled=False,
                 max_particles=4000000)
            call('fluid.settle', domain=name, on_seed=False, request=False)
            try:
                call('timeline.set_frame', frame=0)
                call('fluid.seed', domain=name, seed_min=[-.3, 0, -.3], seed_max=[.3, .2, .3],
                     particles_per_cell=ppc, replace=True)
                levels = [2 * call('fluid.state_digest', domain=name)['centroid'][1]]
                for _ in range(6):
                    call('fluid.run', steps=100, dt=DT, sample_every=0, domains=[name])
                    levels.append(2 * call('fluid.state_digest', domain=name)['centroid'][1])
                ratios = [level / levels[0] for level in levels]
                data['ppc%d' % ppc] = {'levels_by_100': levels, 'vs_seed': ratios}
                print('ppc', ppc, json.dumps([round(r, 4) for r in ratios]), flush=True)
            finally:
                call('fluid.remove_domain', domain=name)
        verdicts = {
            'ppc8_holds': abs(data['ppc8']['vs_seed'][-1] - 1.0) < .02,
            'ppc27_holds': abs(data['ppc27']['vs_seed'][-1] - 1.0) < .02,
            'no_drift': all(abs(data[k]['vs_seed'][-1] - data[k]['vs_seed'][3]) < .01
                            for k in ('ppc8', 'ppc27')),
        }
        data['verdicts'] = verdicts
        for key, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', key, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        call('timeline.set_frame', frame=0)
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
