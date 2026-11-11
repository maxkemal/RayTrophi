"""Water meeting snow keeps flowing, and the answer no longer depends on the step.
(docs/dev/MATTER_SIVI_MPM_BASINC_BAGLASIMI.md, step 2)

A water column collapses onto a snow block in a closed tank, on the liquid's
own clock (always, since 2026-11-11): the porous projection carries the normal
coupling and sim_matter_liquid_drag the pore drag as a rate. Before step 2 the
per-call contact impulse stopped the water within a second (synchronous |p| at
1 s 0.08 kg m/s vs 5.0 for the same water alone) and the own-clock result moved
with the liquid step (1.8 / 0.8 at liquid dt 1/96 / 1/384).

The liquid momentum |p| is averaged over 0.5..1.5 s: a single instant is phase
noise (the same setup read 1.75 and 4.87).
Arms: water alone (no snow) at dt 1/24; snow + water at call dt 1/24, 1/96 and
1/384 (the liquid step follows the call: liquid_steps per call reported).
Step 2f (2026-11-10): the drag and the pressure's push on the snow run on the
MPM grid clock, the pressure on the liquid's. With the drag on the liquid step
the pair relaxed to its common velocity (0.58 u for grounded snow) and the
result drifted with the step: 2.6 / 2.9 / 1.3 at 1/24 / 1/96 / 1/384 against
~0.72 resolved.
Gates:
  * converged: snow arms at 1/24 and 1/96 each within 15 % of 1/384 (the old
    "1/24 vs 1/96" gate passed while both were 3x off);
  * flows: snow arm mean |p| >= 5 % of water alone (water entering snow is
    absorbed by the pores and stops -- the resolved answer is ~11 %, the old
    30 % gate was calibrated on the drifting result);
  * snow_stays: snow centre of mass moves < 2 voxels horizontally;
  * mass: parcels and kg per model unchanged;
  * stable: snow arms' max speed <= 1.2x the water alone's (a 0.3 m dam break
    itself splashes past 3 m/s).
Output: %TEMP%/rt_matter_liquid_drag.json
"""
import json
import math
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

H = 0.025
OUT = Path(tempfile.gettempdir()) / 'rt_matter_liquid_drag.json'


def model_counts(models):
    return {m['model']: [m['particles'], round(m['mass_kg'], 5)]
            for m in models.get('models', []) if m.get('particles')}


def run_arm(call, snow, dt):
    name = 'T_Drag'
    call('fluid.create_domain', name=name, type='matter', domain_min=[-.25, 0, -.25],
         domain_max=[.25, .4, .25], voxel_size=H)
    try:
        call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', thermal_liquid_enabled=False)
        call('fluid.settle', domain=name, on_seed=False, request=False)
        call('fluid.set_seed_source_settings', domain=name,
             settings={'particles_per_cell': 8, 'settle': False})
        if snow:
            call('fluid.add_seed_source', domain=name, settings={
                'name': 'Snow block', 'substance': 'Snow',
                'bounds_min': [-.24, 0.0, -.2], 'bounds_max': [0.0, 0.1, .2]})
        call('fluid.add_seed_source', domain=name, settings={
            'name': 'Water column', 'substance': 'Water',
            'bounds_min': [0.05, 0.0, -.2], 'bounds_max': [.24, 0.3, .2]})
        call('fluid.apply_seed_sources', domain=name)
        call('timeline.set_frame', frame=0)
        m0 = call('fluid.matter_models', domain=name)
        counts0 = model_counts(m0)
        g0 = (m0.get('acceptance_metrics') or {}).get('granular') or {}
        steps = int(round(1.5 / dt))
        sample_every = max(1, int(round((1 / 24) / dt)))
        samples, max_speed, liquid_steps = [], 0.0, None
        for i in range(1, steps + 1):
            call('fluid.step', dt=dt)
            if i % sample_every == 0:
                info = [x for x in call('fluid.list_domains')['domains'] if x['name'] == name][0]
                max_speed = max(max_speed, float(info.get('live_max_speed_m_s', 0.0) or 0.0))
                if i * dt >= 0.5 - 1e-6:
                    m = call('fluid.matter_models', domain=name)
                    p = [x for x in m['models'] if x['model'] == 'fluid'][0]['momentum_kg_m_s']
                    samples.append(math.sqrt(sum(v * v for v in p)))
            if i == 1:
                ms = call('fluid.step_stats', domain=name).get('mixed_stages', {})
                liquid_steps = (ms.get('liquid_steps'), ms.get('grid_steps'))
        m = call('fluid.matter_models', domain=name)
        g = (m.get('acceptance_metrics') or {}).get('granular') or {}
        out = {'snow': snow, 'dt': dt, 'liquid_steps/grid_steps': liquid_steps,
               'mean_p': sum(samples) / max(len(samples), 1), 'samples': len(samples),
               'max_speed': max_speed, 'counts0': counts0, 'counts': model_counts(m),
               'snow_com0': g0.get('dry_center_of_mass'), 'snow_com': g.get('dry_center_of_mass')}
        print(json.dumps(out), flush=True)
        return out
    finally:
        call('fluid.remove_domain', domain=name)


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    data = {}
    try:
        water = run_arm(call, False, 1 / 24)
        coarse = run_arm(call, True, 1 / 24)
        fine = run_arm(call, True, 1 / 96)
        ref = run_arm(call, True, 1 / 384)
        data = {'water': water, 'snow_1_24': coarse, 'snow_1_96': fine, 'snow_1_384': ref}
        drift = 0.0
        if coarse['snow_com'] and coarse['snow_com0']:
            drift = math.hypot(coarse['snow_com'][0] - coarse['snow_com0'][0],
                               coarse['snow_com'][2] - coarse['snow_com0'][2])
        verdicts = {
            'converged': all(abs(a['mean_p'] - ref['mean_p']) <= 0.15 * max(ref['mean_p'], 1e-6)
                             for a in (coarse, fine)),
            'flows': coarse['mean_p'] >= 0.05 * water['mean_p'],
            'snow_stays': drift < 2 * H,
            'mass': all(a['counts'] == a['counts0'] for a in (water, coarse, fine, ref)),
            'stable': max(coarse['max_speed'], fine['max_speed'], ref['max_speed']) <= 1.2 * water['max_speed'],
        }
        data['verdicts'] = verdicts
        print('mean |p|: water %.2f, snow 1/24 %.2f, 1/96 %.2f, 1/384 %.2f; snow drift %.3f m' % (
            water['mean_p'], coarse['mean_p'], fine['mean_p'], ref['mean_p'], drift), flush=True)
        for k, v in verdicts.items():
            print('PASS' if v else 'FAIL', k, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
