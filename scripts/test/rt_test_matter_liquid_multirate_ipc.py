"""Mixed matter: the liquid on its own clock, against the same path at a fine step.
(docs/dev/MATTER_COKLU_ZAMAN_ADIMI.md, MATTER_SIVI_MPM_BASINC_BAGLASIMI.md)

A snow block (MPM, Stomakhin) rests on the floor of a closed matter domain and
a water column beside it collapses onto it. Since 2026-11-11 a liquid + MPM
domain always runs the liquid's transfer and projection on its own CFL clock
(the synchronous A/B switch matter_liquid_multirate was removed); the reference
is the same path driven at a fine call step (1/384 s, liquid step ~2.6 ms).

Gates:
  * switch_refused: fluid.set_param matter_liquid_multirate is an error, not
    silently ignored;
  * own_clock: at 1/24 mixed_stages.liquid_steps < grid_steps / 4;
  * mass: parcels and kg per constitutive model (fluid.matter_models) are the
    same at the start and after 1 s, in both arms;
  * snow_rest: the snow's dry centre of mass agrees with the fine arm within
    two voxels sideways;
  * shape: after 1 s the p05/p95 extents agree with the fine arm within 2 voxels;
  * stable: max speed finite and below 2x the fine arm's;
  * pressure_share: the liquid pressure is under 30 % of the frame's host time
    at 1/24 (it was 95 % when the liquid projected on every elastic substep).
Known, NOT gated here: the liquid momentum near snow still depends on the step
(rt_test_matter_liquid_drag_ipc.py converged, 2026-11-10).
Output: %TEMP%/rt_matter_liquid_multirate.json
"""
import json
import math
from pathlib import Path
import tempfile
import time

from rt_ipc import RtIpc, RtIpcError

H = 0.025
SECONDS = 1.0
OUT = Path(tempfile.gettempdir()) / 'rt_matter_liquid_multirate.json'


def model_counts(models):
    return {m['model']: [m['particles'], round(m['mass_kg'], 6)]
            for m in models.get('models', []) if m.get('particles')}


def run_arm(call, dt):
    name = 'T_Multirate_%d' % round(1 / dt)
    call('fluid.create_domain', name=name, type='matter', domain_min=[-.25, 0, -.25],
         domain_max=[.25, .4, .25], voxel_size=H)
    try:
        call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', thermal_liquid_enabled=False)
        refused = False
        try:
            call('fluid.set_param', domain=name, matter_liquid_multirate=False)
        except RtIpcError:
            refused = True
        call('fluid.settle', domain=name, on_seed=False, request=False)
        call('fluid.set_seed_source_settings', domain=name,
             settings={'particles_per_cell': 8, 'settle': False})
        call('fluid.add_seed_source', domain=name, settings={
            'name': 'Snow block', 'substance': 'Snow',
            'bounds_min': [-.24, 0.0, -.2], 'bounds_max': [0.0, 0.1, .2]})
        call('fluid.add_seed_source', domain=name, settings={
            'name': 'Water column', 'substance': 'Water',
            'bounds_min': [0.05, 0.0, -.2], 'bounds_max': [.24, 0.3, .2]})
        call('fluid.apply_seed_sources', domain=name)
        call('timeline.set_frame', frame=0)
        counts0 = model_counts(call('fluid.matter_models', domain=name))
        calls = int(round(SECONDS / dt))
        pressure_ms, total_ms, max_speed = 0.0, 0.0, 0.0
        steps = {'grid': 0, 'liquid': 0, 'common': 0}
        started = time.time()
        for i in range(calls):
            call('fluid.step', dt=dt)
            st = call('fluid.step_stats', domain=name)
            ms = st.get('mixed_stages', {})
            pressure_ms += float(ms.get('stages', {}).get('pressure', {}).get('host_wall_ms', 0.0))
            total_ms += float(st.get('total_ms', 0.0) or 0.0)
            steps = {'grid': ms.get('grid_steps', 0), 'liquid': ms.get('liquid_steps', 0),
                     'common': ms.get('common_substeps', 0)}
            if i % max(1, calls // 24) == 0:
                info = [x for x in call('fluid.list_domains')['domains'] if x['name'] == name][0]
                max_speed = max(max_speed, float(info.get('live_max_speed_m_s', 0.0) or 0.0))
        digest = call('fluid.state_digest', domain=name)
        models = call('fluid.matter_models', domain=name)
        snow = (models.get('acceptance_metrics') or {}).get('granular') or {}
        out = {'dt': dt, 'switch_refused': refused, 'counts0': counts0,
               'counts': model_counts(models), 'steps_last_call': steps,
               'pressure_ms_per_call': pressure_ms / calls, 'total_ms_per_call': total_ms / calls,
               'wall_s': time.time() - started, 'max_speed': max_speed,
               'p05': digest.get('bounds_p05'), 'p95': digest.get('bounds_p95'),
               'snow_com': snow.get('dry_center_of_mass'),
               'momentum': {m['model']: m.get('momentum_kg_m_s') for m in models.get('models', [])}}
        print(name, json.dumps(out), flush=True)
        return out
    finally:
        call('fluid.remove_domain', domain=name)


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    data = {}
    try:
        coarse = run_arm(call, 1 / 24)
        fine = run_arm(call, 1 / 384)
        data = {'coarse': coarse, 'fine': fine}
        shape = all(abs(a - b) <= 2 * H for a, b in zip(fine['p05'] + fine['p95'],
                                                         coarse['p05'] + coarse['p95']))
        verdicts = {
            'switch_refused': coarse['switch_refused'] and fine['switch_refused'],
            'own_clock': 0 < coarse['steps_last_call']['liquid'] <
                         coarse['steps_last_call']['grid'] / 4,
            'mass': all(a['counts'] == a['counts0'] for a in (coarse, fine)) and
                    coarse['counts0'] == fine['counts0'] and
                    set(coarse['counts0']) == {'fluid', 'granular'},
            'snow_rest': coarse['snow_com'] is not None and fine['snow_com'] is not None and
                         math.hypot(coarse['snow_com'][0] - fine['snow_com'][0],
                                    coarse['snow_com'][2] - fine['snow_com'][2]) < 2 * H,
            'shape': shape,
            'stable': math.isfinite(coarse['max_speed']) and
                      coarse['max_speed'] <= 2.0 * max(fine['max_speed'], 0.1),
            'pressure_share': coarse['pressure_ms_per_call'] <
                              0.30 * max(coarse['total_ms_per_call'], 1e-6),
        }
        data['verdicts'] = verdicts
        print('1/24: %.0f ms/frame, pressure %.0f ms' % (
            coarse['total_ms_per_call'], coarse['pressure_ms_per_call']), flush=True)
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
