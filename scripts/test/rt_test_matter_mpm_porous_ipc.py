"""Slush floats: the MPM lane as a porous solid in the liquid projection.
(docs/dev/MATTER_SIVI_MPM_BASINC_BAGLASIMI.md, step 1)

A snow block (MPM, Stomakhin) lies on the floor of a closed tank, under and
between water. Its pores flood: ice 917 kg/m3 at snow's 400 bulk is a solid
fraction 0.44, so the flooded block weighs 0.44*917 + 0.56*997 = 962 kg/m3,
less than water: it must rise (Archimedes) and float with ~96 % submerged.
The old coupling (sim_matter_contact only: inelastic impulse + friction) had no
pressure force on the snow and kept the block on the floor; it and its A/B
switch were removed 2026-11-11, so the contact arm is gone.

One arm: the porous projection + the pressure's push and pore drag on the MPM
grid clock (always on for liquid + MPM since 2026-11-11).
Gates:
  * porous_active: gpu_status reports the MPM porous projection;
  * floats: the snow centre of mass rises >= 0.08 m in 6 s (2b reached the
    surface in ~4.5 s; 2f rises steadily 5.1 cm -- OPEN, the terminal speed of a
    962 kg/m3 permeable block estimates ~0.2 m/s, much faster than both);
  * near_surface: the snow top (granular bounds_max y) ends within 2 voxels of
    the water surface (lowered by the water the pores took) or above;
  * mass: parcels and kg per model unchanged;
  * stable: snow kinetic energy finite and max speed < 3 m/s.
Output: %TEMP%/rt_matter_mpm_porous.json
"""
import json
import math
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

DT = 1 / 24
H = 0.025
SECONDS = 6.0
WATER_TOP = 0.30
# The flooded pores take water: ice fraction 400/917 of the 0.16 x 0.1 x 0.48 m
# block leaves (1 - 0.436) of it to fill, over the 0.5 x 0.5 m tank.
SURFACE = WATER_TOP - 0.16 * 0.1 * 0.48 * (1.0 - 400.0 / 917.0) / 0.25
OUT = Path(tempfile.gettempdir()) / 'rt_matter_mpm_porous.json'
GAP = 0.001


def model_counts(models):
    return {m['model']: [m['particles'], round(m['mass_kg'], 5)]
            for m in models.get('models', []) if m.get('particles')}


def snow_metrics(models):
    g = (models.get('acceptance_metrics') or {}).get('granular') or {}
    return g.get('dry_center_of_mass'), g.get('bounds_max'), g.get('kinetic_energy_j')


def run_arm(call):
    name = 'T_Porous'
    call('fluid.create_domain', name=name, type='matter', domain_min=[-.25, 0, -.25],
         domain_max=[.25, .45, .25], voxel_size=H)
    try:
        call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', thermal_liquid_enabled=False)
        call('fluid.settle', domain=name, on_seed=False, request=False)
        call('fluid.set_seed_source_settings', domain=name,
             settings={'particles_per_cell': 8, 'settle': False})
        layers = [
            ('Snow block', 'Snow', [-.08, 0.0, -.24], [.08, .1, .24]),
            ('Water left', 'Water', [-.24, 0.0, -.24], [-.08 - GAP, WATER_TOP, .24]),
            ('Water right', 'Water', [.08 + GAP, 0.0, -.24], [.24, WATER_TOP, .24]),
            ('Water top', 'Water', [-.08, .1 + GAP, -.24], [.08, WATER_TOP, .24]),
        ]
        for label, substance, lo, hi in layers:
            call('fluid.add_seed_source', domain=name, settings={
                'name': label, 'substance': substance, 'bounds_min': lo, 'bounds_max': hi})
        call('fluid.apply_seed_sources', domain=name)
        call('timeline.set_frame', frame=0)
        m0 = call('fluid.matter_models', domain=name)
        com0, _, _ = snow_metrics(m0)
        counts0 = model_counts(m0)
        max_speed, status = 0.0, ''
        trace = []
        frames = int(round(SECONDS / DT))
        for f in range(1, frames + 1):
            call('fluid.step', dt=DT)
            info = [x for x in call('fluid.list_domains')['domains'] if x['name'] == name][0]
            max_speed = max(max_speed, float(info.get('live_max_speed_m_s', 0.0) or 0.0))
            if f % 12 == 0:
                m = call('fluid.matter_models', domain=name)
                com, top, _ = snow_metrics(m)
                trace.append((round(f * DT, 2), round(com[1], 4) if com else None,
                              round(top[1], 4) if top else None))
            if f == 1:
                status = call('fluid.step_stats', domain=name).get('gpu_status', '')
        m = call('fluid.matter_models', domain=name)
        com, top, ke = snow_metrics(m)
        out = {'status': status, 'counts0': counts0,
               'counts': model_counts(m), 'com0': com0, 'com': com, 'top': top,
               'snow_ke_j': ke, 'max_speed': max_speed, 'trace': trace}
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
        porous = run_arm(call)
        data = {'porous': porous}
        rise = lambda arm: (arm['com'][1] - arm['com0'][1]) if arm['com'] and arm['com0'] else 0.0
        verdicts = {
            'porous_active': 'MPM porous' in porous['status'],
            'floats': rise(porous) >= 0.08,
            'near_surface': porous['top'] is not None and porous['top'][1] >= SURFACE - 2 * H,
            'mass': porous['counts'] == porous['counts0'] and
                    set(porous['counts0']) == {'fluid', 'granular'},
            'stable': porous['snow_ke_j'] is not None and math.isfinite(porous['snow_ke_j'])
                      and porous['max_speed'] < 3.0,
        }
        data['verdicts'] = verdicts
        print('rise %.3f m, top %.3f m (surface %.3f)' % (
            rise(porous), porous['top'][1] if porous['top'] else -1, SURFACE), flush=True)
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
