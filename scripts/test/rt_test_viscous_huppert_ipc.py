"""Viscous gravity current vs Huppert (1982): honey at real size and density.
(docs/dev/VISKOZ_SIVI_URETIM_KALITESI.md)

48 ml of the built-in Honey (rho 1420 kg/m3, nu 0.007 m2/s -> mu 9.94 Pa s) is
released as a 4 x 3 x 4 cm block on the closed floor of a 24 x 4 x 24 cm domain
at h = 2 mm. Once thin, a viscous puddle spreads as
    r(t) = 0.894 (rho g V^3 t / (3 mu))^(1/8)
with no tunable constant. Parcels carry volume, so the x projection of the
Huppert profile h ~ (1 - r^2/R^2)^(1/3) has its 95 % quantile at 0.7525 R
(0.8055 for a uniform disc; 0.9 was used first and hid a third of the error):
r = half the p05..p95 extent / 0.7525, averaged over x and z.

2026-11-09, fixed 16 red-black GS sweeps: r was 1.31x Huppert at 1 s and 1.51x
at 10 s, exponent 0.186 -- an effective viscosity of 1/9..1/25. 64 sweeps:
1.16x..1.26x. The answer changed with the sweep count: unconverged solver.

Gates:
  * radius_1s / radius_4s: within 10 % of Huppert (the film is still >= 3
    voxels thick here; later it reaches 1-2 voxels and the grid, not the law,
    sets the drag);
  * spreading_exponent: log(r4/r1)/log(4) in [0.09, 0.16] (law: 0.125);
  * viscosity_converged: no GPU viscosity solve hit its iteration cap;
  * finite: parcel count constant, bounds finite.
  * layer_runs_honey: the user's 2026-11-09 setup -- a Water-default domain with
    a Honey seed-source layer. It ran Water's physics (nu 0, free-slip walls,
    viscosity_pcg_iterations 0): honey flowed like water. Now the domain reports
    liquid_physics_substance Honey, nu = Honey's, the viscous solve runs, and
    the radius at 1 s is within 10 % of Huppert like the direct arm. Its solver
    hints follow too: wall slip 0 (Honey), not Water's free-slip 1.
Output: %TEMP%/rt_viscous_huppert.json
"""
import json
import math
from pathlib import Path
import tempfile
import time

from rt_ipc import RtIpc

DT = 1 / 24
H = 0.002
DOMAIN = 'T_Huppert'
BLOCK = (0.04, 0.03, 0.04)
QUANTILE = 0.7525  # x95 / R of the Huppert profile's volume projection
OUT = Path(tempfile.gettempdir()) / 'rt_viscous_huppert.json'


def honey_layer_arm(call, rho, mu):
    """Water-default domain, Honey seed layer: does the liquid run Honey?"""
    name = DOMAIN + '_Layer'
    call('fluid.create_domain', name=name, type='fluid', domain_min=[-.12, 0, -.12],
         domain_max=[.12, .04, .12], voxel_size=H)
    try:
        call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', thermal_liquid_enabled=False)
        call('fluid.settle', domain=name, on_seed=False, request=False)
        call('fluid.set_seed_source_settings', domain=name,
             settings={'particles_per_cell': 8, 'settle': False})
        call('fluid.add_seed_source', domain=name, settings={
            'name': 'Honey layer', 'substance': 'Honey',
            'bounds_min': [-BLOCK[0] / 2, 0.0, -BLOCK[2] / 2],
            'bounds_max': [BLOCK[0] / 2, BLOCK[1], BLOCK[2] / 2]})
        call('fluid.apply_seed_sources', domain=name)
        call('timeline.set_frame', frame=0)
        seeded = call('fluid.state_digest', domain=name)
        iterations = 0
        for _ in range(24):
            call('fluid.step', dt=DT)
            iterations = max(iterations, int(call('fluid.step_stats', domain=name)
                                             .get('viscosity_pcg_iterations', 0) or 0))
        d = call('fluid.state_digest', domain=name)
        info = [x for x in call('fluid.list_domains')['domains'] if x['name'] == name][0]
        volume = seeded['particles'] / 8.0 * info['voxel_size'] ** 3
        rx = (d['bounds_p95'][0] - d['bounds_p05'][0]) / 2 / QUANTILE
        rz = (d['bounds_p95'][2] - d['bounds_p05'][2]) / 2 / QUANTILE
        law = 0.894 * (rho * 9.81 * volume ** 3 * 1.0 / (3 * mu)) ** 0.125
        out = {'liquid_physics_substance': info.get('liquid_physics_substance'),
               'kinematic_viscosity': info.get('kinematic_viscosity'),
               'viscosity_wall_slip': info.get('viscosity_wall_slip'),
               'viscosity_pcg_iterations': iterations, 'particles': seeded['particles'],
               'ratio_1s': ((rx + rz) / 2) / law if law > 0 else None}
        print('layer arm', json.dumps(out), flush=True)
        return out
    finally:
        call('fluid.remove_domain', domain=name)


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    data = {}
    fields = call('substance.get', name='Honey')['fields']
    rho = fields['liquid_density']
    mu = rho * fields['liquid_kinematic_viscosity']
    call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-.12, 0, -.12],
         domain_max=[.12, .04, .12], voxel_size=H)
    try:
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Honey', thermal_liquid_enabled=False)
        call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
        call('timeline.set_frame', frame=0)
        call('fluid.seed', domain=DOMAIN, seed_min=[-BLOCK[0] / 2, 0.0, -BLOCK[2] / 2],
             seed_max=[BLOCK[0] / 2, BLOCK[1], BLOCK[2] / 2], particles_per_cell=8,
             replace=True)
        seeded = call('fluid.state_digest', domain=DOMAIN)
        # The seed snaps to cells: this 4x3x4 cm box lands as 7056 cells = 56.4 ml
        # at h 2 mm, not 48 ml, and r ~ V^(3/8) read 6 % high with the box
        # volume (2026-11-09). Use what was seeded: parcels / ppc cells.
        voxel = [x for x in call('fluid.list_domains')['domains']
                 if x['name'] == DOMAIN][0]['voxel_size']
        volume = seeded['particles'] / 8.0 * voxel ** 3
        rows = []
        unconverged = 0
        max_iterations = 0
        started = time.time()
        for frame in range(1, 97):
            call('fluid.step', dt=DT)
            stats = call('fluid.step_stats', domain=DOMAIN)
            unconverged += int(stats.get('viscosity_pcg_unconverged', 0) or 0)
            max_iterations = max(max_iterations, int(stats.get('viscosity_pcg_iterations', 0) or 0))
            if frame in (24, 48, 96):
                d = call('fluid.state_digest', domain=DOMAIN)
                t = frame * DT
                rx = (d['bounds_p95'][0] - d['bounds_p05'][0]) / 2 / QUANTILE
                rz = (d['bounds_p95'][2] - d['bounds_p05'][2]) / 2 / QUANTILE
                law = 0.894 * (rho * 9.81 * volume ** 3 * t / (3 * mu)) ** 0.125
                # Height next to radius: a smaller radius from LOST volume
                # (parcels compacting) must not pass for a stiffer liquid.
                rows.append({'t': t, 'radius': (rx + rz) / 2, 'huppert': law,
                             'p95_height_m': d['bounds_p95'][1],
                             'particles': d['particles'], 'min': d['bounds_min'],
                             'max': d['bounds_max']})
                print(rows[-1], flush=True)
        r1, r4 = rows[0], rows[2]
        summary = {
            'mu_pa_s': mu, 'volume_m3': volume, 'wall_s': time.time() - started,
            'ratio_1s': r1['radius'] / r1['huppert'], 'ratio_4s': r4['radius'] / r4['huppert'],
            'exponent': math.log(r4['radius'] / r1['radius']) / math.log(4.0),
            'viscosity_pcg_max_iterations': max_iterations,
            'viscosity_pcg_unconverged': unconverged,
        }
        data = {'seeded': seeded, 'rows': rows, 'summary': summary}
        print(json.dumps(summary), flush=True)
        n = seeded['particles']
        verdicts = {
            'radius_1s': abs(summary['ratio_1s'] - 1.0) <= 0.10,
            'radius_4s': abs(summary['ratio_4s'] - 1.0) <= 0.10,
            'spreading_exponent': 0.09 <= summary['exponent'] <= 0.16,
            'viscosity_converged': unconverged == 0 and max_iterations > 0,
            'finite': n > 0 and all(r['particles'] == n and all(
                math.isfinite(v) for v in r['min'] + r['max']) for r in rows),
        }
        layer = honey_layer_arm(call, rho, mu)
        data['layer'] = layer
        verdicts['layer_runs_honey'] = (
            layer['liquid_physics_substance'] == 'Honey' and
            abs((layer['kinematic_viscosity'] or 0) - fields['liquid_kinematic_viscosity']) < 1e-6 and
            # Its solver hints too: Water's free-slip walls let honey slide.
            abs((layer['viscosity_wall_slip'] if layer['viscosity_wall_slip'] is not None else 1.0) -
                fields['solver_viscosity_wall_slip']) < 1e-6 and
            layer['viscosity_pcg_iterations'] > 0 and layer['particles'] > 0 and
            layer['ratio_1s'] is not None and abs(layer['ratio_1s'] - 1.0) <= 0.10)
        data['verdicts'] = verdicts
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        call('fluid.remove_domain', domain=DOMAIN)
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
