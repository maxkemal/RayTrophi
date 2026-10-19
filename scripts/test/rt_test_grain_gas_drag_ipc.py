"""T5 acceptance (2026-10-18): grains feel the Matter domain's own gas.

exchangeMatterGrainGasDrag: Di Felice drag against the gas lane's velocity,
integrated exactly over the frame and handed to the DEM substeps as a force;
the opposite impulse goes onto the gas MAC faces, limited per cell so the air
cannot overshoot the mixture's equilibrium.

Two arms in one closed Matter domain with the same falling Sand:
  still - no gas source. The drag is gated on a gas lane that stepped AND
          moves, so nothing couples: vacuum-calibrated grain scenes (free fall,
          repose, CPU reference) must not change.
  wind  - a horizontal gas source (+x) ON the fall path. Grains must take +x
          drag impulse while they fall through it, the gas must take the
          opposite impulse, and the pair must cancel.

★ 2026-10-19 scene fix: the source used to sit 0.7 m beside the fall path.
In the closed box the jet turned before reaching the sand, and the pile on the
floor sat in the RETURN flow (-x): drag x -0.007 N s against +0.147 N s of
plain air resistance on the fall. The coupling was right, the wind missed. The
x gate is now measured over the fall window (the pile then rests in whatever
the box's circulation is, which is not what this test is about); the totals
are still reported.

Reads fluid.matter_models -> grain_diagnostics.liquid (the same core the
panel and Python read). Scene domains, sources, colliders and fields are
switched off for the run and restored; everything created is removed.
"""

import argparse
import json
import math
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_GrainGas'
SAND = 'T: grain gas sand'
WIND = 'T: grain gas wind'
DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--wind', type=float, default=4.0, help='m/s along +x')
    parser.add_argument('--fall-steps', type=int, default=45,
                        help='steps the sand is airborne (falls 1.6 m in ~0.6 s)')
    args = parser.parse_args()

    client = RtIpc()
    call = client.call
    data = {'arms': {}}
    out = ROOT / 'docs/dev/grain_gas_drag_live.json'

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list'),
                'colliders': call('collider.list')['colliders']}
    assert not original['control']['playing'], 'Pause first'
    created_domain = DOMAIN not in {d['name'] for d in original['domains']}
    created_sources = []

    def run_arm(name, windy):
        call('flow_source.update', name=WIND, enabled=windy)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        arm = {'samples': [], 'coupled_steps': 0, 'drag': [0.0, 0.0, 0.0],
               'fall_drag': [0.0, 0.0, 0.0], 'fall_reaction': [0.0, 0.0, 0.0],
               'reaction': [0.0, 0.0, 0.0], 'max_residual': 0.0,
               'max_relative_speed': 0.0, 'limited_cells': 0, 'grains_max': 0}
        for step in range(1, args.steps + 1):
            call('fluid.step', dt=DT)
            models = call('fluid.matter_models', domain=DOMAIN)
            liquid = (models.get('grain_diagnostics') or {}).get('liquid') or {}
            grains = liquid.get('grains', 0)
            arm['grains_max'] = max(arm['grains_max'], grains)
            coupled = liquid.get('gas_coupled_grains', 0)
            if coupled:
                arm['coupled_steps'] += 1
            for axis in range(3):
                drag = (liquid.get('gas_drag_impulse_n_s') or [0, 0, 0])[axis]
                reaction = (liquid.get('gas_reaction_n_s') or [0, 0, 0])[axis]
                arm['drag'][axis] += drag
                arm['reaction'][axis] += reaction
                if step <= args.fall_steps:
                    arm['fall_drag'][axis] += drag
                    arm['fall_reaction'][axis] += reaction
            arm['max_residual'] = max(arm['max_residual'],
                                      liquid.get('gas_momentum_residual_n_s', 0.0))
            arm['max_relative_speed'] = max(arm['max_relative_speed'],
                                            liquid.get('gas_max_relative_speed_m_s', 0.0))
            arm['limited_cells'] += liquid.get('gas_limited_cells', 0)
            if step % 30 == 0:
                sample = {'step': step, 'grains': grains, 'coupled': coupled,
                          'drag': liquid.get('gas_drag_impulse_n_s'),
                          'reaction': liquid.get('gas_reaction_n_s')}
                arm['samples'].append(sample)
                print(name, json.dumps(sample), flush=True)
        data['arms'][name] = arm
        return arm

    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=False)
        if created_domain:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -.6], domain_max=[1, 2, .6], voxel_size=.1)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Sand', solid_phase=False,
             thermal_liquid_enabled=False)
        call('fluid.set_pore_exchange', domain=DOMAIN, enabled=False,
             wet_response_enabled=False, wet_appearance_enabled=False)
        call('fluid.set_grain_settings', domain=DOMAIN, radius_m=.01, max_substeps=1024)
        call('flow_source.create', name=SAND, domain=DOMAIN, phase='liquid',
             source_mode='point', fluid_substance='Sand', position=[0, 1.6, 0], radius=.08,
             velocity=[0, 0, 0], fluid_velocity_spread=0., use_time_limit=True,
             start_time=0., end_time=.4, fluid_particles_per_second=3000.)
        created_sources.append(SAND)
        # Centred on the fall path (sand leaves y 1.6, falls through y 0.75..1.45).
        call('flow_source.create', name=WIND, domain=DOMAIN, phase='gas',
             position=[0, 1.1, 0], radius=.35, density=.2, temperature=0., fuel=0.,
             velocity=[args.wind, 0, 0])
        created_sources.append(WIND)

        still = run_arm('still', windy=False)
        wind = run_arm('wind', windy=True)

        drag_x = wind['fall_drag'][0]
        drag_size = math.sqrt(sum(v * v for v in wind['drag']))
        verdicts = {
            'grains_born': still['grains_max'] > 0 and wind['grains_max'] > 0,
            # The gate: no moving gas, no air. Vacuum scenes stay bit-identical.
            'still_air_does_not_couple': still['coupled_steps'] == 0,
            'wind_couples': wind['coupled_steps'] > 0,
            'wind_pushes_grains_along_x': drag_x > 0.0,
            'gas_takes_the_opposite': wind['fall_reaction'][0] < 0.0,
            'pair_cancels': wind['max_residual'] <= 1e-4 * max(drag_size, 1e-9) + 1e-9,
        }
        data['verdicts'] = verdicts
        data['reported'] = {'drag_n_s': wind['drag'], 'reaction_n_s': wind['reaction'],
                            'fall_drag_n_s': wind['fall_drag'],
                            'fall_reaction_n_s': wind['fall_reaction'],
                            'limited_cell_steps': wind['limited_cells'],
                            'max_relative_speed_m_s': wind['max_relative_speed']}
        print(json.dumps(data['reported']), flush=True)
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for name in created_sources:
            try:
                call('flow_source.remove', name=name)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        if created_domain:
            try:
                call('fluid.remove_domain', domain=DOMAIN)
            except Exception:  # noqa: BLE001 - leave it disabled instead
                call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=c['enabled'])
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=f['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'],
                 visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        call('timeline.set_frame', frame=0)
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
