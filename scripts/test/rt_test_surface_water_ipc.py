"""Surface water acceptance (2026-10-18): moisture is MASS.

A Wood (Oak) cube (absorbency .55, water_capacity_kg_m2 1.0) stands half in a
Water tank inside a closed Matter domain; a Thermal field heats its top. The
surface soaks water out of the parcels it touches and dries into the domain's
own gas inventory, paying the latent heat.

Expect, from the ledger and msf.water_budget (no panel reads):
  * absorption rows Water -> "surface" (kg taken out of liquid parcels);
  * vaporization rows "surface" -> the domain (kg into its inventory), each
    with latent_required == latent_accounted;
  * held on the surfaces == absorbed - evaporated - vented, within one frame's
    absorption (the device applies queued water one ambient pass later);
  * the hot top dries (evaporated > 0) while the wet bottom holds water.
Scene domains, sources, fields and created objects are restored/removed.
"""

import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_SurfaceWater'
CUBE = 'T_SurfaceWaterCube'
COLLIDER = 'T_SurfaceWaterCollider'
FIELD = 'T: surface water heat'
DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=240)
    args = parser.parse_args()
    client = RtIpc()
    call = client.call
    data = {'samples': []}
    out = ROOT / 'docs/dev/surface_water_live.json'
    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    created = DOMAIN not in {d['name'] for d in original['domains']}
    field = None
    cube = None
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        if created:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
             backend='vulkan', boundary='closed', default_substance='Water',
             thermal_liquid_enabled=True)
        cube = call('scene.add_primitive', type='cube', name=CUBE, size=0.6)
        cube = cube.get('name', CUBE) if isinstance(cube, dict) else (cube or CUBE)
        call('scene.set_transform', name=cube, translation=[0, 0.3, 0])
        call('collider.create', name=COLLIDER, source_mode='mesh_sdf',
             source_object=cube, msf_substance='Wood (Oak)',
             gas_interaction_enabled=True, gas_ignite_on_contact=True)
        field = call('forcefield.create', type='thermal', name=FIELD)['name']
        call('forcefield.set_param', field=field, enabled=True, shape='sphere',
             position=[0, 0.9, 0], inner_radius=0.2, falloff_radius=0.45,
             thermal_delta_kelvin=300.0)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        # Water up to the cube's middle (0.3 m of the 0..0.6 m cube).
        call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True, fill_level=.15)

        absorbed = evaporated = vented = 0.0
        worst_latent = 0.0
        frame_absorbed_max = 0.0
        stale_frames = 0
        for step in range(1, args.steps + 1):
            call('fluid.step', dt=DT)
            ledger = call('matter.exchanges')
            frame_absorbed = 0.0
            for r in ledger['exchanges']:
                if r['kind'] == 'absorption' and r['target'] == 'surface':
                    frame_absorbed += r['target_mass_kg']
                elif r['kind'] == 'vaporization' and r['source'] == 'surface':
                    if r['target'] == 'atmosphere':
                        vented += r['target_mass_kg']
                    else:
                        evaporated += r['target_mass_kg']
                    worst_latent = max(worst_latent,
                                       abs(r['latent_required_j'] - r['latent_accounted_j']))
            absorbed += frame_absorbed
            frame_absorbed_max = max(frame_absorbed_max, frame_absorbed)
            budget = call('msf.water_budget')
            stale_frames += 1 if budget.get('stale') else 0
            if step % 40 == 0:
                fields = [f for f in call('msf.fields')['fields'] if f['object_key'] == cube]
                held = fields[0]['surface_water_kg'] if fields else 0.0
                sample = {'step': step, 'absorbed_kg': absorbed, 'evaporated_kg': evaporated,
                          'vented_kg': vented, 'held_kg': held,
                          'max_saturation': fields[0]['max_water_saturation'] if fields else 0,
                          'max_kelvin': fields[0]['max_kelvin'] if fields else 0,
                          'budget_ran': budget.get('ran'), 'stale': budget.get('stale')}
                data['samples'].append(sample)
                print(json.dumps(sample), flush=True)
        last = data['samples'][-1]
        books = absorbed - evaporated - vented
        tolerance = 1.5 * frame_absorbed_max + 1e-6
        verdicts = {
            'surface_soaks_from_liquid': absorbed > 0.0,
            'surface_holds_water': last['held_kg'] > 0.0,
            'hot_surface_dries_into_domain': evaporated > 0.0,
            'latent_booked': worst_latent < 1e-3,
            'water_mass_closes': abs(last['held_kg'] - books) <= tolerance,
            'budget_not_starved': stale_frames < args.steps // 2,
        }
        data.update({'absorbed_kg': absorbed, 'evaporated_kg': evaporated, 'vented_kg': vented,
                     'books_kg': books, 'tolerance_kg': tolerance,
                     'stale_frames': stale_frames, 'verdicts': verdicts})
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for method, params in (('forcefield.remove', {'field': field}),
                               ('collider.remove', {'name': COLLIDER}),
                               ('scene.delete', {'name': cube})):
            if params.get('field', 1) and params.get('name', 1):
                try:
                    call(method, **params)
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass
        try:
            if created:
                call('fluid.remove_domain', domain=DOMAIN)
        except Exception:  # noqa: BLE001
            pass
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=f['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'], visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        call('timeline.set_frame', frame=0)
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
