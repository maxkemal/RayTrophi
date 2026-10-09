"""Removed scale ceilings, external IPC only (2026-10-08).

A  grid:  max_auto_resolution and max_particles are stored as written (old
          512 / 10M clamps), and a fine voxel actually allocates > 512 cells
          on an axis.
B  grain: > 1M DEM grains step (old 100k, then 1M dispatch cap) over a flat
          terrain collider with > 4096 faces (old face cap). The bucket clear
          runs > 65535 groups, so a wrong 2-D index fold leaves stale buckets:
          the check is energy - kinetic energy may not exceed the potential
          energy released.

Leaves the scene (paused) for a look; does not save. Run with the app open
and an empty scene:  python scripts\\test\\rt_test_scale_ceilings_ipc.py
"""
import argparse
import json
import time

from rt_ipc import RtIpc
from rt_grain_material import install_grain_material

GRID = 'Lim_Grid'
GRAIN = 'Lim_Grain'
SOURCE = 'Lim_Sand'
FLOOR = 'Lim_Floor'
G = 9.81


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grains', type=int, default=1100000)
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--skip-grid', action='store_true')
    parser.add_argument('--no-collider', action='store_true')
    # Births per frame are bounded by the source's attempt budget; a longer
    # window lets the carried-over remainder reach the requested count.
    parser.add_argument('--window-frames', type=int, default=2)
    parser.add_argument('--grain-radius', type=float, default=.01)
    parser.add_argument('--source-radius', type=float, default=1.6)
    # Explicit on/off keeps A/B reproducible when reusing the previous fixture.
    parser.add_argument('--no-sleep', action='store_true')
    args = parser.parse_args()
    client = RtIpc()
    install_grain_material(client)  # old grain keys -> grain substance
    failures = []

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and ('__error' in result or result.get('ok') is False):
            raise RuntimeError((method, result))
        return result

    def check(name, ok, detail):
        print(('PASS ' if ok else 'FAIL ') + name + ' ' + json.dumps(detail), flush=True)
        if not ok:
            failures.append(name)

    assert not call('sim.control_state')['playing'], 'Pause first'
    existing = {d['name'] for d in call('fluid.list_domains')['domains']}

    # ── A: grid knob and particle budget ────────────────────────────────
    if not args.skip_grid:
        if GRID not in existing:
            call('fluid.create_domain', name=GRID, type='matter',
                 domain_min=[-2, 0, -2], domain_max=[2, .2, 2], voxel_size=.1)
        call('gas.set_settings', domain=GRID, enforce_resource_budget=False)
        call('fluid.set_param', domain=GRID, enabled=True, visible=False, backend='vulkan',
             max_auto_resolution=600, max_particles=50000000)
        call('fluid.set_param', domain=GRID, voxel_size=.007)
        info = call('fluid.get', domain=GRID)
        check('A1 knob stored', info.get('max_auto_resolution') == 600,
              {'max_auto_resolution': info.get('max_auto_resolution')})
        check('A2 max_particles stored', info.get('max_particles') == 50000000,
              {'max_particles': info.get('max_particles')})
        call('fluid.step', dt=1/60)
        grids = call('fluid.get_phase_grids', domain=GRID)
        info = call('fluid.get', domain=GRID)
        res = (grids.get('liquid') or grids.get('gas') or {}).get('resolution')
        check('A3 grid > 512 on an axis', bool(res) and max(res) > 512,
              {'resolution': res, 'voxel': info.get('voxel_size'),
               'knob_after_step': info.get('max_auto_resolution')})
        check('A4 knob not written back', info.get('max_auto_resolution') == 600,
              {'max_auto_resolution': info.get('max_auto_resolution')})
        call('fluid.set_param', domain=GRID, enabled=False)
        call('fluid.remove_domain', domain=GRID)

    # ── B: > 1M grains over a > 4096-face collider ──────────────────────
    if GRAIN not in existing:
        call('fluid.create_domain', name=GRAIN, type='matter',
             domain_min=[-2, 0, -2], domain_max=[2, 4, 2], voxel_size=.1)
    call('gas.set_settings', domain=GRAIN, enforce_resource_budget=False)
    call('fluid.set_param', domain=GRAIN, enabled=True, visible=True, backend='vulkan',
         boundary='closed', default_substance='Sand', render_mode='particles',
         solid_phase=False, thermal_liquid_enabled=False,
         max_particles=args.grains + 1000)
    call('fluid.set_pore_exchange', domain=GRAIN, enabled=False,
         wet_response_enabled=False, wet_appearance_enabled=False)
    # Radius is fixed once a grain is born: clear the previous run first.
    call('timeline.set_frame', frame=0)
    call('fluid.reset')
    call('fluid.set_grain_settings', domain=GRAIN, radius_m=args.grain_radius,
         max_substeps=4096, sleep=not args.no_sleep)
    sleep_settings = call('fluid.grain_settings', domain=GRAIN)
    assert sleep_settings['sleep'] == (not args.no_sleep), sleep_settings
    print('SETTINGS ' + json.dumps(sleep_settings), flush=True)
    floor_faces = 0
    if not args.no_collider:
        objects = call('scene.list_objects')
        if FLOOR not in objects:
            call('terrain.create', name=FLOOR, size=3.8, resolution=128, height_scale=.01)
        call('scene.set_transform', name=FLOOR, translation=[0, .3, 0])
        colliders = {c['name'] for c in call('collider.list')['colliders']}
        call('collider.update' if FLOOR in colliders else 'collider.create', name=FLOOR,
             source_mode='mesh_bvh', source_object=FLOOR, enabled=True)
    sources = {s['name'] for s in call('flow_source.list')}
    call('flow_source.update' if SOURCE in sources else 'flow_source.create', name=SOURCE,
         domain=GRAIN, enabled=True, phase='liquid', source_mode='point',
         fluid_substance='Sand', position=[0, 2.2, 0], radius=args.source_radius, velocity=[0, 0, 0],
         fluid_velocity_spread=0., use_time_limit=True, start_time=0., end_time=args.window_frames/60,
         use_particle_limit=True, max_emitted_particles=args.grains,
         fluid_particles_per_second=args.grains * 1.01 * 60 / args.window_frames,
         fluid_temperature_override=True, fluid_temperature_kelvin=293.15)
    call('timeline.set_frame', frame=0)
    call('fluid.reset')

    first = None
    row = None
    for step in range(1, args.steps + 1):
        start = time.perf_counter()
        call('fluid.step', dt=1/60)
        wall = time.perf_counter() - start
        inv = call('fluid.matter_models', domain=GRAIN)
        g = inv['acceptance_metrics']['granular']
        held = inv['mixed_execution']['step_held']
        runtime = inv['grain_diagnostics'].get('runtime') or {}
        floor_faces = runtime.get('collider_faces', floor_faces)
        row = {'step': step, 'wall_s': round(wall, 2), 'grains': g['particles'],
               'held': held, 'status': inv['mixed_execution']['status'],
               'com_y': g['dry_center_of_mass'][1] if g['dry_center_of_mass'] else None,
               'ke_j': g['kinetic_energy_j'], 'mass_kg': g['dry_mass_kg'],
               'substeps': runtime.get('substeps'), 'collider_faces': runtime.get('collider_faces'),
               # Where the wall time goes: grain host stages vs the GPU wait.
               'host_ms': {k: round(v, 1) for k, v in (runtime.get('host_ms') or {}).items()},
               'working_set_mb': (runtime.get('working_set_bytes') or 0) // (1024 * 1024),
               'resident': runtime.get('state_resident'), 'cell_order': runtime.get('cell_order'),
               'sleeping': runtime.get('sleeping_grains'),
               'bounds': [g['bounds_min'], g['bounds_max']]}
        print('STEP ' + json.dumps(row), flush=True)
        if held:
            check('B held', False, {'status': row['status']})
            break
        if first is None and g['particles'] >= args.grains * .99:
            first = row

    if first:
        last = row
        check('B1 > 1M grains stepping', last['grains'] > 1000000 and not last['held'],
              {'grains': last['grains']})
        drop = first['com_y'] - last['com_y']
        potential = last['mass_kg'] * G * max(drop, 0.0)
        # KE now <= KE at full birth + potential released (+10% slack for the
        # contact springs' stored energy turning back into motion).
        bound = first['ke_j'] + 1.1 * potential + 1.0
        check('B2 energy bounded (2-D fold)', last['ke_j'] <= bound,
              {'ke_j': last['ke_j'], 'bound_j': bound, 'com_drop_m': drop})
        inside = all(-2.05 <= v <= 4.05 for corner in last['bounds'] for v in corner)
        check('B3 grains inside domain', inside, {'bounds': last['bounds']})
        if not args.no_collider:
            check('B4 collider > 4096 faces', (last['collider_faces'] or 0) > 4096,
                  {'collider_faces': last['collider_faces']})
    elif not failures:
        check('B births', False, {'last': row})

    print('RESULT ' + ('PASS' if not failures else 'FAIL ' + ', '.join(failures)), flush=True)


if __name__ == '__main__':
    main()
