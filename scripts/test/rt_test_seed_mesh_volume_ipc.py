"""Mesh Volume seed (fluid.seed_mesh): fill the INSIDE of a scene object.

The sand-castle workflow: a (hidden) castle mesh is seeded with matter, so the
material takes the mesh's shape. Run in an EMPTY, paused scene.

Gates:
  * missing_object: seeding a non-existent object fails and seeds nothing;
  * liquid_count: a sphere seeded at 8 PPC emits ~8 x V_sphere / h^3 parcels
    (tessellated sphere -> slightly under; within 15 %), seed_mode reports
    mesh_volume and seed_error is empty;
  * liquid_shape: the seeded top (seed_top_y) is below the sphere top + h/2 —
    a box seed of the bounds would also pass the count only by luck, never this
    together with the count;
  * not_truncated: the partly-inside boundary cells do not read as a budget cut
    (seed_truncated stays false with no Max Particles limit);
  * moved_object: moving the object 0.4 m up before seeding moves the fill
    (seed_top_y follows within one voxel), not the cached old pose;
  * hidden_object: the same seed still works with the object hidden (the
    castle mesh is meant to be invisible);
  * grains (informational unless Sand resolves to DEM): Sand fills the sphere.
Output: %TEMP%/rt_seed_mesh_volume.json
"""
import json
import math
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

DOMAIN = 'T_SeedMesh'
H = 0.05
OUT = Path(tempfile.gettempdir()) / 'rt_seed_mesh_volume.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {'gates': {}}
    gates = data['gates']

    call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-1.5, -1.5, -1.5],
         domain_max=[1.5, 1.5, 1.5], voxel_size=H)
    call('fluid.set_param', domain=DOMAIN, enabled=True, backend='vulkan',
         boundary='closed', default_substance='Water', thermal_liquid_enabled=False)
    call('timeline.set_frame', frame=0)
    sphere = call('scene.add_primitive', type='sphere', name='T_SeedMeshSphere', size=1.0)
    info = call('scene.object_info', name=sphere)
    lo, hi = info['world_min'], info['world_max']
    r = 0.5 * (hi[0] - lo[0])
    cy = 0.5 * (lo[1] + hi[1])
    data['sphere'] = {'name': sphere, 'radius': r, 'centre_y': cy}

    # ── missing object ──
    try:
        call('fluid.seed_mesh', domain=DOMAIN, object='T_NoSuchObject', particles_per_cell=8)
        gates['missing_object'] = False
    except Exception as e:  # noqa: BLE001 - the refusal IS the pass
        data['missing_object_error'] = str(e)
        gates['missing_object'] = 'not found' in str(e)

    # ── liquid fill ──
    call('fluid.seed_mesh', domain=DOMAIN, object=sphere, particles_per_cell=8, replace=True)
    g = call('fluid.get', domain=DOMAIN)
    expected = 8 * (4.0 / 3.0 * math.pi * r ** 3) / H ** 3
    emitted = g.get('seed_emitted_particles', g.get('particle_count', 0))
    data['liquid'] = {k: g.get(k) for k in ('seed_mode', 'seed_mesh_object', 'seed_error',
                                             'seed_emitted_particles', 'seed_requested_particles',
                                             'seed_top_y', 'particle_count', 'seed_truncated')}
    data['liquid']['expected'] = expected
    gates['liquid_count'] = (g.get('seed_mode') == 'mesh_volume' and not g.get('seed_error')
                             and 0.85 * expected <= emitted <= 1.05 * expected)
    gates['liquid_shape'] = g.get('seed_top_y', 1e9) <= cy + r + 0.5 * H
    # A mesh seed's boundary cells are never full; that is shape, not budget.
    gates['not_truncated'] = g.get('seed_truncated') is False

    # ── moved object: the seed must follow the object, not the cached pose ──
    # (2026-11-05: an object moved before seeding was filled at its old place.)
    try:
        t = call('scene.get_transform', name=sphere)
        m = t['matrix'] if isinstance(t, dict) and 'matrix' in t else t
        m = [list(row) for row in m]
        m[1][3] += 0.4
        call('scene.set_transform', name=sphere, matrix=m)
        call('fluid.seed_mesh', domain=DOMAIN, object=sphere, particles_per_cell=8, replace=True)
        gm = call('fluid.get', domain=DOMAIN)
        moved_top = gm.get('seed_top_y', 0.0)
        data['moved'] = {'top_before': g.get('seed_top_y'), 'top_after': moved_top,
                         'emitted': gm.get('seed_emitted_particles')}
        gates['moved_object'] = abs(moved_top - (g.get('seed_top_y', 0.0) + 0.4)) <= H
        m[1][3] -= 0.4
        call('scene.set_transform', name=sphere, matrix=m)
    except Exception as e:  # noqa: BLE001
        data['moved'] = {'error': str(e)}
        gates['moved_object'] = False

    # ── hidden object (the castle is invisible) ──
    call('scene.set_visible', name=sphere, visible=False)
    try:
        call('fluid.seed_mesh', domain=DOMAIN, object=sphere, particles_per_cell=8, replace=True)
        g2 = call('fluid.get', domain=DOMAIN)
        data['hidden'] = {'emitted': g2.get('seed_emitted_particles'), 'error': g2.get('seed_error')}
        gates['hidden_object'] = abs(g2.get('seed_emitted_particles', 0) - emitted) <= 0.02 * emitted
    except Exception as e:  # noqa: BLE001
        data['hidden'] = {'error': str(e)}
        gates['hidden_object'] = False

    # ── grains (informational) ──
    try:
        call('fluid.set_param', domain=DOMAIN, default_substance='Sand')
        call('fluid.seed_mesh', domain=DOMAIN, object=sphere, replace=True)
        g3 = call('fluid.get', domain=DOMAIN)
        data['sand'] = {k: g3.get(k) for k in ('seed_emitted_particles', 'seed_top_y',
                                                'seed_error', 'granular_model_particles')}
    except Exception as e:  # noqa: BLE001
        data['sand'] = {'error': str(e)}

    # ── cleanup ──
    for method, kwargs in (('fluid.remove_domain', {'domain': DOMAIN}),
                           ('fluid.remove_domain', {'name': DOMAIN}),
                           ('scene.delete', {'name': sphere})):
        try:
            call(method, **kwargs)
        except Exception:  # noqa: BLE001
            pass

    OUT.write_text(json.dumps(data, indent=2))
    passed = sum(1 for v in gates.values() if v)
    for k, v in gates.items():
        print(f"{'PASS' if v else 'FAIL'}  {k}")
    print(f"{passed}/{len(gates)} gates, details: {OUT}")


if __name__ == '__main__':
    main()
