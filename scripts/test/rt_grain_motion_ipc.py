"""Grains under force fields and moving colliders (2026-10-07).

Until this build a grain domain held its step ("static colliders, gravity
only") as soon as a force field existed or a collider moved. Now:
  - force fields act on every grain as an acceleration (same fields and mask
    as the liquid: affects_fluid), and reach the liquid lane as before;
  - every collider face carries a vertex velocity (from its own previous
    vertices: rigid, rotating or skinned; bone proxies use their twist) and
    sweeps through the DEM substeps; contacts use the relative velocity.

Arms (each from fluid.reset, same pour, 24 fps):
  rest   no field, sphere parked outside the pile        -> reference
  wind   infinite wind +x, 4 m/s^2                       -> COM drifts +x
  sweep  sphere (r .15) driven through the pile at 1 m/s -> pile pushed +x,
         collider_speed_max ~ 1 m/s, no held step

The sphere is a scene object with an object-bound mesh_bvh collider, moved by
scene.set_transform: that is how an animated object reaches the solver.
collider.update is an authoring edit and invalidates the simulation (frame
cache + timeline resync), so driving the sphere with it restarted the pour
every frame (first live run: 78 grains -> 17 frozen until the sphere stopped).

    python scripts/test/rt_grain_motion_ipc.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_ipc import RtIpc  # noqa: E402
from rt_grain_material import install_grain_material  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'docs/dev/grain_motion_live.json'
# The sphere primitive's size is its DIAMETER (radius .075). The pile settles to
# about one grain layer (top ~.05 m); at y .15 the sphere passed 2.5 cm above it
# and the sweep arm measured nothing (2026-10-08). Centre at .07: it cuts the layer.
SPHERE_Y = .07
DOMAIN, SOURCE, FIELD, SPHERE = 'GrainMotion', 'GrainMotionSand', 'GrainMotionWind', 'GrainMotionSphere'
FPS = 24.


def main():
    client = RtIpc()
    install_grain_material(client)  # old grain keys -> grain substance

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    original = {'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'colliders': call('collider.list')['colliders']}
    assert not call('sim.control_state')['playing'], 'Pause the application first'
    data = {'arms': {}}
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=False)
        if DOMAIN not in {d['name'] for d in original['domains']}:
            call('fluid.create_domain', name=DOMAIN, type='matter', domain_min=[-.6, 0, -.6],
                 domain_max=[.6, 1.2, .6], voxel_size=.1)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True, backend='vulkan',
             boundary='closed', default_substance='Sand', solid_phase=False,
             thermal_liquid_enabled=False)
        call('fluid.set_pore_exchange', domain=DOMAIN, enabled=False, wet_response_enabled=False)
        call('fluid.reset')
        call('fluid.set_grain_settings', domain=DOMAIN, radius_m=.025,
             stiffness_n_m=20000., restitution=.5, friction=.5, rolling_friction=.02,
             max_substeps=1024)
        exists = SOURCE in {s['name'] for s in call('flow_source.list')}
        call('flow_source.update' if exists else 'flow_source.create', name=SOURCE, domain=DOMAIN,
             enabled=True, phase='liquid', source_mode='point', fluid_substance='Sand',
             position=[0, .4, 0], radius=.08,
             velocity=[0, 0, 0], fluid_velocity_spread=0., use_particle_limit=True,
             max_emitted_particles=300, use_time_limit=True, start_time=0., end_time=.5,
             fluid_particles_per_second=600.)
        if FIELD not in {f.get('name') for f in call('forcefield.list')}:
            call('forcefield.create', type='wind', name=FIELD)
        call('forcefield.set_param', field=FIELD, enabled=False, shape='infinite',
             direction=[1, 0, 0], strength=4., use_noise=False, fluid_surface_drag=False,
             linear_drag=0., quadratic_drag=0.,
             affects_fluid=True, affects_gas=False, affects_particles=False,
             affects_cloth=False, affects_rigidbody=False)
        sphere_object = call('scene.add_primitive', type='sphere', name=SPHERE + 'Mesh', size=.15)
        data['sphere_object'] = sphere_object
        call('scene.set_transform', name=sphere_object, translation=[-.5, SPHERE_Y, 0.])
        colliders = {c['name'] for c in call('collider.list')['colliders']}
        call('collider.update' if SPHERE in colliders else 'collider.create', name=SPHERE,
             source_mode='mesh_bvh', source_object=sphere_object,
             enabled=True, fluid_collision_enabled=True)

        def sample():
            inv = call('fluid.matter_models', domain=DOMAIN)
            g = inv['acceptance_metrics']['granular']
            runtime = (inv.get('grain_diagnostics') or {}).get('runtime') or {}
            return {'grains': g.get('particles'), 'com': g.get('dry_center_of_mass'),
                    'held': inv['mixed_execution']['step_held'],
                    'status': inv['mixed_execution']['status'][:160],
                    'collider_faces': runtime.get('collider_faces'),
                    'collider_speed_max': runtime.get('collider_speed_max_m_s'),
                    'field_acceleration_max': runtime.get('field_acceleration_max_m_s2'),
                    'truncated': runtime.get('collider_manifold_truncated'),
                    'motion_ms': (runtime.get('host_ms') or {}).get('motion')}

        def run(arm, wind, sweep):
            call('forcefield.set_param', field=FIELD, enabled=wind)
            call('scene.set_transform', name=sphere_object, translation=[-.5, SPHERE_Y, 0.])
            call('fluid.reset')
            rows = []
            for frame in range(1, 73):
                if sweep and frame > 24:
                    # 1 m/s along +x, through the pile centre line (z = 0).
                    x = -.5 + (frame - 24) / FPS
                    call('scene.set_transform', name=sphere_object,
                         translation=[min(x, .4), SPHERE_Y, 0.])
                call('fluid.step', dt=1/FPS)
                row = sample()
                row['frame'] = frame
                rows.append(row)
                assert not row['held'], (arm, frame, row['status'])
            data['arms'][arm] = rows
            LOG.write_text(json.dumps(data, indent=1), encoding='utf-8')
            last = rows[-1]
            print(f"{arm:<6} grains {last['grains']} com {[round(v, 4) for v in last['com']]} "
                  f"faces {last['collider_faces']} collider max "
                  f"{max((r['collider_speed_max'] or 0) for r in rows):.3f} m/s "
                  f"field max {max((r['field_acceleration_max'] or 0) for r in rows):.3f} m/s2 "
                  f"truncated {sum((r['truncated'] or 0) for r in rows)} "
                  f"motion ms max {max((r['motion_ms'] or 0) for r in rows):.2f}",
                  flush=True)
            return rows

        rest = run('rest', False, False)
        wind = run('wind', True, False)
        sweep = run('sweep', False, True)
        x0, xw, xs = rest[-1]['com'][0], wind[-1]['com'][0], sweep[-1]['com'][0]
        field_max = max((r['field_acceleration_max'] or 0) for r in wind)
        speed_max = max((r['collider_speed_max'] or 0) for r in sweep[30:])
        print(f"RESULT com x: rest {x0:.4f}, wind {xw:.4f} (+{xw-x0:.4f}), sweep {xs:.4f} "
              f"(+{xs-x0:.4f}); wind field {field_max:.3f} m/s2; sphere {speed_max:.3f} m/s", flush=True)
        assert abs(field_max - 4.) < .05, ('grains did not see the wind field', field_max)
        assert max((r['field_acceleration_max'] or 0) for r in rest) == 0, 'field leaked into rest arm'
        assert xw - x0 > .05, ('wind did not move the grains', x0, xw)
        assert abs(speed_max - 1.) < .1, ('collider velocity not measured', speed_max)
        assert xs - x0 > .03, ('moving sphere did not push the pile', x0, xs)
        print('PASS grain force fields + moving collider', flush=True)
    finally:
        call('forcefield.set_param', field=FIELD, enabled=False)
        call('collider.update', name=SPHERE, enabled=False)
        if data.get('sphere_object'):
            call('scene.delete', name=data['sphere_object'])
        call('flow_source.update', name=SOURCE, enabled=False)
        call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s.get('enabled', True))
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d.get('enabled', True),
                 visible=d.get('visible', True))
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=c.get('enabled', True))
        call('fluid.reset')


if __name__ == '__main__':
    main()
