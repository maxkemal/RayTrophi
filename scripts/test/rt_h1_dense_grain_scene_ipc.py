"""Dense dry-grain cost matrix + a reusable visual scene, external IPC only.

Does not save project or build. Benchmark: same profile/volume/dt, hidden output,
30 steps (0.5 s), first five discarded from timing. Final visual: finite stream
onto a transformed flat ramp, distinct owned materials and camera, frame0 paused.
"""
import argparse
import base64
import json
import statistics
import time
from pathlib import Path

from rt_ipc import RtIpc
from grain_units import restitution_of

DOMAIN = 'H1_Dense_Grain_Lab'
SOURCE = 'H1_Dense_Sand_Stream'
ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'docs/dev/matter_h1_dense_grain_scene_live.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', nargs='+', type=int, default=[1024, 4096, 16384])
    parser.add_argument('--visual-only', action='store_true')
    args = parser.parse_args()
    assert all(1 <= n <= 40000 for n in args.counts)
    client = RtIpc()
    data = {'arms': [], 'dt_s': 1/60, 'duration_s': .5, 'warmup_steps': 5,
            'completed': False, 'visual_ready': False}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and ('__error' in result or result.get('ok') is False):
            raise RuntimeError((method, result))
        return result

    def save():
        temporary = LOG.with_suffix('.tmp')
        temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
        for attempt in range(10):
            try:
                temporary.replace(LOG)
                return
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(.1)

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'colliders': call('collider.list')['colliders'],
                'shading': call('viewport.shading'), 'camera': call('camera.get')}
    assert not original['control']['playing'], 'Pause first'
    data['original'] = original
    domain_ready = source_ready = False
    ramp_collider = None
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=False)
        # Only prior owned H1 fixture geometry is moved out of this camera.
        data['moved_previous_fixtures'] = []
        for name in call('scene.list_objects'):
            if name in ('Cube', 'Default_Cube') or name.startswith((
                'H1_Grain_Flat_Ramp', 'H1_Grain_Corner_', 'H1_Dense_Ground', 'H1_Dense_Ramp')):
                transform = call('scene.get_transform', name=name)
                data['moved_previous_fixtures'].append({'name': name, 'transform': transform})
                call('scene.set_transform', name=name, translation=[20, 0, 0])
        call('viewport.set_shading', mode='material')
        if DOMAIN not in {d['name'] for d in original['domains']}:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1.7, 0, -1.7], domain_max=[1.7, 3.2, 1.7], voxel_size=.1)
        domain_ready = True
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
             backend='vulkan', boundary='closed', preset='sand', render_mode='particles',
             granular_enabled=True, solid_phase=False, thermal_liquid_enabled=False,
             max_particles=40000, voxel_size=.1,
             domain_min=[-1.7, 0, -1.7], domain_max=[1.7, 3.2, 1.7])
        call('fluid.set_pore_exchange', domain=DOMAIN, enabled=False,
             wet_response_enabled=False, wet_appearance_enabled=False)
        call('fluid.reset')
        call('fluid.set_grain_settings', domain=DOMAIN, enabled=True, radius_m=.025,
             stiffness_n_m=200000., restitution=restitution_of(8., 200000.), sliding_damping_n_s_m=4.,
             friction=.5, rolling_friction=.02, twisting_friction=.1, max_substeps=4096)
        data['profile'] = call('fluid.grain_settings', domain=DOMAIN)
        exists = SOURCE in {s['name'] for s in call('flow_source.list')}
        call('flow_source.update' if exists else 'flow_source.create', name=SOURCE,
             domain=DOMAIN, enabled=False, phase='liquid', source_mode='point',
             fluid_substance='Sand', initial_constitutive_model='granular',
             position=[0, 1.5, 0], radius=1.1, velocity=[0, 0, 0], fluid_velocity_spread=0.,
             use_time_limit=True, start_time=0., end_time=1/30, use_particle_limit=True,
             max_emitted_particles=16384, fluid_particles_per_second=16384*1.01*60,
             fluid_temperature_override=True, fluid_temperature_kelvin=293.15)
        source_ready = True
        if not args.visual_only:
            for count in args.counts:
                call('timeline.set_frame', frame=0)
                call('fluid.reset')
                call('flow_source.update', name=SOURCE, enabled=True,
                     max_emitted_particles=count, fluid_particles_per_second=count*1.01*60)
                control = call('sim.control_state')
                arm = {'requested': count, 'samples': []}
                data['arms'].append(arm)
                print('START dense', count, flush=True)
                for step in range(1, 31):
                    call('fluid.step', dt=1/60)
                    assert call('sim.control_state') == control
                    inventory = call('fluid.matter_models', domain=DOMAIN)
                    stats = call('fluid.step_stats', domain=DOMAIN)
                    g = inventory['acceptance_metrics']['granular']
                    arm['samples'].append({'step': step, 'stats': stats, 'granular': g,
                        'identity_hash': inventory['particle_id_hash'],
                        'runtime': inventory['grain_diagnostics']['runtime']})
                    assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                    assert g['particles'] == count, ('birth count', count, g['particles'])
                    if step % 10 == 0:
                        save()
                        print('dense', count, step, stats['total_ms'], flush=True)
                samples = arm['samples'][5:]
                ms = sorted(s['stats']['total_ms'] for s in samples)
                arm['summary'] = {'particles': count, 'median_sim_ms': statistics.median(ms),
                    'p95_sim_ms': ms[int(.95*(len(ms)-1))],
                    'median_dispatch': statistics.median(s['stats']['dispatch_calls'] for s in samples),
                    'median_upload_bytes': statistics.median(s['stats']['upload_bytes'] for s in samples),
                    'median_download_bytes': statistics.median(s['stats']['download_bytes'] for s in samples),
                    'dry_mass_drift_kg': max(s['granular']['dry_mass_kg'] for s in arm['samples'])-
                        min(s['granular']['dry_mass_kg'] for s in arm['samples']),
                    'last_runtime': arm['samples'][-1]['runtime']}
                assert arm['summary']['dry_mass_drift_kg'] <= 1e-5
                print('RESULT', json.dumps(arm['summary']), flush=True)
                save()
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        sand = call('material.create', type='substance:Sand', name='H1_Dense_Dry_Sand')
        floor_mat = call('material.create', type='principled', name='H1_Dense_Floor')
        ramp_mat = call('material.create', type='principled', name='H1_Dense_Ramp')
        floor = call('scene.add_primitive', type='plane', name='H1_Dense_Ground', size=3.4)
        ramp = call('scene.add_primitive', type='cube', name='H1_Dense_Ramp', size=1.)
        call('material.assign', object_name=floor, material_name=floor_mat)
        call('material.assign', object_name=ramp, material_name=ramp_mat)
        call('material.set', object_name=floor, param='base_color', value=[.12, .16, .19])
        call('material.set', object_name=ramp, param='base_color', value=[.22, .29, .36])
        call('scene.set_transform', name=floor, translation=[0, 0, 0])
        call('scene.set_transform', name=ramp, translation=[-.2, .65, 0],
             rotation=[0, 0, -20], scale=[1.8, .10, 1.3])
        ramp_collider = 'H1_Dense_Ramp_Collider'
        exists = ramp_collider in {c['name'] for c in call('collider.list')['colliders']}
        call('collider.update' if exists else 'collider.create', name=ramp_collider,
             source_mode='mesh_bvh', source_object=ramp, enabled=True)
        call('fluid.set_substance_material', domain=DOMAIN, substance='Sand',
             material=sand, representation='splat')
        call('fluid.set_splat_geometry', domain=DOMAIN, geometry='icosphere',
             granular_physical_carriers=True, virtual_grains=1)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True)
        call('flow_source.update', name=SOURCE, enabled=True, position=[-.6, 1.9, 0],
             radius=.32, max_emitted_particles=16384, fluid_particles_per_second=4096.,
             start_time=0., end_time=12., velocity=[0, -.1, 0])
        call('camera.set_position', position=[4.0, 2.8, 4.6])
        call('camera.set_target', target=[0, .85, 0])
        data['visual_objects'] = {'floor': floor, 'ramp': ramp, 'sand_material': sand}
        # Real simulation sample for visual inspection, then restore clean start.
        data['visual_samples'] = []
        control = call('sim.control_state')
        for step in range(1, 121):
            call('fluid.step', dt=1/60)
            if step % 30 == 0:
                assert call('sim.control_state') == control
                inventory = call('fluid.matter_models', domain=DOMAIN)
                data['visual_samples'].append(inventory)
                save()
                assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
                print('VISUAL', step, inventory['acceptance_metrics']['granular']['particles'], flush=True)
        call('viewport.capture', enabled=True)
        call('viewport.render_frames', count=2)
        shot = call('viewport.get_screenshot')
        if shot.get('image_base64'):
            image = ROOT / 'docs/dev/matter_h1_dense_grain_visual.jpg'
            image.write_bytes(base64.b64decode(shot['image_base64']))
            data['screenshot'] = str(image)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        data['visual_ready'] = True
        data['completed'] = True
    finally:
        if not data['visual_ready']:
            if source_ready:
                call('flow_source.update', name=SOURCE, enabled=False)
            if domain_ready:
                call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
            if ramp_collider:
                call('collider.update', name=ramp_collider, enabled=False)
            for c in original['colliders']:
                call('collider.update', name=c['name'], enabled=c['enabled'])
            for d in original['domains']:
                call('fluid.set_param', domain=d['name'], enabled=d['enabled'], visible=d['visible'])
            for s in original['sources']:
                call('flow_source.update', name=s['name'], enabled=s['enabled'])
        data['final_control'] = call('sim.control_state')
        save()
        client.close()


if __name__ == '__main__':
    main()
