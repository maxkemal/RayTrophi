"""Sub-voxel collider contact (docs/dev/COLLIDER_TEMAS_YUZEYI.md, P1-P3 + S3-S4).

A 10 cm Gelatin cube is dropped onto a collider and left to rest; its lowest
parcel must end ON the true surface, not on the voxel above it. The surface is
nearly flat everywhere (a 0.9 m sphere whose top is at y = 0.1) so the jelly
does not slide off. Voxel h = 2.5 cm: the old cell mask could only say
"somewhere in this 2.5 cm cell".

Arms:
  plane     analytic plane at y = 0.1                    (P1 analytic phi)
  sphere    analytic sphere, top at 0.1                  (P1)
  sdf       the same sphere as a mesh, mesh_sdf, Auto    (P2 GPU cook +
            resolution (cook cell <= h/2)                  trilinear phi)
  shell     the same mesh as mesh_bvh, thickness 2 cm    (P3 GPU shell query)

Gates:
  * plane_on_surface / sphere_on_surface: |lowest - surface| <= h/4;
  * plane_keeps_height: the jelly stays >= 6 cm tall (not piled into a band);
  * sdf_cooked_on_gpu: collider.get sdf_cooked_on_gpu, Auto dims >= 64 on y;
  * sdf_matches_analytic: |sdf rest - sphere rest| <= h/4;
  * shell_on_gpu: collider_query_gpu on some frame and never an error (a
    static collider hits the voxel cache, so later frames run no query);
  * shell_on_surface: |lowest - (surface + thickness)| <= h/4.
Output: %TEMP%/rt_collider_phi.json
"""
import json
import math
from pathlib import Path
import tempfile
import time

from rt_ipc import RtIpc

DT = 1 / 24
FRAMES = 36
VOXEL = 0.025
TOP = 0.10
RADIUS = 0.9
THICK = 0.02
DOMAIN = 'T_ColliderPhi'
OUT = Path(tempfile.gettempdir()) / 'rt_collider_phi.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}

    def drop(label, collider, mesh=None, wait_sdf=False):
        made = []
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-.4, 0, -.4],
             domain_max=[.4, .7, .4], voxel_size=VOXEL)
        try:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 backend='vulkan', boundary='closed', default_substance='Gelatin',
                 thermal_liquid_enabled=False)
            call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
            name = 'T_ColliderPhi_' + label
            obj = None
            if mesh:
                obj = call('scene.add_primitive', type='sphere', name=name + '_Mesh', size=1.0)
                obj = obj.get('name', name) if isinstance(obj, dict) else (obj or name)
                made.append(('object', obj))
                info = call('scene.object_info', name=obj)
                extent = info['world_max'][1] - info['world_min'][1]
                scale = 2 * RADIUS / max(extent, 1e-6)
                call('scene.set_transform', name=obj, translation=[0, TOP - RADIUS, 0],
                     scale=[scale, scale, scale])
                collider['source_object'] = obj
            call('collider.create', name=name, enabled=True, fluid_collision_enabled=True,
                 **collider)
            made.append(('collider', name))
            cook = {}
            if wait_sdf:
                start = time.time()
                while time.time() - start < 120:
                    cook = call('collider.get', name=name)
                    if cook.get('sdf_ready'):
                        break
                    call('fluid.step', dt=1e-4)  # the publish tick runs GPU cooks
                    time.sleep(0.2)
            call('timeline.set_frame', frame=0)
            call('fluid.seed', domain=DOMAIN, seed_min=[-.05, TOP + .10, -.05],
                 seed_max=[.05, TOP + .20, .05], particles_per_cell=8, replace=True)
            samples = []
            for frame in range(FRAMES):
                call('fluid.step', dt=DT)
                d = call('fluid.state_digest', domain=DOMAIN)
                st = call('fluid.step_stats', domain=DOMAIN)
                samples.append({'frame': frame + 1, 'min_y': d['bounds_min'][1],
                                'p05_y': d['bounds_p05'][1], 'p95_y': d['bounds_p95'][1],
                                'speed': d['mean_speed'],
                                'particles': d['particles'],
                                'query_gpu': st.get('collider_query_gpu'),
                                'query_ms': st.get('collider_query_ms'),
                                'query_error': st.get('collider_query_error')})
            if wait_sdf:
                cook = call('collider.get', name=name)
            data[label] = {'samples': samples,
                           'cook': {k: cook.get(k) for k in ('sdf_ready', 'sdf_dims',
                                    'sdf_cooked_on_gpu', 'sdf_cook_ms')}}
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            # The last third: settled.
            tail = samples[-FRAMES // 3:]
            return sum(s['min_y'] for s in tail) / len(tail), samples, cook
        finally:
            for kind, name in reversed(made):
                if kind == 'collider':
                    call('collider.remove', name=name)
                else:
                    call('scene.delete', name=name)
            call('fluid.remove_domain', domain=DOMAIN)

    try:
        plane, plane_samples, _ = drop('plane', {'source_mode': 'plane', 'plane_y': TOP})
        sphere, _, _ = drop('sphere', {'source_mode': 'sphere',
                                       'sphere_center': [0, TOP - RADIUS, 0],
                                       'sphere_radius': RADIUS})
        sdf, _, cook = drop('sdf', {'source_mode': 'mesh_sdf', 'sdf_resolution_mode': 4},
                            mesh=True, wait_sdf=True)
        shell, shell_samples, _ = drop('shell', {'source_mode': 'mesh_bvh', 'thickness': THICK},
                                       mesh=True)
        dims = cook.get('sdf_dims') or [0, 0, 0]
        summary = {'plane_gap_m': plane - TOP, 'sphere_gap_m': sphere - TOP,
                   'plane_height_m': plane_samples[-1]['p95_y'] - plane_samples[-1]['p05_y'],
                   'sdf_gap_m': sdf - TOP, 'shell_gap_m': shell - (TOP + THICK),
                   'sdf_dims': dims, 'sdf_cooked_on_gpu': cook.get('sdf_cooked_on_gpu'),
                   'sdf_cook_ms': cook.get('sdf_cook_ms'),
                   'shell_query_ms': max((s['query_ms'] or 0) for s in shell_samples),
                   'shell_query_error': next((s['query_error'] for s in shell_samples
                                              if s['query_error']), '')}
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        q = VOXEL / 4
        verdicts = {
            'plane_on_surface': abs(summary['plane_gap_m']) <= q,
            # 2026-11-02: a parcel-only contact piled the jelly into its h/2
            # band (8 cm -> 1.15 cm). A 10 cm Gelatin cube keeps >= 6 cm.
            'plane_keeps_height': summary['plane_height_m'] >= 0.06,
            'sphere_on_surface': abs(summary['sphere_gap_m']) <= q,
            'sdf_cooked_on_gpu': bool(summary['sdf_cooked_on_gpu']) and dims[1] >= 64,
            'sdf_matches_analytic': abs(sdf - sphere) <= q,
            'shell_on_gpu': any(s['query_gpu'] for s in shell_samples) and
                            not summary['shell_query_error'],
            'shell_on_surface': abs(summary['shell_gap_m']) <= q,
            'finite': all(math.isfinite(v) for v in (plane, sphere, sdf, shell)),
        }
        data['verdicts'] = verdicts
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
