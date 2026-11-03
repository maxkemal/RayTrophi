"""MPM bodies against moving colliders (docs/dev/COLLIDER_TEMAS_YUZEYI.md).

Chewing Gum (fracture 1, yield 2: the user scene) on mesh_sdf plates driven by
scene.set_transform, which is how an animated object reaches the solver
(rt_grain_motion_ipc.py says why not collider.update). Arms:

  rise     the plate rises at 0.1 m/s with the gum resting on it (step 1).
           Old exe: the gum's top never moved while the plate's cells stepped
           up a voxel at a time; by frame 34 the block was ONE layer
           (min = max = 0.2375 m). Live 16:51: 5/5, gum rose 0.102 m.
  slide    the plate slides +x at 0.1 m/s; contact_friction 0.8 (step 3).
  slip     the same with contact_friction 0: the gum stays behind.
  stretch  a second plate is lowered onto the gum, both plates glued
           (contact_adhesion_pa 1e6, the order of chewing gum's tack), then
           the top plate rises: the gum stretches between them (step 3).
  draw     the same stretch with fracture 0.5: drawn to ~2.5x (Hencky ~0.9)
           it must tear. Tearing reads the elastic strain PLUS the creep/yield
           flow since 2026-11-02; on elastic strain alone a gum never tore.

Gates:
  * gum_rides: rise - centroid rises >= 70 % of the plate's rise;
  * gum_keeps_height: rise - final p05-p95 height >= 60 % of the settled one;
  * gum_on_plate: rise - lowest 5 % within 2 voxels of the plate top;
  * friction_carries: slide - centroid x moves >= 60 % of the plate's travel;
  * frictionless_slips: slip - centroid x moves <= 25 % of it;
  * gum_stretches: stretch - height grows >= 30 %, and the gum still touches
    both plates (p05 within 2 voxels of the lower top, p95 of the upper bottom);
  * drawn_gum_tears: draw - detached parcels > 2 %;
  * intact / finite: detached <= 5 %, counts constant, bounds finite.
Output: %TEMP%/rt_mpm_collider.json
"""
import json
import math
from pathlib import Path
import tempfile
import time

from rt_ipc import RtIpc

DT = 1 / 24
SETTLE_FRAMES = 12
MOVE_FRAMES = 24
SPEED = 0.1  # m/s
VOXEL = 0.025
DOMAIN = 'T_MpmCollider'
SUBSTANCE = 'T_MpmCollider_Gum'
PLATE_Y = 0.10
HALF = (0.15, 0.025, 0.15)
GUM_HEIGHT = 0.08
GLUE_PA = 1.0e6
OUT = Path(tempfile.gettempdir()) / 'rt_mpm_collider.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}
    scale = [2 * HALF[0], 2 * HALF[1], 2 * HALF[2]]

    def plate(name, y, friction, glue):
        obj = call('scene.add_primitive', type='cube', name=name, size=1.0)
        obj = obj.get('name', name) if isinstance(obj, dict) else (obj or name)
        call('scene.set_transform', name=obj, translation=[0, y, 0], scale=scale)
        call('collider.create', name=name + '_Collider', source_mode='mesh_sdf',
             source_object=obj, enabled=True, fluid_collision_enabled=True,
             contact_friction=friction, contact_adhesion_pa=glue)
        # Wait for the cook before seeding. The slide arm measured 0.00005 m
        # instead of 0.1 m in two full smoke runs (never alone): a cook late
        # under load lets the gum fall through the plate onto the floor.
        start = time.time()
        while not call('collider.get', name=name + '_Collider').get('sdf_ready'):
            if time.time() - start > 120:
                raise RuntimeError(name + ': SDF cook did not finish in 120 s')
            call('fluid.step', dt=1e-4)  # the publish tick runs GPU cooks
            time.sleep(0.1)
        # The scene may rename the object (".001"); the collider keeps `name`.
        return obj, name + '_Collider'

    def arm(label, friction=0.5, glue=0.0, motion='rise', top_plate=False):
        made = []
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-.4, 0, -.4],
             domain_max=[.4, .8, .4], voxel_size=VOXEL)
        try:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 backend='vulkan', boundary='closed', default_substance=SUBSTANCE,
                 thermal_liquid_enabled=False)
            call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
            lower, lower_collider = plate('T_MpmCollider_Lower', PLATE_Y, friction, glue)
            made.append((lower, lower_collider))
            top = PLATE_Y + HALF[1]
            upper_y = top + GUM_HEIGHT + HALF[1]
            upper = None
            if top_plate:
                upper, upper_collider = plate('T_MpmCollider_Upper', upper_y, friction, glue)
                made.append((upper, upper_collider))
            call('timeline.set_frame', frame=0)
            call('fluid.seed', domain=DOMAIN, seed_min=[-.06, top + .005, -.06],
                 seed_max=[.06, top + GUM_HEIGHT - .005, .06], particles_per_cell=8,
                 replace=True)
            seeded = call('fluid.state_digest', domain=DOMAIN)
            samples = []

            def sample(frame, lower_y, upper_pos):
                d = call('fluid.state_digest', domain=DOMAIN)
                r = {x['name']: x for x in call('fluid.list_domains')['domains']}[DOMAIN]
                samples.append({
                    'frame': frame, 'lower_top': lower_y + HALF[1],
                    'upper_bottom': None if upper_pos is None else upper_pos - HALF[1],
                    'particles': d['particles'], 'centroid': d['centroid'],
                    'p05': d['bounds_p05'], 'p95': d['bounds_p95'],
                    'min': d['bounds_min'], 'max': d['bounds_max'],
                    'mean_speed': d['mean_speed'], 'detached': r.get('granular_detached'),
                    'mpm_law': r.get('mpm_law')})

            for frame in range(1, SETTLE_FRAMES + 1):
                call('fluid.step', dt=DT)
                sample(frame, PLATE_Y, upper_y if top_plate else None)
            start = samples[-1]
            x = 0.0
            lower_y, up = PLATE_Y, upper_y
            for frame in range(1, MOVE_FRAMES + 1):
                travel = SPEED * frame * DT
                if motion == 'rise':
                    lower_y = PLATE_Y + travel
                    call('scene.set_transform', name=lower, translation=[0, lower_y, 0],
                         scale=scale)
                elif motion == 'slide':
                    x = travel
                    call('scene.set_transform', name=lower, translation=[x, PLATE_Y, 0],
                         scale=scale)
                elif motion == 'stretch':
                    up = upper_y + travel
                    call('scene.set_transform', name=upper, translation=[0, up, 0],
                         scale=scale)
                call('fluid.step', dt=DT)
                sample(SETTLE_FRAMES + frame, lower_y, up if top_plate else None)
            data[label] = {'seeded': seeded, 'samples': samples}
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return seeded, start, samples[-1], samples, SPEED * MOVE_FRAMES * DT
        finally:
            for obj, collider in made:
                call('collider.remove', name=collider)
                call('scene.delete', name=obj)
            call('fluid.remove_domain', domain=DOMAIN)

    def height(s):
        return s['p95'][1] - s['p05'][1]

    def healthy(seeded, samples):
        n = seeded['particles']
        return n > 0 and all(s['particles'] == n and all(
            math.isfinite(v) for v in s['min'] + s['max']) for s in samples) and \
            max((s['detached'] or 0) for s in samples) <= 0.05 * n

    created = False
    try:
        call('substance.derive', name=SUBSTANCE, based_on='Chewing Gum')
        created = True
        call('substance.set', name=SUBSTANCE,
             fields={'elastic_fracture_strain': 1.0, 'elastic_yield_strain': 2.0})
        r_seed, r_start, r_end, r_samples, travel = arm('rise', motion='rise')
        s_seed, s_start, s_end, s_samples, _ = arm('slide', friction=0.8, motion='slide')
        f_seed, f_start, f_end, f_samples, _ = arm('slip', friction=0.0, motion='slide')
        t_seed, t_start, t_end, t_samples, _ = arm('stretch', glue=GLUE_PA,
                                                    motion='stretch', top_plate=True)
        call('substance.set', name=SUBSTANCE, fields={'elastic_fracture_strain': 0.5})
        d_seed, d_start, d_end, d_samples, _ = arm('draw', glue=GLUE_PA,
                                                 motion='stretch', top_plate=True)
        summary = {
            'travel_m': travel,
            'rise_gum_m': r_end['centroid'][1] - r_start['centroid'][1],
            'rise_height': [height(r_start), height(r_end)],
            'rise_gap_m': r_end['p05'][1] - r_end['lower_top'],
            'slide_gum_x_m': s_end['centroid'][0] - s_start['centroid'][0],
            'slip_gum_x_m': f_end['centroid'][0] - f_start['centroid'][0],
            'stretch_height': [height(t_start), height(t_end)],
            'stretch_lower_gap_m': t_end['p05'][1] - t_end['lower_top'],
            'stretch_upper_gap_m': t_end['upper_bottom'] - t_end['p95'][1],
            'draw_detached': max((s['detached'] or 0) for s in d_samples),
            'draw_particles': d_seed['particles'],
            'mpm_law': r_end['mpm_law'],
        }
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        verdicts = {
            'gum_rides': summary['rise_gum_m'] >= 0.7 * travel,
            'gum_keeps_height': summary['rise_height'][1] >= 0.6 * summary['rise_height'][0],
            'gum_on_plate': abs(summary['rise_gap_m']) <= 2 * VOXEL,
            'friction_carries': summary['slide_gum_x_m'] >= 0.6 * travel,
            'frictionless_slips': summary['slip_gum_x_m'] <= 0.25 * travel,
            'gum_stretches': summary['stretch_height'][1] >= 1.3 * summary['stretch_height'][0]
                             and abs(summary['stretch_lower_gap_m']) <= 2 * VOXEL
                             and abs(summary['stretch_upper_gap_m']) <= 2 * VOXEL,
            'drawn_gum_tears': summary['draw_detached'] > 0.02 * summary['draw_particles'],
            'intact_finite': all(healthy(a, b) for a, b in (
                (r_seed, r_samples), (s_seed, s_samples), (f_seed, f_samples),
                (t_seed, t_samples))),
        }
        data['verdicts'] = verdicts
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        if created:
            call('substance.remove', name=SUBSTANCE)
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
