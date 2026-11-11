"""Liquid against a collider that moves away: the separating wall
(sim_matter_liquid_wall.glsl, docs/dev/COLLIDER_TEMAS_YUZEYI.md "Sivi seridi").

A full-width lower plate (mesh_sdf, nothing can fall past it) carries a liquid
block; a smaller upper plate rests on the block's top and rises 0.1 m in 1 s.

  honey_glued    Honey, contact_adhesion_pa 1e6 on both plates: the projection
                 holds the faces the liquid would leave, the block stretches
                 after the upper plate. Old exe: G2P read the rising plate as a
                 still wall and the liquid stayed down (2026-11-08 user scene:
                 chocolate lifted ~0 under a rising sphere).
  water_free     Water, adhesion 0: the faces open (the gap is air, p = 0),
                 the water lets go and nothing is sucked up.
  water_rests    Water on the lower plate only, no motion: held faces keep the
                 pool on the plate (no leak, no pumping at rest).
  honey_fast     The glued arm at 0.3 m/s (0.5 voxels a frame): the plate's
                 voxels jump past the liquid and an AIR cell opens between them
                 every frame; only the adhesion band (phi < h, glued colliders)
                 holds across it. Without it 1 MPa lifted no chocolate under a
                 rising sphere (2026-11-08 live, h 0.1 m, 5 cm a frame).

Gates (physical, not "the bridge holds 1 s": an 8 cm honey bridge drains at an
extension rate rho g L / (3 mu) ~ 37 1/s, it SHOULD neck and drop within ~0.2 s;
2026-11-08 live it followed the rising plate 6 frames, then broke):
  * glued_holds: honey hanging under a STATIC glued plate keeps its top layer on
    the plate (max y within 1 voxel of its bottom after 12 frames);
  * unglued_falls: the same with water and adhesion 0: all of it falls (max y
    >= 2 voxels below the plate);
  * glued_follows: the rising glued plate carries the contact layer in the first
    6 moving frames (max y within 1 voxel of the plate bottom at frame 8);
  * fast_follows: the same at 0.3 m/s (0.5 voxels a frame: the binary voxel
    jumps a cell every other frame and opens the air gap the adhesion band
    closes) at its 4th moving frame, within 2 voxels;
  * free_lets_go: water between unglued plates is not lifted (max y >= 2 voxels
    below the rising plate at the end);
  * rest_on_plate: water spreading on the plate never leaks below its top
    (min y >= plate top - 1 voxel);
  * finite: counts constant, bounds finite.
Output: %TEMP%/rt_liquid_wall.json
"""
import json
import math
from pathlib import Path
import tempfile
import time

from rt_ipc import RtIpc

DT = 1 / 24
SETTLE_FRAMES = 2
MOVE_FRAMES = 24
REST_FRAMES = 36
SPEED = 0.1  # m/s
# The user scene's regime: a sphere at 1.25 m/s over h 0.1 m = 0.52 voxels a
# frame, so its voxels jump a cell every other frame. Faster than ~1 voxel a
# frame outruns the one-cell band (open: sub-frame collider motion).
FAST = 0.3
VOXEL = 0.025
DOMAIN = 'T_LiquidWall'
PLATE_Y = 0.10
LOWER_HALF = (0.40, 0.025, 0.40)
UPPER_HALF = (0.10, 0.025, 0.10)
BLOCK = 0.08
GLUE_PA = 1.0e6
OUT = Path(tempfile.gettempdir()) / 'rt_liquid_wall.json'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}

    def plate(name, y, half, glue):
        scale = [2 * half[0], 2 * half[1], 2 * half[2]]
        obj = call('scene.add_primitive', type='cube', name=name, size=1.0)
        obj = obj.get('name', name) if isinstance(obj, dict) else (obj or name)
        call('scene.set_transform', name=obj, translation=[0, y, 0], scale=scale)
        call('collider.create', name=name + '_Collider', source_mode='mesh_sdf',
             source_object=obj, enabled=True, fluid_collision_enabled=True,
             contact_adhesion_pa=glue)
        start = time.time()
        while not call('collider.get', name=name + '_Collider').get('sdf_ready'):
            if time.time() - start > 120:
                raise RuntimeError(name + ': SDF cook did not finish in 120 s')
            call('fluid.step', dt=1e-4)
            time.sleep(0.1)
        return obj, name + '_Collider', scale

    def arm(label, substance, glue, upper, speed=SPEED, move_frames=MOVE_FRAMES):
        made = []
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-.4, 0, -.4],
             domain_max=[.4, .6, .4], voxel_size=VOXEL)
        try:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 backend='vulkan', boundary='closed', default_substance=substance,
                 thermal_liquid_enabled=False)
            call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
            lower, lower_c, _ = plate('T_LiquidWall_Lower', PLATE_Y, LOWER_HALF, glue)
            made.append((lower, lower_c))
            top = PLATE_Y + LOWER_HALF[1]
            upper_y = top + BLOCK + UPPER_HALF[1]
            obj = scale = None
            if upper:
                obj, upper_c, scale = plate('T_LiquidWall_Upper', upper_y, UPPER_HALF, glue)
                made.append((obj, upper_c))
            call('timeline.set_frame', frame=0)
            call('fluid.seed', domain=DOMAIN, seed_min=[-.06, top + .002, -.06],
                 seed_max=[.06, top + BLOCK - .002, .06], particles_per_cell=8, replace=True)
            seeded = call('fluid.state_digest', domain=DOMAIN)
            samples = []
            frames = SETTLE_FRAMES + (move_frames if upper else REST_FRAMES)
            for frame in range(1, frames + 1):
                y = upper_y
                if upper and frame > SETTLE_FRAMES:
                    y = upper_y + speed * (frame - SETTLE_FRAMES) * DT
                    call('scene.set_transform', name=obj, translation=[0, y, 0], scale=scale)
                call('fluid.step', dt=DT)
                d = call('fluid.state_digest', domain=DOMAIN)
                samples.append({'frame': frame, 'plate_top': top,
                                'upper_bottom': y - UPPER_HALF[1] if upper else None,
                                'particles': d['particles'], 'p05': d['bounds_p05'],
                                'p95': d['bounds_p95'], 'min': d['bounds_min'],
                                'max': d['bounds_max'], 'mean_speed': d['mean_speed']})
            data[label] = {'seeded': seeded, 'samples': samples}
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            return seeded, samples
        finally:
            for o, c in made:
                call('collider.remove', name=c)
                call('scene.delete', name=o)
            call('fluid.remove_domain', domain=DOMAIN)

    def healthy(seeded, samples):
        n = seeded['particles']
        return n > 0 and all(s['particles'] == n and all(
            math.isfinite(v) for v in s['min'] + s['max']) for s in samples)

    def hang(label, substance, glue):
        made = []
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-.4, 0, -.4],
             domain_max=[.4, .6, .4], voxel_size=VOXEL)
        try:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 backend='vulkan', boundary='closed', default_substance=substance,
                 thermal_liquid_enabled=False)
            call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
            plate_y = 0.30
            obj, col, _ = plate('T_LiquidWall_Hang', plate_y, UPPER_HALF, glue)
            made.append((obj, col))
            bottom = plate_y - UPPER_HALF[1]
            call('timeline.set_frame', frame=0)
            call('fluid.seed', domain=DOMAIN, seed_min=[-.06, bottom - 0.085, -.06],
                 seed_max=[.06, bottom - 0.001, .06], particles_per_cell=8, replace=True)
            seeded = call('fluid.state_digest', domain=DOMAIN)
            samples = []
            for frame in range(1, 13):
                call('fluid.step', dt=DT)
                d = call('fluid.state_digest', domain=DOMAIN)
                samples.append({'frame': frame, 'plate_bottom': bottom,
                                'particles': d['particles'], 'min': d['bounds_min'],
                                'max': d['bounds_max'], 'p05': d['bounds_p05'],
                                'p95': d['bounds_p95']})
            data[label] = {'seeded': seeded, 'samples': samples}
            return seeded, samples
        finally:
            for o, c in made:
                call('collider.remove', name=c)
                call('scene.delete', name=o)
            call('fluid.remove_domain', domain=DOMAIN)

    try:
        g_seed, g = hang('honey_hangs', 'Honey', GLUE_PA)
        u_seed, u = hang('water_drops', 'Water', 0.0)
        h_seed, h = arm('honey_glued', 'Honey', GLUE_PA, True, move_frames=6)
        f_seed, fa = arm('honey_fast', 'Honey', GLUE_PA, True, speed=FAST, move_frames=4)
        w_seed, w = arm('water_free', 'Water', 0.0, True)
        r_seed, r = arm('water_rests', 'Water', 0.0, False)
        summary = {
            'hang_gap_m': g[-1]['plate_bottom'] - g[-1]['max'][1],
            'drop_gap_m': u[-1]['plate_bottom'] - u[-1]['max'][1],
            'follow_gap_m': h[-1]['upper_bottom'] - h[-1]['max'][1],
            'follow_travel_m': SPEED * 6 * DT,
            'fast_gap_m': fa[-1]['upper_bottom'] - fa[-1]['max'][1],
            'fast_travel_m': FAST * 4 * DT,
            'water_top_gap_m': w[-1]['upper_bottom'] - w[-1]['max'][1],
            'rest_min_m': min(s_['min'][1] for s_ in r),
            'rest_plate_top_m': r[-1]['plate_top'],
            'rest_mean_speed': r[-1]['mean_speed'],
        }
        data['summary'] = summary
        print(json.dumps(summary), flush=True)
        verdicts = {
            'glued_holds': summary['hang_gap_m'] <= VOXEL,
            'unglued_falls': summary['drop_gap_m'] >= 2 * VOXEL,
            'glued_follows': summary['follow_gap_m'] <= VOXEL,
            'fast_follows': summary['fast_gap_m'] <= 2 * VOXEL,
            'free_lets_go': summary['water_top_gap_m'] >= 2 * VOXEL,
            'rest_on_plate': summary['rest_min_m'] >= summary['rest_plate_top_m'] - VOXEL,
            'finite': all(healthy(a_, b_) for a_, b_ in ((g_seed, g), (u_seed, u),
                          (h_seed, h), (f_seed, fa), (w_seed, w), (r_seed, r))),
        }
        data['verdicts'] = verdicts
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
