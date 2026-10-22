"""A slab falling several cells per frame stays a slab (2026-10-20).

User scene: a 2.25 m water slab seeded mid-air in a 5 m Matter domain
(5.7 cm voxels), played at 24 fps. Between frames 10 and 14 it flattened into
a one-parcel sheet hanging mid-air. Cause: the advection read the grid
velocity past the cells that held liquid at the frame's transfer (sparse:
absent pages; dense: past the extrapolation band) as 0, so parcels that
travelled 4 cells within the frame stopped. Since then a parcel outside those
cells keeps its own velocity.

The same scene at dt = 1/24, one fluid.run per frame (the timeline's step).
Gates, for every frame before the slab reaches the floor:
  * moves_with_its_speed: mean per-frame displacement >= 85 % of v * dt;
  * stays_a_slab: the pressure solve's liquid keeps its size - CG iterations
    do not fall below half of the first frame's.
Empty scene, Pause first.
"""
import json

from rt_ipc import RtIpc

D = 'T_FastFall'
DT = 1 / 24


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    rows = []
    try:
        call('fluid.create_domain', name=D, type='matter', domain_min=[-2.5, -2.5, -2.5],
             domain_max=[2.5, 2.5, 2.5], voxel_size=5 / 88)
        call('fluid.set_param', domain=D, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', max_particles=100000)
        call('timeline.set_frame', frame=0)
        call('fluid.seed', domain=D, seed_min=[-1.125, .375, -1.125],
             seed_max=[1.125, 2.125, 1.125], particles_per_cell=8, replace=True)
        previous_y = call('fluid.state_digest', domain=D)['centroid'][1]
        for frame in range(1, 15):
            out = call('fluid.run', steps=1, dt=DT, sample_every=0, domains=[D],
                       fields=['live_motion_mean_m', 'live_mean_speed_m_s',
                               'pressure_cg_iterations'])
            r = out['samples'][-1]['domains'][D]
            y = call('fluid.state_digest', domain=D)['centroid'][1]
            rows.append({'frame': frame, 'centroid_y': y, 'drop': previous_y - y,
                         'speed': r['live_mean_speed_m_s'], 'move': r['live_motion_mean_m'],
                         'cg': r['pressure_cg_iterations']})
            previous_y = y
            print(json.dumps(rows[-1]), flush=True)
        # Frame 1 has no motion sample (the probe needs a previous frame).
        moving = [r for r in rows[1:] if r['centroid_y'] > -2.0]
        first_cg = max(rows[0]['cg'], 1)
        verdicts = {
            'moves_with_its_speed': all(r['move'] >= .85 * r['speed'] * DT for r in moving),
            'stays_a_slab': all(r['cg'] >= .5 * first_cg for r in moving),
        }
        for key, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', key, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        call('fluid.remove_domain', domain=D)
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
