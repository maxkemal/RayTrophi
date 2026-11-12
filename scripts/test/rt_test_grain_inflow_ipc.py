"""DEM grain inflow nozzle (planGrainInflow): flux = A |v| at ~40 % solids.

Run with Studio paused in an EMPTY scene:
    python scripts/test/rt_test_grain_inflow_ipc.py
Creates and removes its own domain and source.
"""
import json
import math
from rt_ipc import RtIpc

DOMAIN = 'T_GrainInflow'
SOURCE = 'T_GrainInflow_Source'
DT = 1 / 48
STEPS = 30
NOZZLE = 0.06
GRAIN = 0.01
SPEED = 1.0


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    results = {}
    try:
        call('fluid.create_domain', name=DOMAIN, type='matter',
             domain_min=[-.5, 0, -.5], domain_max=[.5, 1.5, .5], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Sand', solid_phase=False,
             thermal_liquid_enabled=False)
        call('fluid.set_grain_settings', domain=DOMAIN, radius_m=GRAIN, max_substeps=1024)
        settings = call('fluid.grain_settings', domain=DOMAIN)
        call('flow_source.create', name=SOURCE, domain=DOMAIN, phase='liquid',
             source_mode='point', fluid_substance='Sand', liquid_emission='inflow',
             position=[0, 1.2, 0], radius=NOZZLE, velocity=[0, -SPEED, 0],
             fluid_velocity_spread=0., fluid_particles_per_second=0.)
        call('timeline.set_frame', frame=0)
        per_step = []
        last = 0
        status = ''
        for _ in range(STEPS):
            call('fluid.step', dt=DT)
            info = call('flow_source.get', name=SOURCE)
            status = status or info['liquid_emission_status']
            per_step.append(info['total_emitted_particles'] - last)
            last = info['total_emitted_particles']
        digest = call('fluid.state_digest', domain=DOMAIN)
        fastest = call('fluid.particle_extremes', domain=DOMAIN, count=1).get('fastest') or [{}]
        # Largest radius the nozzle plans with: radius_m unless the size
        # distribution widens it (printed so a mismatch is readable).
        radius = max(float(v) for k, v in settings.items()
                     if 'radius' in k and isinstance(v, (int, float)) and 0 < v < 1) \
            if any('radius' in k for k in settings) else GRAIN
        pitch = 2.3 * radius
        reach = NOZZLE - radius
        expected = SPEED * STEPS * DT / pitch * math.pi * reach ** 2 / (0.8660254 * pitch ** 2)
        tail = per_step[STEPS // 2:]
        mean_tail = sum(tail) / len(tail)
        results = {
            'emitted': last, 'expected': expected, 'particles': digest['particles'],
            'per_step_tail_mean': mean_tail, 'per_step_max': max(per_step),
            'fastest_speed': fastest[0].get('speed'), 'status': status,
            'planned_radius': radius,
            'radius_settings': {k: v for k, v in settings.items() if 'radius' in k},
        }
        print(json.dumps(results), flush=True)
        gates = {
            'pours': last > 0,
            'no_status': not status,
            'flux_matches_area_speed': abs(last / expected - 1) <= 0.15,
            'steady_no_backlog': max(per_step) <= 2.0 * mean_tail + 8,
            'all_grains_alive': digest['particles'] == last,
            'no_overlap_blast': (fastest[0].get('speed') or 0.0) <=
                SPEED + 9.81 * STEPS * DT + 0.5,
        }
        for name, ok in gates.items():
            print(('PASS ' if ok else 'FAIL ') + name, flush=True)
    finally:
        try:
            call('flow_source.remove', name=SOURCE)
        except Exception:
            pass
        try:
            call('fluid.remove_domain', domain=DOMAIN)
        except Exception:
            pass
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
