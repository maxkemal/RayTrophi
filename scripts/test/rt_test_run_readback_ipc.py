"""Reads right after fluid.run (diagnostic, D1 2026-10-20).

Twice a read right after a long fluid.run came back empty (matter_models
0 kg, list_domains with no substances, step_stats blank) while the run's own
samples were full. Suspect: the frame loop, seeing the timeline still at
frame 0 after N steps, restores frame 0 (empty for an emitter scene).
Gate: every read after runs of 30 / 120 / 300 steps shows the particles the
run's last sample reported, and a non-zero mass. The print shows the timeline
frame next to each read. Empty scene, Pause first.
"""
import time

from rt_ipc import RtIpc

D = 'T_RunRead'


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    failures = []

    def read(tag, expected):
        row = {d['name']: d for d in call('fluid.list_domains')['domains']}[D]
        mm = call('fluid.matter_models', domain=D)
        frame = call('sim.control_state').get('frame')
        mass = sum(m.get('mass_kg', 0.0) for m in mm.get('models', []))
        print(tag, 'frame', frame, 'particles', row.get('particle_count'),
              'substances', len(row.get('substances') or []), 'mass', round(mass, 3), flush=True)
        if row.get('particle_count') != expected or not mass > 0.0:
            failures.append(tag)

    call('fluid.create_domain', name=D, type='matter', domain_min=[-.3, 0, -.3],
         domain_max=[.3, .6, .3], voxel_size=.05)
    call('fluid.set_param', domain=D, enabled=True, visible=False, backend='vulkan',
         boundary='closed', default_substance='Water', max_particles=60000)
    call('fluid.settle', domain=D, on_seed=False, request=False)
    call('flow_source.create', name='T_RunRead_src', domain=D, phase='liquid',
         source_mode='point', fluid_substance='Water', position=[0, .3, 0], radius=.1,
         velocity=[0, 0, 0], fluid_particles_per_second=6000., use_time_limit=True,
         start_time=0., end_time=.5)
    try:
        for steps in (30, 120, 300):
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            out = call('fluid.run', steps=steps, dt=1 / 60, sample_every=0, domains=[D],
                       fields=['particle_count'])
            expected = out['samples'][-1]['domains'][D]['particle_count']
            print('run', steps, 'last sample particles', expected, flush=True)
            read('run%d read1' % steps, expected)
            read('run%d read2' % steps, expected)
            time.sleep(.5)
            read('run%d read3 +0.5s' % steps, expected)
        print('PASS' if not failures else 'FAIL', 'reads_after_run_match', failures, flush=True)
    finally:
        call('flow_source.remove', name='T_RunRead_src')
        call('fluid.remove_domain', domain=D)
        call('timeline.set_frame', frame=0)
        client.close()
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
