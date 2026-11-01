"""D2.6 real frame-loop controls; empty disposable scene, user build, external IPC.

No fluid.step/run is used. Polling yields to the app's ordinary viewport loop.
Restores the incoming timeline/live mode and leaves the empty scene paused.
"""
import time

from rt_ipc import RtIpc

DOMAIN = 'T_Transport'


def main():
    client = RtIpc()
    call = client.call
    created = False
    source_created = False
    initial = call('timeline.transport')
    assert not initial['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    checks = {}
    try:
        call('sim.set_mode', mode='timeline')
        before = call('timeline.transport')
        ok, _ = client.try_call('sim.set_mode', mode='invalid_transport_mode')
        after = call('timeline.transport')
        checks['invalid_mode_unchanged'] = not ok and (
            before['playing'] == after['playing'] and before['mode'] == after['mode'])
        call('fluid.create_domain', name=DOMAIN, type='fluid',
             domain_min=[-0.2, 0, -0.2], domain_max=[0.2, 0.4, 0.2], voxel_size=0.1)
        created = True
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
             backend='vulkan', default_substance='Water', boundary='closed')
        call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
        call('timeline.set_frame', frame=0)
        call('fluid.seed', domain=DOMAIN, fill_level=0.5, particles_per_cell=8,
             replace=True, persistent=True)
        start = call('timeline.play')
        deadline = time.monotonic() + 15
        state = start
        while state['frame'] < start['frame'] + 3 and time.monotonic() < deadline:
            time.sleep(0.05)
            state = call('timeline.transport')
        paused = call('timeline.pause')
        checks['play_advances_real_loop'] = state['frame'] >= start['frame'] + 3
        checks['pause_reports_stopped'] = not paused['playing']
        time.sleep(0.2)
        checks['pause_holds_frame'] = call('timeline.transport')['frame'] == paused['frame']
        checks['cache_state_reported'] = all(key in paused['cache'] for key in (
            'valid', 'baking', 'ram_frames', 'has_range', 'budget_reached'))
        call('flow_source.create', name=DOMAIN + '_source', domain=DOMAIN,
             phase='liquid', source_mode='point', fluid_substance='Water',
             position=[0, 0.25, 0], radius=0.05, fluid_particles_per_second=120)
        source_created = True
        count = call('fluid.state_digest', domain=DOMAIN)['particles']
        live = call('sim.set_mode', mode='live')
        checks['live_mode_roundtrip'] = live['mode'] == 'live' and not live['playing']
        deadline = time.monotonic() + 5
        digest = call('fluid.state_digest', domain=DOMAIN)
        while digest['particles'] <= count and time.monotonic() < deadline:
            time.sleep(0.05)
            digest = call('fluid.state_digest', domain=DOMAIN)
        checks['live_steps_without_play'] = digest['particles'] > count and (
            call('timeline.transport')['frame'] == live['frame'])
        timeline = call('sim.set_mode', mode='timeline')
        checks['timeline_mode_roundtrip'] = timeline['mode'] == 'timeline'
        for name, passed in checks.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(checks.values()):
            raise SystemExit(1)
    finally:
        try:
            call('timeline.pause')
            call('sim.set_mode', mode='timeline')
            if source_created:
                call('flow_source.remove', name=DOMAIN + '_source')
            if created:
                call('fluid.remove_domain', domain=DOMAIN)
            call('timeline.set_frame', frame=0)
            call('sim.set_mode', mode=initial['mode'])
        finally:
            client.close()


if __name__ == '__main__':
    main()
