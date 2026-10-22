"""MPM + DEM in one Matter domain (2026-10-20): Snow must fall next to Sand.

Bug: on the shared Matter clock (MPM parcels + DEM grains in one domain) the
per-substep force integration re-uploaded the host particles, rewinding every
substep to the frame start. Snow alone fell; Snow beside Sand stayed at its
emitter with its birth speed.

Three runs of 40 frames with equal emitters: snow only, sand only, both. With
equal counts the snow centroid in the mixed run is 2 * mixed - sand_only.
Gate: the snow drops in the mixed run at least half as far as alone.
Empty scene, Pause first.
"""
from rt_ipc import RtIpc

DT = 1 / 60
DOMAIN = 'T_MpmDem'
BIRTH_Y = 1.6


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    call('fluid.create_domain', name=DOMAIN, type='matter', domain_min=[-1, 0, -1],
         domain_max=[1, 2, 1], voxel_size=.05)
    call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
         boundary='closed', max_particles=40000)
    # Birth temperature and phase follow the substance (read-only on the source).
    sources = {'T_MpmDem_sand': ('Sand', [.4, BIRTH_Y, 0]),
               'T_MpmDem_snow': ('Snow', [-.4, BIRTH_Y, 0])}
    try:
        for name, (substance, pos) in sources.items():
            call('flow_source.create', name=name, domain=DOMAIN, phase='liquid',
                 source_mode='point', fluid_substance=substance, position=pos, radius=.2,
                 velocity=[0, -.5, 0], fluid_particles_per_second=1000.,
                 use_time_limit=True, start_time=0., end_time=.3)

        def probe(enabled):
            for name in sources:
                call('flow_source.update', name=name, enabled=name in enabled)
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('fluid.run', steps=40, dt=DT, sample_every=0, domains=[DOMAIN])
            digest = call('fluid.state_digest', domain=DOMAIN)
            status = call('fluid.matter_models', domain=DOMAIN)['mixed_execution']['status']
            return digest['particles'], digest['centroid'][1], status

        snow = probe({'T_MpmDem_snow'})
        sand = probe({'T_MpmDem_sand'})
        both = probe(set(sources))
        snow_mixed_y = 2 * both[1] - sand[1]
        print('snow only', snow, flush=True)
        print('sand only', sand, flush=True)
        print('both', both, 'snow y in mixed run ~ %.3f' % snow_mixed_y, flush=True)
        drop_alone = BIRTH_Y - snow[1]
        drop_mixed = BIRTH_Y - snow_mixed_y
        checks = {
            'snow_falls_alone': drop_alone > .2,
            'mixed_run_uses_shared_clock': 'MPM + grain' in both[2],
            'equal_counts': both[0] == snow[0] + sand[0],
            'snow_falls_beside_sand': drop_mixed >= .5 * drop_alone,
        }
        for key, passed in checks.items():
            print('PASS' if passed else 'FAIL', key, flush=True)
        if not all(checks.values()):
            raise SystemExit(1)
    finally:
        for name in sources:
            try:
                call('flow_source.remove', name=name)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        call('fluid.remove_domain', domain=DOMAIN)
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
