"""Builds the fog/rain EYE-TEST scene and leaves it in the app (no cleanup).

Same physics as rt_test_fog_ipc.py: a closed matter domain with a Water tank,
one corner boiled (+400 K thermal field), the opposite side held cold
(-60 K). Paused at frame 0, seeded; press Play to watch.

What is drawn today (2026-10-11):
  * steam: the gas density tracer's volume (the domain's fog view);
  * rain only once it is liquid again (rain-out parcels, absorbed rain).
Fog water and falling rain are physics-only; no renderer reads them yet.

Optional: --steps N pre-steps the sim so there is something to see at once.
"""

import argparse

from rt_ipc import RtIpc

DOMAIN = 'Gorsel_Buhar_Yagmur'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=0)
    args = parser.parse_args()
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    if DOMAIN not in {d['name'] for d in call('fluid.list_domains')['domains']}:
        call('fluid.create_domain', name=DOMAIN, type='matter',
             domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=.05)
    call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
         backend='vulkan', boundary='closed', default_substance='Water',
         thermal_liquid_enabled=True)
    fields = {f['name'] for f in call('forcefield.list')}
    for name, position, inner, falloff, delta in (
            ('Gorsel: kaynatici', [-.8, -.6, -.8], .6, .9, 400.0),
            ('Gorsel: soguk', [.8, -.5, .8], 1.0, 1.3, -60.0)):
        if name not in fields:
            name = call('forcefield.create', type='thermal', name=name)['name']
        call('forcefield.set_param', field=name, enabled=True, shape='sphere',
             position=position, inner_radius=inner, falloff_radius=falloff,
             thermal_delta_kelvin=delta)
    call('timeline.set_frame', frame=0)
    call('fluid.reset')
    # fluid.step never runs a pending seed (the frame loop does): seed here.
    call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True, fill_level=.1)
    for _ in range(args.steps):
        call('fluid.step', dt=1 / 60)
    d = {x['name']: x for x in call('fluid.list_domains')['domains']}[DOMAIN]
    print('ready:', DOMAIN, 'particles', d['particle_count'],
          'vapour_kg', d.get('gas_phase_mass_kg'), 'rain_kg', d.get('gas_rain_kg'))
    client.close()


if __name__ == '__main__':
    main()
