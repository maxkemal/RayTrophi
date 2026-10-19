"""P1 acceptance (2026-10-19): walls and air are films, fields own their rate.

Short on purpose (MATTER_ROADMAP_2026_10 test rule): a few numeric gates.
  * retired keys are refused: fluid.set_param thermal_*_cooling_rate,
    substance.set cooling_rate;
  * a Thermal field reports thermal_rate_per_s (default 2/s) and takes edits;
  * 330 K water poured into a closed plain Fluid domain (walls + open air
    surface) cools through films: it loses heat to the ambient (booked as
    thermal_heat_to_ambient_j > 0) but SLOWLY - a 5 cm parcel on a wall has
    tau = rho c h / 500 ~ 420 s, so the mean drops well under 2 K in 2 s.
    The old 2/s contact rate dropped wall parcels ~25 K/s.
"""
import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_ThermalFilms'
WATER = 'T: films water'
FIELD = 'T: films field'
DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]


def main():
    client = RtIpc()
    call = client.call
    data = {}

    def refused(method, **params):
        try:
            call(method, **params)
        except Exception as error:  # noqa: BLE001 - any rejection counts
            return str(error)
        return None

    assert not call('sim.control_state')['playing'], 'Pause first'
    field = None
    try:
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-.4, 0, -.4],
             domain_max=[.4, .6, .4], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', thermal_liquid_enabled=True)
        data['refused_rate'] = refused('fluid.set_param', domain=DOMAIN,
                                       thermal_contact_cooling_rate=2.0)
        data['refused_cooling'] = refused('substance.set', name='Water',
                                          fields={'cooling_rate': 0.1})
        field = call('forcefield.create', type='thermal', name=FIELD)['name']
        call('forcefield.set_param', field=field, enabled=False, thermal_rate_per_s=0.5)
        fields = {f['name']: f for f in call('forcefield.list')}
        data['field_rate'] = fields.get(field, {}).get('thermal_rate_per_s')
        call('forcefield.remove', field=field)
        field = None

        call('flow_source.create', name=WATER, domain=DOMAIN, phase='liquid',
             source_mode='point', fluid_substance='Water', position=[0, .12, 0], radius=.15,
             velocity=[0, 0, 0], fluid_velocity_spread=0., fluid_particles_per_second=24000.,
             fluid_temperature_override=True, fluid_temperature_kelvin=330.,
             use_time_limit=True, start_time=0., end_time=.3,
             use_particle_limit=True, max_emitted_particles=6000)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        to_ambient = 0.0
        means = []
        for step in range(1, 181):
            call('fluid.step', dt=DT)
            row = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
            if step > 30:  # after the pour
                to_ambient += row.get('thermal_heat_to_ambient_j', 0.0)
                means.append(row.get('thermal_mean_kelvin', 0.0))
        data.update({'to_ambient_j': to_ambient, 'mean_first': means[0], 'mean_last': means[-1]})
        drop_per_s = (means[0] - means[-1]) / ((len(means) - 1) * DT)
        data['drop_k_per_s'] = drop_per_s
        print(json.dumps(data), flush=True)
        verdicts = {
            'retired_rate_refused': bool(data['refused_rate']),
            'retired_cooling_refused': bool(data['refused_cooling']),
            'field_rate_reported': data['field_rate'] is not None and
                                   abs(data['field_rate'] - 0.5) < 1e-6,
            'walls_take_heat': to_ambient > 0.0,
            'cools_at_film_rate': 0.0 < drop_per_s < 1.0,
        }
        data['verdicts'] = verdicts
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for method, params in (('forcefield.remove', {'field': field}),
                               ('flow_source.remove', {'name': WATER}),
                               ('fluid.remove_domain', {'domain': DOMAIN})):
            if all(v is not None for v in params.values()):
                try:
                    call(method, **params)
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
