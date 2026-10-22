"""Snow takes heat from what touches it (2026-10-20), two arms, each with a control.

  sand  Snow (born 263 K, solid, MPM) lies on the floor; DEM Sand pours onto
        it at 900 K (control: Sand at 263 K). Before the fix the grain heat
        pass counted MPM parcels as nobody (Role::None): 600 K sand gave the
        snow nothing and it never melted. Gates: the hot run melts snow and
        pays its latent heat, clearly more than the control; the grain heat
        ledger closes.
  water Snow on the floor, Water poured onto it at 330 K (control: 274 K).
        Before the fix parcels sharing a cell never traded heat (cell faces
        only), so snow inside warm water kept its 263 K. Gates: in-cell
        exchange moves heat, the warm run melts snow with latent heat, more
        than the control.
  boil  Water at 293 K, Sand poured into it at 900 K (control 340 K). The
        grain heat now lands on the parcels at the contact (D1), so water
        touching hot sand boils before the cell's pool reaches 373 K.
        No particle ends hotter than the 900 K sand (contact deposit cap,
        2026-10-21).
  mix   Snow, 293 K water and 293 K sand together: no particle passes 293 K.
        The proximity share once carried a parcel past the grains whenever
        the cell's pool mixed two temperatures (snow + water).

Numbers go to docs/dev/snow_heat_live.json. Empty scene, Pause first.
--arm sand | water | boil | mix runs one arm (repeatable); the default runs all.
"""
import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc

DT = 1 / 60
STEPS = 300
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/dev/snow_heat_live.json'
DOMAIN = 'T_SnowHeat'
SUMS = ('thermal_fusion_absorbed_j', 'thermal_melted_this_frame', 'thermal_in_cell_heat_j',
        'thermal_grain_heat_to_liquid_j', 'thermal_grain_energy_residual_j',
        'thermal_evaporated_kg')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('sand', 'water', 'boil', 'mix'), action='append')
    args = parser.parse_args()
    arms = args.arm or ['sand', 'water', 'boil', 'mix']
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}
    sources = []

    def source(name, substance, pos, radius, start, end, rate, limit, kelvin=None):
        params = dict(name=name, domain=DOMAIN, phase='liquid', source_mode='point',
                      fluid_substance=substance, position=pos, radius=radius,
                      velocity=[0, 0, 0], fluid_velocity_spread=0.,
                      fluid_particles_per_second=rate, use_time_limit=True,
                      start_time=start, end_time=end, use_particle_limit=True,
                      max_emitted_particles=limit)
        if kelvin is not None:
            params.update(fluid_temperature_override=True, fluid_temperature_kelvin=kelvin)
        call('flow_source.create', **params)
        sources.append(name)

    def run_case(arm, kelvin):
        call('fluid.create_domain', name=DOMAIN, type='matter', domain_min=[-.3, 0, -.3],
             domain_max=[.3, .6, .3], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', solid_phase=False,
             thermal_liquid_enabled=True, max_particles=60000)
        call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
        try:
            # Snow is born at its own solid temperature (substanceAmbientBirthKelvin).
            # A thin layer: 1500 hot 1 cm grains (~6 kg) must be able to melt
            # part of it in 5 s through their films (tau ~80 s per grain).
            if arm == 'boil':
                # Water at the room's 293 K, hot sand poured in (D1 contact deposit).
                source('T_SnowHeat_pool', 'Water', [0, .08, 0], .14, 0., .3, 20000., 3000)
            else:
                snow = 600 if arm == 'sand' else 3000
                source('T_SnowHeat_snow', 'Snow', [0, .08, 0], .14, 0., .3, 20000., snow)
            if arm in ('sand', 'boil', 'mix'):
                source('T_SnowHeat_sand', 'Sand', [0, .3, 0], .12, .5, .8, 5000., 1500, kelvin)
                call('fluid.set_grain_settings', domain=DOMAIN, radius_m=.01,
                     fluid_coupling=True, max_substeps=2048)
            if arm in ('water', 'mix'):
                source('T_SnowHeat_water', 'Water', [0, .35, 0], .12, .5, .8, 24000., 4000, kelvin)
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            result = call('fluid.run', steps=STEPS, dt=DT, sample_every=0, domains=[DOMAIN],
                          sum_fields=list(SUMS), max_fields=['thermal_grain_max_kelvin', 'thermal_max_kelvin'])
            sums = result['sums'].get(DOMAIN, {})
            case = {key: sums.get(key, 0.0) for key in SUMS}
            # Hottest particle over the run, grains and parcels alike.
            case['max_kelvin'] = result['max'].get(DOMAIN, {}).get('thermal_max_kelvin', 0.0)
            case['step_ms_median'] = result['step_ms']['median']
            return case
        finally:
            while sources:
                call('flow_source.remove', name=sources.pop())
            call('fluid.remove_domain', domain=DOMAIN)

    verdicts = {}
    try:
        if 'sand' in arms:
            hot, cold = run_case('sand', 900.), run_case('sand', 263.)
            data['sand'] = {'hot': hot, 'cold': cold}
            print('sand hot', json.dumps(hot), flush=True)
            print('sand cold', json.dumps(cold), flush=True)
            verdicts.update({
                'sand_heats_snow_pool': hot['thermal_grain_heat_to_liquid_j'] > 0.0,
                # thermal_melted_this_frame counts parcels crossing the half-solid
                # threshold; in-cell mixing melts many parcels part-way at once
                # (mushy), so the latent heat paid is the gate.
                'hot_sand_melts_snow': hot['thermal_fusion_absorbed_j'] > 1.0e4,
                'hot_sand_beats_control': hot['thermal_fusion_absorbed_j'] >
                    3.0 * max(cold['thermal_fusion_absorbed_j'], 1.0),
                # Residual is summed over every step: a joule per step is rounding.
                'grain_ledger_closes': abs(hot['thermal_grain_energy_residual_j']) <= STEPS * 1.0,
                'snow_never_above_hot_sand': 0.0 < hot['max_kelvin'] <= 900.0 + 0.5,
            })
        if 'water' in arms:
            warm, cold = run_case('water', 330.), run_case('water', 274.)
            data['water'] = {'warm': warm, 'cold': cold}
            print('water warm', json.dumps(warm), flush=True)
            print('water cold', json.dumps(cold), flush=True)
            verdicts.update({
                'in_cell_exchange_moves_heat': warm['thermal_in_cell_heat_j'] > 0.0,
                'warm_water_melts_snow': warm['thermal_fusion_absorbed_j'] > 1.0e4,
                'warm_water_beats_control': warm['thermal_fusion_absorbed_j'] >
                    3.0 * max(cold['thermal_fusion_absorbed_j'], 1.0),
            })
        if 'boil' in arms:
            # D1: the grain heat lands on the parcels at the contact, so water
            # touching 900 K sand reaches its boiling point long before the
            # cell's pool does (before D1: 0.008 kg in the user's 4-source scene).
            hot, cold = run_case('boil', 900.), run_case('boil', 340.)
            data['boil'] = {'hot': hot, 'cold': cold}
            print('boil hot', json.dumps(hot), flush=True)
            print('boil cold', json.dumps(cold), flush=True)
            verdicts.update({
                'hot_sand_boils_water': hot['thermal_evaporated_kg'] > 0.01,
                'boil_beats_control': hot['thermal_evaporated_kg'] >
                    10.0 * max(cold['thermal_evaporated_kg'], 1.0e-4),
                'boil_ledger_closes': abs(hot['thermal_grain_energy_residual_j']) <= STEPS * 1.0,
                # Heat cannot carry a parcel past its source: the contact
                # deposit once put a parcel at 629.8 K beside 600 K sand.
                'parcel_never_above_hot_sand': 0.0 < hot['max_kelvin'] <= 900.0 + 0.5,
            })
        if 'mix' in arms:
            # Snow + water + sand, the sand no warmer than the water (user's
            # 4-source scene, 2026-10-21): the cell pool mixes 263 K snow with
            # 293 K water, the grains warm the pool, and the proximity share
            # carried a 293 K water parcel beside them to 296 K (600 K sand:
            # 734 K). Nothing here is above 293 K, so nothing may end there.
            mix = run_case('mix', 293.)
            data['mix'] = mix
            print('mix', json.dumps(mix), flush=True)
            verdicts.update({
                'mix_parcel_never_above_sources': 0.0 < mix['max_kelvin'] <= 293.0 + 0.5,
                'mix_ledger_closes': abs(mix['thermal_grain_energy_residual_j']) <= STEPS * 1.0,
            })
        data['verdicts'] = verdicts
        for key, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', key, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        call('timeline.set_frame', frame=0)
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
