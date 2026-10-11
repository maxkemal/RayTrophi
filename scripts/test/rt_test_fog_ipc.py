"""Fog water + steam condensation acceptance (2026-10-11).

A closed matter domain holds a Water tank. One corner boils (+400 K field),
the opposite side is held cold (-60 K field). The air above carries steam.
Expect:
  * fog water forms in the air (gas_condensate_kg > 0) and its latent heat
    warms the air (fog_latent_to_air_j > 0 on some step);
  * vapour <-> fog exchanges are balanced in the ledger;
  * surface condensation is no longer negligible: it returns more than 1%
    of what boiled (2026-10-11 before this batch: 0.010 kg of 68 kg);
  * closed box, water conserved: vapour + fog water + rain water in the air
    equals what left the liquid (vaporization, mist) minus what came back
    (surface condensation, rain) within 2%.
  * part 3 (rain): fog water stays near the domain's ceiling (the first live
    run held 41.7 kg/m^3 suspended); the excess autoconverts into rain water
    and the rain reaches the liquid.
  * part 4: boiled Water stays in the vapour pool (gas_foreign_kg stays 0:
    only non-condensable gas goes there).
  * part 4, --wind S: a +x wind field (affects gas only) of strength S. Rain
    rides the air now, so its mass-weighted centre (gas_rain_centroid, mean
    over the run) must sit more than one voxel downwind of the no-wind run's.
    The wind run uses an OPEN boundary: in a closed box a uniform body force is
    a pure gradient, the pressure projection removes it and no wind blows
    (first live run 2026-10-11: centroid shift 0.0001 m).
    Run once WITHOUT --wind first: that run writes the baseline.
Scene domains, sources and force fields are switched off for the run and
restored afterwards.
"""

import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_Fog'
DT = 1 / 60
STEPS = 240
FOG_CEILING = 5.0e-3   # kg/m^3, APICSolverParams::fog_ceiling_kg_m3 default
VOXEL = .05
ROOT = Path(__file__).resolve().parents[2]


def balanced(row):
    return (abs(row['target_mass_kg'] - row['source_mass_kg']) < 1e-9 and
            abs(row['target_energy_j'] + row['latent_required_j'] -
                row['source_energy_j'] - row['latent_accounted_j']) < 1e-3)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wind', type=float, default=0.0,
                        help='strength of a +x gas wind field; 0 = baseline run')
    args = parser.parse_args()
    client = RtIpc()
    call = client.call
    data = {'samples': [], 'wind': args.wind}
    baseline_path = ROOT / 'docs/dev/fog_live.json'
    out = ROOT / 'docs/dev/fog_wind_live.json' if args.wind > 0 else baseline_path

    def save():
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')

    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    created_domain = DOMAIN not in {d['name'] for d in original['domains']}
    fields = []
    try:
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=False)
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        if created_domain:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=VOXEL)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
             backend='vulkan', boundary='open' if args.wind > 0 else 'closed',
             default_substance='Water',
             thermal_liquid_enabled=True, fog_ceiling_kg_m3=FOG_CEILING)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True, fill_level=.1)
        for name, position, inner, falloff, delta in (
                ('T: fog boil', [-.8, -.6, -.8], .6, .9, 400.0),
                ('T: fog cold', [.8, -.5, .8], 1.0, 1.3, -60.0)):
            created = call('forcefield.create', type='thermal', name=name)['name']
            call('forcefield.set_param', field=created, enabled=True, shape='sphere',
                 position=position, inner_radius=inner, falloff_radius=falloff,
                 thermal_delta_kelvin=delta)
            fields.append(created)
        if args.wind > 0:
            created = call('forcefield.create', type='wind', name='T: fog wind')['name']
            call('forcefield.set_param', field=created, enabled=True, shape='infinite',
                 direction=[1, 0, 0], strength=args.wind, use_noise=False,
                 affects_fluid=False, affects_gas=True, affects_particles=False,
                 affects_cloth=False, affects_rigidbody=False)
            fields.append(created)

        max_foreign = 0.0
        rain_x_weighted = rain_weight = 0.0
        boiled = surface_condensed = rained = mist = 0.0
        fog_condensed = fog_evaporated = autoconverted = absorbed = 0.0
        max_fog = max_fog_mean = 0.0
        latent_to_air = False
        measured = False
        unbalanced = []
        for step in range(1, STEPS + 1):
            call('fluid.step', dt=DT)
            for r in call('matter.exchanges')['exchanges']:
                if r['kind'] == 'vaporization' and r['source'] == DOMAIN:
                    boiled += r['source_mass_kg']
                elif r['kind'] == 'mist_to_gas' and r['source'] == DOMAIN:
                    mist += r['source_mass_kg']   # liquid droplets into the fog pool
                elif r['kind'] == 'condensation' and r['source'] == DOMAIN:
                    surface_condensed += r['target_mass_kg']
                elif r['kind'] == 'precipitation':
                    rained += r['target_mass_kg']
                elif r['kind'] == 'autoconversion' and r['source'] == DOMAIN + ':fog':
                    autoconverted += r['target_mass_kg']
                elif r['kind'] == 'condensation' and r['source'] == DOMAIN + ':vapour':
                    fog_condensed += r['target_mass_kg']
                elif r['kind'] == 'vaporization' and r['source'] == DOMAIN + ':fog':
                    fog_evaporated += r['target_mass_kg']
                if r['kind'] in ('condensation', 'vaporization', 'precipitation',
                                 'autoconversion') and not balanced(r):
                    unbalanced.append(r)
            d = {x['name']: x for x in call('fluid.list_domains')['domains']}[DOMAIN]
            max_fog = max(max_fog, d.get('gas_condensate_kg', 0.0))
            max_fog_mean = max(max_fog_mean, d.get('gas_condensate_mean_kg_m3', 0.0))
            absorbed += d.get('rain_absorbed_kg', 0.0)
            max_foreign = max(max_foreign, d.get('gas_foreign_kg', 0.0))
            rain_now = d.get('gas_rain_kg', 0.0)
            if rain_now > 0.0:
                rain_x_weighted += rain_now * d['gas_rain_centroid'][0]
                rain_weight += rain_now
            latent_to_air = latent_to_air or d.get('fog_latent_to_air_j', 0.0) > 0.0
            measured = measured or d.get('saturation_measured', False)
            if step % 20 == 0 or step == 1:
                sample = {'step': step, 'particles': d['particle_count'],
                          'vapour_kg': d['gas_phase_mass_kg'],
                          'fog_kg': d.get('gas_condensate_kg'),
                          'fog_mean_kg_m3': d.get('gas_condensate_mean_kg_m3'),
                          'rain_kg': d.get('gas_rain_kg'),
                          'rain_cells': d.get('gas_rain_cells'),
                          'rain_centroid': d.get('gas_rain_centroid'),
                          'foreign_kg': d.get('gas_foreign_kg'),
                          'autoconverted_kg': autoconverted,
                          'boiled_kg': boiled, 'surface_condensed_kg': surface_condensed,
                          'rained_kg': rained, 'fog_condensed_kg': fog_condensed,
                          'fog_evaporated_kg': fog_evaporated}
                data['samples'].append(sample)
                print(json.dumps(sample), flush=True)
        last = data['samples'][-1]
        in_air = last['vapour_kg'] + (last['fog_kg'] or 0.0) + (last['rain_kg'] or 0.0)
        expected_air = boiled + mist - surface_condensed - rained
        verdicts = {
            'saturation_ran': measured,
            'fog_forms': max_fog > 0.0 and fog_condensed > 0.0,
            'latent_heats_air': latent_to_air,
            'exchanges_balanced': not unbalanced,
            'surface_condensation_substantial': boiled > 0.0 and
                surface_condensed > .01 * boiled,
            # An open box (wind mode) loses air water through its walls.
            'water_conserved': args.wind > 0 or (expected_air > 0.0 and
                abs(in_air - expected_air) <= .02 * expected_air),
            # Advection after the cell pass may concentrate it a little;
            # 41.7 kg/m^3 (before the ceiling) is four orders above this bar.
            'fog_capped': 0.0 < max_fog_mean <= 2.0 * FOG_CEILING,
            'rain_forms': autoconverted > 0.0,
            'rain_reaches_liquid': rained > 0.0 and absorbed > 0.0,
            'water_stays_vapour': max_foreign == 0.0,
        }
        rain_x = rain_x_weighted / rain_weight if rain_weight > 0.0 else None
        data['rain_centroid_x'] = rain_x
        if args.wind > 0:
            baseline = None
            if baseline_path.exists():
                baseline = json.loads(baseline_path.read_text(encoding='utf-8')).get('rain_centroid_x')
            data['baseline_rain_centroid_x'] = baseline
            verdicts['rain_drifts_downwind'] = (
                rain_x is not None and baseline is not None and rain_x - baseline > VOXEL)
        data.update({'boiled_kg': boiled, 'surface_condensed_kg': surface_condensed,
                     'rained_kg': rained, 'mist_kg': mist, 'fog_condensed_kg': fog_condensed,
                     'fog_evaporated_kg': fog_evaporated, 'max_fog_kg': max_fog,
                     'max_fog_mean_kg_m3': max_fog_mean, 'autoconverted_kg': autoconverted,
                     'rain_absorbed_kg': absorbed, 'max_foreign_kg': max_foreign,
                     'in_air_kg': in_air, 'expected_air_kg': expected_air,
                     'unbalanced': unbalanced[:5], 'verdicts': verdicts})
        save()
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for f in fields:
            call('forcefield.remove', field=f)
        if created_domain:
            try:
                call('fluid.remove_domain', domain=DOMAIN)
            except Exception:  # noqa: BLE001 - leave it disabled instead
                call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        else:
            call('fluid.set_param', domain=DOMAIN, enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=f['enabled'])
        for domain in original['domains']:
            call('fluid.set_param', domain=domain['name'], enabled=domain['enabled'],
                 visible=domain['visible'])
        for source in original['sources']:
            call('flow_source.update', name=source['name'], enabled=source['enabled'])
        call('timeline.set_frame', frame=0)
        save()
        client.close()


if __name__ == '__main__':
    main()
