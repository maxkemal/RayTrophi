"""P9 acceptance (2026-10-20): the matter domain's known holes, three short arms.

  ice_melt   Ice grains (4 mm, 263 K) dropped into 360 K water. The melt path
             existed (Ice is a solid-skeleton substance: its fusion fraction
             runs, and at zero the parcel becomes liquid) but was never seen
             live. Gates: the melt point is reached, latent heat is paid,
             grains melt into the liquid, mass is conserved.
  ice_floor  Water born at 180 K in a closed Matter domain freezes on the floor
             (held ice). Sand falls on it. Before P9 grains met triangles only
             and fell through. Gates: held-ice faces exist; the sand rests
             above the floor (on the ice), not on it.
  pore_heat  Wet (absorbing) cold Sand (280 K) into 340 K water. Absorbed
             water now brings its own heat into the grain (pore_heat_mixed_j),
             and the grain-air film reads the vapour the hot water makes
             (thermal_grain_air_humid; a device-owned inventory used to read
             as dry air).
Numbers go to docs/dev/p9_matter_live.json. Empty scene, Pause first.
Steps run through fluid.run (one IPC call per chunk, 2026-10-20); --arm runs
one arm (melt | floor | pore), the default runs all three.
"""
import argparse
import json
from pathlib import Path

from rt_ipc import RtIpc
from rt_grain_material import install_grain_material

DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/dev/p9_matter_live.json'


def find_key(node, key):
    """First value of `key` anywhere in a JSON tree."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = find_key(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = find_key(value, key)
            if found is not None:
                return found
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=('melt', 'floor', 'pore'), action='append')
    args = parser.parse_args()
    arms = set(args.arm or ('melt', 'floor', 'pore'))
    client = install_grain_material(RtIpc())
    call = client.call
    data = {}
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'

    def row(name):
        return {d['name']: d for d in call('fluid.list_domains')['domains']}[name]

    def run(name, steps, sums=(), maxima=()):
        return call('fluid.run', steps=steps, dt=DT, sample_every=0, domains=[name],
                    sum_fields=list(sums), max_fields=list(maxima))

    def total_mass(name):
        inv = call('fluid.matter_models', domain=name)
        return sum(m.get('mass_kg', 0.0) for m in inv.get('models', [])), inv

    def domain(name):
        call('fluid.create_domain', name=name, type='matter', domain_min=[-.3, 0, -.3],
             domain_max=[.3, .6, .3], voxel_size=.05)
        call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', solid_phase=False,
             thermal_liquid_enabled=True, max_particles=60000)
        call('fluid.set_pore_exchange', domain=name, enabled=False,
             wet_response_enabled=False, wet_appearance_enabled=False)
        call('fluid.settle', domain=name, on_seed=False, request=False)

    def source(name, dom, substance, pos, radius, kelvin, start, end, rate, limit):
        call('flow_source.create', name=name, domain=dom, phase='liquid', source_mode='point',
             fluid_substance=substance, position=pos, radius=radius, velocity=[0, 0, 0],
             fluid_velocity_spread=0., fluid_particles_per_second=rate,
             fluid_temperature_override=True, fluid_temperature_kelvin=kelvin,
             use_time_limit=True, start_time=start, end_time=end,
             use_particle_limit=True, max_emitted_particles=limit)

    sources = []
    try:
        # ── ice_melt ────────────────────────────────────────────────────────
        name = 'T_P9_Melt'
        if 'melt' in arms:
            domain(name)
        try:
            if 'melt' not in arms:
                raise StopIteration
            source('T_P9_melt_water', name, 'Water', [0, .12, 0], .18, 360., 0., .5, 24000., 9000)
            source('T_P9_melt_ice', name, 'Ice', [0, .45, 0], .08, 263., .6, .9, 1000., 200)
            sources += ['T_P9_melt_water', 'T_P9_melt_ice']
            call('fluid.set_grain_settings', domain=name, radius_m=.004, fluid_coupling=True,
                 max_substeps=2048)
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            sums = ('thermal_fusion_absorbed_j', 'thermal_melted_this_frame')
            peaks = ('thermal_grain_count', 'thermal_grain_max_kelvin')
            first = run(name, 70, sums, peaks)
            mass_after_pour, _ = total_mass(name)
            rest = run(name, 650, sums, peaks)
            mass_end, inv = total_mass(name)
            fs, rs = first['sums'].get(name, {}), rest['sums'].get(name, {})
            fm, rm = first['max'].get(name, {}), rest['max'].get(name, {})
            absorbed = fs.get('thermal_fusion_absorbed_j', 0.0) + rs.get('thermal_fusion_absorbed_j', 0.0)
            melted = int(fs.get('thermal_melted_this_frame', 0) + rs.get('thermal_melted_this_frame', 0))
            grains_peak = max(fm.get('thermal_grain_count', 0), rm.get('thermal_grain_count', 0))
            grain_max_k = max(fm.get('thermal_grain_max_kelvin', 0.0), rm.get('thermal_grain_max_kelvin', 0.0))
            grains_end = row(name).get('thermal_grain_count', 0)
            data['ice_melt'] = {'absorbed_j': absorbed, 'melted_events': melted,
                                'grains_peak': grains_peak, 'grains_end': grains_end,
                                'grain_max_k': grain_max_k, 'mass_after_pour_kg': mass_after_pour,
                                'mass_end_kg': mass_end}
            print('ice_melt', json.dumps(data['ice_melt']), flush=True)
        except StopIteration:
            pass
        finally:
            if 'melt' in arms:
                for s in ('T_P9_melt_water', 'T_P9_melt_ice'):
                    call('flow_source.remove', name=s)
                    sources.remove(s)
                call('fluid.remove_domain', domain=name)

        # ── ice_floor ───────────────────────────────────────────────────────
        name = 'T_P9_Floor'
        if 'floor' in arms:
            domain(name)
        try:
            if 'floor' not in arms:
                raise StopIteration
            source('T_P9_floor_water', name, 'Water', [0, .1, 0], .25, 180., 0., .4, 24000., 6000)
            source('T_P9_floor_sand', name, 'Sand', [0, .5, 0], .08, 180., 1.5, 1.8, 1000., 200)
            sources += ['T_P9_floor_water', 'T_P9_floor_sand']
            call('fluid.set_grain_settings', domain=name, radius_m=.01, fluid_coupling=True,
                 max_substeps=2048)
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            ice_faces_peak, frozen_peak = 0, 0
            held = None
            for chunk in range(10):
                result = run(name, 30, (), ('thermal_frozen_particles',))
                frozen_peak = max(frozen_peak, int(result['max'].get(name, {}).get(
                    'thermal_frozen_particles', 0)))
                inv = call('fluid.matter_models', domain=name)
                ice_faces_peak = max(ice_faces_peak, find_key(inv, 'ice_faces') or 0)
                held = (inv.get('mixed_execution') or {}).get('status')
            g = inv['acceptance_metrics']['granular']
            data['ice_floor'] = {'frozen_peak': frozen_peak, 'ice_faces_peak': ice_faces_peak,
                                 'sand_particles': g['particles'],
                                 'sand_bounds_min': g.get('bounds_min'),
                                 'sand_com': g.get('dry_center_of_mass'),
                                 'mixed_status_last': held}
            print('ice_floor', json.dumps(data['ice_floor']), flush=True)
        except StopIteration:
            pass
        finally:
            if 'floor' in arms:
                for s in ('T_P9_floor_water', 'T_P9_floor_sand'):
                    call('flow_source.remove', name=s)
                    sources.remove(s)
                call('fluid.remove_domain', domain=name)

        # ── pore_heat ───────────────────────────────────────────────────────
        name = 'T_P9_Pore'
        if 'pore' in arms:
            domain(name)
        try:
            if 'pore' not in arms:
                raise StopIteration
            source('T_P9_pore_water', name, 'Water', [0, .12, 0], .18, 340., 0., .5, 24000., 9000)
            source('T_P9_pore_sand', name, 'Sand', [0, .45, 0], .08, 280., .6, .9, 1000., 300)
            sources += ['T_P9_pore_water', 'T_P9_pore_sand']
            call('fluid.set_grain_settings', domain=name, radius_m=.01, fluid_coupling=True,
                 max_substeps=2048, wet_grains=True)
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            mixed, humid_steps = 0.0, 0
            for chunk in range(36):
                run(name, 10)
                r = row(name)
                humid_steps += 1 if r.get('thermal_grain_air_humid') else 0
                inv = call('fluid.matter_models', domain=name)
                mixed += find_key(inv, 'pore_heat_mixed_j') or 0.0
            data['pore_heat'] = {'pore_heat_mixed_j_sampled': mixed, 'humid_steps': humid_steps,
                                 'grain_mean_k_end': r.get('thermal_grain_mean_kelvin')}
            print('pore_heat', json.dumps(data['pore_heat']), flush=True)
        except StopIteration:
            pass
        finally:
            if 'pore' in arms:
                for s in ('T_P9_pore_water', 'T_P9_pore_sand'):
                    call('flow_source.remove', name=s)
                    sources.remove(s)
                call('fluid.remove_domain', domain=name)

        m = data.get('ice_melt')
        f = data.get('ice_floor')
        p = data.get('pore_heat')
        verdicts = {}
        if m is not None:
            verdicts.update({
            'ice_reaches_melt_point': m['grain_max_k'] >= 272.0,
            'latent_heat_paid': m['absorbed_j'] > 0.0,
            'ice_grains_melt': m['melted_events'] > 0 and m['grains_end'] < m['grains_peak'],
            'melt_conserves_mass': m['mass_after_pour_kg'] is not None and
                abs(m['mass_end_kg'] - m['mass_after_pour_kg']) <= 1e-4 * m['mass_after_pour_kg'],
            })
        if f is not None:
            verdicts.update({
            'held_ice_has_faces': f['frozen_peak'] > 0 and f['ice_faces_peak'] > 0,
            # 1 cm grains on the floor sit at y ~ 0.01; on the ice they sit at
            # least a voxel (5 cm) higher. Hanging where they were born (y ~ .45)
            # is the 8th-build failure: the step was held.
            'sand_rests_on_ice': f['sand_particles'] > 0 and f['sand_bounds_min'] is not None and
                                 .05 < f['sand_bounds_min'][1] < .4,
            })
        if p is not None:
            verdicts.update({
            'absorbed_water_brings_heat': p['pore_heat_mixed_j_sampled'] > 0.0,
            'grain_air_reads_vapour': p['humid_steps'] > 0,
            })
        data['verdicts'] = verdicts
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        for key, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', key, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for s in sources:
            try:
                call('flow_source.remove', name=s)
            except Exception:  # noqa: BLE001 - best effort cleanup
                pass
        call('timeline.set_frame', frame=0)
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
