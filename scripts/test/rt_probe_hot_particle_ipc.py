"""Which particle runs past every heat source? (2026-10-21)

Steps the open scene's domain one fluid.step at a time and reads
fluid.particle_extremes after each. The bound is the hottest birth temperature
of the enabled flow sources aimed at the domain (and the domain ambient): heat
cannot carry a particle past its hottest source. The first steps where the
hottest particle passes it are printed with what that particle is.

Probe, not a gate: run it in the scene that shows the reading
(the user's 4-source scene: Water, Snow, 600 K Sand, gas). Pause first.
Full rows go to %TEMP%/rt_probe_hot_particle.json.
"""
import argparse
import json
import os
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', default='Physics Domain 1')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--dt', type=float, default=1 / 24)
    parser.add_argument('--events', type=int, default=5, help='stop after this many hits')
    parser.add_argument('--margin', type=float, default=0.5, help='K above the bound')
    args = parser.parse_args()
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    domain = next(d for d in call('fluid.list_domains')['domains'] if d['name'] == args.domain)
    sources = [s for s in call('flow_source.list')
               if s['enabled'] and s['domain'] == args.domain and s['phase'] != 'gas']
    bound = max([float(domain.get('thermal_ambient_kelvin', 293.0))] +
                [float(s['fluid_birth_kelvin']) for s in sources])
    print('bound %.1f K from %s' % (bound, [(s['name'], s['fluid_substance'],
                                            s['fluid_birth_kelvin']) for s in sources]), flush=True)
    out = {'bound_kelvin': bound, 'events': []}
    try:
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        for step in range(1, args.steps + 1):
            call('fluid.step', dt=args.dt)
            ex = call('fluid.particle_extremes', domain=args.domain, count=4)
            hot = ex['hottest']
            if not hot or hot[0]['kelvin'] <= bound + args.margin:
                continue
            over = [p for p in hot if p['kelvin'] > bound + args.margin]
            # Rain parcels are born at their air cell's rain temperature: show
            # the cells too (fluid.gas_cell_extremes).
            gas = call('fluid.gas_cell_extremes', domain=args.domain, count=3)
            out['events'].append({'step': step, 'rows': over, 'gas': gas})
            print('step %d: %d over %.1f K' % (step, len(over), bound), flush=True)
            for p in over:
                print('  %.1f K %s model=%s transport=%s mass_fraction=%.3f rest_mass=%.4g '
                      'solid=%.2f frozen=%s id=%d pos=(%.2f, %.2f, %.2f)' % (
                          p['kelvin'], p['substance'] or '(untagged)', p['model'],
                          p['granular_transport'] or '-', p['mass_fraction'],
                          p['rest_mass_kg'], p['fusion_solid_fraction'], p['frozen'],
                          p['particle_id'], *p['position']), flush=True)
            print('  gas: max vapour/pure %.2f' % gas.get('max_vapour_over_pure', 0.0), flush=True)
            for label in ('hottest', 'most_vapour'):
                for c in gas.get(label, []):
                    print('   %-11s T %.1f K (Tv %.1f) vapour %.3g kg/m3 (pure %.3g) fog %.3g kg '
                          'rain %.3g kg @ %.1f K ijk=%s' % (
                              label, c['air_kelvin'], c['virtual_kelvin'], c['vapour_kg_m3'],
                              c['pure_vapour_kg_m3'], c['condensate_kg'], c['rain_kg'],
                              c['rain_kelvin'], c['ijk']), flush=True)
            if len(out['events']) >= args.events:
                break
        if not out['events']:
            print('no particle passed %.1f K in %d steps' % (bound, args.steps), flush=True)
    finally:
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        path = Path(os.environ.get('TEMP', '.')) / 'rt_probe_hot_particle.json'
        path.write_text(json.dumps(out, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
