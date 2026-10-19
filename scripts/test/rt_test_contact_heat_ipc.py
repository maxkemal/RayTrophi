"""Collider (MSF) <-> liquid contact heat acceptance (2026-10-10).

Hot iron dropped into water. An Iron cube (mesh SDF collider with an MSF
shell) is heated by a Thermal field while the closed matter domain is still
EMPTY; then the field is removed and the tank is filled. The only heat the
water can get is from the cube.
Expect: the shell is hot before the fill (msf.fields max_kelvin), the water
touches it (thermal_solid_contact_particles > 0), heat flows solid -> liquid
(thermal_heat_to_solid_j < 0) and the water's hottest parcel rises above
ambient.
Why not "a stove under the pot": the MSF has no in-solid conduction and, even
if it had, iron's diffusivity (~2.3e-5 m^2/s) needs ~h^2/alpha = 4000 s to move
heat 0.3 m. A 5 s test of that geometry could only pass by being unphysical.
Scene domains, sources, fields and the created objects are restored/removed.
"""

import json
from pathlib import Path

from rt_ipc import RtIpc

DOMAIN = 'T_ContactHeat'
CUBE = 'T_ContactCube'
COLLIDER = 'T_ContactCollider'
FIELD = 'T: contact stove'
DT = 1 / 60
ROOT = Path(__file__).resolve().parents[2]


def main():
    client = RtIpc()
    call = client.call
    data = {'samples': []}
    original = {'control': call('sim.control_state'),
                'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'fields': call('forcefield.list')}
    assert not original['control']['playing'], 'Pause first'
    created = DOMAIN not in {d['name'] for d in original['domains']}
    field = None
    cube = None
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=False)
        if created:
            call('fluid.create_domain', name=DOMAIN, type='matter',
                 domain_min=[-1, 0, -1], domain_max=[1, 2, 1], voxel_size=.05)
        call('fluid.set_param', domain=DOMAIN, enabled=True, visible=True,
             backend='vulkan', boundary='closed', default_substance='Water',
             thermal_liquid_enabled=True)
        cube = call('scene.add_primitive', type='cube', name=CUBE, size=0.6)
        cube = cube.get('name', CUBE) if isinstance(cube, dict) else (cube or CUBE)
        call('collider.create', name=COLLIDER, source_mode='mesh_sdf',
             source_object=cube, msf_substance='Iron',
             # The MSF shell only exists behind these two gates today; every
             # gas rate stays 0, so nothing is injected into the domain.
             gas_interaction_enabled=True, gas_ignite_on_contact=True)
        field = call('forcefield.create', type='thermal', name=FIELD)['name']
        call('forcefield.set_param', field=field, enabled=True, shape='sphere',
             position=[0, 0, 0], inner_radius=0.6, falloff_radius=0.9,
             thermal_delta_kelvin=400.0)
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        for step in range(1, 181):            # heat the dry cube
            call('fluid.step', dt=DT)
        # ★ msf.fields reads the host copy, and an EMPTY domain requests no
        # readback: the copy can date from the field's creation (2026-10-19:
        # 293.7 K reported while the shell was ~390 K). The call requests one;
        # one more heated step lands it. temperature_fresh says it did.
        call('msf.fields')
        call('fluid.step', dt=DT)
        shell = [f for f in call('msf.fields')['fields'] if f['substance'] == 'Iron']
        data['shell_before_fill'] = shell[0] if shell else None
        print(json.dumps({'shell_before_fill': {k: shell[0].get(k) for k in
                          ('mean_kelvin', 'max_kelvin', 'temperature_valid', 'temperature_fresh',
                           'thermal_depth_m', 'surface_area_m2')} if shell else None}),
              flush=True)
        call('forcefield.remove', field=field)
        field = None
        call('fluid.seed', domain=DOMAIN, particles_per_cell=4, replace=True, fill_level=.15)
        for step in range(181, 481):
            call('fluid.step', dt=DT)
            if step % 50 == 0:
                r = {d['name']: d for d in call('fluid.list_domains')['domains']}[DOMAIN]
                s = {'step': step,
                     'contact': r.get('thermal_solid_contact_particles', 0),
                     'heat_to_solid_j': r.get('thermal_heat_to_solid_j', 0.0),
                     'max_kelvin': r['particle_max_kelvin'],
                     'field_heated': r.get('thermal_field_heated_particles', 0),
                     # What the touching parcels saw (2026-10-18 instrument).
                     'shell_seen_kelvin': r.get('thermal_contact_shell_kelvin'),
                     'parcel_seen_kelvin': r.get('thermal_contact_parcel_kelvin'),
                     'shell_cells': r.get('thermal_contact_shell_cells')}
                data['samples'].append(s)
                print(json.dumps(s), flush=True)
        shell_before = data['shell_before_fill'] or {}
        verdicts = {
            'shell_hot_before_fill': bool(shell_before.get('temperature_fresh')) and
                                     shell_before.get('max_kelvin', 0) > 350.0,
            'liquid_touches_shell': any(s['contact'] > 0 for s in data['samples']),
            'field_misses_water': all(s['field_heated'] == 0 for s in data['samples']),
            'heat_flows_solid_to_liquid': any(s['heat_to_solid_j'] < 0 for s in data['samples']),
            # 2026-10-19 root: the wetted faces are the ones INSIDE the domain,
            # and the gas pass used to pin them to the room air (294 K seen while
            # the shell mean was 348 K). With the air film against the body's
            # capacity they heat like the rest of the cube.
            'wetted_shell_is_hot': (data['samples'][0]['shell_seen_kelvin'] or 0.0) > 330.0,
            # A solid iron body, not a 1 mm sheet: ~1 m^2 wetted at ~+45 K under a
            # 500 W/m^2K film is ~24 kW, ~400 J per 1/60 s step. The hollow-shell
            # model drained after a few steps and gave ~15 J.
            'shell_is_a_body': data['samples'][0]['heat_to_solid_j'] < -100.0,
            # A 5 cm parcel (130 J/K) next to the wall warms ~0.002 K/step, so the
            # hottest one lands near +0.5 K in 5 s, not tens of kelvin.
            'water_warms_through_cube': max(s['max_kelvin'] for s in data['samples']) > 293.1,
        }
        data['verdicts'] = verdicts
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        for method, params in (('forcefield.remove', {'field': field}),
                               ('collider.remove', {'name': COLLIDER}),
                               ('scene.delete', {'name': cube})):
            if params.get('field', 1) and params.get('name', 1):
                try:
                    call(method, **params)
                except Exception:  # noqa: BLE001 - best effort cleanup
                    pass
        try:
            if created:
                call('fluid.remove_domain', domain=DOMAIN)
        except Exception:  # noqa: BLE001
            pass
        for f in original['fields']:
            call('forcefield.set_param', field=f['name'], enabled=f['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'], visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        call('timeline.set_frame', frame=0)
        (ROOT / 'docs/dev/contact_heat_acceptance_live.json').write_text(
            json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
