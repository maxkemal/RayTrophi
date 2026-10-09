"""Revision-24 weak/strong impact A/B on a small settled DEM bed.

External IPC, empty paused scene, no colliders or fields. Does not start the app.
All sources are scheduled before physics; introducing a projectile does not edit
the live scene, reset the bed, or deliberately wake it. Run after the user build.
"""
import json
import math
import uuid
from pathlib import Path

from rt_ipc import RtIpc
from rt_grain_material import install_grain_material

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/dev/grain_sleep_transfer_live.json'
BED = 27
RADIUS = .025


def main():
    client = RtIpc()
    install_grain_material(client)  # old grain keys -> grain substance
    suffix = uuid.uuid4().hex[:8]
    domain = 'SleepTransfer_' + suffix
    weak = 'SleepTransferWeak_' + suffix
    sources = []
    domain_created = material_created = False
    results = {}

    def save():
        OUT.write_text(json.dumps(results, indent=2), encoding='utf-8')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    try:
        control = client.call('sim.control_state')
        assert not control['playing'], 'Pause the timeline'
        assert not client.call('fluid.list_domains')['domains'], 'Use an empty scene'
        assert not any(c.get('enabled', True) for c in
                       client.call('collider.list')['colliders']), 'Disable colliders'
        assert not any(f.get('enabled', True) for f in
                       client.call('forcefield.list')), 'Disable force fields'
        client.call('substance.derive', name=weak, based_on='Sand')
        material_created = True
        density = client.call('substance.get', name='Sand')['fields']['density']
        client.call('substance.set', name=weak, fields={'density': density * .001})
        client.call('fluid.create_domain', name=domain, type='matter',
                    domain_min=[-.3, 0, -.3], domain_max=[.4, .4, .4], voxel_size=.05)
        domain_created = True
        client.call('fluid.set_param', domain=domain, backend='vulkan', visible=False,
                    boundary='closed', default_substance='Sand', solid_phase=False,
                    thermal_liquid_enabled=False)
        client.call('fluid.set_pore_exchange', domain=domain, enabled=False,
                    wet_response_enabled=False, wet_appearance_enabled=False)

        def make_source(position, material, velocity, start):
            name = 'SleepTransferSource_' + suffix + '_' + str(len(sources))
            client.call('flow_source.create', name=name, domain=domain, enabled=True,
                        phase='liquid', source_mode='point', fluid_substance=material,
                        position=position, radius=.000001, velocity=velocity,
                        fluid_velocity_spread=0., use_particle_limit=True,
                        max_emitted_particles=1, fluid_particles_per_second=600.,
                        use_time_limit=True, start_time=start, end_time=start+.1,
                        fluid_temperature_override=True, fluid_temperature_kelvin=293.15)
            sources.append(name)

        def sample():
            inv = client.call('fluid.matter_models', domain=domain)
            assert not inv['mixed_execution']['step_held'], inv['mixed_execution']
            diag = inv['grain_diagnostics']
            assert diag['contact_shader_revision'] >= 24, 'Build revision-24 C++ and shaders'
            g, r = inv['acceptance_metrics']['granular'], diag['runtime']
            assert r['dispatches'] == 4*r['substeps'], r
            assert client.call('sim.control_state') == control
            return {'particles': g['particles'], 'mass_kg': g['dry_mass_kg'],
                    'com': g['dry_center_of_mass'], 'energy_j': g['kinetic_energy_j'],
                    'sleeping': r['sleeping_grains'], 'contacts': r['contacts_last_substep'],
                    'substeps': r['substeps'], 'gpu_wait_ms': r['host_ms']['gpu_wait'],
                    'history_reset': r.get('history_reset'),
                    'history_reset_reason': r.get('history_reset_reason')}

        for strength, material, speed in [('weak', weak, .1), ('strong', 'Sand', 1.)]:
            for sleep in (False, True):
                for name in sources:
                    client.call('flow_source.remove', name=name)
                sources.clear()
                client.call('timeline.set_frame', frame=0)
                client.call('fluid.reset')
                client.call('fluid.set_grain_settings', domain=domain, enabled=True,
                            radius_m=RADIUS, stiffness_n_m=2000., restitution=.1,
                            sliding_damping_n_s_m=0., friction=.5, rolling_friction=.3,
                            twisting_friction=0., tangential_stiffness_ratio=2/7,
                            contact_resolution=24, packing_fraction=.6, max_substeps=4096,
                            sleep=sleep, sleep_speed_m_s=.002, sleep_time_s=.2)
                for y in range(3):
                    for x in range(3):
                        for z in range(3):
                            make_source([x*.0504, RADIUS+y*.0504, z*.0504],
                                        'Sand', [0, 0, 0], 0.)
                make_source([-.0503, RADIUS, .0504], material, [speed, 0, 0], 1.1)
                client.call('fluid.reset')
                control = client.call('sim.control_state')
                for step in range(60):
                    client.call('fluid.step', dt=1/60)
                    if step == 0:
                        sample()  # Reject an old binary before the warm-up.
                before = sample()
                key = strength + ('_on' if sleep else '_off')
                samples = []
                results[key] = {'before': before, 'samples': samples}
                save()
                assert before['particles'] == BED, before
                if sleep:
                    assert before['sleeping'] >= BED-4, ('bed did not settle', before)
                for _ in range(30):
                    client.call('fluid.step', dt=1/60)
                    samples.append(sample())
                    save()
                born = [s for s in samples if s['particles'] == BED+1]
                assert len(born) >= 10, ('projectile not born', samples)
                mass_added = born[-1]['mass_kg'] - before['mass_kg']
                mass_ratio = mass_added / (before['mass_kg']/BED)
                if strength == 'weak':
                    assert 0 < mass_ratio < .01, mass_ratio
                else:
                    assert abs(mass_ratio-1.) < .01, mass_ratio
                assert max(s['mass_kg'] for s in born) - min(s['mass_kg'] for s in born) < 1e-6
                assert all(s['contacts'] > 0 for s in born), born
                minimum_sleep = min(s['sleeping'] for s in born)
                if sleep and strength == 'weak':
                    assert minimum_sleep >= BED-4, ('weak impact woke the bed', minimum_sleep)
                if sleep and strength == 'strong':
                    assert minimum_sleep < BED, ('strong impact did not wake a target', born)
                if not sleep:
                    assert all(s['sleeping'] == 0 for s in samples), samples
                results[key].update(mass_ratio=mass_ratio, minimum_sleep=minimum_sleep)
                save()
                print('PASS', key, 'mass ratio', mass_ratio, 'minimum sleep', minimum_sleep,
                      flush=True)
            on, off = results[strength+'_on'], results[strength+'_off']
            # Same timed pour, tiny birth jitter: compare aggregate trajectory,
            # not identities or a selected best final image. No gate relaxation.
            for a, b in zip(on['samples'], off['samples']):
                assert a['particles'] == b['particles'], (a, b)
                assert math.dist(a['com'], b['com']) <= .001, (strength, a, b)
                energy_tolerance = .00001 + .25*max(a['energy_j'], b['energy_j'])
                assert abs(a['energy_j']-b['energy_j']) <= energy_tolerance, (strength, a, b)
        print('PASS revision-24 transfer: weak impact preserves sleeping core, strong '
              'impact wakes targets, A/B population/mass/COM/energy gates')
    finally:
        try:
            for name in sources:
                client.call('flow_source.remove', name=name)
            if domain_created:
                client.call('fluid.remove_domain', domain=domain)
            if material_created:
                client.call('substance.remove', name=weak)
        finally:
            client.close()


if __name__ == '__main__':
    main()
