"""Revision-24 DEM sleep acceptance, external named-pipe process only.

Requires an empty paused scene, downward gravity and no active colliders/fields.
Tests physical free fall at deliberately loose sleep settings, and settled
contact diagnostics with sleep on/off. For wet bridges, support motion and
large-N performance also run the existing wet/motion/scale/repose gates.
Creates/removes only its own domain and source; never starts the application.
"""

import json
import uuid

from rt_ipc import RtIpc
from rt_grain_material import install_grain_material


def main():
    client = RtIpc()
    install_grain_material(client)  # old grain keys -> grain substance
    suffix = uuid.uuid4().hex[:8]
    domain = 'SleepAcceptance_' + suffix
    source = 'SleepAcceptanceSource_' + suffix
    domain_created = source_created = False
    results = {}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    def sample():
        inventory = call('fluid.matter_models', domain=domain)
        assert not inventory['mixed_execution']['step_held'], inventory['mixed_execution']
        diagnostic = inventory['grain_diagnostics']
        assert diagnostic['contact_shader_revision'] >= 24, 'Rebuild revision-24 C++ and grain shaders'
        grain = inventory['acceptance_metrics']['granular']
        runtime = diagnostic['runtime']
        assert grain['particles'] == 1, grain
        assert runtime['dispatches'] == 4 * runtime['substeps'], runtime
        return {'shader_revision': diagnostic['contact_shader_revision'],
                'position': grain['dry_center_of_mass'], 'mass_kg': grain['dry_mass_kg'],
                'energy_j': grain['kinetic_energy_j'], 'runtime': runtime}

    try:
        control = call('sim.control_state')
        assert not control['playing'], 'Pause the timeline'
        assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
        assert not any(c.get('enabled', True) for c in call('collider.list')['colliders']), \
            'Disable existing colliders'
        assert not any(f.get('enabled', True) for f in call('forcefield.list')), \
            'Disable existing force fields'
        call('fluid.create_domain', name=domain, type='matter',
             domain_min=[-.5, 0, -.5], domain_max=[.5, 4, .5], voxel_size=.1)
        domain_created = True
        call('fluid.set_param', domain=domain, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Sand', solid_phase=False,
             thermal_liquid_enabled=False)
        call('fluid.set_pore_exchange', domain=domain, enabled=False,
             wet_response_enabled=False, wet_appearance_enabled=False)
        call('flow_source.create', name=source, domain=domain, enabled=False,
             phase='liquid', source_mode='point', fluid_substance='Sand',
             position=[0, 2, 0], radius=.0001, velocity=[0, 0, 0],
             fluid_velocity_spread=0., use_particle_limit=True, max_emitted_particles=1,
             use_time_limit=False, fluid_particles_per_second=200.,
             fluid_temperature_override=True, fluid_temperature_kelvin=293.15)
        source_created = True

        def run_arm(name, sleep, height, frames, speed, duration):
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('fluid.set_grain_settings', domain=domain, radius_m=.025,
                 stiffness_n_m=20000., restitution=.1, sliding_damping_n_s_m=4.,
                 friction=.5, rolling_friction=.1, twisting_friction=0.,
                 tangential_stiffness_ratio=2/7, contact_resolution=24,
                 packing_fraction=.6, max_substeps=1024,
                 sleep=sleep, sleep_speed_m_s=speed, sleep_time_s=duration)
            call('flow_source.update', name=source, enabled=True, position=[0, height, 0])
            samples = []
            frame_control = call('sim.control_state')
            for frame in range(frames):
                call('fluid.step', dt=1/60)
                assert call('sim.control_state') == frame_control
                if frame == 0 or frame == frames - 1:
                    samples.append(sample())
            results[name] = samples
            assert abs(samples[-1]['mass_kg'] - samples[0]['mass_kg']) < 1e-7
            return samples

        # Speed alone permits sleep during the first .01 s of this fall; the
        # force-balance gate must reject it, independent of a loose speed knob.
        falling = {}
        for sleep in (False, True):
            samples = run_arm('fall_' + str(sleep), sleep, 2., 13, 1., .01)
            drop = samples[0]['position'][1] - samples[-1]['position'][1]
            assert drop > .1, ('unsupported grain froze', sleep, drop)
            assert samples[-1]['runtime']['sleeping_grains'] == 0, samples[-1]
            falling[sleep] = drop
        # Compare displacement: the source's tiny birth jitter is not a sleep effect.
        assert abs(falling[True] - falling[False]) <= 2e-5, falling
        print('PASS sleep free fall: loose settings cannot freeze an unsupported grain')

        settled = {}
        for sleep in (False, True):
            samples = run_arm('settle_' + str(sleep), sleep, .025, 120, .002, .2)
            last = samples[-1]
            runtime = last['runtime']
            assert runtime['contacts_last_substep'] >= 1, last
            assert runtime['sticking_contacts_last_substep'] >= 1, last
            assert runtime['max_contacts_per_grain'] >= 1, last
            assert runtime['cost_last_substep']['contactless_grains'] == 0, last
            assert runtime['sleeping_grains'] == (1 if sleep else 0), last
            assert .025 * .7 <= last['position'][1] <= .025 * 1.1, last
            settled[sleep] = last
        assert abs(settled[True]['position'][1] - settled[False]['position'][1]) <= 2e-5
        print('PASS sleeping support: contact/sticking/CFL diagnostics remain physical')
        print(json.dumps(results))
        print('PASS grain sleep acceptance; wet/motion/repose/large-N gates still required')
    finally:
        try:
            if source_created:
                call('flow_source.remove', name=source)
            if domain_created:
                call('fluid.remove_domain', domain=domain)
        finally:
            client.close()


if __name__ == '__main__':
    main()
