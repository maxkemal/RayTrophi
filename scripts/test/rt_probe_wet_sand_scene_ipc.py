"""External IPC diagnostic of the user's paused, empty wet Sand seed scene.

No build or application launch. Each invocation runs one controlled arm.
Derived-material arms restore the original default material and delete their
temporary material, leaving the domain empty and paused. Baseline preserves
its measured live end state. Raw data is written after every sample.
"""
import argparse
import json
import time
from pathlib import Path

from rt_ipc import RtIpc
from rt_domain_material import remove_material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm', choices=['baseline', 'dry', 'low_rebound',
                                         'fine_radius', 'fine_low_rebound'],
                        default='baseline')
    parser.add_argument('--steps', type=int, default=96)
    parser.add_argument('--dt', type=float, default=1 / 60)
    parser.add_argument('--sample-every', type=int, default=8)
    parser.add_argument('--tag', default='')
    parser.add_argument('--contact-resolution', type=int)
    parser.add_argument('--reset', action='store_true',
                        help='Reset live simulation before this comparison arm')
    args = parser.parse_args()
    if args.dt <= 0 or args.sample_every < 1 or args.steps < 1:
        parser.error('dt, sample-every and steps must be positive')
    client = RtIpc()
    root = Path(__file__).resolve().parents[2]
    label = args.tag or args.arm
    if not all(c.isalnum() or c in '_-' for c in label):
        parser.error('tag must contain only letters, numbers, underscores or hyphens')
    output = root / ('docs/dev/wet_sand_scene_' + label + '_20261023.json')
    data = {'arm': args.arm, 'dt': args.dt, 'samples': []}
    temporary = 'Diagnostic Wet Sand ' + args.arm
    changed = False
    contact_changed = False

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and result.get('ok') is False:
            raise RuntimeError((method, result))
        return result

    def save():
        output.write_text(json.dumps(data, indent=2), encoding='utf-8')

    try:
        data['control_before'] = call('sim.control_state')
        assert not data['control_before']['playing'], 'Pause the scene first'
        if args.reset:
            call('fluid.reset')
        domains = call('fluid.list_domains')['domains']
        active = [d for d in domains if d['enabled']]
        assert len(active) == 1, 'Requires exactly one enabled domain'
        original = active[0]
        name = original['name']
        assert original['particle_count'] == 0, 'Reset existing simulation first'
        assert original['default_substance'] == 'Sand', 'Requires original Sand'
        data['domain_before'] = original
        data['sources'] = call('flow_source.list')
        assert not any(s['enabled'] for s in data['sources']), 'Active source found'
        data['colliders'] = call('collider.list')
        data['material_before'] = call('substance.get', name='Sand')
        data['models_before'] = call('fluid.matter_models', domain=name)
        save()
        if args.contact_resolution is not None:
            call('fluid.set_grain_settings', domain=name,
                 contact_resolution=args.contact_resolution)
            contact_changed = True
        if args.arm not in ('baseline', 'dry'):
            call('substance.derive', name=temporary, based_on='Sand')
            fields = {}
            if args.arm in ('low_rebound', 'fine_low_rebound'):
                fields['grain_restitution'] = 0.1
            if args.arm in ('fine_radius', 'fine_low_rebound'):
                fields['grain_real_radius_m'] = 0.0001
            call('substance.set', name=temporary, fields=fields)
            call('fluid.set_param', domain=name, default_substance=temporary)
            changed = True
        saturation = 0.0 if args.arm == 'dry' else original['seed_grain_saturation']
        data['seed_result'] = call(
            'fluid.seed', domain=name, seed_min=original['seed_min'],
            seed_max=original['seed_max'],
            particles_per_cell=original['seed_particles_per_cell'],
            grain_saturation=saturation, replace=True)
        for step in range(args.steps + 1):
            elapsed = None
            if step:
                start = time.perf_counter()
                result = call('fluid.step', dt=data['dt'])
                elapsed = time.perf_counter() - start
            if step % args.sample_every and step != args.steps:
                continue
            models = call('fluid.matter_models', domain=name)
            row = next(d for d in call('fluid.list_domains')['domains'] if d['name'] == name)
            sample = {'step': step, 'seconds': step * data['dt'],
                      'host_seconds': elapsed, 'models': models,
                      'digest': call('fluid.state_digest', domain=name), 'domain': row}
            data['samples'].append(sample)
            save()
            grain = models['grain_diagnostics']
            granular = models['acceptance_metrics'].get('granular', {})
            print(json.dumps({'arm': args.arm, 'step': step,
                              'kinetic_energy_j': granular.get('kinetic_energy_j'),
                              'bridges': grain['liquid']['wet']['liquid_bridges_last_substep'],
                              'host_seconds': elapsed}), flush=True)
            if models['mixed_execution']['step_held']:
                raise RuntimeError(models['mixed_execution'])
    finally:
        if contact_changed:
            call('fluid.set_grain_settings', domain=name,
                 contact_resolution=data['models_before']['grain_settings']['contact_resolution'])
        if changed:
            call('fluid.clear', domain=name)
            call('fluid.set_param', domain=name,
                 default_substance=original['default_substance'])
            remove_material(call, temporary)
        if args.arm == 'dry':
            call('fluid.set_param', domain=name,
                 seed_grain_saturation=original['seed_grain_saturation'])
        data['control_after'] = call('sim.control_state')
        save()
        client.close()


if __name__ == '__main__':
    main()
