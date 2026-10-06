"""Exercise the application's C++ grain reference through external IPC.

Default is read-only. --preview adds final pile spheres beside the scene; it
never clears, saves, plays or resets the user's simulation. No Python physics.
"""
import argparse
import json
import math
from pathlib import Path

from rt_ipc import RtIpc


def body(identity, position, velocity=(0, 0, 0), saturation=0):
    return dict(id=identity, position=list(position), velocity=list(velocity),
                radius_m=.05, mass_kg=.2, saturation=saturation)


def magnitude(vector):
    return math.sqrt(sum(x * x for x in vector))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', action='store_true')
    parser.add_argument('--log', type=Path, default=Path(__file__).resolve().parents[2] /
                        'docs/dev/matter_h1_grain_reference_live.json')
    args = parser.parse_args()
    client = RtIpc()
    data = {'completed': False, 'runs': {}, 'gates': {}, 'preview_objects': []}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError(result)
        return result

    def run(name, **config):
        config.setdefault('maximum_dt_s', .0001)
        config.setdefault('sample_interval_s', .02)
        result = call('fluid.grain_reference', **config)
        assert result['backend'] == 'cpu_grain_reference'
        assert result['scene_mutated'] is False and result['production_dem_enabled'] is False
        frames = result['frames']
        baseline = frames[0]
        assert abs(frames[-1]['seconds'] - config.get('duration_s', 1)) < 1e-6
        ids = [b['id'] for b in baseline['bodies']]
        for frame in frames:
            assert [b['id'] for b in frame['bodies']] == ids
            assert frame['mass_kg'] == baseline['mass_kg']
            for current, original in zip(frame['bodies'], baseline['bodies']):
                assert current['saturation'] == original['saturation']
                assert current['mass_kg'] == original['mass_kg']
                for key in ('position', 'velocity', 'angular_velocity'):
                    assert all(math.isfinite(x) for x in current[key])
        data['runs'][name] = {'config': config, 'result': result}
        args.log.write_text(json.dumps(data, indent=2), encoding='utf-8')
        print(name, 'steps', result['micro_steps'], 'contacts', result['contact_evaluations'],
              'max overlap/r', result['maximum_overlap_ratio'], flush=True)
        return result

    def gate(name, passed):
        data['gates'][name] = bool(passed)

    try:
        data['control_before'] = call('sim.control_state')
        fall = run('free_fall', bodies=[body(1, (0, 1, 0))], plane=None, duration_s=.2)
        final = fall['frames'][-1]['bodies'][0]
        gate('gravity_velocity', abs(final['velocity'][1] + 9.81 * .2) < .001)
        gate('gravity_position', abs(final['position'][1] - (1 - .5 * 9.81 * .2**2)) < .001)
        gate('free_fall_no_spin', magnitude(final['angular_velocity']) < 1e-8)

        pair_config = dict(bodies=[body(1, (-.08, 0, 0), (1, .2, 0)),
                                   body(2, (.08, 0, 0), (-1, -.2, 0))],
                           plane=None, gravity=[0, 0, 0], duration_s=.2)
        pair = run('oblique_pair', **pair_config)
        first, last = pair['frames'][0], pair['frames'][-1]
        gate('pair_contacts', pair['contact_evaluations'] > 0)
        gate('pair_linear_momentum', magnitude(last['momentum_kg_m_s']) < 1e-5)
        gate('pair_angular_momentum', magnitude([x - y for x, y in zip(
            last['angular_momentum_kg_m2_s'], first['angular_momentum_kg_m2_s'])]) < 1e-4)
        gate('pair_dissipates', last['kinetic_energy_j'] <= first['kinetic_energy_j'] * 1.02)
        gate('pair_spins', any(magnitude(b['angular_velocity']) > .01 for b in last['bodies']))
        gate('pair_overlap', pair['maximum_overlap_ratio'] < .15)

        # k=10 kN/m admits ~15.5% radius compression for this drop/mass.
        # Resolve the fixture's <15% overlap target by authoring 20 kN/m,
        # verified against dt-half; do not loosen the gate or alter the core.
        bounce = run('plane_bounce', bodies=[body(1, (0, .2, 0))], duration_s=.5,
                     contact=dict(normal_stiffness_n_m=20000,
                                  tangential_stiffness_n_m=5000))
        gate('bounce', any(f['bodies'][0]['velocity'][1] > .1 for f in bounce['frames']))
        gate('bounce_overlap', bounce['maximum_overlap_ratio'] < .15)

        angle = math.radians(20)
        normal = [math.sin(angle), math.cos(angle), 0]
        incline = run('incline_roll', bodies=[body(1, [x * .05 for x in normal])],
                      plane=dict(normal=normal, offset_m=0), duration_s=.4,
                      contact=dict(rolling_friction=0))
        final = incline['frames'][-1]['bodies'][0]
        gate('incline_moves', final['position'][0] > .005)
        gate('incline_spins', magnitude(final['angular_velocity']) > .1)

        # Two separate spheres sharing one group, only the second is wet.
        slide = run('local_dry_wet', bodies=[body(1, (-1, .05, 0), (2, 0, 0)),
                    body(2, (1, .05, 0), (2, 0, 0), saturation=1)],
                    duration_s=.1, contact=dict(dry_friction=.6, saturated_friction=.05,
                                              rolling_friction=0))
        final = slide['frames'][-1]['bodies']
        gate('local_wet_friction', final[1]['velocity'][0] > final[0]['velocity'][0] + .05)
        gate('local_saturation_preserved', [b['saturation'] for b in final] == [0, 1])

        pile_bodies = [body(1 + x + 3*z + 9*y,
                           ((x - 1)*.105 + y*.007, .06 + y*.105, (z - 1)*.105),
                           saturation=.5 if x == 2 else 0)
                       for y in range(3) for z in range(3) for x in range(3)]
        pile = run('pile', bodies=pile_bodies, duration_s=.4)
        half = run('pile_dt_half', bodies=pile_bodies, duration_s=.4, maximum_dt_s=.00005)
        gate('pile_dt_actually_halved',
             half['maximum_micro_dt_s'] <= pile['maximum_micro_dt_s'] / 1.9)
        a, b = pile['frames'][-1]['bodies'], half['frames'][-1]['bodies']
        displacement = max(magnitude([x-y for x, y in zip(p['position'], q['position'])])
                           for p, q in zip(a, b))
        data['pile_dt_max_displacement_m'] = displacement
        gate('pile_dt_convergence_2mm', displacement < .002)
        gate('pile_overlap', max(pile['maximum_overlap_ratio'], half['maximum_overlap_ratio']) < .15)
        gate('pile_contacts', pile['contact_evaluations'] > 0)

        # Validate actual API rejection rather than merely validating this probe.
        invalid = [dict(bodies=[body(1, (0, 1, 0))], duration_s=True),
                   dict(bodies=[body(1, (0, 1, 0)), body(1, (.2, 1, 0))]),
                   dict(bodies=[body(1, (0, 1, 0), saturation=2)]),
                   dict(bodies=[dict(id=1, position=[0, 1, 0], mass_kg=.2)]),
                   dict(bodies=[body(1, (0, 1, 0))], unexpected=1)]
        rejected = []
        for config in invalid:
            try:
                call('fluid.grain_reference', **config)
                rejected.append(False)
            except Exception as error:
                rejected.append('grain_reference_' in str(error))
        gate('invalid_input_rejections', all(rejected))
        data['control_after'] = call('sim.control_state')
        gate('scene_control_unchanged', data['control_before'] == data['control_after'])
        data['completed'] = True
        data['all_gates_passed'] = all(data['gates'].values())
        print(json.dumps(data['gates'], indent=2), flush=True)
        if args.preview:
            if not data['all_gates_passed']:
                raise RuntimeError('Preview withheld: numerical gates failed')
            data['preview_snapshot_only'] = True
            materials = {wet: call('material.create', type='principled',
                                  name='H1_Reference_Wet' if wet else 'H1_Reference_Dry')
                         for wet in (False, True)}
            # New names only. Existing scene objects and physics remain untouched.
            for grain in pile['frames'][-1]['bodies']:
                name = call('scene.add_primitive', type='sphere',
                            name=f"H1_Reference_Grain_{grain['id']}", size=2*grain['radius_m'])
                data['preview_objects'].append(name)
                position = grain['position'][:]
                position[0] += 6
                call('scene.set_transform', name=name, translation=position)
                # Primitive defaults share material 0. Assign new owned materials
                # before coloring so the user's original materials cannot change.
                call('material.assign', object_name=name,
                     material_name=materials[bool(grain['saturation'])])
                call('material.set', object_name=name, param='base_color',
                     value=[.16, .10, .04] if grain['saturation'] else [.65, .48, .25])
            data['preview_camera_before'] = call('camera.get')
            ground_material = call('material.create', type='principled',
                                   name='H1_Reference_Ground')
            ground = call('scene.add_primitive', type='plane',
                          name='H1_Reference_Ground', size=1.2)
            call('material.assign', object_name=ground, material_name=ground_material)
            call('material.set', object_name=ground, param='base_color', value=[.17, .19, .22])
            call('scene.set_transform', name=ground, translation=[6, 0, 0])
            call('camera.set_position', position=[6.75, .65, 1.05])
            call('camera.set_target', target=[6, .13, 0])
            data['preview_ground_object'] = ground
            data['preview_camera_after'] = call('camera.get')
        if not data['all_gates_passed']:
            raise AssertionError('H1 reference gates failed; inspect log, do not claim production DEM')
    except Exception as error:
        data['error'] = str(error)
        raise
    finally:
        args.log.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
