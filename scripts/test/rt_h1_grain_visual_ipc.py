"""Replay actual C++ reference grain trajectories on the existing H1 preview.

External process only. Does not use timeline Play, embedded scripts, Python
physics, scene clearing or saving. Leaves the initial pose ready for another run.
"""
import argparse
import json
import time
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loops', type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.loops <= 20:
        parser.error('--loops must be 1..20')
    directory = Path(__file__).resolve().parents[2] / 'docs/dev'
    baseline = json.loads((directory / 'matter_h1_grain_reference_live.json')
                          .read_text(encoding='utf-8'))
    assert baseline['all_gates_passed'], 'Run H1 acceptance first'
    names = baseline['preview_objects']
    assert len(names) == 27, 'Run acceptance with --preview first'
    bodies = [dict(id=1+x+3*z+9*y,
                   position=[(x-1)*.105+y*.007, .4+y*.105, (z-1)*.105],
                   velocity=[(x-1)*.2, 0, (z-1)*.2],
                   radius_m=.05, mass_kg=.2, saturation=.5 if x == 2 else 0)
              for y in range(3) for z in range(3) for x in range(3)]
    config = dict(bodies=bodies, duration_s=1.2, maximum_dt_s=.0001,
                  sample_interval_s=.05, contact=dict(normal_stiffness_n_m=40000,
                  tangential_stiffness_n_m=10000, normal_damping_n_s_m=4,
                  tangential_damping_n_s_m=1, rolling_friction=.02))
    client = RtIpc()
    data = {'config': config, 'loops_requested': args.loops, 'loops_completed': 0}

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and ('__error' in result or result.get('ok') is False):
            raise RuntimeError(result)
        return result

    def pose(frame):
        calls = []
        for name, grain in zip(names, frame['bodies']):
            position = grain['position'][:]
            position[0] += 6
            calls.append(dict(method='scene.set_transform',
                              params=dict(name=name, translation=position)))
        result = call('batch', calls=calls)
        if len(result) != 27 or any('error' in item for item in result):
            raise RuntimeError(result)

    frames = None
    try:
        data['control_before'] = call('sim.control_state')
        assert not data['control_before']['playing'], 'Pause the timeline first'
        result = call('fluid.grain_reference', **config)
        data['reference'] = result
        frames = result['frames']
        assert result['backend'] == 'cpu_grain_reference' and not result['scene_mutated']
        assert all(f['mass_kg'] == frames[0]['mass_kg'] for f in frames)
        assert result['maximum_overlap_ratio'] < .2, 'Visual fixture too soft/deeply overlapping'
        call('camera.set_position', position=[6.8, .8, 1.4])
        call('camera.set_target', target=[6, .28, 0])
        for loop in range(args.loops):
            pose(frames[0])
            time.sleep(1)
            started = time.monotonic()
            for frame in frames[1:]:
                remaining = frame['seconds'] - (time.monotonic() - started)
                if remaining > 0:
                    time.sleep(remaining)
                pose(frame)
            time.sleep(1)
            data['loops_completed'] += 1
            print('visual replay', loop+1, '/', args.loops, flush=True)
        data['control_after'] = call('sim.control_state')
        assert data['control_after'] == data['control_before'], 'Simulation control changed'
    except Exception as error:
        data['error'] = str(error)
        raise
    finally:
        if frames is not None:
            try:
                pose(frames[0])
                data['left_initial_pose'] = True
            except Exception as error:
                data['restore_error'] = str(error)
        (directory / 'matter_h1_grain_visual_live.json').write_text(
            json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
