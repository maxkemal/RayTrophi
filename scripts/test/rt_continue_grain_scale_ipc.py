"""Continue an existing scale fixture without resetting its material or state.

External IPC only. Use after rt_test_scale_ceilings_ipc.py when the poured pile
has not settled in the first measurement window. No scene authoring/builds.
The reference log supplies the original full-birth energy/mass bound.
"""

import argparse
import json
import time
from pathlib import Path

from rt_ipc import RtIpc
from rt_grain_material import install_grain_material


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-log', required=True)
    parser.add_argument('--start-step', type=int, required=True)
    parser.add_argument('--steps', type=int, default=90)
    parser.add_argument('--grains', type=int, default=1300000)
    parser.add_argument('--sleep', choices=('on', 'off'), required=True)
    args = parser.parse_args()
    if args.steps <= 0 or args.start_step < 0:
        parser.error('steps must be positive and start-step nonnegative')
    rows = [json.loads(line[5:]) for line in
            Path(args.reference_log).read_text(encoding='utf-8-sig').splitlines()
            if line.startswith('STEP ')]
    first = next(row for row in rows if row['grains'] >= args.grains * .99)
    client = RtIpc()
    install_grain_material(client)  # old grain keys -> grain substance
    try:
        assert not client.call('sim.control_state')['playing'], 'Pause the timeline'
        settings = client.call('fluid.grain_settings', domain='Lim_Grain')
        assert settings['sleep'] == (args.sleep == 'on'), settings
        for index in range(args.steps):
            start = time.perf_counter()
            client.call('fluid.step', dt=1/60)
            wall = time.perf_counter() - start
            inv = client.call('fluid.matter_models', domain='Lim_Grain')
            g = inv['acceptance_metrics']['granular']
            runtime = inv['grain_diagnostics']['runtime']
            row = {'step': args.start_step + index + 1, 'wall_s': round(wall, 2),
                   'grains': g['particles'], 'held': inv['mixed_execution']['step_held'],
                   'status': inv['mixed_execution']['status'],
                   'com_y': g['dry_center_of_mass'][1], 'ke_j': g['kinetic_energy_j'],
                   'mass_kg': g['dry_mass_kg'], 'substeps': runtime['substeps'],
                   'collider_faces': runtime['collider_faces'],
                   'host_ms': {k: round(v, 1) for k, v in runtime['host_ms'].items()},
                   'working_set_mb': runtime['working_set_bytes'] // (1024 * 1024),
                   'resident': runtime['state_resident'], 'cell_order': runtime['cell_order'],
                   'sleeping': runtime['sleeping_grains'],
                   'bounds': [g['bounds_min'], g['bounds_max']]}
            print('STEP ' + json.dumps(row), flush=True)
            assert not row['held'], row
            assert row['grains'] == args.grains, row
            assert abs(row['mass_kg'] - first['mass_kg']) <= first['mass_kg'] * 1e-6, row
            bound = first['ke_j'] + 1.1 * row['mass_kg'] * 9.81 * \
                max(first['com_y'] - row['com_y'], 0) + 1
            assert row['ke_j'] <= bound, (row, bound)
            assert all(-2.05 <= value <= 4.05 for corner in row['bounds'] for value in corner)
        print('RESULT PASS continuation: population, mass, energy bound and domain containment')
    finally:
        client.close()


if __name__ == '__main__':
    main()
