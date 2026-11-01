"""R1 + R3 probe: automatic landmarks and geodesic voxel weights.

    python scripts/test/rt_probe_rig_auto_ipc.py [character] [fit_mesh] [--apply]

Without --apply nothing in the scene changes: it prints the proposed landmarks' confidence and
a geodesic weight preview. With --apply it measures the range-of-motion test, applies geodesic
weights (one undo step), measures again and prints both, so the weight change is judged by the
same instrument. Undo in the app to return to the previous weights.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc  # noqa: E402

args = [a for a in sys.argv[1:] if not a.startswith('--')]
APPLY = '--apply' in sys.argv
CHARACTER = args[0] if args else 'Rig'


def rom_line(report):
    s = report['summary']
    return 'min %.3f | pinched %d | stretched %d | folded %d' % (
        s['min_area_ratio'], s['pinched_motions'], s['stretched_motions'], s['folded_motions'])


def main():
    ipc = RtIpc()
    checks = []
    mesh = args[1] if len(args) > 1 else None
    if mesh is None:
        targets = ipc.call('rig.list_fit_targets')
        rows = targets.get('targets', targets) if isinstance(targets, dict) else targets
        mesh = rows[0]['name'] if rows and isinstance(rows[0], dict) else (rows[0] if rows else None)
    if mesh:
        ok, auto = ipc.try_call('rig.auto_landmarks', character=CHARACTER, mesh=mesh)
        if ok:
            print('auto_landmarks:', auto['method'], 'forward', auto['forward_check'],
                  'mean %.2f' % auto['mean_confidence'], 'low', auto['low_confidence'])
            for warning in auto['warnings']:
                print('  warning:', warning)
            ok_preview, fit = ipc.try_call('rig.preview_fit', character=CHARACTER, mesh=mesh,
                                           landmarks=auto['landmarks'], axes_confirmed=True)
            print('preview_fit:', fit if not ok_preview else
                  'outside_bounds %d can_commit %s' % (fit['outside_bounds'], fit['can_commit']))
            checks.append(('auto landmarks accepted by preview_fit', ok_preview and fit['can_commit']))
            checks.append(('mesh faces +Z', auto['forward_check'] != '-Z'))
        else:
            print('auto_landmarks refused:', auto, '(an already skinned target is refused by design)')
    else:
        print('no fit target in the scene; skipping auto landmarks')

    ok, geo = ipc.try_call('rig.preview_geodesic_weights', character=CHARACTER)
    if ok:
        print('geodesic: voxel %.4g grid %s interior %d closed %s fallback %d bones_outside %d' % (
            geo['voxel_size'], geo['grid'], geo['interior_voxels'], geo['closed_volume'],
            geo['fallback_vertices'], geo['bones_outside_mesh']))
        checks.append(('geodesic preview ran', True))
        checks.append(('no unweighted fallback vertices', geo['fallback_vertices'] == 0))
        if APPLY:
            before = ipc.call('rig.preview_rom_test', character=CHARACTER)
            ipc.call('rig.apply_geodesic_weights', character=CHARACTER,
                     rig_revision=geo['rig_revision'])
            after = ipc.call('rig.preview_rom_test', character=CHARACTER)
            binding = ipc.call('rig.get_binding', character=CHARACTER)
            print('ROM before:', rom_line(before))
            print('ROM after: ', rom_line(after))
            checks.append(('binding records geodesic_voxel_v1',
                           binding.get('algorithm') == 'geodesic_voxel_v1'))
    else:
        print('geodesic preview refused:', geo)
        checks.append(('geodesic preview ran', False))

    failed = 0
    for name, passed in checks:
        failed += 0 if passed else 1
        print(('PASS ' if passed else 'FAIL ') + name)
    print('%d/%d PASS' % (len(checks) - failed, len(checks)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
