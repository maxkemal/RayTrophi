"""Range-of-motion probe: deformation report for a bound rig, plus the matching clip.

    python scripts/test/rt_probe_rig_rom_ipc.py [character] [clip]

The report is the instrument; the thresholds below only catch a broken pipeline (unknown skin
indices, nothing measured, total collapse). Judge weight quality from the worst-motion table:
a min area ratio below .5 is a pinch, folded triangles are skin turning inside out.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc  # noqa: E402

CHARACTER = sys.argv[1] if len(sys.argv) > 1 else 'Rig'
CLIP = sys.argv[2] if len(sys.argv) > 2 else 'ROM_Test'


def main():
    ipc = RtIpc()
    report = ipc.call('rig.preview_rom_test', character=CHARACTER)
    summary = report['summary']
    print('motions %d | meshes %d | vertices %d | triangles %d' % (
        report['motions'], report['meshes'], report['vertices'], report['triangles']))
    print('summary:', summary)
    print('%-28s %-22s %6s %6s %6s  %s' % ('motion', 'bone', 'min', 'max', 'folds', 'worst bone'))
    for row in report['worst']:
        print('%-28s %-22s %6.2f %6.2f %6d  %s' % (
            row['motion'], row['bone'][-22:], row['min_area_ratio'], row['max_area_ratio'],
            row['folded_triangles'], row['worst']['bone']))
    state = ipc.call('rig.get_pose_state', character=CHARACTER)
    created = ipc.try_call('rig.create_rom_clip', character=CHARACTER, name=CLIP,
                           rig_revision=report['rig_revision'], fps=float(state.get('fps', 24)))
    print('clip:', created)
    checks = [
        ('every motion measured', report['motions'] > 0 and len(report['rows']) == report['motions']),
        ('triangles measured', report['triangles'] > 0),
        ('skin rows reference only this rig', summary['unknown_influences'] == 0),
        ('no total collapse (min area ratio > .05)', summary['min_area_ratio'] > .05),
    ]
    failed = 0
    for name, ok in checks:
        failed += 0 if ok else 1
        print(('PASS ' if ok else 'FAIL ') + name)
    print('%d/%d PASS' % (len(checks) - failed, len(checks)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
