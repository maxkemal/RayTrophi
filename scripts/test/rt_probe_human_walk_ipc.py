"""Human Walk v2 probe: recipe metrics plus one sampled cycle of the left leg.

Run from outside the app while a humanoid rig is in Pose mode:
    python scripts/test/rt_probe_human_walk_ipc.py [character] [clip]

It generates (or replaces) the clip with default recipe values, then reads the posed bones
frame by frame. The stance knee column is the one that matters: a constant value means the
pelvis solve is pinning the supporting leg at one length, which reads as a stiff, bent stick.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc  # noqa: E402

CHARACTER = sys.argv[1] if len(sys.argv) > 1 else 'Rig'
CLIP = sys.argv[2] if len(sys.argv) > 2 else 'Walk_Loop'
RECIPE = dict(cadence=100, cycles=2, stride=.35, step_height=.06, body_bounce=float(os.environ.get('RT_WALK_BOUNCE', 0)),
              arm_swing=.7, leg_reach=float(os.environ.get('RT_WALK_REACH', .998)))


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def length(a):
    return math.sqrt(sum(x * x for x in a))


def flexion(a, mid, b):
    u, v = sub(a, mid), sub(b, mid)
    cosine = sum(u[i] * v[i] for i in range(3)) / (length(u) * length(v))
    return 180 - math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def main():
    ipc = RtIpc()
    state = ipc.call('rig.get_pose_state', character=CHARACTER)
    fps = float(state.get('fps', 24))
    # Anatomy v8: fill missing bend hints / foot pivots so the walk reads stored data.
    print('derive_limb_hints:', ipc.call('rig.derive_limb_hints', character=CHARACTER))
    plan = ipc.call('rig.preview_human_walk', character=CHARACTER, fps=fps, **RECIPE)
    print('pivots:', plan.get('foot_pivots_source'), 'differ:', plan.get('foot_pivots_differ_from_toes'),
          'knee_pole:', plan.get('knee_pole'))
    height = plan['character_height']
    metrics = plan['metrics']
    print('metrics:', metrics)
    ipc.call('rig.create_human_walk_clip', character=CHARACTER, name=CLIP, fps=fps,
             rig_revision=plan['rig_revision'], **RECIPE)
    cycle = plan['frames'] // RECIPE['cycles']
    prefix = CHARACTER + '_'
    rows = []
    print(' f  pelvisY  knee  ankleY  tipY')
    for frame in range(0, cycle + 1):
        ipc.call('rig.set_pose_frame', character=CHARACTER, frame=frame)
        bones = {b['name']: b['world_position'] for b in ipc.call('rig.list_bones', character=CHARACTER)}
        hip, knee, ankle = bones[prefix + 'LeftThigh'], bones[prefix + 'LeftShin'], bones[prefix + 'LeftFoot']
        tip = bones[prefix + 'LeftToeEnd']
        row = (frame, bones[prefix + 'Pelvis'][1], flexion(hip, knee, ankle), ankle[1], tip[1])
        rows.append(row)
        print('%2d  %6.3f  %5.1f  %6.3f  %6.3f' % row)

    # Loading window: contact until the heel starts to rise (60% of the 62% stance). Pre-swing
    # frames bend the knee anyway and would make a pinned leg look like it varies.
    loading = rows[:int(cycle * .62 * .6) + 1]
    swing = rows[int(cycle * .62):]
    stance_knees = [r[2] for r in loading]
    checks = [
        ('foot target error <= .005 h', metrics['foot_target_error_fraction'] <= .005),
        ('ground penetration <= .002 h', metrics['ground_penetration_max'] <= .002 * height),
        ('no unreachable frames', metrics['unreachable_frames'] == 0),
        ('pivots read from anatomy and match the toes',
         plan.get('foot_pivots_source') == 'anatomy' and not plan.get('foot_pivots_differ_from_toes')),
        ('knee pole from anatomy bend hint', plan.get('knee_pole') == 'anatomy_bend_hint'),
        ('loading knee varies >= 8 deg before heel rise (not pinned to leg_reach)',
         max(stance_knees) - min(stance_knees) >= 8),
        ('heel-contact knee <= 12 deg', rows[0][2] <= 12),
        ('swing knee peak 50..75 deg', 50 <= max(r[2] for r in swing) <= 75),
        # A pelvis that drops several centimetres in one frame at contact reads as a thud.
        ('pelvis step <= .006 h per frame',
         max(abs(rows[i + 1][1] - rows[i][1]) for i in range(len(rows) - 1)) <= .006 * height),
    ]
    failed = 0
    for name, ok in checks:
        failed += 0 if ok else 1
        print(('PASS ' if ok else 'FAIL ') + name)
    print('%d/%d PASS' % (len(checks) - failed, len(checks)))
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
