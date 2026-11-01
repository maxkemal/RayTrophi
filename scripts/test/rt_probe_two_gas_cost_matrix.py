"""Reversible paused two-gas visibility matrix through the external IPC client."""
import argparse
import copy
import json
from pathlib import Path

from rt_ipc import RtIpc
from rt_probe_volume_split_cost import capture, context


def stable_context(value):
    value = copy.deepcopy(value)
    for material in value['materials'].values():
        material.pop('visible', None)
    return value


def run(client, targets, save, initial_both=None):
    if client.call('sim.control_state')['playing']:
        raise RuntimeError('Timeline must already be paused')
    before = context(client, targets)
    original = {target: before['materials'][target]['visible'] for target in targets}
    report = dict(targets=targets, baseline=before, original_visibility=original,
                  cases=[], complete=False, visibility_restored=False)
    reference = stable_context(before)
    cases = [('A_and_B', [True, True]), ('gas_hidden', [False, False]),
             ('A_only', [True, False]), ('B_only', [False, True]),
             ('A_and_B_repeat', [True, True])]
    try:
        for name, visible in cases:
            print('Starting case:', name, flush=True)
            for target, enabled in zip(targets, visible):
                current = client.call('volume.get_material', target=target)['visible']
                if current != enabled:
                    client.call('volume.set_material', target=target, settings={'visible': enabled})
            if stable_context(context(client, targets)) != reference:
                raise RuntimeError('Non-visibility context changed before ' + name)
            if name == 'A_and_B' and initial_both is not None:
                if initial_both.get('complete') is not True or initial_both['context'] != before:
                    raise RuntimeError('Initial both-visible capture does not match current context')
                row = copy.deepcopy(initial_both)
            else:
                row = capture(client, targets)
            if stable_context(row['context']) != reference:
                raise RuntimeError('Non-visibility context changed during ' + name)
            row.update(case=name, visible=visible)
            report['cases'].append(row)
            save(report)
            print(json.dumps(dict(case=name, medians=[r['gpu_median_ms']
                for r in row['timing_cases']])), flush=True)
        report['complete'] = True
        return report
    finally:
        try:
            for target, visible in original.items():
                current = client.call('volume.get_material', target=target)['visible']
                if current != visible:
                    client.call('volume.set_material', target=target, settings={'visible': visible})
            after = context(client, targets)
            report['visibility_restored'] = after['materials'] == before['materials']
            report['scene_unchanged'] = after == before
        finally:
            save(report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-both', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    initial = json.loads(args.initial_both.read_text(encoding='utf-8'))
    save = lambda report: args.output.write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    client = RtIpc()
    try:
        run(client, initial['targets'], save, initial)
    finally:
        client.close()
    print('Report:', args.output)


if __name__ == '__main__':
    main()
