"""Run every H1 grain acceptance arm in order and print one short summary.

The arms are the ones listed at the top of docs/dev/NEXT_BUILD_CHECKS.md. Each
runs as its own process against the running app (same as by hand). Full output
and each arm's JSON log go to docs/dev/grain_suite/ (the arms share one JSON
path and would overwrite each other); the console gets one line per arm plus
the lines that carry the numbers to report. Bring back only the summary.

    python scripts/test/rt_h1_grain_suite.py            # everything
    python scripts/test/rt_h1_grain_suite.py --quick    # skips dense, repose, g2
    python scripts/test/rt_h1_grain_suite.py --only settle coexist
"""
import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST = Path(__file__).resolve().parent
OUT = ROOT / 'docs/dev/grain_suite'
RUNTIME_LOG = ROOT / 'docs/dev/matter_h1_grain_runtime_live.json'
DENSE_LOG = ROOT / 'docs/dev/matter_h1_dense_grain_scene_live.json'

# name, script, args, JSON log the arm writes, slow
ARMS = [
    ('contracts', 'check_matter_grain_contracts.py', [], None, False),
    ('panel_fields', 'check_domain_panel_fields.py', [], None, False),
    ('readiness', 'rt_h1_grain_runtime_ipc.py', ['--readiness-only'], RUNTIME_LOG, False),
    ('base', 'rt_h1_grain_runtime_ipc.py', [], RUNTIME_LOG, False),
    ('history', 'rt_h1_grain_runtime_ipc.py', ['--history-only'], RUNTIME_LOG, False),
    ('static', 'rt_h1_grain_runtime_ipc.py', ['--static-only'], RUNTIME_LOG, False),
    ('convergence', 'rt_h1_grain_runtime_ipc.py', ['--convergence-only'], RUNTIME_LOG, False),
    ('extended', 'rt_h1_grain_runtime_ipc.py', ['--extended-only'], RUNTIME_LOG, False),
    ('settle', 'rt_h1_grain_runtime_ipc.py', ['--settle-only', '--settle-normal-damping', '8'],
     RUNTIME_LOG, False),
    ('coexist', 'rt_h1_grain_runtime_ipc.py', ['--coexist-only'], RUNTIME_LOG, False),
    ('coexist_hydrostatic', 'rt_h1_grain_runtime_ipc.py',
     ['--coexist-only', '--coexist-hydrostatic'], RUNTIME_LOG, False),
    ('porous', 'rt_h1_grain_runtime_ipc.py', ['--porous-only'], RUNTIME_LOG, False),
    ('wet', 'rt_h1_grain_runtime_ipc.py', ['--wet-only'], RUNTIME_LOG, False),
    ('motion', 'rt_grain_motion_ipc.py', [], None, False),
    ('dense', 'rt_h1_dense_grain_scene_ipc.py', ['--counts', '4096', '16384', '--steps', '120'],
     DENSE_LOG, True),
    ('repose', 'rt_h1_grain_runtime_ipc.py', ['--repose-only'], RUNTIME_LOG, True),
    ('g2_dry_wet', 'rt_g2_dry_wet_compare_ipc.py', ['--expect-empty-fluid-skipped'], None, True),
]

# Lines worth reporting: verdicts, results and the tables they print.
KEYS = ('PASS', 'FAIL', 'RESULT', 'rejected', 'ACCEPTED', 'Error', 'assert',
        'count ', 'coexist', 'porous', 'convergence', 'free fall', 'repose', 'wet')


def key_lines(text, table_after=None):
    lines = [l.rstrip() for l in text.splitlines() if l.strip()]
    picked = [l for l in lines if any(k in l for k in KEYS)]
    if table_after:
        for n, l in enumerate(lines):
            if l.startswith(table_after):
                picked += lines[n:]
                break
    # A traceback: its last frame (file:line) and the error line say where and why.
    frames = [l for l in lines if l.lstrip().startswith('File "')]
    if frames and frames[-1] not in picked:
        picked.append(frames[-1])
    if lines and lines[-1] not in picked:
        picked.append(lines[-1])
    seen, out = set(), []
    for l in picked:
        if l not in seen:
            seen.add(l)
            out.append(l if len(l) <= 220 else l[:217] + '...')
    return out[-14:]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--quick', action='store_true', help='skip the slow arms')
    parser.add_argument('--only', nargs='+', choices=[a[0] for a in ARMS])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    arms = [a for a in ARMS if (not args.only or a[0] in args.only) and
            not (args.quick and a[4] and not args.only)]
    summary = []
    for name, script, extra, log, _ in arms:
        started = time.time()
        print(f'... {name}', flush=True)
        run = subprocess.run([sys.executable, str(TEST / script), *extra], cwd=ROOT,
                             capture_output=True, text=True, encoding='utf-8', errors='replace')
        seconds = time.time() - started
        text = run.stdout + ('\n' + run.stderr if run.stderr.strip() else '')
        (OUT / f'{name}.log').write_text(text, encoding='utf-8')
        if log and log.exists():
            shutil.copyfile(log, OUT / f'{name}.json')
        status = 'PASS' if run.returncode == 0 else 'FAIL'
        summary.append((name, status, seconds, key_lines(text, 'count ' if name == 'dense' else None)))
        print(f'{status} {name} ({seconds:.0f} s)', flush=True)
    print('\n==== H1 grain suite summary (bring back from here) ====')
    for name, status, seconds, lines in summary:
        print(f'{status:<4} {name} ({seconds:.0f} s)')
        for line in lines:
            print('     ' + line)
    failed = [s[0] for s in summary if s[1] != 'PASS']
    print(f'==== {len(summary) - len(failed)}/{len(summary)} PASS'
          + (f'; failed: {", ".join(failed)}' if failed else '') +
          f'; full logs: {OUT.relative_to(ROOT)}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
