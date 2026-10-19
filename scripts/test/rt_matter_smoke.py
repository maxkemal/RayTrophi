"""Matter batch smoke run: every quick acceptance test, one line each.

Runs each test as its own process against the running app (Pause first).
Full output goes to %TEMP%/rt_matter_smoke/<name>.log; the console gets one
line per test - PASS n/n, or the failing gate names - so a run costs a few
lines to read. Open a log only for a test that failed.

    python scripts/test/rt_matter_smoke.py            # quick arms
    python scripts/test/rt_matter_smoke.py --with-h1  # + the long H1 suite
    python scripts/test/rt_matter_smoke.py --only grain_heat contact_heat
"""
import argparse
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

TEST = Path(__file__).resolve().parent
OUT = Path(tempfile.gettempdir()) / 'rt_matter_smoke'

# name, script, args. Order: independent and fast first (NEXT_BUILD_CHECKS).
ARMS = [
    ('substance', 'rt_test_substance_profiles_ipc.py', []),
    ('thermal_films', 'rt_test_thermal_films_ipc.py', []),
    ('phase3_freeze', 'rt_test_fluid_phase3_boundary_freeze_ipc.py', []),
    ('gas_drag', 'rt_test_grain_gas_drag_ipc.py', []),
    ('contact_heat', 'rt_test_contact_heat_ipc.py', []),
    # Twice: the second run is what a deleted-name or leftover-state bug breaks.
    ('contact_heat_again', 'rt_test_contact_heat_ipc.py', []),
    ('surface_water', 'rt_test_surface_water_ipc.py', []),
    ('fire_steam', 'rt_test_fire_steam_quench_ipc.py', []),
    ('grain_heat', 'rt_test_grain_heat_ipc.py', []),
    ('grain_readiness', 'rt_h1_grain_runtime_ipc.py', ['--readiness-only']),
]
H1 = ('h1_suite', 'rt_h1_grain_suite.py', [])
GATE = re.compile(r'^(PASS|FAIL)\s+(\S+)')


def run(name, script, args, timeout):
    OUT.mkdir(parents=True, exist_ok=True)
    log = OUT / f'{name}.log'
    start = time.time()
    try:
        proc = subprocess.run([sys.executable, str(TEST / script), *args], cwd=TEST,
                              capture_output=True, text=True, encoding='utf-8',
                              errors='replace', timeout=timeout,
                              env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
        text, code = proc.stdout + proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as error:
        text, code = (error.stdout or '') + '\nTIMEOUT', -1
        if isinstance(text, bytes):
            text = text.decode('utf-8', 'replace')
    log.write_text(text, encoding='utf-8')
    gates = [m.groups() for m in map(GATE.match, text.splitlines()) if m]
    failed = [g for verdict, g in gates if verdict == 'FAIL']
    passed = sum(1 for verdict, _ in gates if verdict == 'PASS')
    seconds = time.time() - start
    if code == 0 and not failed:
        summary = f'PASS {passed}/{len(gates)}' if gates else 'PASS'
    elif failed:
        summary = f'FAIL {passed}/{len(gates)}: ' + ', '.join(failed[:6])
    else:
        last = next((l for l in reversed(text.splitlines()) if l.strip()), '')
        summary = f'ERROR (exit {code}): {last[:110]}'
    print(f'{name:<20} {summary}  ({seconds:.0f} s)', flush=True)
    return code == 0 and not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-h1', action='store_true')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--timeout', type=int, default=1200, help='seconds per test')
    args = parser.parse_args()
    arms = ARMS + ([H1] if args.with_h1 else [])
    if args.only:
        arms = [a for a in arms if a[0] in args.only]
    ok = sum(run(name, script, extra, args.timeout) for name, script, extra in arms)
    print(f'==== {ok}/{len(arms)} tests pass; logs: {OUT}', flush=True)
    sys.exit(0 if ok == len(arms) else 1)


if __name__ == '__main__':
    main()
