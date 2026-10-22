"""Matter batch smoke run: every quick acceptance test, one line each.

Runs each test as its own process against the running app (Pause first).
Full output goes to %TEMP%/rt_matter_smoke/<name>.log; the console gets one
line per test - PASS n/n, or the failing gate names - so a run costs a few
lines to read. Open a log only for a test that failed.

    python scripts/test/rt_matter_smoke.py            # quick arms
    python scripts/test/rt_matter_smoke.py --with-h1  # + the long H1 suite
    python scripts/test/rt_matter_smoke.py --only grain_heat contact_heat
    python scripts/test/rt_matter_smoke.py --restart-app   # fresh app first
    python scripts/test/rt_matter_smoke.py --quick         # per-batch tier (~10 min)

After every test, domains / flow sources / force fields whose names start
with "T_", "T: ", "H1_" or "G2_" (test-created) are removed, so a test that timed out
cannot leave a scene that slows or breaks the next one (2026-10-20: a timed
out P9 arm left a domain that made grain_heat 10x slower and sleep_transfer
refuse to start). --restart-app restarts RayTrophi first: a long session
was measured 10x slower than a fresh one; it closes the open scene, so it is
off by default.
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
    ('seed_settle', 'rt_test_seed_settle_ipc.py', []),
    ('sleep_transfer', 'rt_test_grain_sleep_transfer_ipc.py', []),
    ('liquid_column', 'rt_test_liquid_column_ipc.py', []),
    ('p9_melt', 'rt_test_p9_matter_ipc.py', ['--arm', 'melt']),
    ('p9_floor', 'rt_test_p9_matter_ipc.py', ['--arm', 'floor']),
    ('mpm_dem_clock', 'rt_test_mpm_dem_shared_clock_ipc.py', []),
    ('snow_heat', 'rt_test_snow_heat_ipc.py', []),
    ('liquid_density', 'rt_test_liquid_density_ipc.py', []),
    ('run_readback', 'rt_test_run_readback_ipc.py', []),
    ('fast_fall', 'rt_test_fast_fall_ipc.py', []),
    ('p9_pore', 'rt_test_p9_matter_ipc.py', ['--arm', 'pore']),
    ('t0', 'rt_t0_matter_vs_specialized_perf.py', ['--rounds', '1']),
]
H1 = ('h1_suite', 'rt_h1_grain_suite.py', [])
# Per-batch tier (2026-10-20 rule: at most ~10 minutes). The rest run once,
# at the end of the plan.
QUICK = ('substance', 'thermal_films', 'phase3_freeze', 'fire_steam', 'grain_readiness',
         'seed_settle', 'p9_melt', 'p9_floor', 'mpm_dem_clock', 'run_readback', 'fast_fall')
GATE = re.compile(r'^(PASS|FAIL)\s+(\S+)')


def cleanup():
    """Remove test-created entities; never touches user-named ones."""
    sys.path.insert(0, str(TEST))
    try:
        from rt_ipc import RtIpc
        client = RtIpc()
    except Exception as error:  # noqa: BLE001 - app gone: nothing to clean
        return f'cleanup skipped ({error})'
    removed = []
    # T_/T: = acceptance tests; H1_/G2_ = suites that keep their domain for reuse.
    test = lambda n: n.startswith(('T_', 'T: ', 'H1_', 'G2_'))
    try:
        for src in client.call('flow_source.list'):
            if test(src['name']):
                client.call('flow_source.remove', name=src['name'])
                removed.append(src['name'])
        for field in client.call('forcefield.list'):
            if test(field['name']):
                client.call('forcefield.remove', field=field['name'])
                removed.append(field['name'])
        for dom in client.call('fluid.list_domains')['domains']:
            if test(dom['name']):
                client.call('fluid.remove_domain', domain=dom['name'])
                removed.append(dom['name'])
        for sub in client.call('substance.list').get('substances', []):
            name = sub.get('name', '')
            if test(name) and not sub.get('builtin'):
                try:
                    client.call('substance.remove', name=name)
                    removed.append(name)
                except Exception:  # noqa: BLE001 - still referenced: keep it
                    pass
        client.call('timeline.set_frame', frame=0)
    except Exception as error:  # noqa: BLE001 - report, do not hide
        return f'cleanup error: {error}'
    finally:
        client.close()
    return ('removed ' + ', '.join(removed)) if removed else ''


def restart_app():
    root = TEST.parents[1]
    subprocess.run(['powershell', '-NoProfile', '-Command',
                    'Get-Process RayTrophiStudio -ErrorAction SilentlyContinue | Stop-Process -Force'],
                   check=False)
    time.sleep(2)
    subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
                    str(root / 'scripts/ipc/Start-RayTrophi.ps1')], check=False,
                   capture_output=True)


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
    leftover = cleanup()
    note = f'  [{leftover}]' if leftover else ''
    print(f'{name:<20} {summary}  ({seconds:.0f} s){note}', flush=True)
    return code == 0 and not failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--with-h1', action='store_true')
    parser.add_argument('--only', nargs='*')
    parser.add_argument('--timeout', type=int, default=1200, help='seconds per test')
    parser.add_argument('--quick', action='store_true',
                        help='per-batch tier, about 10 minutes')
    parser.add_argument('--restart-app', action='store_true',
                        help='restart RayTrophi first (closes the open scene)')
    args = parser.parse_args()
    if args.restart_app:
        restart_app()
    arms = ARMS + ([H1] if args.with_h1 else [])
    if args.quick:
        arms = [a for a in arms if a[0] in QUICK]
    if args.only:
        arms = [a for a in arms if a[0] in args.only]
    ok = sum(run(name, script, extra, args.timeout) for name, script, extra in arms)
    print(f'==== {ok}/{len(arms)} tests pass; logs: {OUT}', flush=True)
    sys.exit(0 if ok == len(arms) else 1)


if __name__ == '__main__':
    main()
