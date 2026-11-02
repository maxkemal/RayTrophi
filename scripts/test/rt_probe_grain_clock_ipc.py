"""What sets the grain substep, frame by frame (2026-10-27).

Steps the open scene's Matter domain one fluid.step at a time and prints, every
--every steps, the DEM substep count and what set it
(grain_diagnostics.cfl_terms): stiffest spring and its source, smallest
bridge volume, contacts assumed, stiffness scale; the shared clock split
(continuum substeps x DEM substeps each); whether the step held; and the
grain step's own host time (not the IPC wall clock, which measures frame
cadence).

Probe, not a gate. Run it in the 4-source scene (E:\\rt_k\\matter4emiter.rtp:
Water, Snow, 600 K Sand, gas), paused. Before 2026-10-27 that scene held at
step ~32 (554 > 512 substeps, wet bridge stiffness). Full rows go to
%TEMP%/rt_probe_grain_clock.json.
"""
import argparse
import json
import math
import os
from pathlib import Path
import statistics

from rt_ipc import RtIpc


def clock_consistent(clock, authored_limit=None):
    if not clock.get('enabled'):
        return None
    values = [clock.get(key) for key in ('transport_steps', 'continuum_substeps',
                                        'dem_substeps_per_continuum', 'continuum_grid_steps')]
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in values):
        return False
    dem, continuum, per, grid = values
    if authored_limit is not None:
        if not isinstance(authored_limit, int) or isinstance(authored_limit, bool):
            return False
        if authored_limit > 0 and dem > authored_limit:
            return False
    return dem == continuum * per and grid <= continuum


def summarize(rows):
    timings = {}
    for key in sorted({key for row in rows for key in row.get('host_ms', {})}):
        values = sorted(row['host_ms'][key] for row in rows
                        if isinstance(row.get('host_ms', {}).get(key), (int, float))
                        and math.isfinite(row['host_ms'][key]) and row['host_ms'][key] >= 0)
        if values:
            timings[key] = {'samples': len(values), 'median': statistics.median(values),
                            'p95': values[math.ceil(0.95 * len(values)) - 1]}
    shared = [row for row in rows if row.get('clock_consistent') is not None]
    return {'sampled_frames': len(rows), 'held_frames': sum(row['held'] for row in rows),
            'shared_clock_samples': len(shared),
            'invalid_clock_frames': [row['step'] for row in shared
                                     if not row['clock_consistent']],
            'host_ms': timings,
            'timing_note': 'Nested/overlapping durations; do not sum these fields. '
                           'These are sampled solver times, not IPC wall times.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', default='Physics Domain 1')
    parser.add_argument('--steps', type=int, default=120)
    parser.add_argument('--every', type=int, default=6)
    parser.add_argument('--dt', type=float, default=1 / 24)
    parser.add_argument('--require-clock', action='store_true',
                        help='fail if held, no shared-clock sample, or inconsistent schedule')
    args = parser.parse_args()
    if args.steps <= 0 or args.every <= 0 or not math.isfinite(args.dt) or args.dt <= 0:
        parser.error('steps, every and finite dt must be positive')
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    rows = []
    held_at = None
    try:
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        for step in range(1, args.steps + 1):
            call('fluid.step', dt=args.dt)
            m = call('fluid.matter_models', domain=args.domain)
            execution = m.get('mixed_execution') or {}
            held = bool(execution.get('step_held'))
            if step % args.every and not held and step != args.steps:
                continue
            diagnostics = m.get('grain_diagnostics') or {}
            runtime = diagnostics.get('runtime') or {}
            terms = diagnostics.get('cfl_terms') or {}
            clock = runtime.get('common_clock') or {}
            pore = m.get('pore_exchange') or {}
            row = {'step': step, 'held': held, 'status': execution.get('status'),
                   'substeps': runtime.get('substeps'), 'limit': runtime.get('substep_limit'),
                   'continuum': clock.get('continuum_substeps'),
                   'dem_per_continuum': clock.get('dem_substeps_per_continuum'),
                   'contacts_measured': runtime.get('max_contacts_per_grain'),
                   'contacts_exceeded': runtime.get('cfl_contacts_exceeded'),
                   'grain_step_ms': (runtime.get('host_ms') or {}).get('step_total'),
                   'host_ms': runtime.get('host_ms') or {},
                   'common_clock': clock,
                   # substep_limit names the limiting term; the authored cap is in cfl_terms.
                   'clock_consistent': clock_consistent(clock, terms.get('max_substeps')),
                   'cost_last_substep': runtime.get('cost_last_substep') or {},
                   'upload_bytes': runtime.get('upload_bytes'),
                   'download_bytes': runtime.get('download_bytes'),
                   'pore_water_kg': pore.get('pore_water_kg'), 'cfl_terms': terms}
            rows.append(row)
            print('%3d %s sub %s (%s) = %s x %s | %s %.0f N/m, Vb %.3g m3, contacts %s/%s%s, '
                  'scale %s | %s ms' % (
                      step, 'HELD' if held else 'ok  ', row['substeps'], row['limit'],
                      row['continuum'], row['dem_per_continuum'],
                      terms.get('effective_source'), terms.get('effective_stiffness_n_m') or 0.0,
                      terms.get('minimum_bridge_volume_m3') or 0.0,
                      row['contacts_measured'], terms.get('contacts_assumed'),
                      ' EXCEEDED' if row['contacts_exceeded'] else '',
                      terms.get('stiffness_scale'), row['grain_step_ms']), flush=True)
            print('    grid=%s contactless=%s host_ms=%s' % (
                clock.get('continuum_grid_steps'),
                row['cost_last_substep'].get('contactless_grains'),
                json.dumps(row['host_ms'], sort_keys=True)), flush=True)
            if held:
                held_at = step
                print('  ' + str(row['status']), flush=True)
                break
        print('held at step %s' % held_at if held_at else
              'never held in %d steps' % args.steps, flush=True)
    finally:
        try:
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
        finally:
            path = Path(os.environ.get('TEMP', '.')) / 'rt_probe_grain_clock.json'
            path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
            client.close()
    summary = summarize(rows)
    summary_path = path.with_name('rt_probe_grain_clock_summary.json')
    summary_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if args.require_clock and (summary['held_frames'] or summary['invalid_clock_frames']
                               or not summary['shared_clock_samples']):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
