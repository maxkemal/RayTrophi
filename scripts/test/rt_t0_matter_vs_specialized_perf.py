"""T0: does a Matter domain carrying one phase cost what the specialized domain costs?

Decision gate for docs/dev/MADDE_TIPLERI_TASARIMI.md karar 1: the Gas and Liquid
domain types stay until a Matter domain is measured as fast. Nothing measured
this before 2026-10-07.

Arms (same bounds, voxel, backend, source, frame count):
  gas/gas        type='gas',    gas flow source
  gas/matter     type='matter', the same gas flow source, no liquid source
  liquid/fluid   type='fluid',  liquid flow source
  liquid/matter  type='matter', the same liquid flow source, no gas source

Per arm: the solver's own stats (gas.step_stats / fluid.step_stats total_ms:
THE comparison), client wall ms per fluid.step (information only: every IPC
call is serviced in the UI frame loop, so it is the app's frame cadence and
moves with the viewport; see docs/dev/MADDE_TIPLERI_TASARIMI.md §6b), device memory
before/after, and a result fingerprint (gas: active density cells and plume;
liquid: particle count) so "faster" cannot hide "did less".

    python scripts/test/rt_t0_matter_vs_specialized_perf.py [--frames 96] [--voxel .04]
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_ipc import RtIpc  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LOG = ROOT / 'docs/dev/t0_matter_vs_specialized_live.json'
FPS = 24.
LOW, HIGH = [-1., 0., -1.], [1., 2., 1.]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=96)
    ap.add_argument('--voxel', type=float, default=.04)
    ap.add_argument('--warmup', type=int, default=8, help='frames left out of the timing')
    ap.add_argument('--rounds', type=int, default=3,
                    help='each round runs all four arms; the order alternates so GPU '
                         'warm-up and clock drift do not favour one arm')
    args = ap.parse_args()
    client = RtIpc()

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    assert not call('sim.control_state')['playing'], 'Pause the application first'
    original = {'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list')}
    data = {'frames': args.frames, 'voxel': args.voxel, 'runs': []}
    created_domains, created_sources = [], []

    def device_bytes():
        # Per-device rows; a missing table is an error, not a 0 MiB reading.
        devices = call('perf.get_gpu_memory')['devices']
        assert devices, 'perf.get_gpu_memory returned no devices'
        return sum(d['device_local_bytes'] for d in devices)

    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=False, visible=False)

        def run(arm, domain_type, phase):
            name = 'T0_' + arm.replace('/', '_')
            source = name + '_Src'
            before = device_bytes()
            existing = {d['name'] for d in call('fluid.list_domains')['domains']}
            if name not in existing:
                call('fluid.create_domain', name=name, type=domain_type,
                     domain_min=LOW, domain_max=HIGH, voxel_size=args.voxel)
                created_domains.append(name)
            call('fluid.set_param', domain=name, enabled=True, visible=False,
                 backend='vulkan', boundary='closed')
            if phase == 'gas':
                spec = dict(domain=name, phase='gas', position=[0, .3, 0], radius=.25,
                            density=1., temperature=.4, fuel=0., velocity=[0, .5, 0])
            else:
                spec = dict(domain=name, phase='liquid', source_mode='point',
                            position=[0, 1.2, 0], radius=.15, velocity=[0, -1., 0],
                            fluid_particles_per_second=20000.)
            names = {s['name'] for s in call('flow_source.list')}
            if source in names:
                call('flow_source.update', name=source, enabled=True, **spec)
            else:
                call('flow_source.create', name=source, **spec)
                created_sources.append(source)
            call('fluid.reset')
            wall, solver, other = [], [], []
            # The other phase of a Matter domain: what an unused phase costs.
            other_method = None
            if domain_type == 'matter':
                other_method = 'fluid.step_stats' if phase == 'gas' else 'gas.step_stats'
            for frame in range(args.frames):
                t0 = time.perf_counter()
                call('fluid.step', dt=1 / FPS)
                ms = (time.perf_counter() - t0) * 1e3
                stats = call('gas.step_stats' if phase == 'gas' else 'fluid.step_stats',
                             domain=name)
                extra = call(other_method, domain=name) if other_method else None
                if frame >= args.warmup:
                    wall.append(ms)
                    solver.append(stats)
                    if extra is not None:
                        other.append(extra)
            info = call('fluid.get', domain=name)
            after = device_bytes()
            row = {
                'type': domain_type, 'phase': phase, 'phases': info.get('phases'),
                'grid': [info.get(k) for k in ('nx', 'ny', 'nz')],
                'wall_ms_median': statistics.median(wall),
                'wall_ms_p90': sorted(wall)[int(.9 * (len(wall) - 1))],
                'measured_all': all(s.get('measured', True) for s in solver),
                'solver_last': solver[-1],
                'solver_total_ms_median': statistics.median(
                    s.get('total_ms', 0.0) for s in solver),
                'other_phase_last': other[-1] if other else None,
                'other_phase_total_ms_median': statistics.median(
                    s.get('total_ms', 0.0) for s in other) if other else None,
                'device_bytes_delta': after - before,
                'particles': info.get('particle_count'),
                'gas_cells': info.get('active_density_cells'),
            }
            row['arm'] = arm
            data['runs'].append(row)
            LOG.write_text(json.dumps(data, indent=1), encoding='utf-8')
            call('flow_source.update', name=source, enabled=False)
            call('fluid.set_param', domain=name, enabled=False)
            other_ms = row['other_phase_total_ms_median']
            delta = row['device_bytes_delta']
            memory = 'not tracked' if delta == 0 else '+%.1f MiB' % (delta / 2**20)
            print(f"{arm:<14} solver {row['solver_total_ms_median']:6.2f} ms  other phase "
                  f"{'-' if other_ms is None else f'{other_ms:6.2f}'} ms  "
                  f"wall median {row['wall_ms_median']:7.2f} ms  p90 "
                  f"{row['wall_ms_p90']:7.2f}  dev "
                  f"{memory}  "
                  f"particles {row['particles']}  gas cells {row['gas_cells']}  "
                  f"phases {row['phases']}  measured {row['measured_all']}", flush=True)
            return row

        arms = [('gas/gas', 'gas', 'gas'), ('gas/matter', 'matter', 'gas'),
                ('liquid/fluid', 'fluid', 'liquid'), ('liquid/matter', 'matter', 'liquid')]
        for round_index in range(args.rounds):
            order = arms if round_index % 2 == 0 else list(reversed(arms))
            print(f"-- round {round_index + 1}/{args.rounds}", flush=True)
            for arm in order:
                run(*arm)

        def med(arm, key):
            values = [r[key] for r in data['runs'] if r['arm'] == arm and r[key] is not None]
            return statistics.median(values) if values else None

        summary = {}
        for label, a, b in (('gas', 'gas/gas', 'gas/matter'),
                            ('liquid', 'liquid/fluid', 'liquid/matter')):
            row = {k: {arm: med(arm, k) for arm in (a, b)}
                   for k in ('wall_ms_median', 'solver_total_ms_median',
                             'other_phase_total_ms_median')}
            row['wall_ratio'] = row['wall_ms_median'][b] / max(row['wall_ms_median'][a], 1e-6)
            row['solver_ratio'] = (row['solver_total_ms_median'][b] /
                                   max(row['solver_total_ms_median'][a], 1e-6))
            summary[label] = row
            # Solver total_ms is the comparison. Client wall time is the app's UI
            # frame cadence (every IPC call is serviced in the frame loop) and
            # moves with whatever the viewport draws: information only.
            print(f"RESULT {label}: solver {row['solver_total_ms_median'][a]:.2f} -> "
                  f"{row['solver_total_ms_median'][b]:.2f} ms ({row['solver_ratio']:.2f}x); "
                  f"unused phase in matter {row['other_phase_total_ms_median'][b]} ms; "
                  f"(client wall {row['wall_ms_median'][a]:.2f} -> {row['wall_ms_median'][b]:.2f} ms, "
                  f"frame cadence, not solver cost)", flush=True)
        data['summary'] = summary
        LOG.write_text(json.dumps(data, indent=1), encoding='utf-8')
    finally:
        for s in created_sources:
            try:
                call('flow_source.remove', name=s)
            except RuntimeError:
                pass
        for d in created_domains:
            try:
                call('fluid.remove_domain', domain=d)
            except RuntimeError:
                pass
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s.get('enabled', True))
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d.get('enabled', True),
                 visible=d.get('visible', True))
        call('fluid.reset')
        client.close()


if __name__ == '__main__':
    main()
