"""T0: does a Matter domain carrying one phase cost what the specialized domain costs?

Decision gate for docs/dev/MADDE_TIPLERI_TASARIMI.md karar 1: the Gas and Liquid
domain types stay until a Matter domain is measured as fast. Nothing measured
this before 2026-10-07.

Arms (same bounds, voxel, backend, source, frame count):
  gas/gas        type='gas',    gas flow source
  gas/matter     type='matter', the same gas flow source, no liquid source
  liquid/fluid   type='fluid',  liquid flow source
  liquid/matter  type='matter', the same liquid flow source, no gas source

Per arm (P8, 2026-10-20): the frames run in ONE fluid.run call, and the
comparison is fluid.run's server-side step_ms (the simulation world's step,
no IPC cadence in it - §6b showed the client wall clock is the frame loop).
The solver's own step_stats total_ms after the last frame is kept too. A
result fingerprint makes "faster" unable to hide "did less": liquid particle
count, centroid and mean speed (fluid.state_digest); gas active density
cells and max density.

Decision (MADDE_TIPLERI karar 1 / T6): a phase is EQUAL when the Matter step
costs <= 1.10x the specialist step and the fingerprint agrees (counts within
2%, liquid centroid within one voxel, mean speed within 10%).

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
            # The other phase of a Matter domain: what an unused phase costs.
            other_method = None
            if domain_type == 'matter':
                other_method = 'fluid.step_stats' if phase == 'gas' else 'gas.step_stats'
            t0 = time.perf_counter()
            result = call('fluid.run', steps=args.frames, dt=1 / FPS, sample_every=0,
                          domains=[name])
            wall_total = (time.perf_counter() - t0) * 1e3
            step_ms = result['step_ms']['all'][args.warmup:]
            solver = call('gas.step_stats' if phase == 'gas' else 'fluid.step_stats', domain=name)
            other = call(other_method, domain=name) if other_method else None
            last = result['samples'][-1]['domains'][name]
            digest = call('fluid.state_digest', domain=name) if phase == 'liquid' else {}
            info = call('fluid.get', domain=name)
            after = device_bytes()
            row = {
                'type': domain_type, 'phase': phase, 'phases': info.get('phases'),
                'grid': [info.get(k) for k in ('nx', 'ny', 'nz')],
                'step_ms_median': statistics.median(step_ms),
                'step_ms_p90': sorted(step_ms)[int(.9 * (len(step_ms) - 1))],
                'client_wall_ms_per_frame': wall_total / args.frames,
                'measured_all': solver.get('measured', True),
                'solver_last': solver,
                'solver_total_ms_last': solver.get('total_ms'),
                'other_phase_last': other,
                'other_phase_total_ms_last': other.get('total_ms') if other else None,
                'device_bytes_delta': after - before,
                'particles': last.get('particle_count'),
                'gas_cells': last.get('active_density_cells'),
                'max_density': last.get('max_density'),
                'centroid': digest.get('centroid'),
                'mean_speed': digest.get('mean_speed'),
            }
            row['arm'] = arm
            data['runs'].append(row)
            LOG.write_text(json.dumps(data, indent=1), encoding='utf-8')
            call('flow_source.update', name=source, enabled=False)
            call('fluid.set_param', domain=name, enabled=False)
            delta = row['device_bytes_delta']
            memory = 'not tracked' if delta == 0 else '+%.1f MiB' % (delta / 2**20)
            print(f"{arm:<14} step {row['step_ms_median']:6.2f} ms (p90 {row['step_ms_p90']:6.2f})  "
                  f"solver last {row['solver_total_ms_last']}  dev {memory}  "
                  f"particles {row['particles']}  gas cells {row['gas_cells']}  "
                  f"phases {row['phases']}", flush=True)
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

        def last_of(arm, key):
            rows = [r for r in data['runs'] if r['arm'] == arm]
            return rows[-1][key] if rows else None

        def close(a, b, rel):
            if a is None or b is None:
                return False
            return abs(a - b) <= rel * max(abs(a), abs(b), 1e-9)

        summary = {}
        for label, a, b in (('gas', 'gas/gas', 'gas/matter'),
                            ('liquid', 'liquid/fluid', 'liquid/matter')):
            ms = {arm: med(arm, 'step_ms_median') for arm in (a, b)}
            ratio = ms[b] / max(ms[a], 1e-6)
            if label == 'gas':
                agrees = (close(last_of(a, 'gas_cells'), last_of(b, 'gas_cells'), .02) and
                          close(last_of(a, 'max_density'), last_of(b, 'max_density'), .02))
            else:
                ca, cb = last_of(a, 'centroid'), last_of(b, 'centroid')
                centroid_ok = ca is not None and cb is not None and \
                    max(abs(x - y) for x, y in zip(ca, cb)) <= args.voxel
                agrees = (close(last_of(a, 'particles'), last_of(b, 'particles'), .02) and
                          centroid_ok and
                          close(last_of(a, 'mean_speed'), last_of(b, 'mean_speed'), .10))
            equal = ratio <= 1.10 and agrees
            summary[label] = {'step_ms_median': ms, 'ratio': ratio, 'result_agrees': agrees,
                              'equal': equal}
            print(f"RESULT {label}: step {ms[a]:.2f} -> {ms[b]:.2f} ms ({ratio:.2f}x), "
                  f"result agrees {agrees} -> {'EQUAL' if equal else 'NOT EQUAL'}", flush=True)
            print(('PASS' if equal else 'FAIL'), f't0_{label}_equal', flush=True)
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
