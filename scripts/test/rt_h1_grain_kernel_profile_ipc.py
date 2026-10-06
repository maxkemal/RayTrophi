"""Per-kernel GPU time of the dry grain step; external IPC, after user build.

Uses the runtime test's isolated domain/source (created there), disables every
other authoring item for the run and restores it. Frame0 paused, no save.
Reports perf.gpu_kernel_timings per grain count so per-substep kernel cost is
separated from dispatch count. Not an acceptance gate.
"""
import argparse
import json
from pathlib import Path
from rt_ipc import RtIpc
from grain_units import restitution_of

DOMAIN = 'H1_Grain_Runtime'
SOURCE = 'H1_Grain_Runtime_Source'
LOG = Path(__file__).resolve().parents[2] / 'docs/dev/matter_h1_grain_kernel_profile_live.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', nargs='+', type=int, default=[256, 1024])
    parser.add_argument('--steps', type=int, default=36)
    parser.add_argument('--stiffness', type=float, default=100000.)
    parser.add_argument('--tangential', type=float, default=2/7)
    parser.add_argument('--rolling', type=float, default=.02)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    client = RtIpc()

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and (result.get('ok') is False or '__error' in result):
            raise RuntimeError((method, result))
        return result

    original = {'domains': call('fluid.list_domains')['domains'],
                'sources': call('flow_source.list'),
                'colliders': call('collider.list')['colliders']}
    assert not call('sim.control_state')['playing'], 'Pause application first'
    assert DOMAIN in {d['name'] for d in original['domains']}, 'run rt_h1_grain_runtime_ipc.py first'
    data = {'args': vars(args), 'arms': []}
    try:
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=False)
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['name'] == DOMAIN, visible=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=False)
        for count in args.counts:
            call('timeline.set_frame', frame=0)
            call('fluid.reset')
            call('fluid.set_grain_settings', domain=DOMAIN, enabled=True, radius_m=.025,
                 stiffness_n_m=args.stiffness, restitution=restitution_of(4., args.stiffness), sliding_damping_n_s_m=4.,
                 friction=.5, rolling_friction=args.rolling, twisting_friction=0.,
                 tangential_stiffness_ratio=args.tangential, contact_resolution=24,
                 max_substeps=4096)
            call('flow_source.update', name=SOURCE, position=[0, .65, 0], radius=.43,
                 velocity=[0, 0, 0], max_emitted_particles=count,
                 fluid_particles_per_second=count*1.01*60, start_time=0., end_time=1/30,
                 enabled=True)
            call('perf.set_gpu_kernel_timing', enabled=True)
            call('perf.gpu_kernel_timings', reset=True)
            stats = []
            for _ in range(args.steps):
                call('fluid.step', dt=1/60)
                stats.append(call('fluid.step_stats', domain=DOMAIN)['total_ms'])
            timings = call('perf.gpu_kernel_timings', reset=True)
            call('perf.set_gpu_kernel_timing', enabled=False)
            runtime = call('fluid.matter_models', domain=DOMAIN)['grain_diagnostics']['runtime']
            grain = [k for k in timings['kernels'] if 'grain' in k['kernel']]
            arm = {'count': count, 'kernels': grain, 'runtime': runtime,
                   'median_total_ms': sorted(stats[5:])[len(stats[5:])//2]}
            for k in grain:
                k['ms_per_call'] = k['ms']/max(k['calls'], 1)
            data['arms'].append(arm)
            print(json.dumps(arm), flush=True)
    finally:
        call('flow_source.update', name=SOURCE, enabled=False)
        for c in original['colliders']:
            call('collider.update', name=c['name'], enabled=c['enabled'])
        for d in original['domains']:
            call('fluid.set_param', domain=d['name'], enabled=d['enabled'], visible=d['visible'])
        for s in original['sources']:
            call('flow_source.update', name=s['name'], enabled=s['enabled'])
        call('timeline.set_frame', frame=0)
        call('fluid.reset')
        out = LOG if not args.tag else LOG.with_name(LOG.stem + '_' + args.tag + '.json')
        out.write_text(json.dumps(data, indent=2), encoding='utf-8')
        client.close()


if __name__ == '__main__':
    main()
