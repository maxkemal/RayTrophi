"""Probe: what an AWAKE MPM frame costs, by kernel (docs/dev/MPM_UYUYAN_KAROLAR.md).

The resting snow frame was measured bound by its serial substep chain
(~50 substeps x ~36 dispatches). This probe reads the moving case: a snow
block falling in the user's scene scale (closed 5 m domain, h 0.1 m), sleep
off, with a tiny-particle control so chain latency separates from work.

Prints per arm: wall total_ms, GPU total_ms, dispatches/frame, substeps, and
the kernel table (ms/frame, calls/frame). Output: %TEMP%/rt_mpm_chain.json
"""
import argparse
import json
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

DT = 1 / 24
DOMAIN = 'T_MpmChain'
OUT = Path(tempfile.gettempdir()) / 'rt_mpm_chain.json'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--frames', type=int, default=6)
    ap.add_argument('--warm', type=int, default=2)
    args = ap.parse_args()
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}

    def arm(label, size, ppc):
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-2.5, -2.5, -2.5],
             domain_max=[2.5, 2.5, 2.5], voxel_size=0.1)
        try:
            call('fluid.set_param', domain=DOMAIN, enabled=True, visible=False,
                 backend='vulkan', boundary='closed', default_substance='Snow',
                 thermal_liquid_enabled=False)
            call('fluid.set_grain_settings', domain=DOMAIN, sleep=False)
            call('timeline.set_frame', frame=0)
            h = size / 2
            call('fluid.seed', domain=DOMAIN, seed_min=[-h, 0.0, -h],
                 seed_max=[h, size, h], particles_per_cell=ppc, replace=True)
            for _ in range(args.warm):
                call('fluid.step', dt=DT)
            call('perf.set_gpu_kernel_timing', enabled=True)
            call('perf.gpu_kernel_timings', reset=True)
            wall = []
            last = {}
            for _ in range(args.frames):
                call('fluid.step', dt=DT)
                last = call('fluid.step_stats', domain=DOMAIN)
                wall.append(last.get('total_ms') or 0.0)
            timing = call('perf.gpu_kernel_timings', reset=True)
            call('perf.set_gpu_kernel_timing', enabled=False)
            n = args.frames
            kernels = [{'kernel': k['kernel'], 'ms': k['ms'] / n, 'calls': k['calls'] / n}
                       for k in timing.get('kernels', [])]
            d = call('fluid.state_digest', domain=DOMAIN)
            row = {
                'particles': d['particles'],
                'wall_ms': sum(wall) / max(len(wall), 1),
                'gpu_ms': (timing.get('total_ms') or 0.0) / n,
                'dispatches': sum(k['calls'] for k in kernels),
                'substeps': last.get('mixed_common_substeps'),
                'grid_steps': last.get('granular_solver_substeps'),
                'gpu_status': last.get('gpu_status'),
                'kernels': kernels,
            }
            data[label] = row
            OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
            print(json.dumps({k: v for k, v in row.items() if k != 'kernels'}), flush=True)
            for k in kernels[:40]:
                print(f"  {k['kernel']:<44} {k['ms']:8.2f} ms {k['calls']:7.1f} calls", flush=True)
            return row
        finally:
            call('fluid.remove_domain', domain=DOMAIN)

    try:
        arm('tiny', 0.3, 1)
        arm('block', 2.25, 8)
    finally:
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
