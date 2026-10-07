"""Probe of the open scene: why do sand clumps stay on the water surface?

Runs against whatever scene is open (no reset, no save). For one grain domain
it steps three arms of N frames each from the current state, changing one
coupling setting live (settings are live-editable since 2026-10-07):

  A  as authored
  B  volume_exclusion off  (no pressure force from the liquid's acceleration;
     hydrostatic Archimedes instead)
  C  fluid_coupling off    (grains and liquid ignore each other)

and restores the authored settings at the end. A grain is denser than water
(2667 kg/m3 at packing .6), so a clump that keeps floating in A but sinks in
B is held up by the measured-acceleration pressure force (a coupling
artefact), one that sinks only in C by drag; floating in all three = contacts
(grains resting on grains / a collider), not the liquid.

    python scripts/test/rt_grain_float_probe.py                  # first grain domain
    python scripts/test/rt_grain_float_probe.py --domain Matter --frames 90

The timeline advances by 3N frames; reload the scene afterwards if needed.

    python scripts/test/rt_grain_float_probe.py --cost --frames 120

--cost only steps the scene as authored and prints where a frame goes: the
liquid lane (p2g/pressure/g2p/advect), the grain step by host stage
(order/prepare/gpu_wait/publish/merge/coupling), transfers and host<->GPU
synchronisation points. --no-coupling repeats it with fluid_coupling off
(restored after) to show whether the grain reaction sets the liquid CFL. A GPU that never runs full is usually waiting for
the CPU between these synchronisation points.
"""
import statistics
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rt_ipc import RtIpc  # noqa: E402

LOG = Path(__file__).resolve().parents[2] / 'docs/dev/grain_float_probe_live.json'


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--domain')
    parser.add_argument('--frames', type=int, default=90)
    # One fluid.step = one timeline frame of the scene. fluid.step does not move
    # the timeline, and there is no fps query yet: pass the scene's rate.
    parser.add_argument('--fps', type=float, default=24.)
    parser.add_argument('--cost', action='store_true')
    parser.add_argument('--no-coupling', action='store_true',
                        help='with --cost: fluid_coupling off for the run, restored after')
    parser.add_argument('--compare', action='store_true',
                        help='cost coupled vs uncoupled, each from fluid.reset (same frames)')
    args = parser.parse_args()
    client = RtIpc()

    def call(method, **params):
        result = client.call(method, **params)
        if isinstance(result, dict) and ('__error' in result or result.get('ok') is False):
            raise RuntimeError((method, result))
        return result

    domain = args.domain
    if not domain:
        for d in call('fluid.list_domains')['domains']:
            if call('fluid.grain_settings', domain=d['name']).get('enabled'):
                domain = d['name']
                break
    assert domain, 'no domain with discrete grains enabled'
    authored = call('fluid.grain_settings', domain=domain)
    print('domain', domain, 'wet_grains', authored.get('wet_grains'),
          'volume_exclusion', authored.get('volume_exclusion'),
          'fluid_coupling', authored.get('fluid_coupling'), flush=True)

    def sample(arm, frame):
        inv = call('fluid.matter_models', domain=domain)
        g = inv['acceptance_metrics']['granular']
        gd = inv.get('grain_diagnostics') or {}
        liquid = gd.get('liquid') or {}
        shape = liquid.get('shape') or {}
        wet = liquid.get('wet') or {}
        return {'arm': arm, 'frame': frame, 'grains': g.get('particles'),
                'com_y': (g.get('dry_center_of_mass') or [0, 0, 0])[1],
                'top_y': (g.get('bounds_max') or [0, 0, 0])[1],
                'liquid_surface_p95': shape.get('surface_p95_m'),
                'coupled': liquid.get('coupled_grains'),
                'pressure_y': (liquid.get('pressure_impulse_n_s') or [0, 0, 0])[1],
                'buoyancy_y': (liquid.get('buoyancy_impulse_n_s') or [0, 0, 0])[1],
                'drag_y': (liquid.get('drag_impulse_n_s') or [0, 0, 0])[1],
                'max_submerged': liquid.get('max_submerged_fraction'),
                'max_solid_fraction': liquid.get('max_solid_fraction'),
                'bridges': wet.get('liquid_bridges_last_substep'),
                'held': inv['mixed_execution']['step_held'],
                'status': inv['mixed_execution']['status'][:90]}

    if args.compare:
        # Both runs start from fluid.reset (frame 0, empty emitters), so they
        # see the same frames and particle counts; only coupling differs.
        results = {}
        for name, coupled in (('coupled', True), ('uncoupled', False)):
            call('fluid.reset')
            call('fluid.set_grain_settings', domain=domain,
                 fluid_coupling=authored['fluid_coupling'] and coupled,
                 volume_exclusion=authored['volume_exclusion'] and coupled)
            try:
                print(f'\n==== {name}', flush=True)
                results[name] = cost(call, args, domain)
            finally:
                call('fluid.set_grain_settings', domain=domain, fluid_coupling=authored['fluid_coupling'],
                     volume_exclusion=authored['volume_exclusion'])
        call('fluid.reset')
        print(f"\n==== compare from frame 0, {args.frames} frames at {args.fps:g} fps (medians after warm-up)")
        print(f"{'':<16}{'total ms':>10}{'pressure':>10}{'liq sub':>9}{'max m/s':>9}{'p99 m/s':>9}"
              f"{'parcels':>9}{'grains':>8}")
        for name, r in results.items():
            print(f"{name:<16}{r['total_ms']:10.2f}{r['pressure_ms']:10.2f}{r['liquid_substeps']:9.0f}"
                  f"{r['speed_max']:9.2f}{r['speed_p99']:9.2f}{r['particles'] or 0:9}{r['grains'] or 0:8}")
        return
    if args.cost and args.no_coupling:
        # Same scene with grains and liquid ignoring each other: if the liquid's
        # speed max drops, the grain reaction is what drives the liquid CFL.
        call('fluid.set_grain_settings', domain=domain, fluid_coupling=False, volume_exclusion=False)
        try:
            return cost(call, args, domain)
        finally:
            call('fluid.set_grain_settings', domain=domain, fluid_coupling=authored['fluid_coupling'],
                 volume_exclusion=authored['volume_exclusion'])
    if args.cost:
        return cost(call, args, domain)

    run_arms(call, args, domain, authored, sample)


def cost(call, args, domain):
    if True:
        frames = []
        # Per-kernel GPU time: batch_end waits for the GPU, so it hides which
        # kernels the frame is spent in. Timed over the frames after warm-up.
        timing = True
        try:
            call('perf.set_gpu_kernel_timing', enabled=True)
        except RuntimeError:
            timing = False
        for frame in range(1, args.frames + 1):
            if timing and frame == 11:
                call('perf.gpu_kernel_timings', reset=True)
            call('fluid.step', dt=1/args.fps)
            stats = call('fluid.step_stats', domain=domain)
            inv = call('fluid.matter_models', domain=domain)
            runtime = ((inv.get('grain_diagnostics') or {}).get('runtime')) or {}
            liquid = ((inv.get('grain_diagnostics') or {}).get('liquid')) or {}
            frames.append({'stats': stats, 'host_ms': runtime.get('host_ms', {}),
                           'liquid_speed_max': liquid.get('speed_max_m_s'),
                           'liquid_speed_p99': liquid.get('speed_p99_m_s'),
                           'liquid_substeps': liquid.get('liquid_substeps'),
                           'substeps': runtime.get('substeps'), 'limit': runtime.get('substep_limit'),
                           'grains': inv['acceptance_metrics']['granular'].get('particles'),
                           'held': inv['mixed_execution']['step_held']})
            if frames[-1]['held']:
                print('HELD', inv['mixed_execution']['status'], flush=True)
                break
        kernels = []
        if timing:
            timed = call('perf.gpu_kernel_timings', reset=True)
            call('perf.set_gpu_kernel_timing', enabled=False)
            kernels = sorted(timed.get('kernels', []), key=lambda k: -k.get('ms', 0.))
        LOG.with_name('grain_cost_probe_live.json').write_text(
            json.dumps({'domain': domain, 'frames': frames, 'kernels': kernels}, indent=1),
            encoding='utf-8')
        tail = frames[min(10, len(frames)-1):]
        med = lambda xs: statistics.median(xs) if xs else 0.
        stat = lambda k: med([f['stats'].get(k, 0.) or 0. for f in tail])
        host = lambda k: med([f['host_ms'].get(k, 0.) or 0. for f in tail])
        total = stat('total_ms')
        print(f"\nframes {len(tail)} at {args.fps:g} fps (after warm-up), grains {tail[-1]['grains']}, "
              f"particles {tail[-1]['stats'].get('particle_count')}, grain substeps {tail[-1]['substeps']} "
              f"({tail[-1]['limit']})")
        print(f"step total (median)          {total:8.2f} ms")
        lq = lambda k: med([f[k] for f in tail if isinstance(f.get(k), (int, float))])
        print(f"  liquid substeps {lq('liquid_substeps'):.0f}, parcel speed max {lq('liquid_speed_max'):.2f} m/s, "
              f"p99 {lq('liquid_speed_p99'):.2f} m/s (CFL follows the max)")
        for key in ('p2g_ms', 'pressure_ms', 'g2p_ms', 'advect_ms', 'density_ms'):
            print(f"  liquid {key:<20} {stat(key):8.2f} ms")
        for key in ('order', 'coupling', 'prepare', 'gpu_wait', 'publish', 'merge'):
            print(f"  grain host {key:<16} {host(key):8.2f} ms")
        for key in ('synchronize_ms', 'batch_end_ms', 'upload_call_ms', 'download_call_ms', 'dispatch_call_ms'):
            print(f"  transfer {key:<18} {stat(key):8.2f} ms")
        print(f"  sync points: synchronize {stat('synchronize_calls'):.0f}, batch_end {stat('batch_end_calls'):.0f}, "
              f"dispatches {stat('dispatch_calls'):.0f}; upload {stat('upload_bytes')/1e6:.2f} MB, "
              f"download {stat('download_bytes')/1e6:.2f} MB per frame")
        if kernels:
            timed_frames = max(1, len(frames) - 10)
            total_gpu = sum(k.get('ms', 0.) for k in kernels) / timed_frames
            print(f"\nGPU kernels (ms per frame, {timed_frames} frames), total {total_gpu:.2f} ms:")
            for k in kernels[:15]:
                extra = {key: k[key] for key in ('count', 'calls', 'dispatches') if key in k}
                print(f"  {k.get('kernel', '?'):<44} {k.get('ms', 0.)/timed_frames:8.2f}  {extra}")
        print('log:', LOG.with_name('grain_cost_probe_live.json').relative_to(LOG.parents[2]))
        return {'total_ms': total, 'pressure_ms': stat('pressure_ms'),
                'liquid_substeps': lq('liquid_substeps'), 'speed_max': lq('liquid_speed_max'),
                'speed_p99': lq('liquid_speed_p99'), 'particles': tail[-1]['stats'].get('particle_count'),
                'grains': tail[-1]['grains']}


def run_arms(call, args, domain, authored, sample):
    rows = []
    arms = [('A authored', {}), ('B volume_exclusion off', {'volume_exclusion': False}),
            ('C fluid_coupling off', {'fluid_coupling': False, 'volume_exclusion': False})]
    try:
        for name, patch in arms:
            restore = {k: authored[k] for k in ('volume_exclusion', 'fluid_coupling')}
            call('fluid.set_grain_settings', domain=domain, **{**restore, **patch})
            for frame in range(1, args.frames + 1):
                call('fluid.step', dt=1/args.fps)
                if frame % 15 == 0 or frame == 1:
                    row = sample(name, frame)
                    rows.append(row)
                    print(json.dumps(row), flush=True)
                    if row['held']:
                        break
    finally:
        call('fluid.set_grain_settings', domain=domain,
             volume_exclusion=authored['volume_exclusion'], fluid_coupling=authored['fluid_coupling'])
        LOG.write_text(json.dumps({'domain': domain, 'authored': authored, 'rows': rows}, indent=1),
                       encoding='utf-8')
    print('\narm                      frame  com_y   top_y   surface  pressure_y  buoy_y   drag_y   bridges')
    for r in rows:
        if r['frame'] % 45 and r['frame'] != args.frames:
            continue
        f = lambda v: f"{v:8.4f}" if isinstance(v, (int, float)) else f"{'-':>8}"
        print(f"{r['arm']:<24} {r['frame']:>5} {f(r['com_y'])} {f(r['top_y'])} {f(r['liquid_surface_p95'])} "
              f"{f(r['pressure_y'])}  {f(r['buoyancy_y'])} {f(r['drag_y'])} {r['bridges']}")
    print('log:', LOG.relative_to(LOG.parents[2]))


if __name__ == '__main__':
    main()
