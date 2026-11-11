"""Scale rig (docs/dev/OLCEK_MILYON_PARCACIK.md, S0): how each frame cost item grows with N.

Same voxel (h), same physics, the SCENE grows with N (substep count stays put),
so a slope > 1 in an item means it is not just parcel work. Arms: snow (MPM),
water (pure liquid), snow+water (mixed). For each N: untimed frames for host
time and host stages, then timed frames for GPU kernel time. Temporary domains,
empty scene required, removed afterwards. Not an acceptance gate.

  python rt_probe_scale_ipc.py --counts 100000 1000000 --frames 4
Output: %TEMP%/rt_probe_scale.json
"""
import argparse
import json
import math
import statistics
import tempfile
import time
from pathlib import Path

from rt_ipc import RtIpc

H = 0.025
PPC = 8
HEIGHT = 0.3
OUT = Path(tempfile.gettempdir()) / 'rt_probe_scale.json'


def footprint(count):
    """Square side (m) of a HEIGHT-tall block holding `count` parcels at PPC."""
    volume = count / PPC * H ** 3
    return math.sqrt(volume / HEIGHT)


def seeds_for(arm, count):
    side = footprint(count)
    half = side / 2
    if arm == 'snow':
        return side, [{'name': 'Snow', 'substance': 'Snow',
                       'bounds_min': [-half, 0.0, -half], 'bounds_max': [half, HEIGHT, half]}]
    if arm == 'water':
        return side, [{'name': 'Water', 'substance': 'Water',
                       'bounds_min': [-half, 0.0, -half], 'bounds_max': [half, HEIGHT, half]}]
    # snow+water: half of the parcels each, side by side
    return side, [{'name': 'Snow', 'substance': 'Snow',
                   'bounds_min': [-half, 0.0, -half], 'bounds_max': [0.0, HEIGHT, half]},
                  {'name': 'Water', 'substance': 'Water',
                   'bounds_min': [0.0, 0.0, -half], 'bounds_max': [half, HEIGHT, half]}]


def run(call, arm, count, frames):
    side, seeds = seeds_for(arm, count)
    pad = 2 * H
    name = 'T_Scale_%s_%dk' % (arm.replace('+', '_'), count // 1000)
    lo = [-side / 2 - pad, 0.0, -side / 2 - pad]
    hi = [side / 2 + pad, HEIGHT + 0.15, side / 2 + pad]
    call('fluid.create_domain', name=name, type='matter', domain_min=lo, domain_max=hi, voxel_size=H)
    try:
        call('fluid.set_param', domain=name, enabled=True, visible=False, backend='vulkan',
             boundary='closed', default_substance='Water', thermal_liquid_enabled=False)
        # The default 1 GiB domain budget holds a 1M mixed step silently ("Mixed step
        # held"); the rig measures cost, so lift it and report the working set instead.
        call('gas.set_settings', domain=name, enforce_resource_budget=False)
        call('fluid.settle', domain=name, on_seed=False, request=False)
        call('fluid.set_seed_source_settings', domain=name,
             settings={'particles_per_cell': PPC, 'settle': False})
        for s in seeds:
            call('fluid.add_seed_source', domain=name, settings=s)
        started = time.time()
        call('fluid.apply_seed_sources', domain=name)
        seed_s = time.time() - started
        call('timeline.set_frame', frame=0)
        info = [d for d in call('fluid.list_domains')['domains'] if d['name'] == name][0]
        out = {'arm': arm, 'target': count, 'particles': info.get('particle_count'),
               'grid': [round((b - a) / H) for a, b in zip(lo, hi)], 'seed_s': round(seed_s, 2)}
        call('perf.set_gpu_kernel_timing', enabled=False)
        rows = []
        for _ in range(frames):
            t0 = time.time()
            call('fluid.step', dt=1 / 24)
            wall = (time.time() - t0) * 1000
            st = call('fluid.step_stats', domain=name)
            ms = st.get('mixed_stages') or {}
            rows.append({'total': st.get('total_ms'), 'wall': wall,
                         'grid': ms.get('grid_steps'), 'liquid': ms.get('liquid_steps'),
                         'stages': {k: v.get('host_wall_ms', 0.0) for k, v in (ms.get('stages') or {}).items()},
                         'keys': sorted(k for k, v in st.items() if isinstance(v, (int, float)))})
        body = rows[1:] or rows
        out['total_ms'] = round(statistics.median(r['total'] for r in body), 1)
        out['ipc_wall_ms'] = round(statistics.median(r['wall'] for r in body), 1)
        last = call('fluid.step_stats', domain=name)
        out['gpu_status'] = last.get('gpu_status')
        out['working_set_mb'] = round((last.get('mixed_working_set_bytes') or 0) / 2**20, 1)
        out['grid_steps'] = body[-1]['grid']
        out['liquid_steps'] = body[-1]['liquid']
        stage_names = set().union(*(r['stages'] for r in body))
        out['stages_ms'] = {k: round(statistics.median(r['stages'].get(k, 0.0) for r in body), 1)
                            for k in sorted(stage_names)}
        out['scalar_stats'] = {k: v for k, v in call('fluid.step_stats', domain=name).items()
                               if isinstance(v, (int, float)) and k.endswith('_ms')}
        timed = max(2, frames // 2)
        call('perf.set_gpu_kernel_timing', enabled=True)
        call('perf.gpu_kernel_timings', reset=True)
        for _ in range(timed):
            call('fluid.step', dt=1 / 24)
        t = call('perf.gpu_kernel_timings', reset=True)
        call('perf.set_gpu_kernel_timing', enabled=False)
        ks = sorted(t.get('kernels', []), key=lambda k: -k['ms'])
        out['gpu_ms'] = round(sum(k['ms'] for k in ks) / timed, 1)
        out['gpu_top'] = [[k['kernel'], round(k['ms'] / timed, 1), k['calls'] // timed] for k in ks[:10]]
        print(json.dumps(out), flush=True)
        return out
    finally:
        call('perf.set_gpu_kernel_timing', enabled=False)
        call('fluid.remove_domain', domain=name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--counts', nargs='+', type=int, default=[100000, 1000000])
    p.add_argument('--arms', nargs='+', default=['snow', 'water', 'snow+water'])
    p.add_argument('--frames', type=int, default=4)
    a = p.parse_args()
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Needs an empty scene'
    results = []
    try:
        for arm in a.arms:
            for count in a.counts:
                results.append(run(call, arm, count, a.frames))
    finally:
        call('timeline.set_frame', frame=0)
        OUT.write_text(json.dumps(results, indent=1), encoding='utf-8')
        client.close()
    # slopes: d log(item) / d log(N) between the smallest and largest count
    for arm in a.arms:
        rs = sorted((r for r in results if r['arm'] == arm), key=lambda r: r['particles'] or 0)
        if len(rs) < 2 or not rs[0]['particles']:
            continue
        lo, hi = rs[0], rs[-1]
        ln = math.log(hi['particles'] / lo['particles'])
        def slope(x0, x1):
            return round(math.log(x1 / x0) / ln, 2) if x0 > 0.5 and x1 > 0.5 else None
        print(arm, 'N %d -> %d' % (lo['particles'], hi['particles']),
              'total', slope(lo['total_ms'], hi['total_ms']), 'gpu', slope(lo['gpu_ms'], hi['gpu_ms']),
              {k: slope(lo['stages_ms'].get(k, 0), hi['stages_ms'].get(k, 0)) for k in hi['stages_ms']})


if __name__ == '__main__':
    main()
