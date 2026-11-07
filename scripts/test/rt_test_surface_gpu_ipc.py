"""GPU surface reconstruction (docs/dev/GPU_YUZEY_LEVEL_SET.md, Y1 + Y2).

Arms (closed 5 m domain, h 0.1 m, Vulkan, empty scene, paused):
  water   a resting water block: the reference liquid surface;
  snow    a falling 2.25 m snow block (~100k parcels): the measured cost case;
  bound   the water block with a scene material bound to Water (Output list):
          the per-substance composition must come from the GPU too, and the
          route must stay on the GPU (it used to refuse such domains).

Gates:
  * water_parity / snow_parity: fluid.surface_parity, GPU vs CPU on the same
    particles: no sign mismatch, max |dphi| <= 1e-3 h, band mismatch <= 0.1 %
    of compared cells, uvw residual within 1e-4 m;
  * bound_composition: the bound arm builds a mixture on both sides, with no
    material-pair mismatch and blend weights within 1e-4; route not cpu;
  * route_gpu: with surface_gpu on, fluid.get surface_route is gpu_host or
    gpu_live (never cpu) and surface_gpu_status is empty;
  * route_cpu: surface_gpu=false reports cpu;
  * aniso_refused: anisotropic kernels report cpu with a status naming why;
  * cpu_build_gone: with the GPU route, perf.list sim.render.surface.level_set
    and material_coordinates add (near) nothing over the measured frames, and
    sim.render.surface.gpu runs instead;
  * faster: render_sync per frame (GPU) <= 0.25 x (CPU) on the snow arm.
Informational: surface_route itself (gpu_live needs the Rendered viewport;
Solid/Material read back by design), gpu_ms, upload/download bytes.
Output: %TEMP%/rt_surface_gpu.json
"""
import json
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

DT = 1 / 24
DOMAIN = 'T_SurfaceGpu'
OUT = Path(tempfile.gettempdir()) / 'rt_surface_gpu.json'


def scope(rows, name):
    for r in rows:
        if r['name'] == name:
            return r
    return {'count': 0, 'total_ms': 0.0}


def main():
    client = RtIpc()
    call = client.call
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    data = {}

    def make(substance, y0, height, size, ppc):
        call('fluid.create_domain', name=DOMAIN, type='fluid', domain_min=[-2.5, -2.5, -2.5],
             domain_max=[2.5, 2.5, 2.5], voxel_size=0.1)
        call('fluid.set_param', domain=DOMAIN, enabled=True, backend='vulkan',
             boundary='closed', default_substance=substance, thermal_liquid_enabled=False)
        call('timeline.set_frame', frame=0)
        h = size / 2
        call('fluid.seed', domain=DOMAIN, seed_min=[-h, y0, -h], seed_max=[h, y0 + height, h],
             particles_per_cell=ppc, replace=True)

    def surface(label):
        g = call('fluid.get', domain=DOMAIN)
        row = {k: g.get(k) for k in ('surface_gpu', 'surface_route', 'surface_gpu_status',
                                     'surface_build_ms', 'surface_upload_bytes',
                                     'surface_download_bytes', 'surface_active_cells')}
        data.setdefault('routes', {})[label] = row
        return row

    def frames(n):
        call('perf.reset')
        for _ in range(n):
            call('fluid.step', dt=DT)
        rows = call('perf.list')
        per = lambda name: scope(rows, name)['total_ms'] / max(n, 1)
        return {'render_sync_ms': per('sim.timeline.render_sync'),
                'level_set_ms': per('sim.render.surface.level_set'),
                'uvw_ms': per('sim.render.surface.material_coordinates'),
                'gpu_ms': per('sim.render.surface.gpu'),
                'gpu_calls': scope(rows, 'sim.render.surface.gpu')['count']}

    def parity_gate(p):
        compared = max(p.get('compared_cells') or 0, 1)
        return (p.get('gpu_ok') and p.get('sign_mismatch') == 0 and
                (p.get('max_abs_phi_over_h') or 1.0) <= 1e-3 and
                (p.get('band_mismatch') or 0) <= 0.001 * compared and
                (not p.get('uvw_compared') or (p.get('uvw_max_abs') or 1.0) <= 1e-4))

    verdicts = {}
    try:
        # ── water ──
        make('Water', -2.5, 1.0, 2.0, 8)
        try:
            for _ in range(6):
                call('fluid.step', dt=DT)
            p = call('fluid.surface_parity', domain=DOMAIN)
            data['water_parity'] = p
            verdicts['water_parity'] = parity_gate(p)
        finally:
            call('fluid.remove_domain', domain=DOMAIN)

        # ── water with a scene material bound to its substance ──
        mats = call('material.list')
        mats = mats.get('materials', mats) if isinstance(mats, dict) else mats
        name = next((m.get('name') for m in mats if m.get('name')), None)
        make('Water', -2.5, 1.0, 2.0, 8)
        try:
            if name:
                call('fluid.set_substance_material', domain=DOMAIN, substance='Water',
                     material=name)
            for _ in range(4):
                call('fluid.step', dt=DT)
            p = call('fluid.surface_parity', domain=DOMAIN)
            data['bound_parity'] = p
            bound_route = surface('bound')
            verdicts['bound_composition'] = bool(
                name and parity_gate(p) and p.get('composition_cpu') and
                p.get('composition_gpu') and p.get('composition_slot_mismatch') == 0 and
                (p.get('composition_weight_max_abs') or 1.0) <= 1e-4 and
                bound_route['surface_route'] in ('gpu_host', 'gpu_live'))
        finally:
            call('fluid.remove_domain', domain=DOMAIN)

        # ── snow ──
        make('Snow', 0.0, 2.25, 2.25, 8)
        try:
            call('fluid.set_grain_settings', domain=DOMAIN, sleep=False)
            for _ in range(2):
                call('fluid.step', dt=DT)
            call('fluid.set_param', domain=DOMAIN, surface_gpu=True)
            gpu = frames(6)
            route_gpu = surface('gpu')
            p = call('fluid.surface_parity', domain=DOMAIN)
            data['snow_parity'] = p
            call('fluid.set_param', domain=DOMAIN, surface_gpu=False)
            cpu = frames(6)
            route_cpu = surface('cpu')
            call('fluid.set_param', domain=DOMAIN, surface_gpu=True, anisotropy_enabled=True)
            call('fluid.step', dt=DT)
            route_aniso = surface('aniso')
            call('fluid.set_param', domain=DOMAIN, anisotropy_enabled=False)
            data['frames'] = {'gpu': gpu, 'cpu': cpu}
            verdicts['snow_parity'] = parity_gate(p)
            verdicts['route_gpu'] = (route_gpu['surface_route'] in ('gpu_host', 'gpu_live') and
                                     not route_gpu['surface_gpu_status'])
            verdicts['route_cpu'] = route_cpu['surface_route'] == 'cpu'
            verdicts['aniso_refused'] = (route_aniso['surface_route'] == 'cpu' and
                                         'anisotropic' in (route_aniso['surface_gpu_status'] or ''))
            verdicts['cpu_build_gone'] = (gpu['level_set_ms'] + gpu['uvw_ms'] < 1.0 and
                                          gpu['gpu_calls'] > 0)
            verdicts['faster'] = gpu['render_sync_ms'] <= 0.25 * max(cpu['render_sync_ms'], 1e-6)
        finally:
            call('fluid.remove_domain', domain=DOMAIN)

        data['verdicts'] = verdicts
        print(json.dumps({'frames': data.get('frames'), 'routes': data.get('routes')}), flush=True)
        for name in ('water_parity', 'bound_parity', 'snow_parity'):
            q = data.get(name, {})
            print(name, {k: q.get(k) for k in ('gpu_ok', 'gpu_status', 'compared_cells',
                                               'max_abs_phi_over_h', 'mean_abs_phi',
                                               'sign_mismatch', 'band_mismatch',
                                               'uvw_max_abs', 'composition_cpu',
                                               'composition_gpu', 'composition_slot_mismatch',
                                               'composition_weight_max_abs',
                                               'cpu_ms', 'gpu_ms')}, flush=True)
        for name, passed in verdicts.items():
            print('PASS' if passed else 'FAIL', name, flush=True)
        if not all(verdicts.values()):
            raise SystemExit(1)
    finally:
        OUT.write_text(json.dumps(data, indent=2), encoding='utf-8')
        call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
