"""Hot water on snow with the thermal liquid ON (the honey scene's hold): no
parcel may go below the coldest source, and the mixed step must not hold."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rt_ipc import RtIpc

c = RtIpc(); call = c.call
assert not call('fluid.list_domains')['domains'], 'needs an empty scene'
n = 'T_Thermal'
call('fluid.create_domain', name=n, type='matter', domain_min=[-.25, 0, -.25], domain_max=[.25, .45, .25], voxel_size=.025)
try:
    call('fluid.set_param', domain=n, enabled=True, visible=False, backend='vulkan', boundary='closed',
         default_substance='Water', thermal_liquid_enabled=True)
    call('fluid.settle', domain=n, on_seed=False, request=False)
    call('fluid.set_seed_source_settings', domain=n, settings={'particles_per_cell': 8, 'settle': False})
    call('fluid.add_seed_source', domain=n, settings={'name': 'Snow', 'substance': 'Snow',
         'bounds_min': [-.24, 0, -.2], 'bounds_max': [0, .1, .2]})
    call('fluid.add_seed_source', domain=n, settings={'name': 'Hot water', 'substance': 'Water',
         'bounds_min': [.05, 0, -.2], 'bounds_max': [.24, .3, .2],
         'temperature_mode': 'custom', 'temperature_kelvin': 353.0})
    call('fluid.apply_seed_sources', domain=n)
    call('timeline.set_frame', frame=0)
    worst = 1e9; held = 0
    for f in range(48):
        call('fluid.step', dt=1 / 24)
        st = call('fluid.step_stats', domain=n)
        d = [x for x in call('fluid.list_domains')['domains'] if x['name'] == n][0]
        tmin = d.get('particle_min_kelvin', 0)
        worst = min(worst, tmin)
        if 'held' in (st.get('gpu_status') or ''):
            held += 1
        if f % 8 == 7:
            print(f + 1, 'Tmin %.1f Tmax %.1f mean %.1f frozen %s evaporated %.2e steps %s | %s' % (
                tmin, d.get('particle_max_kelvin', 0), d.get('thermal_mean_kelvin', 0),
                d.get('thermal_frozen_particles'), d.get('thermal_evaporated_kg', 0),
                (st.get('mixed_stages') or {}).get('grid_steps'), (st.get('gpu_status') or '')[:70]), flush=True)
    print('worst Tmin %.1f K, held frames %d' % (worst, held))
    print('PASS' if worst > 250.0 and held == 0 else 'FAIL')
finally:
    call('fluid.remove_domain', domain=n)
    c.close()
