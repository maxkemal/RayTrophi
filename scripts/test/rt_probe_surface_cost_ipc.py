"""Read-only surface/cost snapshot in the already open application.

Run after the user chooses SDF/splat and quality in the UI. Does not step,
reset, edit parameters, launch the application or compile anything.
"""
import argparse
import json
from pathlib import Path
from rt_ipc import RtIpc

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--domain', default='Physics Domain 1')
parser.add_argument('--label', default='surface')
args = parser.parse_args()
client = RtIpc()
try:
    control = client.call('sim.control_state')
    info = client.call('fluid.get', domain=args.domain)
    fields = {k:v for k,v in info.items() if k.startswith('surface_') or
              k in ('particle_count','render_mode','smoothing_iterations',
                    'anisotropy_enabled','kernel_radius_voxels','voxel_size')}
    report = {'label':args.label,'domain':args.domain,'control':control,
              'surface':fields,'fluid':client.call('fluid.step_stats',domain=args.domain),
              'cache':client.call('sim_cache.status'),
              'viewport':client.call('viewport.frame_telemetry')}
    path = Path(__file__).resolve().parents[2]/'docs/dev/surface_cost_snapshots_2026_10_10.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    rows.append(report)
    path.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
finally:
    client.close()
