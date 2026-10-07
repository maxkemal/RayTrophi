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
"""
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

    rows = []
    arms = [('A authored', {}), ('B volume_exclusion off', {'volume_exclusion': False}),
            ('C fluid_coupling off', {'fluid_coupling': False, 'volume_exclusion': False})]
    try:
        for name, patch in arms:
            restore = {k: authored[k] for k in ('volume_exclusion', 'fluid_coupling')}
            call('fluid.set_grain_settings', domain=domain, **{**restore, **patch})
            for frame in range(1, args.frames + 1):
                call('fluid.step', dt=1/60)
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
