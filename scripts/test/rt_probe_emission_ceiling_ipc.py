"""Step the open paused scene until its emitter ceiling binds; preserve settings."""
import argparse
import json
import tempfile
import time
from pathlib import Path

from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', default='Physics Domain 1')
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--seconds', type=float, default=360)
    args = parser.parse_args()
    client = RtIpc()
    rows = []
    started = time.monotonic()
    path = Path(tempfile.gettempdir()) / 'rt_probe_emission_ceiling.json'
    try:
        assert not client.call('sim.control_state')['playing'], 'Pause first'
        for step in range(1, args.steps + 1):
            client.call('fluid.step', dt=1 / 24)
            if step % 12 and step != args.steps:
                continue
            domain = next(d for d in client.call('fluid.list_domains')['domains']
                          if d['name'] == args.domain)
            sources = [s for s in client.call('flow_source.list')
                       if s['domain'] == args.domain]
            models = client.call('fluid.matter_models', domain=args.domain)
            execution = models.get('mixed_execution') or {}
            source_total = sum(s.get('withheld_mass_kg_total', 0) for s in sources)
            row = {
                'step': step,
                'elapsed_s': time.monotonic() - started,
                'particles': domain['particle_count'],
                'max_particles': domain['max_particles'],
                'domain_withheld_kg': domain['emission_withheld_kg_total'],
                'source_withheld_kg': source_total,
                'step_held': execution.get('step_held'),
                'status': execution.get('status'),
                'sources': [{k: s.get(k) for k in (
                    'name', 'total_emitted_particles', 'total_emitted_mass_kg',
                    'withheld_mass_kg_step', 'withheld_mass_kg_total')}
                    for s in sources],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row['domain_withheld_kg'] > 0:
                print('CEILING OBSERVED; source/domain agree:',
                      abs(row['domain_withheld_kg'] - source_total)
                      <= 1e-6 * max(1, source_total), flush=True)
                break
            if row['step_held'] or row['elapsed_s'] >= args.seconds:
                print('STOP: held or wall-time budget; ceiling not observed', flush=True)
                break
        else:
            print('Step budget reached; ceiling not observed', flush=True)
    finally:
        path.write_text(json.dumps(rows, indent=2), encoding='utf-8')
        client.close()
        print('Report:', path, flush=True)


if __name__ == '__main__':
    main()
