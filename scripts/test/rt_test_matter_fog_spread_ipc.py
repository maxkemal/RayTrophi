"""User-run acceptance on a SMALL, PAUSED Matter fixture with isolated Fog carriers.

Does not step/reset/seed/save. Temporarily edits spread and restores it on exit.
The 128 MiB inspection ceiling applies; do not use the large snow scene.
"""
import sys
from rt_ipc import RtIpc


def main():
    domain = sys.argv[1] if len(sys.argv) > 1 else 'Physics Domain 1'
    client = RtIpc()
    before = next(d for d in client.call('fluid.list_domains')['domains']
                  if d['name'] == domain)
    saved = before['fog_spread_voxels']
    try:
        client.call('fluid.set_fog', domain=domain, spread_voxels=0.0)
        raw = client.call('fluid.inspect_fog_fields', domain=domain)
        client.call('fluid.set_fog', domain=domain, spread_voxels=2.5)
        spread = client.call('fluid.inspect_fog_fields', domain=domain)
        key = lambda field: (field['kind'], field['substance_tag'],
                             field['phase'], field['producer_uid'])
        old = {key(field): field for field in raw['fields']}
        new = {key(field): field for field in spread['fields']}
        assert old.keys() == new.keys()
        carriers = [k for k in old if k[0] == 'carrier' and old[k]['active_cells']]
        assert carriers, 'Fixture needs populated isolated Fog carriers'
        assert any(old[k]['density_hash'] != new[k]['density_hash'] for k in carriers)
        assert any(new[k]['active_cells'] > old[k]['active_cells'] for k in carriers)
        for k in old:
            if k[0] != 'carrier':
                assert old[k]['density_hash'] == new[k]['density_hash']
                assert old[k]['temperature_hash'] == new[k]['temperature_hash']
        for value in (-1.0, 7.0):
            try:
                client.call('fluid.set_fog', domain=domain, spread_voxels=value)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Invalid spread accepted')
        final = next(d for d in client.call('fluid.list_domains')['domains']
                     if d['name'] == domain)
        assert final['fog_spread_voxels'] == 2.5
        assert final['particle_count'] == before['particle_count']
        assert spread['state_version'] == raw['state_version']
        print('PASS spread changes carrier footprint, preserves gas and simulation version')
    finally:
        try:
            client.call('fluid.set_fog', domain=domain, spread_voxels=saved)
        finally:
            client.close()


if __name__ == '__main__':
    main()
