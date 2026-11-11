"""External user-run acceptance. Requires a dedicated paused DEM test population.

Does not advance time. Explicitly binds and releases the current loose grains.
Run only after user C++ build, after other agents' live tests have ended.
"""
import argparse
from rt_ipc import RtIpc, RtIpcError


def main(domain):
    call = RtIpc().call
    assert not call('sim.control_state')['playing'], 'Pause the dedicated test scene'

    def bonds(**request):
        return call('fluid.dem_bodies', domain=domain, request=request)

    baseline = bonds(action='snapshot')
    assert not baseline['nodes'], 'Use a dedicated population with no existing bodies'
    physical = call('fluid.state_digest', domain=domain)
    material = dict(young_pa=1e5, poisson=.25, tensile_strength_pa=1e4, shear_strength_pa=1e4)
    body = None
    try:
        for invalid in ({'young_pa': -1}, {'poisson': .5}, {'young_pa': True}):
            try:
                bonds(action='bind', all_current_grains=True,
                      material={**material, **invalid})
            except RtIpcError:
                pass
            else:
                raise AssertionError('invalid material accepted: ' + repr(invalid))
            assert bonds(action='snapshot') == baseline
        created = bonds(action='bind', all_current_grains=True, material=material)
        body = created['body']
        assert body > 0 and created['bonds'] > 0 and created['broken'] == 0
        snapshot = bonds(action='snapshot')
        try:
            bonds(action='bind', all_current_grains=True, material=material)
        except RtIpcError:
            pass
        else:
            raise AssertionError('already bonded population was rebound')
        assert bonds(action='snapshot') == snapshot
        after = call('fluid.state_digest', domain=domain)
        for key in ('particles', 'position_hash', 'velocity_hash'):
            assert after[key] == physical[key], key
    finally:
        if body is not None:
            released = bonds(action='remove', body=body)
            assert released['bodies'] == 0 and released['bonds'] == 0
    print('PASS paused DEM body authoring/validation; no dynamic fracture test performed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('domain')
    main(parser.parse_args().domain)
