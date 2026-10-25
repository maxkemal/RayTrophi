"""After user build: reversible render-only IPC smoke on a paused live Matter scene.

Does not seed, reset, step, launch or save. Restores phase output in finally.
"""
import argparse
from rt_ipc import RtIpc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', required=True)
    args = parser.parse_args()
    client = RtIpc()
    original = None
    try:
        assert not client.call('sim.control_state')['playing'], 'Pause the scene first'
        original = client.call('fluid.get_phase_output', domain=args.domain)
        before = client.call('fluid.state_digest', domain=args.domain)
        for fields in [dict(solid='fog', liquid='sdf', gas='fog'),
                       dict(liquid='splat'), dict(gas='hidden'),
                       dict(solid='follow', liquid='follow', gas='follow')]:
            expected = dict(client.call('fluid.get_phase_output', domain=args.domain), **fields)
            client.call('fluid.set_phase_output', domain=args.domain, **fields)
            assert client.call('fluid.get_phase_output', domain=args.domain) == expected
            assert client.call('fluid.state_digest', domain=args.domain) == before
        snapshot = client.call('fluid.get_phase_output', domain=args.domain)
        for invalid in [dict(solid='invalid'), dict(gas='sdf'), dict(substance='Snow'),
                        dict(liquid=True), dict(solid='sdf', liquid='invalid')]:
            ok, result = client.try_call('fluid.set_phase_output', domain=args.domain, **invalid)
            assert not ok, result
            assert client.call('fluid.get_phase_output', domain=args.domain) == snapshot
        print('PASS: phase routes, patch retention, atomic refusal, physical state unchanged')
    finally:
        if original is not None:
            client.call('fluid.set_phase_output', domain=args.domain, **original)
        client.close()


if __name__ == '__main__':
    main()
