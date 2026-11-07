"""Bounded contact investigation of an OPEN scene; no reset/seed or collider edits.

Run externally after building C++ and all five DEM shaders (revision 38).
Advances the playhead, stops at the first held attempt, restores the probe switch.
"""
import argparse
import json

from rt_ipc import RtIpc


def fingerprint(state):
    return state['particles'], state['position_hash'], state['velocity_hash']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--domain', default='Physics Domain 1')
    parser.add_argument('--steps', type=int, default=1)
    parser.add_argument('--dt', type=float, default=1 / 24)
    parser.add_argument('--require-success', action='store_true',
                        help='Fail the check if any attempted DEM step is held')
    args = parser.parse_args()
    if not 1 <= args.steps <= 32 or not 0 < args.dt <= 1 / 24:
        parser.error('steps must be 1..32 and dt must be (0, 1/24]')

    client = RtIpc()
    previous = None
    try:
        assert not client.call('sim.control_state')['playing'], 'Pause the scene first'
        settings = client.call('fluid.grain_settings', domain=args.domain)
        if 'contact_probe_enabled' not in settings:
            raise RuntimeError('This executable lacks the contact probe; build C++ and shaders')
        models = client.call('fluid.matter_models', domain=args.domain)
        assert models['grain_ownership']['enabled'], 'An active DEM domain is required'
        before = client.call('fluid.state_digest', domain=args.domain)
        assert before['particles'] > 0, 'The domain must already contain grains'
        previous = settings['contact_probe_enabled']
        # The shared JSON validator must reject an integer unchanged.
        ok, _ = client.try_call('fluid.set_grain_settings', domain=args.domain,
                                contact_probe_enabled=1)
        assert not ok, 'The probe switch must require a JSON boolean'
        assert client.call('fluid.grain_settings', domain=args.domain) == settings
        client.call('fluid.set_grain_settings', domain=args.domain, contact_probe_enabled=True)
        assert fingerprint(client.call('fluid.state_digest', domain=args.domain)) == fingerprint(before)
        for index in range(args.steps):
            before = client.call('fluid.state_digest', domain=args.domain)
            client.call('fluid.step', dt=args.dt)
            models = client.call('fluid.matter_models', domain=args.domain)
            stats = client.call('fluid.step_stats', domain=args.domain)
            after = client.call('fluid.state_digest', domain=args.domain)
            grain = models['grain_diagnostics']
            probe = grain['contact_probe']
            print(json.dumps({
                'attempt': index + 1,
                'frame': client.call('sim.control_state')['frame'],
                'status': stats['gpu_status'],
                'total_ms': stats['total_ms'],
                'cfl_terms': grain['cfl_terms'],
                'sleep_work_mode': grain['runtime'].get('sleep_work_mode')
                    if grain['runtime'] else None,
                'contact_probe': probe,
                'host_state_unchanged': fingerprint(before) == fingerprint(after),
            }, ensure_ascii=False), flush=True)
            if models['grain_readiness']['step_held']:
                assert fingerprint(before) == fingerprint(after), 'Held attempt changed host particles'
                if args.require_success:
                    raise RuntimeError('DEM acceptance failed: ' + stats['gpu_status'])
                break
            assert probe['measured'], 'No compatible GPU probe result; check shader revision/status'
            sleep = probe.get('sleep')
            if sleep is not None:
                assert sleep['counts_valid'], 'Sleep counters overflowed; shorten/reduce the probe'
                assert sleep['final_snapshot_complete'], 'Incomplete final-substep sleep snapshot'
                assert sleep['final_sampled_grains'] == sleep['grains']
                expected = sleep['grains'] * grain['runtime']['substeps']
                assert sleep['frame_contact_skips'] + sleep['frame_contact_solves'] == expected
    finally:
        try:
            if previous is not None:
                client.call('fluid.set_grain_settings', domain=args.domain,
                            contact_probe_enabled=previous)
        finally:
            client.close()


if __name__ == '__main__':
    main()
