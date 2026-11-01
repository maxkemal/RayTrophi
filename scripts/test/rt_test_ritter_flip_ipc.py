"""D2 FLIP comparison; empty scene, Pause, external IPC after the user's build.

Ritter's frictionless dry-bed shallow-water front is x=2*sqrt(g*h)*t.
Reference: https://ethz.ch/content/dam/ethz/special-interest/baug/vaw/vaw-dam/
documents/das-institut/mitteilungen/2010-2019/222.pdf (section 2.5.1).
This finite-depth 3D particle test measures the leading carrier, not a surface
quantile. It is a resolution-specific comparison, not a general solver proof.
No built-in profile is edited and no preset decision is applied automatically.
"""
import argparse
import json
import math
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

HEIGHT = 0.15
GRAVITY = 9.81
DT = 1 / 240
DOMAIN = 'T_RitterFlip'
PROFILES = {0.3: 'T_RitterFlip_03', 0.9: 'T_RitterFlip_09'}
REFERENCE_SPEED = 2 * math.sqrt(GRAVITY * HEIGHT)


def slope(samples):
    """Least-squares speed; a free intercept separates startup from speed."""
    mt = sum(t for t, _ in samples) / len(samples)
    mx = sum(x for _, x in samples) / len(samples)
    return sum((t - mt) * (x - mx) for t, x in samples) / sum(
        (t - mt) ** 2 for t, _ in samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path, default=Path(tempfile.gettempdir()) /
                        'rt_ritter_flip.json')
    args = parser.parse_args()
    if args.self_test:
        assert math.isclose(slope([(0, 4), (1, 6), (2, 8)]), 2)
        assert math.isclose(REFERENCE_SPEED ** 2, 4 * GRAVITY * HEIGHT)
        print('PASS: regression and Ritter reference arithmetic; no IPC used')
        return

    client = RtIpc()
    call = client.call
    created = []
    domain_created = False
    results = {}
    try:
        assert not call('sim.control_state')['playing'], 'Pause first'
        assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
        for flip, profile in PROFILES.items():
            call('substance.derive', name=profile, based_on='Water')
            created.append(profile)
            call('substance.set', name=profile, fields={
                'solver_flip_blend': flip, 'solver_viscosity_wall_slip': 1.0})
            fields = call('substance.get', name=profile)['fields']
            assert math.isclose(fields['solver_flip_blend'], flip, abs_tol=1e-6)
            call('fluid.create_domain', name=DOMAIN, type='fluid',
                 domain_min=[-1.2, 0, -0.05], domain_max=[1.8, 0.3, 0.05],
                 voxel_size=0.025)
            domain_created = True
            call('fluid.set_param', domain=DOMAIN, default_substance=profile,
                 enabled=True, visible=False, backend='vulkan', boundary='closed',
                 thermal_liquid_enabled=False, max_particles=200000)
            call('fluid.settle', domain=DOMAIN, on_seed=False, request=False)
            call('timeline.set_frame', frame=0)
            call('fluid.seed', domain=DOMAIN, seed_min=[-1.2, 0, -0.05],
                 seed_max=[0, HEIGHT, 0.05], particles_per_cell=8, replace=True)
            initial = call('fluid.state_digest', domain=DOMAIN)
            assert initial['bounds_measured'] and initial['particles'] > 0
            samples = []
            for block in range(1, 13):
                run = call('fluid.run', steps=8, dt=DT, sample_every=0,
                           domains=[DOMAIN], fields=['particle_count'])
                digest = call('fluid.state_digest', domain=DOMAIN)
                expected = run['samples'][-1]['domains'][DOMAIN]['particle_count']
                assert digest['particles'] == expected == initial['particles'], \
                    'particle loss or a changed state after run'
                assert digest['bounds_measured']
                x = digest['bounds_max'][0]
                assert math.isfinite(x) and x < 1.7, 'front reached downstream wall'
                t = block * 8 * DT
                samples.append((t, x))
            # Tail fit after the release transient; upstream rarefaction remains
            # short of the -1.2 m wall throughout this 0.4 s experiment.
            speed = slope([(t, x) for t, x in samples if t >= 0.15])
            results[str(flip)] = {'samples_s_m': samples, 'speed_m_s': speed,
                                  'relative_error': abs(speed / REFERENCE_SPEED - 1)}
            call('fluid.remove_domain', domain=DOMAIN)
            domain_created = False
        checks = {
            'flip03_ritter_speed': results['0.3']['relative_error'] <= 0.15,
            'flip09_ritter_speed': results['0.9']['relative_error'] <= 0.15,
            'flip03_no_regression': results['0.3']['relative_error'] <=
                                    results['0.9']['relative_error'] + 0.05,
        }
        report = {'height_m': HEIGHT, 'voxel_m': 0.025, 'dt_s': DT,
                  'reference_speed_m_s': REFERENCE_SPEED, 'arms': results,
                  'checks': checks, 'preset_changed': False}
        args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report), flush=True)
        if not all(checks.values()):
            raise SystemExit(1)
    finally:
        if domain_created:
            call('fluid.remove_domain', domain=DOMAIN)
        for profile in reversed(created):
            call('substance.remove', name=profile)
        if created:
            call('timeline.set_frame', frame=0)
        client.close()


if __name__ == '__main__':
    main()
