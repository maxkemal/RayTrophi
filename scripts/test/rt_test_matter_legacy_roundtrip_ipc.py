"""T6 synthetic pre-lane Gas+Fluid project migration and save/open acceptance.

Run only in an empty disposable project, Pause, after the user's build. This
test saves and opens projects under a new TEMP directory; the current project
path changes. It never opens an existing user fixture or overwrites a user file.
The fixture uses today's valid empty project envelope with pre-T6 domain keys;
it tests that schema migration, not every historical project version.
"""
import argparse
import copy
import json
from pathlib import Path
import tempfile

from rt_ipc import RtIpc

GAS, LIQUID = 'T_LegacyGas', 'T_LegacyLiquid'
NAMES = {GAS: ('Gas', 'gas', 0), LIQUID: ('Fluid', 'liquid', 1)}


def domain_rows(payload):
    return [domain for system in payload['particle_simulation']['systems']
            for domain in system['domains']]


def make_legacy(payload):
    payload = copy.deepcopy(payload)
    seen = set()
    for system in payload['particle_simulation']['systems']:
        for index, domain in enumerate(system['domains']):
            if domain['name'] not in NAMES:
                continue
            kind, _, phase = NAMES[domain['name']]
            domain['type'] = kind
            domain.pop('phase_grids', None)
            domain.pop('fluid_fog_shader', None)
            # Pre-T6 source phase followed its owning domain. Deliberately stale
            # persisted phase values must be canonicalized, even when disabled.
            for source in system['flow_sources']:
                if source['domain_index'] == index:
                    source['phase'] = 1 - phase
            seen.add(domain['name'])
    assert seen == set(NAMES), 'fixture must contain both legacy domains'
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        fixture = {'particle_simulation': {'systems': [{
            'domains': [{'name': n, 'type': 'Matter', 'phase_grids': {'version': 2}}
                        for n in NAMES],
            'flow_sources': [{'domain_index': i, 'phase': 0} for i in range(2)]}]}}
        legacy = make_legacy(fixture)
        assert all('phase_grids' not in d for d in domain_rows(legacy))
        assert fixture['particle_simulation']['systems'][0]['domains'][0]['type'] == 'Matter'
        assert [d['type'] for d in domain_rows(legacy)] == ['Gas', 'Fluid']
        print('PASS: synthetic legacy fixture transform; no IPC used')
        return

    client = RtIpc()
    call = client.call
    created_domains, created_sources = [], []
    folder = Path(tempfile.mkdtemp(prefix='rt_matter_legacy_'))
    fixture_path = folder / 'legacy.rtp'
    canonical_path = folder / 'canonical.rtp'
    try:
        assert not call('sim.control_state')['playing'], 'Pause first'
        assert not call('fluid.list_domains')['domains'], 'Use an empty disposable project'
        assert not call('flow_source.list'), 'Use an empty disposable project'
        for name, (_, phase, _) in NAMES.items():
            preset = 'gas' if phase == 'gas' else 'fluid'
            call('fluid.create_domain', name=name, type=preset,
                 domain_min=[-0.2, 0, -0.2], domain_max=[0.2, 0.4, 0.2], voxel_size=0.1)
            created_domains.append(name)
            source = name + '_source'
            call('flow_source.create', name=source, domain=name, phase=phase,
                 source_mode='point', position=[0, 0.2, 0], radius=0.05, enabled=False)
            created_sources.append(source)
        call('project.save', path=str(fixture_path))
        payload = json.loads(fixture_path.read_text(encoding='utf-8'))
        legacy = make_legacy(payload)
        original_shader = next(d for d in domain_rows(legacy) if d['name'] == GAS)['shader']
        fixture_path.write_text(json.dumps(legacy), encoding='utf-8')
        call('project.open', path=str(fixture_path))

        def verify(label):
            call('fluid.step', dt=1 / 60)
            sources = {s['name']: s for s in call('flow_source.list')}
            for name, (_, phase, _) in NAMES.items():
                lanes = call('fluid.get_lanes', domain=name)
                assert lanes['type'] == 'matter' and lanes['measured'], (label, lanes)
                assert lanes['gas_lane_enabled'] == (phase == 'gas'), (label, lanes)
                assert lanes['liquid_lane_enabled'] == (phase == 'liquid'), (label, lanes)
                assert lanes['primary_cells'] > 0 and lanes['secondary_cells'] == 0
                source = sources[name + '_source']
                assert source['phase'] == phase and not source['enabled'], (label, source)

        verify('legacy migration')
        call('project.save', path=str(canonical_path))
        saved = json.loads(canonical_path.read_text(encoding='utf-8'))
        for domain in domain_rows(saved):
            if domain['name'] not in NAMES:
                continue
            assert domain['type'] == 'Matter'
            assert domain['phase_grids']['version'] == 2
        gas = next(d for d in domain_rows(saved) if d['name'] == GAS)
        assert gas['fluid_fog_shader'] == original_shader, 'legacy gas shader lost on migration'
        call('project.open', path=str(canonical_path))
        verify('canonical reopen')
        print('PASS: Gas/Fluid migration, disabled source retarget, single-grid storage, '
              'gas shader preservation, canonical save/open', folder, flush=True)
    finally:
        try:
            for source in reversed(created_sources):
                call('flow_source.remove', name=source)
            for domain in reversed(created_domains):
                call('fluid.remove_domain', domain=domain)
            if created_domains:
                call('timeline.set_frame', frame=0)
        finally:
            client.close()


if __name__ == '__main__':
    main()
