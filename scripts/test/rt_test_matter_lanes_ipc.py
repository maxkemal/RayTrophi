"""T6 lane acceptance. Empty paused scene; no application launch or save.

Creates only T_ domains/sources and removes them in finally. Gas/liquid remain
creation presets, with one canonical Matter identity and no disabled grid.
"""
from rt_ipc import RtIpc


def main():
    client = RtIpc()
    call = client.call
    domains, sources, checks = [], [], {}
    assert not call('sim.control_state')['playing'], 'Pause first'
    assert not call('fluid.list_domains')['domains'], 'Use an empty scene'
    try:
        rows = {}
        for preset in ('gas', 'fluid', 'matter'):
            name = 'T_Lanes_' + preset
            call('fluid.create_domain', name=name, type=preset,
                 domain_min=[-.2, 0, -.2], domain_max=[.2, .4, .2], voxel_size=.1)
            domains.append(name)
            rows[preset] = call('fluid.get_lanes', domain=name)
        checks['canonical_matter'] = all(r['type'] == 'matter' for r in rows.values())
        checks['preset_lanes'] = (
            rows['gas']['gas_lane_enabled'] and not rows['gas']['liquid_lane_enabled']
            and rows['fluid']['liquid_lane_enabled'] and not rows['fluid']['gas_lane_enabled']
            and rows['matter']['gas_lane_enabled'] and rows['matter']['liquid_lane_enabled'])
        # Synchronize before interpreting actual storage counts.
        call('fluid.step', dt=1 / 60)
        measured = [call('fluid.get_lanes', domain=name) for name in domains]
        checks['disabled_grid_absent'] = all(r['measured'] for r in measured) and (
            measured[0]['primary_cells'] > 0 and measured[0]['secondary_cells'] == 0
            and measured[1]['primary_cells'] > 0 and measured[1]['secondary_cells'] == 0
            and measured[2]['secondary_cells'] > 0)
        name = 'T_Lanes_matter'
        before = call('fluid.get_lanes', domain=name)
        ok, _ = client.try_call('fluid.set_lanes', domain=name,
                                gas_lane_enabled=False, liquid_lane_enabled=False)
        checks['empty_lane_set_refused_unchanged'] = not ok and (
            call('fluid.get_lanes', domain=name) == before)
        source = 'T_Lanes_source'
        call('flow_source.create', name=source, domain=name, phase='liquid',
             source_mode='point', fluid_substance='Water', position=[0, .2, 0], radius=.05)
        sources.append(source)
        ok, message = client.try_call('fluid.set_lanes', domain=name,
                                      gas_lane_enabled=True, liquid_lane_enabled=False)
        checks['source_lane_refused_unchanged'] = not ok and 'source' in message and (
            call('fluid.get_lanes', domain=name) == before)
        call('flow_source.remove', name=source)
        sources.remove(source)
        switched = call('fluid.set_lanes', domain=name,
                        gas_lane_enabled=True, liquid_lane_enabled=False)
        checks['empty_domain_switches'] = switched['gas_lane_enabled'] and (
            not switched['liquid_lane_enabled'] and switched['secondary_cells'] == 0)
        auto_name = 'T_Lanes_auto'
        auto = call('flow_source.create', name=auto_name, domain='T_Lanes_fluid',
                    source_mode='point', fluid_substance='Water', enabled=False)
        sources.append(auto_name)
        checks['auto_source_uses_enabled_lane'] = auto['phase'] == 'liquid'
        for key, passed in checks.items():
            print('PASS' if passed else 'FAIL', key, flush=True)
        if not all(checks.values()):
            raise SystemExit(1)
    finally:
        for source in reversed(sources):
            call('flow_source.remove', name=source)
        for name in reversed(domains):
            call('fluid.remove_domain', domain=name)
        client.close()


if __name__ == '__main__':
    main()
