"""Non-build T6 audit: core routing, storage, migration, bindings and registration."""
from pathlib import Path
import ast
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(path):
    return (SOURCE / path).read_text(encoding='utf-8-sig')


def check_script_filters():
    for path in (ROOT / 'scripts/test').glob('*.py'):
        if path.name == 'rt_test_matter_legacy_roundtrip_ipc.py':
            continue  # Deliberately asserts the synthetic pre-T6 file schema.
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            text = ast.unparse(node)
            reads_type = "['type']" in text or ".get('type'" in text
            old_identity = any(repr(value) in text for value in ('gas', 'fluid', 'Gas', 'Fluid'))
            assert not (reads_type and old_identity), (path, node.lineno, text)


def main():
    check_script_filters()
    core = read('src/Physics/SimulationDomainLanes.cpp')
    assert 'domain.gas_lane_enabled' in core and 'domain.liquid_lane_enabled' in core
    assert core.index('if (has_live_state)') < core.index('domain.gas_lane_enabled = gas;')
    assert 'source.phase = liquid ?' in core
    assert 'domain.fluid_fog_shader = domain.shader;' in core
    assert 'domain.shader = domain.fluid_fog_shader;' in core
    assert 'unknown saved simulation domain type: ' in core
    assert 'parseSavedSimulationDomainType(value)' in read('src/Core/ProjectManager.cpp')
    assert 'parseSavedSimulationDomainType(j.at("type").get<int>())' in read(
        'src/Utils/SceneSerializer.cpp')
    manager_load = read('src/Core/ProjectManager.cpp').split(
        'void ProjectManager::deserializeParticleSimulation(', 1)[1]
    assert manager_load.index('addFlowSource(parseFlowSource(') < manager_load.index(
        'synchronizeGridDomainsNow();')
    snapshot_load = read('src/Utils/SceneSerializer.cpp').split(
        'system.runtime->addGridDomain(jsonToDomain(dj));', 1)[1]
    assert snapshot_load.index('addFlowSource(jsonToFlowSource(') < snapshot_load.index(
        'synchronizeGridDomainsNow();')
    assert 'state.type != SimulationDomainType::Matter' in core
    assert 'validateMatterCacheLane(states[i], domains[i], error)' in core
    assert 'legacy Gas/Fluid or changed Matter lanes require a rebake' in core
    assert 'if (available(liquid, liquid))' in core
    types = read('include/ParticleSimulation.h')
    assert 'LegacyGas = 0' in types and 'LegacyFluid = 1' in types
    assert 'simulationDomainHasGas(SimulationDomainType type)' not in types
    for path in SOURCE.rglob('*'):
        if path.suffix.lower() not in (
                '.h', '.hpp', '.hxx', '.hh', '.cpp', '.cc', '.cxx', '.inl', '.ipp', '.tpp'):
            continue
        text = path.read_bytes()
        assert not re.search(rb'SimulationDomainType::(?:Gas|Fluid)\b', text), path
        assert not re.search(rb'simulationDomainHas(?:Gas|Liquid)\([^\r\n]*\.type\)',
                             text), path
    grids = read('include/Fluid/MatterPhaseGrid.h')
    assert 'simulationDomainHasBothPhases(state) && !state.matter_liquid_active' in grids
    assert '!simulationDomainHasBothPhases(state) || state.matter_liquid_active' in grids
    storage = read('src/Physics/Fluid/MatterPhaseConfig.cpp')
    assert 'simulationDomainHasBothPhases(domain) ? layouts.liquid : PhaseLayout{}' in storage
    assert '{"version", 2}' in storage
    assert 'grids.at("gas_lane_enabled").get<bool>()' in storage
    sync = read('src/Physics/Fluid/MatterDomainSynchronization.inl')
    assert 'canonicalizeMatterDomain(domain, flow_sources_, static_cast<int>(i))' in sync
    step = read('src/Physics/ParticleSimulation.cpp')
    assert 'if (!simulationDomainHasGas(state))' in step
    assert 'validateMatterCacheLanes(states, grid_domains_, lane_error)' in step
    assert 'fluid_gas_owner[fluid_index] != static_cast<int>(i)' in step
    assert 'const bool air_through_gas = fluid_gas_owner[i] >= 0;' in read(
        'src/Physics/Fluid/FluidDomainStep.inl')
    cache = read('src/Physics/SimCache.cpp')
    assert 'legacy Gas/Fluid cache rejected: rebake for Matter lanes' in cache
    assert 'writePod(os, static_cast<uint8_t>(simulationDomainHasGas(d)))' in cache
    bridge = read('include/scene_data.h')
    ram = bridge[bridge.index('bool restoreSimFrame(int frame,'):]
    assert ram.index('MatterCacheRestore::validateRamFrame(') < ram.index('restoreSoftFrame(frame)')
    disk = bridge[bridge.index('bool restoreSimFrameFromDisk('):]
    assert disk.index('MatterCacheRestore::stageDiskFrame(') < disk.index('setGridDomainStates(')
    assert 'sim_cache_valid_ = false;' in disk[:disk.index('setGridDomainStates(')]
    staged = read('include/Fluid/MatterCacheRestore.h')
    assert staged.index('validateMatterCacheLanes(loaded.domains') < staged.index('output = std::move(candidate)')
    assert '!RayTrophiSim::simulationDomainHasLiquid(state) &&' in bridge
    assert 'dense_gpu_view.valid()' in bridge
    api = read('src/Api/RtApiMatterLanes.cpp')
    assert 'RayTrophiSim::setMatterDomainLanes(domain, runtime.flowSources()' in api
    assert 'runtime.synchronizeGridDomainsNow();' in api
    assert 'runtime.resetGridDomainStates();' not in api
    assert 'rtapi::setMatterDomainLanes' in read('src/UI/MatterLaneControls.cpp')
    assert 'rtapi::setMatterDomainLanes' in read('src/Api/RtPythonMatterLanes.cpp')
    assert 'rtapi::setMatterDomainLanes' in read('src/Api/RtIpcMatterLanes.cpp')
    assert 'fluid.get_lanes' in read('src/Api/RtIpcSecurity.cpp')
    for name in ('GranularVirtualScene.cpp', 'GranularVirtualRepresentation.cpp',
                 'GranularVirtualSurface.cpp'):
        assert not (SOURCE / 'src/Physics/Fluid' / name).exists(), name
    for name in ('RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters'):
        document = ET.parse(ROOT / 'RayTrophiStudio' / name)
        entries = {node.attrib['Include'].replace('\\', '/')
                   for node in document.iter() if 'Include' in node.attrib}
        for path in ('src/Physics/SimulationDomainLanes.cpp', 'src/Api/RtApiMatterLanes.cpp',
                     'src/Api/RtIpcMatterLanes.cpp', 'src/Api/RtPythonMatterLanes.cpp',
                     'src/UI/MatterLaneControls.cpp'):
            assert 'source/' + path in entries, (name, path)
    overlay = json.loads((ROOT / 'scripts/ipc_descriptor_overlay.json').read_text('utf-8'))
    for method in ('fluid.get_lanes', 'fluid.set_lanes'):
        assert overlay[method]['summary']
    ast.parse((ROOT / 'scripts/test/rt_test_matter_lanes_ipc.py').read_text('utf-8'))
    print('PASS: Matter lanes, single-grid storage, legacy migration/cache refusal, UI/API/IPC')


if __name__ == '__main__':
    main()
