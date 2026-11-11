"""Static integration/ABI checks only; not numerical or live physics acceptance."""
from pathlib import Path
import hashlib
import json
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(relative):
    return (SOURCE / relative).read_text(encoding='utf-8')


def main():
    projects = [ET.parse(ROOT / 'RayTrophiStudio' / name)
                for name in ('RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters')]
    modules = list((SOURCE / 'include/Fluid').glob('DemBond*.h'))
    modules += list((SOURCE / 'src/Physics/Fluid').glob('DemBond*.cpp'))
    modules += [SOURCE / path for path in ('include/Api/DemBondApi.h',
                'src/Api/RtApiDemBonds.cpp', 'src/UI/DemBondControls.h',
                'src/UI/DemBondControls.cpp')]
    for path in modules:
        relative = 'source/' + path.relative_to(SOURCE).as_posix()
        for document in projects:
            entries = [e.attrib.get('Include', '').replace('\\', '/')
                       for e in document.getroot().iter()]
            assert entries.count(relative) == 1, relative
        assert len(path.read_text('utf-8').splitlines()) < 2000, path

    registry = read('src/Device/SimulationComputeVulkan.cpp')
    bulk = read('shaders/compile_sim_shaders.bat')
    shader = read('shaders/sim_dem_bond.glsl')
    assert sorted(map(int, re.findall(r'binding\s*=\s*(\d+)', shader))) == list(range(10))
    assert 'uvec4 meta;' in shader and 'vec4 step;' in shader
    for name in ('evaluate', 'kick', 'orient'):
        kernel = 'sim_dem_bond_' + name
        assert re.search(r'"' + kernel + r'"\s*,\s*"' + kernel +
                         r'\.spv"\s*,\s*10\s*,\s*32', registry), kernel
        assert kernel in bulk
        source_binary = SOURCE / 'shaders' / (kernel + '.spv')
        deployed = ROOT / 'x64/Release/shaders' / (kernel + '.spv')
        assert source_binary.is_file() and deployed.is_file(), kernel
        assert source_binary.read_bytes()[:4] == bytes.fromhex('03022307'), kernel
        assert hashlib.sha256(source_binary.read_bytes()).digest() == \
            hashlib.sha256(deployed.read_bytes()).digest(), kernel
    # The concurrent agent owns these rows. Never "fix" them back to old counts.
    assert re.search(r'"sim_fluid_divergence_porous\.spv"\s*,\s*12\s*,\s*52', registry)
    assert re.search(r'"sim_sparse_mac_divergence_porous\.spv"\s*,\s*14\s*,\s*52', registry)

    gpu = read('src/Physics/Fluid/DemBondGpu.cpp')
    driver = read('src/Physics/Fluid/MatterGrainGpu.cpp')
    assert 'sizeof(EdgeRow) == 96' in gpu and 'sizeof(Constants) == 32' in gpu
    assert 'runtime.bond_buffers' in gpu and 'working_bytes += capacity' in gpu
    before = driver.index('bonds.before(')
    assert before < driver.index('!mpm_contact->step(', before)
    assert before < driver.index('compute.dispatch(command)', before) < driver.index('bonds.after(')
    assert driver.index('bonds.readback(error)') < driver.index('p.position = std::move(new_position)')
    assert '!bonds.constrains_motion && wall.adhesion_pa' in driver
    assert 'bonds.maximum_dt' in driver and 'fracture stiffness is never softened' in driver
    assert 'std::hypot(1.0 / bonds.maximum_dt' in driver
    assert 'bonds.working_bytes' in driver
    readback = gpu[gpu.index('bool DemBondGpuFrame::readback'):]
    assert 'downloadBuffer' not in gpu[:gpu.index('bool DemBondGpuFrame::readback')]
    assert 'compute.synchronize()' in readback
    assert 'identities != p.particle_id' in gpu
    assert 'canonical_edge_count != p.dem_bonds.edges.size()' in gpu
    assert 'edge_indices.push_back(canonical)' in gpu
    assert 'next.edges[edge_indices[e]]' in gpu
    assert 'if (!constrains_motion)' in gpu
    assert 'bonds.extraDispatchesPerStep()' in driver
    assert 'dot(poses[i], poses[i]) < 0.5' in shader

    particles = read('include/Fluid/FluidParticles.h')
    assert 'DemBond::Network dem_bonds' in particles
    assert 'dem_bonds = {};' in particles
    coordinator = read('src/Physics/Fluid/MatterGrainStep.inl')
    assert 'grains.dem_bonds = state.particles.dem_bonds' in coordinator
    assert 'state.particles.dem_bonds = std::move(grains.dem_bonds)' in coordinator
    assert 'solid_weight_source_valid = false' in coordinator
    assert 'DemBond::retireNonDem' in read('src/Physics/Fluid/FluidDomainStep.inl')
    cache = read('src/Physics/SimCache.cpp')
    assert 'DemBond::writeCache' in cache and 'DemBond::readCache' in cache
    assert 'kVersion = 20u' in read('include/SimCache.h')
    binary = read('src/Physics/Fluid/DemBondCache.cpp')
    assert 'sizeof(DiskEdge) == 176' in binary and 'sizeof(DiskNode) == 48' in binary
    assert 'to_cbor' not in binary and 'encode(network)' not in binary
    assert 'validate(next, error)' in binary
    assert 'version == 19u' in read('include/SimCache.h')
    assert 'dem_bonds.edges' in read('include/SimFrameCacheMemory.h')
    network = read('src/Physics/Fluid/DemBondNetwork.cpp')
    assert 'next.edges.size() >= maximum_edges' in network
    assert 'maximum_degree' in network and 'byte_budget' in network
    assert 'retired_energy_j += edge.elastic_energy_j' in network
    seed = read('src/Physics/Fluid/MatterSeedSources.cpp')
    assert 'DemBond::seedJson(source.bond_body)' in seed
    assert 'DemBond::parseSeed(value, candidate.bond_body, error)' in seed
    assert 'DemBond::bindSeed(initial, begin' in seed
    assert 'matterSeedSourceToJson(source).dump()' in seed
    assert 'Seed Settle off' in seed
    assert 'editDemBondSeed' in read('src/UI/MatterSeedSourcesUI.cpp')
    sync = read('src/Physics/Fluid/MatterDomainSynchronization.inl')
    apply_block = sync[sync.index('if (applied && i < grid_domain_compute_buffers_.size())'):]
    assert apply_block.index('invalidateMatterGrainAfterReset(') < \
        apply_block.index('invalidateDeviceCopies()')
    assert 'rtapi::demBodies' in read('src/UI/DemBondControls.cpp')
    assert 'rtapi::demBodies' in read('src/Api/RtIpcMatterModels.cpp')
    assert 'rtapi::demBodies' in read('src/Api/RtPythonMatterModels.cpp')
    api = read('src/Api/RtApiDemBonds.cpp')
    assert 'ui.timeline.isPlaying()' in api and 'invalidateSimCacheAfterCurrentFrame' in api
    assert 'captureSimFrame(frame, true)' in api
    assert 'force_velocity_keyframe' in read('include/scene_data.h')
    overlay = json.loads((ROOT / 'scripts/ipc_descriptor_overlay.json').read_text('utf-8'))
    assert overlay['fluid.dem_bodies']['params']['request']['type'] == 'object'
    print('PASS: DEM bond source integration, canonical/cache lifecycle, seed persistence, '
          'shared UI/Python/IPC, ABI/deployed shader hashes; physics tests pending')


if __name__ == '__main__':
    main()
