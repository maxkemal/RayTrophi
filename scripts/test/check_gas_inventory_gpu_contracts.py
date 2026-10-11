"""Read-only source contracts; no compilation and no claim of GPU acceptance."""
from pathlib import Path
import ast
import re
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(path):
    return (SOURCE / path).read_text(encoding='utf-8')


def main():
    registry = read('src/Device/SimulationComputeVulkan.cpp')
    build = read('shaders/compile_sim_shaders.bat')
    kernels = [('advect_scalar', 7, 28), ('maccormack_scalar', 7, 28),
               ('conserve', 6, 16), ('clear', 2, 4), ('deposit', 3, 4), ('momentum', 7, 20),
               ('summary', 4, 36), ('surface_exchange', 11, 36), ('surface_commit', 8, 8),
               ('saturation', 11, 112), ('rain_fall', 4, 32)]
    for suffix, bindings, size in kernels:
        name = 'sim_inventory_' + suffix
        pattern = (r'"' + name + r'"\s*,\s*"' + name +
                   r'\.spv"\s*,\s*' + str(bindings) + r'\s*,\s*' + str(size))
        assert re.search(pattern, registry), name
        assert name + '.comp' in build and name + '.spv' in build, name
        shader = read('shaders/' + name + '.comp')
        if suffix not in ('advect_scalar', 'maccormack_scalar'):
            assert 'gl_GlobalInvocationID.y' in shader, name
        code = re.sub(r'//[^\n]*|/\*.*?\*/', '', shader, flags=re.S)
        assert not re.search(r'\b(?:float|double|uint|int|Result)\s+(?:output|input|active)\b', code), name
        assert code.count('{') == code.count('}') and code.count('(') == code.count(')'), name
    gate = registry.split('if (!m_has_shader_float64 &&', 1)[1].split('return false;', 1)[0]
    for suffix in ('conserve', 'summary', 'surface_exchange'):
        assert 'kernel == "sim_inventory_' + suffix + '"' in gate
    for scheme in ('advect', 'maccormack'):
        common = read(f'shaders/sim_grid_{scheme}_scalar.glsl')
        assert '#ifndef INVENTORY_SCALAR' in common
        assert 'gl_GlobalInvocationID.y' in common
        assert '#define INVENTORY_SCALAR' in read(f'shaders/sim_inventory_{scheme}_scalar.comp')
        assert '#define INVENTORY_SCALAR' not in read(f'shaders/sim_grid_{scheme}_scalar.comp')
    host = read('src/Physics/Fluid/GasInventoryGpu.cpp')
    for forbidden in ('downloadBuffer(', 'uploadBuffer(', 'gas_phase_mass_kg', 'gas_phase_energy_j'):
        assert forbidden not in host, forbidden
    assert 'static_assert(sizeof(AdvectionConstants) == 28)' in host
    assert 'static_assert(sizeof(ReductionConstants) == 16)' in host
    assert host.index('std::swap(scratch.fields[0]') > host.index('rc.phase = 2;')
    owner = read('src/Physics/Fluid/GasInventoryDevice.cpp')
    for contract in ('snapshot_download_bytes', 'query_download_bytes', 'summary_valid',
                     'ensureGasInventoryStorage', 'pending.empty()', 'current->grid_upload_bytes = 0'):
        assert contract in owner, contract
    surface = read('src/Physics/Fluid/GasSurfaceExchangeGpu.cpp')
    readback = surface.split('// No gas kg/J grid comes back.', 1)[1]
    assert readback.index('compute.beginTransferBatch()') < readback.index('compute.downloadBuffer(')
    assert readback.index('compute.endTransferBatch()') < readback.index('validateGasSurfaceResults(')
    summary = owner.split('double totals[8]{};', 1)[1]
    assert summary.index('compute.beginTransferBatch()') < summary.index('compute.downloadBuffer(')
    assert summary.index('compute.endTransferBatch()') < summary.index('owner->summary =')
    assert surface.index('validateGasSurfaceResults(plan, results)') < surface.index('command.kernel = "sim_inventory_surface_commit"')
    assert 'evaporating && !ensureGasInventoryStorage' in surface
    driver = read('src/Physics/ParticleSimulation.cpp')
    for contract in ('releaseGasInventoryGpu', 'holdGasInventory', 'snapshotGasInventory',
                     'exchangeGasSurfaceGpuDomains', 'gpu_surface_exchange_ms'):
        assert contract in driver, contract
    assert 'releaseGasInventoryDomainsForBackendChange' in read('src/Physics/ParticleDeviceResidency.cpp')
    assert 'captureGasInventoryHostImages' in read('src/Physics/SimCache.cpp')
    assert 'captureGasInventoryFrame' in read('include/scene_data.h')
    for path in ['src/Api/RtApiParticle.cpp', 'src/Api/RtIpc.cpp', 'src/Api/RtPython.cpp']:
        for key in ('inventory_transport', 'inventory_step_held', 'inventory_error',
                    'inventory_grid_upload_bytes', 'inventory_grid_download_bytes',
                    'gpu_surface_exchange_ms'):
            assert key in read(path), (path, key)
    for suffix in ['vcxproj', 'vcxproj.filters']:
        project = ROOT / f'RayTrophiStudio/RayTrophiStudio.{suffix}'
        ET.parse(project)
        project_text = project.read_text(encoding='utf-8')
        for name in ('GasInventoryGpu', 'GasInventoryDevice', 'GasInventoryDiagnostics',
                     'GasInventoryFrameCapture', 'GasSurfaceExchangeGpu'):
            assert name + '.cpp' in project_text, name
    for name in ('rt_test_gas_inventory_gpu_ipc.py', 'check_gas_inventory_math.py'):
        ast.parse((ROOT / 'scripts/test' / name).read_text(encoding='utf-8'))
    print('PASS GPU inventory ABI/registration/residency/consumer source contracts; no GPU execution')


if __name__ == '__main__':
    main()
