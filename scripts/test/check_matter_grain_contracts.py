"""Non-build contracts for the opt-in dry grain runtime; not numerical GPU acceptance."""
import ast
import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(path):
    return (SOURCE / path).read_text(encoding='utf-8')


def main():
    gpu = read('src/Physics/Fluid/MatterGrainGpu.cpp')
    shader = read('shaders/sim_matter_grain.glsl')
    header = read('include/Fluid/MatterGrain.h')
    assert sorted(map(int, re.findall(r'binding\s*=\s*(\d+)', shader))) == list(range(14))
    push = re.search(r'uniform Constants\s*\{(.*?)\}\s*pc;', shader, re.S).group(1)
    assert len(re.findall(r'\b(?:uvec4|vec4)\s+\w+;', push)) == 6
    assert 'static_assert(sizeof(Constants) == 96)' in gpu
    # Host CFL, history slots and the overflow gate share one contact budget.
    budget = int(re.search(r'kMatterGrainContactBudget = (\d+);', header).group(1))
    assert f'const uint SLOTS = {budget}u;' in shader and f'const float BUDGET = {budget}.0;' in shader
    revision = int(re.search(r'kMatterGrainShaderRevision = (\d+);', header).group(1))
    assert f'const uint REVISION = {revision}u;' in shader
    assert 'diagnostics[1] != kMatterGrainShaderRevision' in gpu
    assert 'if (g_contacts > SLOTS) atomicOr(diagnostics[0],1u);' in shader
    assert '48.0' not in shader, 'old 48-contact bound left in shader'
    assert gpu.index(' CFL exceeds max_substeps') < gpu.index('ensureParticles(compute, runtime')
    assert 'substeps += substeps & 1u;' in gpu
    begin = gpu.index('if (!dispatchMatterGrainStages(')
    end = gpu.index('auto result = p;')
    assert 'downloadBuffer' not in gpu[begin:end]
    assert 'uploadBuffer' not in gpu[begin:end]
    stages = read('include/Fluid/MatterGrainStages.h')
    assert stages.index('sim_matter_grain_clear') < stages.index('sim_matter_grain_hash')
    assert stages.index('sim_matter_grain_hash') < stages.index('sim_matter_grain_step')
    assert '(substeps & 1u)' in stages
    # Fused step: reads bank k&1 and bucket table k%3, writes the other bank,
    # inserts into (k+1)%3 and clears (k+2)%3 (last read in k-1).
    capacity = int(re.search(r'kMatterGrainBucketCapacity = (\d+);', header).group(1))
    assert f'const uint BUCKET = {capacity}u;' in shader
    assert 'uint table = pc.substep.x % 3u;' in shader
    assert 'bucket_counts[((pc.substep.x+2u)%3u)*pc.meta.y+i] = 0u;' in shader
    assert shader.index('bucket_counts[((pc.substep.x+2u)%3u)') < shader.index('if (i >= pc.meta.x) return;')
    assert 'insert(i,p,(pc.substep.x+1u)%3u);' in shader
    assert shader.index('store(i,dst,p,v,w);') < shader.index('insert(i,p,(pc.substep.x+1u)%3u);')
    assert 'else atomicOr(diagnostics[0],4u);' in shader and '(overflow & 4u)' in gpu
    assert 'threads = std::max(constants.count, runtime.buckets);' in gpu
    assert 'threads = kMatterGrainBucketTables * runtime.buckets;' in gpu
    assert 'heads[' not in shader and 'links[' not in shader
    assert 'g_history_valid = history_owner[bank*pc.meta.x+i] == ids[i];' in shader
    assert 'tangential = kt > 0.0 && coulomb > 0.0 ? -(ft-damping)/kt : vec3(0.0);' in shader
    assert 'rolling = limit > 0.0 ? -(rt-roll_damping)/kr : vec3(0.0);' in shader
    assert '2 * kMatterGrainContactBudget * 2 * 4 * sizeof(uint32_t)' in gpu
    assert '2.8125 * mu_r * mu_r * k' in gpu
    assert 'nodes[stack[--size]]' in shader and 'patch_count == 8u' in shader
    assert 'manifold_count == 4u' in shader
    assert 'previous->collider_fingerprint != fingerprint' in gpu
    assert 'all(equal(cell(pj),wanted))' in shader
    diagnostics = read('src/Physics/Fluid/MatterGrainDiagnostics.cpp')
    assert '{"pile", matterGrainPileProfile(centres, params.radius_m)}' in diagnostics
    assert 'cross(arm,ft)' in shader
    assert 'affines[b+1]=w.z' in shader
    assert 'BUDGET*dt*inverse_pair_inertia' in shader
    assert 'twist_limit' in shader and 'twist_torque*n' in shader
    assert 'pore_water_mass_kg[i] != 0.0f' in gpu
    assert gpu.index('if (!ok || overflow)') < gpu.index('p = std::move(result)')
    geometry = read('src/Physics/Fluid/MatterGrainGeometry.cpp')
    assert 'TriangleMesh*' in geometry and 'get_attribute_data<Vec3>' in geometry
    assert 'shared_ptr<Triangle>' not in geometry and 'dynamic_cast<const Triangle*' not in geometry
    step = read('src/Physics/Fluid/FluidDomainStep.inl')
    assert 'grain_collider_mesh_resolver_' in step
    assert 'if (fluid_params.grain.enabled)' in step
    assert 'runMatterGrainStep(' in step
    assert 'grain candidate' in read('src/Physics/Fluid/MatterGrainStep.inl')
    births = read('src/Physics/Fluid/MatterDomainSources.inl')
    assert births.index('grain_filter.accept(spawn_pos)') < births.index('state.particles.emit(')
    assert 'nullptr, nullptr, grain_rest_mass, emit_model);' in births
    assert 'ensureMatterGrainRestMasses(' in step
    assert 'source.fluid_emit_sample_serial += attempts;' in births
    assert 'source.fluid_emit_accumulator += static_cast<float>(emit_count -' in births
    birth_filter = read('src/Physics/Fluid/MatterGrainBirth.cpp')
    assert 'cells_[own].push_back(p);' in birth_filter
    assert 'p.y < low_.y + radius_' in birth_filter
    api = read('src/Api/RtApiMatterGrain.cpp')
    assert api.index('patchMatterGrainParams(') < api.index('d.fluid_params.grain = candidate')
    for method in ('grain_settings', 'set_grain_settings'):
        assert f'fluid.{method}' in read('src/Api/RtIpcMatterModels.cpp')
        assert f'fluid.def("{method}"' in read('src/Api/RtPythonMatterModels.cpp')
    assert 'matterGrainSettings(' in read('src/UI/MatterGrainControls.cpp')
    serializer = read('src/Utils/SceneSerializer.cpp')
    assert 'matterGrainParamsToJson' in serializer and 'matterGrainParamsFromJson' in serializer
    registry = read('src/Device/SimulationComputeVulkan.cpp')
    for stage in ('clear', 'hash', 'step'):
        kernel = 'sim_matter_grain_' + stage
        assert f'"{kernel}.spv", 14, 96' in registry
        assert kernel in read('shaders/compile_sim_shaders.bat')
        assert f'GRAIN_{stage.upper()}' in read(f'shaders/{kernel}.comp')
    for dead in ('contact', 'integrate', 'integrate_hash'):
        assert not (SOURCE / f'shaders/sim_matter_grain_{dead}.comp').exists()
        assert f'sim_matter_grain_{dead}' not in registry
    assert 'm_descriptorCache.clear();' in registry and 'm_recordedDispatches >= MAX_DESC_SETS' in registry
    for project in ('RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters'):
        ET.parse(ROOT / 'RayTrophiStudio' / project)
    ast.parse((ROOT / 'scripts/test/rt_h1_grain_runtime_ipc.py').read_text(encoding='utf-8'))
    print('PASS dry grain source: fused ping-pong ABI/order, bucket rotation, contact budget, history, grain mass, pile, UI/API/save wiring')


if __name__ == '__main__':
    main()
