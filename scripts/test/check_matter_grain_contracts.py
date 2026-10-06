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
    assert sorted(map(int, re.findall(r'binding\s*=\s*(\d+)', shader))) == list(range(15))
    push = re.search(r'uniform Constants\s*\{(.*?)\}\s*pc;', shader, re.S).group(1)
    assert len(re.findall(r'\b(?:uvec4|vec4)\s+\w+;', push)) == 7
    assert 'static_assert(sizeof(Constants) == 112)' in gpu
    # Host CFL, history slots and the overflow gate share one contact budget.
    budget = int(re.search(r'kMatterGrainContactBudget = (\d+);', header).group(1))
    assert f'const uint SLOTS = {budget}u;' in shader and f'const float BUDGET = {budget}.0;' in shader
    revision = int(re.search(r'kMatterGrainShaderRevision = (\d+);', header).group(1))
    assert f'const uint REVISION = {revision}u;' in shader
    assert 'diagnostics[1] != kMatterGrainShaderRevision' in gpu
    assert 'g_contacts > SLOTS) atomicOr(diagnostics[0],1u);' in shader
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
    step_section = shader[shader.index('#elif defined(GRAIN_STEP)'):]
    assert step_section.index('bucket_counts[((pc.substep.x+2u)%3u)') < step_section.index('if (i >= pc.meta.x) return;')
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
    assert 'runtime.collider_fingerprint != fingerprint' in gpu
    assert 'all(equal(cell(pj),wanted))' in shader
    diagnostics = read('src/Physics/Fluid/MatterGrainDiagnostics.cpp')
    assert '{"pile", matterGrainPileProfile(centres, params.radius_m)}' in diagnostics
    assert 'cross(arm,ft)' in shader
    assert 'affines[b+1]=w.z' in shader
    assert 'BUDGET*dt*inverse_pair_inertia' in shader
    assert 'twist_limit' in shader and 'twist_torque*n' in shader
    assert '(!params.wet_grains && water != 0.0f)' in gpu
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
        assert f'"{kernel}.spv", 15, 112' in registry
        assert kernel in read('shaders/compile_sim_shaders.bat')
        assert f'GRAIN_{stage.upper()}' in read(f'shaders/{kernel}.comp')
    for dead in ('contact', 'integrate', 'integrate_hash'):
        assert not (SOURCE / f'shaders/sim_matter_grain_{dead}.comp').exists()
        assert f'sim_matter_grain_{dead}' not in registry
    assert 'm_descriptorCache.clear();' in registry and 'm_recordedDispatches >= MAX_DESC_SETS' in registry
    for project in ('RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters'):
        ET.parse(ROOT / 'RayTrophiStudio' / project)
    runtime_test = (ROOT / 'scripts/test/rt_h1_grain_runtime_ipc.py').read_text(encoding='utf-8')
    ast.parse(runtime_test)
    # Per-carrier transport owner: grains have their own bank-0 buffers, the
    # liquid lane steps the liquid subset, coupling returns every impulse.
    assert 'buffers.fluid_positions' not in gpu, 'grain step must not alias the liquid particle buffers'
    assert 'static_assert(sizeof(handles) / sizeof(handles[0]) == 15);' in gpu
    assert 'runtime.coupling};' in gpu
    assert 'layout(std430, binding = 14) buffer Coupling { vec4 coupling[]; };' in shader
    assert 'vec3 relative = (drag.xyz-v)/(1.0+dt*drag.w*(im+1.0/M));' in shader
    assert 'vec3 settled = (m*v+M*(drag.xyz-relative))/(m+M);' in shader
    assert shader.index('vec4 lift = coupling[3u*i+1u];') < shader.index('p += dt*v;')
    coordinator = read('src/Physics/Fluid/MatterGrainStep.inl')
    order = ['partitionMatterGrainOwners(', 'runGpuFluidParticleIntegrateForces(',
             'runMatterGpuStep(', 'buildMatterGrainLiquidField(', 'prepareMatterGrainCoupling(',
             'stepMatterGrainGpu(', 'applyMatterGrainLiquidReaction(', 'mergeMatterGrainOwners(']
    positions = [coordinator.index(token) for token in order]
    assert positions == sorted(positions), 'coexistence stage order'
    assert 'std::swap(state.particles, liquid);' in coordinator and '~SwapBack()' in coordinator
    coupling = read('src/Physics/Fluid/MatterGrainCoupling.cpp')
    assert 'std::stable_sort(grains.begin(), grains.end()' in coupling
    assert 'impulse[0][c] -= s * gain.x;' in coupling
    assert 'w[n] * sphere / grains_in_cell * f.mass[c[n]]' in coupling
    assert 'std::max(f.solid[c[n]], sphere)' in coupling
    for project in ('RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters'):
        text = (ROOT / 'RayTrophiStudio' / project).read_text(encoding='utf-8')
        assert 'MatterGrainCoupling.cpp' in text and 'MatterGrainCoupling.h' in text
    assert 'MatterConstitutiveModel::Granular' in birth_filter, 'liquid parcels must not block grain births'
    assert 'emit_model == RayTrophiSim::Fluid::MatterConstitutiveModel::Granular;' in births
    params_src = read('src/Physics/Fluid/MatterGrainParams.cpp')
    assert '"fluid_coupling"' in params_src and '"drag_viscosity_pa_s"' in params_src
    ui = read('src/UI/MatterGrainControls.cpp')
    assert 'p.fluid_coupling' in ui and 'p.drag_viscosity_pa_s' in ui
    bridge = read('src/Physics/ParticleRenderBridge.cpp')
    assert 'const float d = grain ? diam : liquid_diam;' in bridge
    assert '--coexist-only' in runtime_test
    # B9a: device history is valid only for the state it was published with.
    assert 'std::memcmp(&was, &p.position[i], sizeof(Vec3)) != 0' in gpu
    assert gpu.index('reset_reason = "host_state_changed";') < gpu.index('const bool history_reset = runtime.history_fresh;')
    assert gpu.index('p = std::move(result);') < gpu.index('runtime.published_positions = p.position;')
    # B8: history follows identity across any order; grains sorted by cell.
    assert gpu.index('"sim_matter_grain_permute", "sim_matter_grain_permute_copy"') < gpu.index('if (!dispatchMatterGrainStages(')
    assert 'GRAIN_PERMUTE_COPY' in read('shaders/sim_matter_grain_permute_copy.comp')
    for kernel in ('sim_matter_grain_permute', 'sim_matter_grain_permute_copy'):
        assert f'"{kernel}.spv", 15, 112' in registry and kernel in read('shaders/compile_sim_shaders.bat')
    assert 'orderMatterGrainsByCell(' in coordinator
    # B4: residency + transfer accounting.
    assert gpu.index('const bool resident =') < gpu.index('if (!resident) {')
    assert 'runtime.published_masses = std::move(masses);' in gpu
    assert "resident_tail_samples" in runtime_test
    assert '--history-only' in runtime_test
    contact = read('src/Physics/Fluid/GranularContact.cpp')
    assert '2.25f * params.rolling_friction * params.rolling_friction' in contact
    assert 'stop_torque' not in contact, 'CPU reference still has the kinetic-only rolling torque'
    # B5: porous projection + pressure force.
    assert '"sim_fluid_divergence_porous.spv",    11, 52' in registry
    porous = read('shaders/sim_fluid_divergence_porous.comp')
    assert 'return porous(' in porous
    pressure_src = read('src/Physics/Fluid/FluidGpuPressure.inl')
    assert 'vulkan_variational && gpu_buffers.porous_solid_velocity' in pressure_src
    assert 'sim_fluid_divergence_porous' in read('shaders/compile_sim_shaders.bat')
    assert coordinator.index('applyMatterGrainPorosity(') < coordinator.index('runMatterGpuStep(')
    # B5 pressure force comes from the liquid's measured acceleration, not a
    # pressure buffer: the liquid lane integrates gravity once per frame and
    # projects in substeps, so no single pressure field carries the load.
    assert 'downloadBuffer(buffers.pressure' not in coordinator
    assert 'frame_pressure' not in coordinator
    assert coordinator.index('liquid_start_velocity = exclude') < coordinator.index('runMatterGpuStep(')
    assert coordinator.index('runMatterGpuStep(') < coordinator.index('applyMatterGrainLiquidAccelerationForce(')
    mixed = read('src/Physics/Fluid/MatterGpuStep.inl')
    porous_kernel = read('shaders/sim_fluid_divergence_porous.comp')
    # Density correction must target the pore volume, or it refills the pores.
    assert 'max(count - eps * ppc, 0.0) - max(count - ppc, 0.0)' in porous_kernel
    coupling = read('src/Physics/Fluid/MatterGrainCoupling.cpp')
    assert 'f.volume[c[n]] / (.5 * pores)' in coupling, 'submersion must be occupancy, not volume'
    assert 'has_granular ? static_cast<double>(elastic.required_substeps)' in mixed
    assert coordinator.index('prepareMatterGrainCoupling(') < coordinator.index('applyMatterGrainLiquidAccelerationForce(')
    assert '~RestorePorosity()' in coordinator and 'restoreMatterGrainPorosity(grid, backup);' in coordinator
    assert '"volume_exclusion"' in params_src and 'p.volume_exclusion' in ui
    assert '--porous-only' in runtime_test
    # B6: wet grains.
    assert 'return ivec3(floor((p-pc.low_radius.xyz)/pc.wet.x));' in shader
    assert 'void bridge(uint i, uint j, float d, vec3 n, float r, inout vec3 force)' in shader
    assert 'if (d < 2.0*r+pc.wet.z && d > 1e-9) bridge(i,j,d,separation/d,r,f);' in shader
    assert 'constants.wet[0] = 2.0f * params.radius_m + skin;' in gpu
    assert 'coupling_rows[12 * i + 11] = p.pore_water_mass_kg[i] / 1000.0f;' in gpu
    assert coordinator.index('applyMatterGrainLiquidReaction(') < coordinator.index('exchangeMatterGrainWater(')
    assert coordinator.index('exchangeMatterGrainWater(') < coordinator.index('mergeMatterGrainOwners(state.particles')
    assert 'initMatterGrainBirthWater(' in births and '"birth_saturation"' in params_src
    assert 'p.wet_grains' in ui and '--wet-only' in runtime_test
    # B7: XPBD candidate in the same runtime.
    assert 'bool xpbd = pc.wet.w > .5;' in shader and 'void xpbdConstraint(' in shader
    assert 'g_alpha = 1.0/(pc.high_stiffness.w*h*h);' in shader
    assert 'if (!xpbd && g_contacts > SLOTS) atomicOr(diagnostics[0],1u);' in shader
    assert 'constants.wet[3] = xpbd ? 1.0f : 0.0f;' in gpu
    assert '"solver_kind"' in params_src and 'p.solver_kind' in ui and '--xpbd-compare' in runtime_test
    print('PASS dry grain source: fused ping-pong ABI/order, bucket rotation, contact budget, history, grain mass, pile, per-carrier owner + liquid coupling, UI/API/save wiring')


if __name__ == '__main__':
    main()
