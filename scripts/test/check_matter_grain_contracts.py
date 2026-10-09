"""Non-build contracts for the opt-in dry grain runtime; not numerical GPU acceptance."""
import ast
import re
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT = next(parent for parent in Path(__file__).resolve().parents
            if (parent / 'RayTrophiStudio/source').is_dir())
SOURCE = ROOT / 'RayTrophiStudio/source'


def read(path):
    return (SOURCE / path).read_text(encoding='utf-8')


def main():
    gpu = read('src/Physics/Fluid/MatterGrainGpu.cpp')
    shader = read('shaders/sim_matter_grain.glsl')
    header = read('include/Fluid/MatterGrain.h')
    bindings = sorted(map(int, re.findall(r'binding\s*=\s*(\d+)', shader)))
    assert bindings == list(range(17))
    # Host handle array and every grain kernel table entry carry the same count.
    assert 'static_assert(kGrainBindings == 17);' in gpu
    table = read('src/Device/SimulationComputeVulkan.cpp')
    entries = re.findall(r'\{ "sim_matter_grain_\w+", "sim_matter_grain_\w+\.spv", (\d+), 128 \}', table)
    assert len(entries) == 4 and set(entries) == {'17'}, entries
    push = re.search(r'uniform Constants\s*\{(.*?)\}\s*pc;', shader, re.S).group(1)
    assert len(re.findall(r'\b(?:uvec4|vec4)\s+\w+;', push)) == 8
    assert 'static_assert(sizeof(Constants) == 128)' in gpu
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
    begin = gpu.index('const auto dispatch_stage =')
    end = gpu.index('std::vector<Vec3> new_position(count), new_velocity(count);')
    assert 'auto result = p;' not in gpu, 'full FluidParticles copy at publication is back'
    assert 'downloadBuffer' not in gpu[begin:end]
    assert 'uploadBuffer' not in gpu[begin:end]
    stages = read('include/Fluid/MatterGrainStages.h')
    # Substep group: conditional Verlet-list rebuild, then the fused step.
    group = stages[stages.index('bool dispatchMatterGrainSubstep('):stages.index('bool dispatchMatterGrainStages(')]
    assert group.index('sim_matter_grain_list_clear') < group.index('sim_matter_grain_hash') <         group.index('sim_matter_grain_list_build') < group.index('sim_matter_grain_step')
    frame = stages[stages.index('bool dispatchMatterGrainStages('):]
    assert 'sim_matter_grain_clear' not in frame and 'dispatchMatterGrainSubstep(step, dispatch)' in frame
    assert '(substeps & 1u)' in stages
    assert 'dispatchMatterGrainSubstep(index, dispatch_stage)' in gpu, 'common clock skips the list rebuild'
    capacity = int(re.search(r'kMatterGrainBucketCapacity = (\d+);', header).group(1))
    assert f'const uint BUCKET = {capacity}u;' in shader
    list_capacity = int(re.search(r'kMatterGrainListCapacity = (\d+);', header).group(1))
    assert f'const uint LIST = {list_capacity}u;' in shader
    assert 'kMatterGrainBucketTables = 1;' in header
    # Rebuild trigger: forced on substep 0, raised by the step for k+1, word k+2
    # zeroed by list_clear of k (last read in k-1, next written in k+1).
    assert 'diagnostics[12] = 1u;' in gpu
    assert 'if (i == 0u) diagnostics[flagWord(pc.substep.x+2u)] = 0u;' in shader
    assert 'uint next_flag = flagWord(pc.substep.x+1u);' in shader
    assert 'float half_skin = .5*(pc.wet.x-2.0*r-pc.wet.z);' in shader
    assert 'kMatterGrainListSkinRadii * params.radius_m' in gpu
    step_section = shader[shader.index('#elif defined(GRAIN_STEP)'):]
    assert 'bucket_counts' not in step_section, 'the step reads neighbours from the list only'
    assert shader.index('store(i,dst,p,v,w);') < shader.index('uint next_flag = flagWord(pc.substep.x+1u);')
    assert 'else atomicOr(diagnostics[0],4u);' in shader and '(overflow & 4u)' in gpu
    assert 'atomicOr(diagnostics[0],8u)' in shader and '(overflow & 8u)' in gpu
    assert 'threads = kMatterGrainBucketTables * runtime.buckets;' in gpu
    assert 'heads[' not in shader and 'links[' not in shader
    # Single-bank in-place history (revision 20): a grain's own block, valid
    # only for its owner id and never on the frame the host marks it FRESH.
    assert '} else if (history_blocks[ownerWord()] == ids[i]) {' in shader
    assert 'if (pc.substep.x == 0u) history_blocks[i] = g_block;' in shader
    assert 'uint open_slots = ~(g_old_mask|g_allocated) & ((1u<<SLOTS)-1u);' in shader
    assert 'history_blocks[ownerWord()+1u] = g_touched|g_allocated;' in shader
    # Sleeping grains (revision 24): last-substep contact audits, physical
    # eligibility and one CAS commit preserve a concurrent neighbour audit request.
    assert 'if (sleepOn() && valid_block && can_sleep && input_slow) {' in shader
    sleep_shader = read('shaders/sim_matter_grain_sleep.glsl')
    assert 'grainSleepCommit(rest_word, next_rest);' in shader
    assert 'atomicCompSwap(history_blocks[word], previous, desired)' in sleep_shader
    assert 'if (sleeping && !audit_requested && !grainSleepAudit())' in shader
    assert 'pc.substep.x == pc.substep.w ||' in sleep_shader
    assert 'grainSleepBalanced(acceleration,t*ii,r,p,im,g_contacts)' in shader
    assert 'runtime.sleep_context = sleep_context;' in gpu
    assert 'runtime.collider_velocity_offset == 0 && !common_driver &&' in gpu
    assert '"sleep_speed_m_s"' in read('src/Physics/Fluid/MatterGrainParams.cpp')
    assert 'fluid.set_grain_settings(sleep=...)' in read('src/UI/MatterGrainControls.cpp')
    assert 'kMatterGrainContactBudget * 7 * sizeof(uint32_t)' in gpu
    assert 'constants.history_blocks = static_cast<uint32_t>(runtime.capacity);' in gpu
    # The map the next frame reads through moves only with a publication.
    assert gpu.index('runtime.published_ids = p.particle_id;') < gpu.index('runtime.history_block[i] = blocks[i] & ~kFreshHistoryBlock;')
    assert 'tangential = kt > 0.0 && coulomb > 0.0 ? -(ft-damping)/kt : vec3(0.0);' in shader
    assert 'rolling = limit > 0.0 ? -(rt-roll_damping)/kr : vec3(0.0);' in shader
    assert '2.8125 * mu_r * mu_r * k' in gpu
    assert 'nodes[stack[--size]]' in shader and 'patch_count == 8u' in shader
    assert 'manifold_count == 4u' in shader
    assert 'runtime.collider_fingerprint != fingerprint' in gpu
    assert 'if (!all(equal(cell(pj),wanted))) continue;' in shader
    diagnostics = read('src/Physics/Fluid/MatterGrainDiagnostics.cpp')
    assert '{"pile", matterGrainPileProfile(centres, params.radius_m)}' in diagnostics
    assert 'cross(arm,ft)' in shader
    assert 'affines[b+1]=w.z' in shader
    assert 'BUDGET*dt*inverse_pair_inertia' in shader
    assert 'twist_limit' in shader and 'twist_torque*n' in shader
    assert '(!params.wet_grains && water != 0.0f)' in gpu
    assert gpu.index('if (!ok || overflow)') < gpu.index('p.position = std::move(new_position);')
    geometry = read('src/Physics/Fluid/MatterGrainGeometry.cpp')
    assert 'TriangleMesh*' in geometry and 'get_attribute_data<Vec3>' in geometry
    assert 'shared_ptr<Triangle>' not in geometry and 'dynamic_cast<const Triangle*' not in geometry
    step = read('src/Physics/Fluid/FluidDomainStep.inl')
    assert 'grain_collider_mesh_resolver_' in step
    assert 'if (fluid_params.grain.enabled)' in step
    assert 'runMatterGrainStep(' in step
    # Collider faces are bounded by index width only (no 4096 budget, 2026-10-08).
    assert 'Fluid::kMatterGrainMaxColliderFaces' in read('src/Physics/Fluid/MatterGrainStep.inl')
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
    for stage in ('list_clear', 'hash', 'list_build', 'step'):
        kernel = 'sim_matter_grain_' + stage
        assert f'"{kernel}.spv", 17, 128' in registry
        assert kernel in read('shaders/compile_sim_shaders.bat')
        assert f'GRAIN_{stage.upper()}' in read(f'shaders/{kernel}.comp')
    for dead in ('contact', 'integrate', 'integrate_hash', 'clear', 'permute', 'permute_copy'):
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
    assert 'static_assert(kGrainBindings == 17);' in gpu
    assert 'runtime.coupling,' in gpu and 'runtime.neighbour_list, runtime.build_positions};' in gpu
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
    # Grain order is (cell, identity), total and independent of the partition's
    # canonical order; host copies are column-wise (1.3M-grain host cost, 2026-10-09).
    assert 'std::stable_sort(grains.begin(), grains.end()' not in coupling
    assert 'return a.cell != b.cell ? a.cell < b.cell : a.id < b.id;' in coupling
    assert 'out.gatherFrom(particles, order);' in coupling and 'particles.copyRangeFrom(nl, grains);' in coupling
    assert 'unordered_set' not in coupling and 'unordered_map' not in gpu
    assert 'const bool rows_resident = zero_rows && runtime.coupling_zero;' in gpu
    assert 'impulse[0][c] -= s * gain.x;' in coupling
    assert 'w[n] * sphere / grains_in_cell * f.mass[c[n]]' in coupling
    assert 'std::max(f.solid[c[n]], sphere)' in coupling
    for project in ('RayTrophiStudio.vcxproj', 'RayTrophiStudio.vcxproj.filters'):
        text = (ROOT / 'RayTrophiStudio' / project).read_text(encoding='utf-8')
        assert 'MatterGrainCoupling.cpp' in text and 'MatterGrainCoupling.h' in text
    assert 'MatterConstitutiveModel::Granular' in birth_filter, 'liquid parcels must not block grain births'
    assert 'Fluid::substanceTransportOwner(' in births
    assert 'Fluid::MatterTransportOwner::Grain;' in births
    assert 'substanceTransportOwner(' in birth_filter
    params_src = read('src/Physics/Fluid/MatterGrainParams.cpp')
    assert '"fluid_coupling"' in params_src
    # One physical value, one home: drag reads the liquid substance viscosity,
    # restitution replaced the per-domain normal damping, and (2026-10-09) the
    # whole grain material is the substance's (MADDE_UI_TEK_OTORITE U2).
    assert 'drag_viscosity_pa_s was removed' in params_src
    assert "normal_damping_n_s_m was replaced by the substance's grain_restitution" in params_src
    assert '"normal_damping_n_s_m", "drag_viscosity_pa_s"' in params_src
    assert '{"restitution", "substance grain_restitution"}' in params_src
    assert 'p.restitution = std::clamp(grain.grain_restitution' in params_src
    coupling_src = read('src/Physics/Fluid/MatterGrainCoupling.cpp')
    assert 'liquid_kinematic_viscosity' in coupling_src and 'viscous_mass / liquid_mass' in coupling_src
    # 2026-10-07: with volume exclusion the pressure force reaches the liquid
    # through the porous projection only; also kicking the liquid with it fed
    # back through the next frame's measured acceleration (50 m/s parcels).
    assert 'pressure_in_projection ? drag[g].drag_impulse' in coupling_src
    # 2026-10-07: force fields and moving colliders. The step no longer holds
    # for them; colliders carry vertex velocities and sweep through substeps.
    step_src = read('src/Physics/Fluid/MatterGrainStep.inl')
    assert 'static colliders, gravity only' not in step_src
    assert 'forces->evaluateAt(grains.position[g]' in step_src
    assert 'KinematicConsumerGranular' in step_src and 'collider_previous_vertices' in step_src
    assert 'fields ? force_buffer : nullptr' in step_src
    motion_shader = read('shaders/sim_matter_grain.glsl')
    assert 'v+cross(w,arm)-manifold_v[m]' in motion_shader and 'a -= va*time_left' in motion_shader
    assert 'triangles * 6 * sizeof(Vec3)' in gpu
    shader_src = read('shaders/sim_matter_grain.glsl')
    assert 'float cn = 2.0*pc.step_contact.y*sqrt(k/inverse_normal_mass);' in shader_src
    assert 'im+jm,f,t);' in shader_src and shader_src.count('im,f,t);') == 2
    ui = read('src/UI/MatterGrainControls.cpp')
    # Matter tab shows the substance's material read-only; the domain edits none of it.
    assert 'p.fluid_coupling' in ui and 's->grain_restitution' in ui and 'p.restitution' not in ui
    bridge = read('src/Physics/ParticleRenderBridge.cpp')
    assert 'const float d = grain ? diam : liquid_diam;' in bridge
    assert '--coexist-only' in runtime_test
    # B9a: device history is valid only for the state it was published with.
    assert 'std::memcmp(&was, &p.position[i], sizeof(Vec3)) != 0' in gpu
    assert gpu.index('reset_reason = "host_state_changed";') < gpu.index('const bool history_reset = runtime.history_fresh;')
    assert gpu.index('p.position = std::move(new_position);') < gpu.index('runtime.published_positions = p.position;')
    # B8: history follows identity across any order; grains sorted by cell.
    # Each grain keeps its block, so a reorder uploads the map only.
    assert 'blocks[i] = runtime.history_block[previous_index[i]];' in gpu
    assert 'GRAIN_PERMUTE' not in shader and 'sim_matter_grain_permute' not in gpu
    for dead in ('sim_matter_grain_clear', 'sim_matter_grain_permute'):
        assert dead not in read('shaders/compile_sim_shaders.bat')
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
    porous = read('shaders/sim_fluid_divergence_porous.glsl')
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
    porous_kernel = read('shaders/sim_fluid_divergence_porous.glsl')
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
    # Hash cell = list cutoff: the bridge reach stays inside it (plus the skin).
    assert 'constants.wet[0] = 2.0f * params.radius_m + rupture_cap +' in gpu
    assert 'coupling_rows[12 * i + 11] = p.pore_water_mass_kg[i] / 1000.0f;' in gpu
    assert coordinator.index('applyMatterGrainLiquidReaction(') < coordinator.index('exchangeMatterGrainWater(')
    assert coordinator.index('exchangeMatterGrainWater(') < coordinator.index('mergeMatterGrainOwners(state.particles')
    # Wet at birth is the source's; wet grains are derived (ownership.wet_grains).
    assert 'initMatterGrainBirthWater(' in births and 'source.grain_birth_saturation' in births
    assert '{"birth_saturation", "flow source grain_birth_saturation"}' in params_src
    assert 'ownership.wet_grains' in ui and 'source.grain_birth_saturation' in ui
    assert '--wet-only' in runtime_test
    # B7 closed (H1-G0, 2026-10-06): XPBD failed the 16k accuracy/cost gate and
    # was removed; DEM is the only grain solver, the old keys are rejected.
    assert 'xpbd' not in shader.lower() and 'solver_kind' not in gpu
    assert 'it.key() == "solver_kind" || it.key() == "xpbd_substeps"' in params_src
    assert 'd["matter_grain"] = RayTrophiSim::Fluid::matterGrainParamsToJson' in read('src/Core/ProjectManager.cpp'), 'project save drops grain settings'
    assert '"solver_kind", "xpbd_substeps"}) {' in params_src and "solver_kind='xpbd'" in runtime_test
    # Domain panel (2026-09-27 six-tab decision): grain material in Matter,
    # solver + coupling + readiness in Solvers, report in Measure; the legacy
    # MPM granular block is not offered while grains own the granular phase;
    # every blocker is a lock with its reason, from the core rule.
    panel = read('src/UI/scene_ui_simulation_domains.cpp')
    tab = lambda name: panel.index(f'BeginTabItem("{name}")')
    assert tab('Matter') < panel.index('drawMatterGrainMaterial(domain,') < tab('Environment')
    assert tab('Solvers') < panel.index('drawMatterGrainSolver(domain') < tab('Output')
    assert tab('Measure') < panel.index('drawMatterGrainReport(domain')
    assert 'drawMatterGrainControls' not in panel
    assert 'Legacy MPM granular material is not used' not in panel
    for reason in ('Grains need the Vulkan backend.', 'Grains need a Closed boundary.',
                   'Solid phase cannot run with grains yet.'):
        assert reason in panel, reason
    assert 'const bool reseed_locked = fp.granular_enabled || fp.grain.enabled;' in panel
    assert 'ownership.blockers' in ui and 'ImGui::DragFloat' not in ui and 'ImGui::SliderFloat' not in ui
    # 2026-10-09: no grain switch. Grains follow the substances in reach of the
    # domain; the step decides before the sources emit, panel and IPC read the
    # same struct, and a wanted-but-blocked DEM names its blockers.
    assert 'Enable discrete grains' not in ui and not re.search(r'\bp\.enabled', ui)
    assert '{"enabled", "derived: grains run when' in params_src
    assert '{"enabled", p.enabled}' not in params_src
    # U3: stiffness follows the radius; the scale is the authored value.
    assert 'p.stiffness_n_m = p.stiffness_scale * kMatterGrainStiffnessPerRadius * p.radius_m;' in params_src
    assert '{"stiffness_scale", p.stiffness_scale}' in params_src and '"stiffness_n_m", p.' not in params_src
    # U2: the step copies the DEM substance's material every step.
    sim_src = read('src/Physics/ParticleSimulation.cpp')
    assert 'Fluid::applyMatterGrainSubstance(grain, *profile,' in sim_src
    assert sim_src.index('Fluid::applyMatterGrainSubstance(') < sim_src.index('    injectFlowSourcesIntoGridDomains(')
    library = read('src/Physics/SubstanceLibrary.cpp')
    for field in ('grain_friction', 'grain_rolling_friction', 'grain_restitution',
                  'grain_packing_fraction', 'grain_water_capacity_fraction',
                  'liquid_surface_tension_n_m', 'granular_compaction_hardening'):
        assert 'RT_FLOAT(' + field + ',' in library, field
    owner = read('src/Physics/Fluid/MatterSubstanceState.cpp')
    assert 'profile->granular_transport == MatterGranularTransport::Dem' in owner
    assert 'matterGrainBlockers(domain)' in owner and 'result.reset_pending = true;' in owner
    sim = read('src/Physics/ParticleSimulation.cpp')
    assert sim.index('grain.enabled = ownership.enabled;') < sim.index('    injectFlowSourcesIntoGridDomains(')
    assert '"grain_ownership"' in read('src/Api/RtApiMatterModels.cpp')
    # Substance editor (MADDE_UI_TEK_OTORITE U1): the dragged value is held until
    # release, DEM substances say their continuum fields are unused, sections
    # follow what the substance can use and none is unreachable.
    editor = read('src/UI/scene_ui_substance_editor.hpp')
    assert 'pending_key == edit_key' in editor and 'IsItemDeactivatedAfterEdit()) apply' in editor
    assert 'Grains (DEM) use the grain material below' in editor and 'Show unused sections' in editor
    assert '{"Grains (DEM)", "Grain material (DEM)", dem, nullptr}' in editor
    assert '"Script key: %s"' in editor and '##substance_filter' in editor
    assert 'grain_locked' in read('src/UI/scene_ui_fluid_thermal.cpp')
    assert 'desc.fluid_params.grain.enabled' in read('src/UI/ParticleBillboardBuilder.cpp')
    cross = read('shaders/sim_grain_mpm.glsl')
    driver = read('src/Physics/Fluid/MatterGrainMpmContact.cpp')
    assert len(re.findall(r'binding\s*=\s*\d+', cross)) == 13
    assert 'static_assert(sizeof(Constants) == 32)' in driver
    assert 'pairImpulse(i, j) : -pairImpulse(j, i)' in cross
    assert cross.index('CONTACT_GATHER') < cross.index('CONTACT_APPLY')
    assert 'CONTACT_BUDGET' not in cross
    assert 'max(degrees[grain], degrees[mpm])' in cross
    assert 'count > 1000000' not in driver
    assert 'groups.groups_y' in driver
    assert 'compile simulation shaders' in driver
    assert 'compute_.uploadBuffer(buffers_[6], metadata.data()' in driver
    assert 'uploadBuffer(buffers_[4]' not in driver and 'uploadBuffer(buffers_[5]' not in driver
    assert 'mpm_contact->step(runtime, substep, error)' in gpu
    assert gpu.index('mpm_contact->step(') < gpu.index('command.kernel = kernel;',
        gpu.index('const auto dispatch_stage ='))
    assert 'report.mpm_parcels == 0' in read('src/Physics/Fluid/MatterGrainStep.inl')
    for stage in ('init', 'clear', 'hash', 'count', 'gather', 'apply'):
        kernel = f'sim_grain_mpm_{stage}'
        assert f'{{ "{kernel}", "{kernel}.spv", 13, 32 }}' in registry
        assert kernel in read('shaders/compile_sim_shaders.bat')
        assert f'#define CONTACT_{stage.upper()}' in read(f'shaders/{kernel}.comp')
    common = read('src/Physics/Fluid/MatterGpuStep.inl')
    assert common.index('clock->contact(substep') < common.index('runGpuFluidAdvectTail(')
    assert 'substep % grid_stride == 0' in common
    assert 'request > 4096' not in common
    assert 'bool publish_velocity = true' in read('src/Physics/ParticleSimulation.cpp')
    liquid_gpu = read('src/Physics/Fluid/MatterGrainFluidGpuCoupling.cpp')
    liquid_shader = read('shaders/sim_grain_fluid.glsl')
    assert len(re.findall(r'binding\s*=\s*\d+', liquid_shader)) == 21
    assert 'std::isfinite(impulse[axis])' in liquid_gpu
    assert 'static_assert(sizeof(Constants) == 96)' in liquid_gpu
    assert 'grainVelocity(i)' in liquid_shader
    assert 'firstFluid(c)' in liquid_shader and 'solidVolume(c)' in liquid_shader
    assert 'previous_bucket' in liquid_shader
    assert 'atomicCompSwap' in liquid_shader
    assert 'FLUID_SOLID' in liquid_shader
    assert 'fieldLeader(c)' in liquid_shader
    assert 'storage_->handles[i] = buffers_[i]' in liquid_gpu
    assert 'compute_.getBufferSize(buffers_[i])' in liquid_gpu
    assert 'liquid_contact.react(index, step_dt' in coordinator
    assert 'mpm_contact.publish(report, error, !shared_clock)' in coordinator
    for stage in ('clear', 'hash', 'cells', 'solid', 'refresh', 'delta', 'reaction', 'apply'):
        kernel = f'sim_grain_fluid_{stage}'
        assert f'"{kernel}.spv", 21, 96' in registry
        assert f'#define FLUID_{stage.upper()}' in read(f'shaders/{kernel}.comp')
    print('PASS dry grain source: fused ping-pong ABI/order, Verlet list rebuild, contact budget, history, grain mass, pile, per-carrier owner + liquid coupling, UI/API/save wiring')


if __name__ == '__main__':
    main()
