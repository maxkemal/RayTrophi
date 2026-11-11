"""Source-only check: no build, shader compilation, app or GPU execution."""
from pathlib import Path
import re
import xml.etree.ElementTree as ET


def main():
    project = Path(__file__).resolve().parents[2] / "RayTrophiStudio"
    source = project / "source"
    def read(path):
        return (source / path).read_text(encoding="utf-8-sig")
    for entry in (source / "shaders").glob("*.comp"):
        raw = entry.read_text(encoding="utf-8-sig")
        raw = re.sub(r"/\*.*?\*/|//[^\n]*", "", raw, flags=re.S).lstrip()
        assert raw.startswith("#version "), (entry.name, "#version must occur first")
    for body in (source / "shaders").glob("sim_fluid_*.glsl"):
        assert "#version" not in body.read_text(encoding="utf-8-sig"), body.name
    shader = read("shaders/sim_matter_partition.comp")
    dispatch = read("src/Physics/Fluid/MatterGpuPartition.cpp")
    registry = read("src/Device/SimulationComputeVulkan.cpp")
    particle_source = read("src/Physics/ParticleSimulation.cpp")
    namespace_start = particle_source.index("namespace RayTrophiSim {")
    for include in re.finditer(r'^#include "[^"\n]+\.h"', particle_source, re.M):
        assert include.start() < namespace_start, (
            "Header included inside RayTrophiSim namespace", include.group())
    assert sorted(map(int, re.findall(r"binding = (\d+)", shader))) == [0, 1, 2, 3]
    assert "static_assert(sizeof(Constants) == 8)" in dispatch
    assert '"sim_matter_partition.spv", 4, 8' in registry
    assert "sim_matter_partition.comp" in read("shaders/compile_sim_shaders.bat")
    assert "downloadBuffer(" not in dispatch
    assert "synchronize(" not in dispatch
    assert "stepMixedMatter(" not in dispatch
    assert "compute.backendType() != ComputeBackendType::VulkanCompute" in dispatch
    assert "budget_bytes && total > budget_bytes" in dispatch
    assert "candidate.valid()" in dispatch
    assert "fluid_index[slot] = index;" in shader
    assert "granular_index[slot] = index;" in shader
    # Elastic (3) shares the MPM lane with granular (2) since 2026-11-02
    # (docs/dev/ELASTIK_MPM.md); only an unknown model is counted aside.
    assert "resolved == 2u || resolved == 3u" in shader
    assert "atomicAdd(count[3], 1u)" in shader
    specs = {
        "p2g": (10, 40), "g2p": (12, 72), "stress_update": (18, 128),
        "stress_p2g": (9, 52), "settle": (7, 52), "advect": (11, 104),
        "occupancy": (6, 40), "liquid_drag": (19, 32), "copy": (2, 4),
        "clear": (1, 4), "zero_faces": (4, 36), "pores": (13, 40),
    }
    def expand(filename, defines=None):
        defines = set() if defines is None else defines
        active = [True]
        output = []
        for line in read("shaders/" + filename).splitlines():
            stripped = line.strip()
            if stripped.startswith("#if ") and "defined(" in stripped:
                # `#if defined(A) && !defined(B)`: defined() terms joined by && / ||.
                expr = re.sub(r"defined\((\w+)\)",
                              lambda m: str(m.group(1) in defines), stripped[4:])
                expr = expr.replace("&&", " and ").replace("||", " or ").replace("!", " not ")
                active.append(active[-1] and bool(eval(expr, {"__builtins__": {}})))
            elif stripped.startswith("#ifdef "):
                active.append(active[-1] and stripped.split()[1] in defines)
            elif stripped.startswith("#ifndef "):
                active.append(active[-1] and stripped.split()[1] not in defines)
            elif stripped == "#else":
                active[-1] = active[-2] and not active[-1]
            elif stripped == "#endif":
                active.pop()
            elif active[-1] and stripped.startswith("#define "):
                defines.add(stripped.split()[1])
            elif active[-1] and stripped.startswith("#include "):
                output.append(expand(stripped.split('"')[1], defines))
            elif active[-1]:
                output.append(line)
        assert len(active) == 1, filename
        return "\n".join(output)
    for name, (bindings, push_bytes) in specs.items():
        compiled_source = expand(f"sim_matter_{name}.comp")
        actual = sorted(map(int, re.findall(r"binding\s*=\s*(\d+)", compiled_source)))
        assert actual == list(range(bindings)), (name, actual)
        push = re.search(r"layout\(push_constant\).*?\{(.*?)\}\s*pc;",
                         compiled_source, re.S).group(1)
        push = re.sub(r"//[^\n]*", "", push)
        size = len(re.findall(r"\b(?:int|uint|float)\s+\w+", push)) * 4
        assert size == push_bytes, (name, size, push_bytes)
        assert compiled_source.count("#version 450") == 1, name
        assert f'"sim_matter_{name}.spv", {bindings}, {push_bytes}' in registry, name
        assert f"sim_matter_{name}" in read("shaders/compile_sim_shaders.bat")
    driver = read("src/Physics/Fluid/MatterGpuStep.inl")
    assert driver.index("working_set >") < driver.index("if (!ensure(primary))")
    assert driver.index("runGpuFluidMGPCGPressure(") < driver.index('drag.kernel = Fluid::macKernel(liquid_mac, "sim_matter_liquid_drag"')
    assert driver.index('drag.kernel = Fluid::macKernel(liquid_mac, "sim_matter_liquid_drag"') < driver.index("runGpuFluidG2P(")
    assert "sim_matter_contact" not in read("src/Device/SimulationComputeVulkan.cpp")
    # Publication (scale S1 slice A, 2026-11-11): staged columns, validated, then
    # swapped in; the adaptive snow sizing check comes before both.
    assert driver.index("runGpuFluidG2P(") < driver.index("static thread_local PublicationStaging staging_scratch;")
    assert driver.index("retry_at_ceiling = true;") < driver.index("mixed GPU produced nonfinite canonical particle state") \
        < driver.index("particles.position.swap(staging.position);")
    assert "granular_result.commitParticle(particles, static_cast<std::size_t>(i));" in driver
    assert "particles.advanceMaterialCoordinates();" in driver
    # The ceiling re-run happens in the wrapper, after the first attempt's scope ended.
    assert "bool runMatterGpuStep(" in driver and "static bool runMatterGpuStepOnce(" in driver
    assert "CPU fallback" in driver
    assert "Fluid::MatterGpuParticleLease shared_particles;" in driver
    assert "shared_particles.bind(*lanes[1], primary, count, error)" in driver
    assert "Fluid::enqueueGranularSidecars(*compute, lanes[1]->granular," in driver
    assert "granular_result.canCommit(particles, count)" in driver
    lease = read("include/Fluid/MatterGpuParticleLease.h")
    assert "~MatterGpuParticleLease()" in lease and "restore();" in lease
    assert "source.fluid_uploaded_particle_count != count" in lease
    assert "target_->fluid_uploaded_particle_count = uploaded_;" in lease
    assert "if (download_transport)" in read("src/Physics/Fluid/FluidGpuTransferStages.inl")
    assert "state->fluid_stats.mixed_model_step" in read("src/Api/RtApiMatterModels.cpp")
    assert "common_substeps" in read("src/Api/RtApiMatterModels.cpp")
    assert "runMatterGpuStep(" in read("src/Physics/Fluid/FluidDomainStep.inl")
    # 2026-11-02: an MPM-bearing mixed step takes gravity per substep, not as
    # one frame kick (a resting Dough rang at ~0.4 g*frame_dt).
    domain_step = read("src/Physics/Fluid/FluidDomainStep.inl")
    assert "mpm_substep_forces ? &substep_forces : nullptr" in domain_step
    assert domain_step.index("} else if (mpm_substep_forces) {") < domain_step.index(
        "} else if (!force_fields_require_cpu) {")
    assert "substep_forces->frame_start_seconds + substep * dt" in driver
    # 2026-11-02: the MPM lane's walls move and let go (COLLIDER_TEMAS_YUZEYI.md).
    assert "runGpuMpmSolidFaces(grid, compute, *lanes[1], closed_boundary, step_dt," in driver
    assert driver.count("solid_faces(lane, lane_dt)") == 1
    assert "solid_faces(lane, lane_dt, false)" in driver
    mpm_wall = read("shaders/sim_matter_mpm_boundary.comp")
    assert '"sim_matter_mpm_boundary", "sim_matter_mpm_boundary.spv", 13, 52' in read(
        "src/Device/SimulationComputeVulkan.cpp")
    # The GRID carries contact: grid-SDF collision per face (phi_face < 0,
    # n = grad phi, Coulomb on approach, adhesion while leaving), the old
    # moving/separating rule only for walls without a distance. A parcel-only
    # contact piled a jelly into the h/2 band (2026-11-02 live).
    assert "float phiContact(float v, ivec3 low, ivec3 high, int component)" in mpm_wall
    assert "vt *= max(slip - shear, 0.0) / slip;" in mpm_wall
    assert "shear = (band ? 0.0 : mu * (-vn)) + hold;" in mpm_wall
    assert "float tension = min(vn, hold);" in mpm_wall
    assert "rel = vt + n * vn_kept;" in mpm_wall
    assert "(plainWall(low) || plainWall(f))" in mpm_wall
    # A ridge face (phi_face < 0, no slope: a shell thinner than ~2h) blocks
    # flow across instead of being skipped (a jelly fell through, 2026-11-03).
    assert "return solidVelocity(clamp(low, ivec3(0)," in mpm_wall
    # Solid recovery must not teleport MPM parcels resting on a collider with a
    # distance (crushed a jelly on a dome; the snow-sticks report); they take
    # the true-surface push the same frame, and both counts reach step_stats.
    domain_step = read("src/Physics/Fluid/FluidDomainStep.inl")
    assert "recoverParticlesFromSolidCells(state.particles, state.grid, 6,\n" \
        "                                                          true, auto_is_mpm);" in domain_step
    assert "mpm_surface_push_count = Fluid::applyMpmParticleContact(" in domain_step
    assert "if (keep_mpm_on_phi && cell < grid.solid_phi.size() &&" in read(
        "src/Physics/Fluid/FluidParticleSolidRecovery.cpp")
    assert 'j["mpm_surface_pushed_particles"]' in read("src/Api/RtIpc.cpp")
    assert 'method == "fluid.sample_solid_phi"' in read("src/Api/RtIpc.cpp")
    # The P2G skips the face mass gradient (its only reader, sim_matter_contact,
    # was removed 2026-11-11): bit 31 of the lane word.
    p2g = read("shaders/sim_fluid_p2g_scatter.glsl")
    assert "const bool write_gradient = (pc.matter_lane & 0x80000000u) == 0u;" in p2g
    assert "view.mass_gradient_wanted = false;" in read("src/Physics/Fluid/MatterGpuStep.inl")
    # Clustered P2G (same-address contention): loaded only where the device
    # reports clustered subgroup ops; the per-lane kernels stay the fallback.
    vk = read("src/Device/SimulationComputeVulkan.cpp")
    assert "(subgroup.supportedOperations & VK_SUBGROUP_FEATURE_CLUSTERED_BIT) != 0" in vk
    assert 'spv = std::string(d.name) + "_clustered.spv";' in vk
    assert "#define P2G_CLUSTERED" in read("shaders/sim_matter_p2g_clustered.comp")
    assert "#define P2G_CLUSTERED" in read("shaders/sim_matter_stress_p2g_clustered.comp")
    # Domain sleep: every parcel resting for kMpmSleepFrames runs no substep,
    # never under a force field. Partial freezing was measured slower (serial
    # substep chain) and removed; nothing leaves a lane any more.
    sleep_src = read("src/Physics/Fluid/MatterMpmSleep.cpp")
    assert "out.domain_asleep = state.rest_frames >= kMpmSleepFrames;" in sleep_src
    step = read("src/Physics/Fluid/MatterGpuStep.inl")
    assert "if (count > 0 && no_force_field && mpm_sleep.domain_asleep) {" in step
    # An asleep frame recounts the granular counters from the parcels: a blank
    # stats block reported a resting pile as 0 asleep (snow_support, 2026-11-04).
    assert "Fluid::Granular::countMaterialState(particles, asleep_stats);" in step
    assert "frozen_cells" not in step and "frozen_particle" not in step
    assert "kMatterPartitionFrozen" not in read("include/Fluid/MatterGpuPartition.h")
    # Settle damps and sleeps only SUPPORTED parcels (mean acceleration along
    # gravity < tolerance * g, measured before damping). Pressure alone hung a
    # mid-air snow block from its top (2026-11-03 live).
    settle = read("shaders/sim_fluid_granular_settle.glsl")
    assert "if(pressure<=pc.pressure_threshold || !supported)" in settle
    assert "supported=accel<pc.support_tolerance*g;" in settle
    assert "inline bool updateSupport(const SettleParams& sp" in read(
        "include/Fluid/GranularConstitutive.h")
    assert "static_assert(sizeof(SettleConstants)==48);" in read(
        "include/Fluid/GranularGpuDispatch.h")
    # The SDF cook pad is isotropic and >= 3 voxels (a 15 %-per-axis pad left a
    # 5 cm plate 7.5 mm of field; a glued gum let go); readers take the stored
    # mesh size, never sdf_extents / 1.3.
    assert "sdf_extents / 1.3f" not in read("src/Physics/ParticleSimulation.cpp")
    assert "pad_len = std::max(pad_len, 3.0f * domain_h);" in read("include/scene_data.h")
    # Matrix4x4::determinant applied the cofactor sign twice (det of a z
    # rotation = cos 2theta): every tilted collider OBB was inverted scaled.
    assert "m[0][0] * cofactor(0, 0) + m[0][1] * cofactor(0, 1) + m[0][2] * cofactor(0, 2) + m[0][3] * cofactor(0, 3)" in read(
        "src/Math/Matrix4x4.cpp")
    assert "Fluid::copyMatterGpuFloat(*compute, velocity[axis], snapshot[axis]," in read(
        "src/Physics/ParticleSimulation.cpp")
    particle_sim = read("src/Physics/ParticleSimulation.cpp")
    assert "grid.solid_contact_friction[ci] = active_contact_friction;" in particle_sim
    assert "sig = hashF(sig, c.contact_adhesion_pa);" in particle_sim
    assert "constants.pass" not in particle_sim and "adhesionBand" not in mpm_wall
    # Sub-voxel contact (COLLIDER_TEMAS_YUZEYI.md P1-P3, S3-S4).
    vk = read("src/Device/SimulationComputeVulkan.cpp")
    assert '"sim_collider_bvh_query", "sim_collider_bvh_query.spv", 3, 80' in vk
    assert '"sim_matter_mpm_particle_contact", "sim_matter_mpm_particle_contact.spv", 10, 48' in vk
    bat = read("shaders/compile_sim_shaders.bat")
    assert "sim_collider_bvh_query" in bat and "sim_matter_mpm_particle_contact" in bat
    assert "FluidGpuOccupancy granular_occupancy(state, compute, lanes[1], &mpm_solid_cells);" in driver
    assert "runGpuMpmParticleContact(grid, compute, *lanes[1], dt," in driver
    assert "writePhi(i, j, k, obbDistance(obb.local_to_world, lc, cc, local_mn, local_mx," in particle_sim
    assert "sampleCookedSdf(c, tx, ty, tz, gradient)" in particle_sim
    assert "(*gpu_mesh_query)(c, lattice, distance)" in particle_sim
    assert "&gpu_mesh_query);" in read("src/Physics/Fluid/FluidDomainStep.inl")
    scene = read("include/scene_data.h")
    assert "runPendingGpuSdfCooks();" in scene and "launchCpuSdfCook(std::move(job));" in scene
    contact = read("shaders/sim_matter_mpm_particle_contact.comp")
    # The parcel contact is a safety push only (support is the grid's).
    assert "if (d >= 0.0) return;" in contact and "glue * pc.dt" not in contact
    query = read("shaders/sim_collider_bvh_query.comp")
    assert "vec3(0.5060, 0.7071, 0.4943)" in query  # the CPU cook's probe rays
    for path in ("src/Api/RtIpc.cpp", "src/Api/RtPython.cpp"):
        text = read(path)
        assert "collider_query_gpu" in text and "sdf_cooked_on_gpu" in text, path
    assert "ColliderGpuQuery.cpp" in (project / "RayTrophiStudio.vcxproj").read_text(encoding="utf-8")
    # Tearing reads elastic strain + accumulated creep/yield flow, both backends.
    assert "float tear=stretch+flow;" in read("shaders/sim_fluid_granular_stress_update.glsl")
    assert "const float tear = stretch + flow;" in read("include/Fluid/GranularConstitutive.h")
    assert "uploadGpuSolidContact(grid, *compute, *lanes[1])" in driver
    for path in ("src/Api/RtIpc.cpp", "src/Api/RtPython.cpp", "src/Core/ProjectManager.cpp",
                 "src/Utils/SceneSerializer.cpp", "src/UI/scene_ui_forcefield.hpp"):
        assert "contact_adhesion_pa" in read(path), path
    assert "sim_matter_mpm_boundary" in read("shaders/compile_sim_shaders.bat")
    cpu = read("src/Physics/Fluid/APICFluidSolver.cpp")
    assert "enforceMpmContact(grid, dt, params.granular_density);" in cpu
    assert "float face(float v, const int lo[3], const int hi[3], int axis) const {" in cpu
    assert "ColliderGpuQuery" in read("src/Physics/Fluid/ColliderGpuQuery.cpp") and \
        "alive(compute, output_)" in read("src/Physics/Fluid/ColliderGpuQuery.cpp")
    assert driver.index("if (substep_forces) {") < driver.index("runGpuFluidP2G(state, compute, *lanes[lane]")
    assert 'm_descLayouts[bufferCount] == VK_NULL_HANDLE' in registry
    for file in ["RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"]:
        paths = [n.attrib["Include"] for n in ET.parse(project / file).iter()
                 if "Include" in n.attrib]
        for module in ["MatterMacContact", "MatterMixedStep", "MatterGpuPartition",
                       "MatterGpuModelView", "MatterGranularLoad"]:
            for path in [f"source\\include\\Fluid\\{module}.h",
                         f"source\\src\\Physics\\Fluid\\{module}.cpp"]:
                assert paths.count(path) == 1, (file, path)
        assert paths.count(r"source\shaders\sim_matter_partition.comp") == 1
    assert "measureMatterGranularLoad(particles, legacy_granular)" in driver
    assert "load.overburden_pressure, load.softening_min" in driver
    for field in ["velocity_damping", "affine_damping"]:
        assert f"models[lane].{field} = Fluid::Granular::timeScaledSubstepDamping(" in driver
        assert f"fluid_params.{field} = Fluid::Granular::timeScaledSubstepDamping(" in read(
            "src/Physics/Fluid/FluidDomainStep.inl")
    assert "lane == 1 ? frame_dt" in driver
    # has_fluid = any non-MPM parcel (2026-11-02): resolveSingleMatterModel said
    # "not all granular", which counted an all-elastic domain as liquid.
    assert "const bool has_fluid = std::any_of(" in driver
    assert "return !mpm_parcel(" in driver
    assert "if (liquid_step && !runGpuFluidMGPCGPressure(" in driver
    assert "if (liquid_multirate && !compute->dispatch(drag))" in driver
    # Separate transfer, gather and advection loops each skip the empty lane;
    # transfer and gather skip it between liquid steps (liquid_step implies has_fluid).
    assert driver.count("if (lane == 0 && !has_fluid)") == 2
    assert driver.count("if (lane == 0 && !liquid_step)") == 2
    assert driver.index("clock->contact(substep") < driver.index("runGpuFluidAdvectTail(")
    assert "mixed GPU empty fluid field clear failed" in driver
    assert driver.index("mixed GPU empty fluid field clear failed") < driver.index(
        "for (int substep = 0; substep < substeps;")
    assert "stats.pressure_on_gpu = has_fluid;" in driver
    for field in ["granular_wave_substeps", "granular_strain_substeps",
                  "granular_strain_rate", "granular_overburden_pressure",
                  "granular_young_modulus_for_load", "granular_load_measured"]:
        assert "stats." + field in driver
    forces = read("src/Physics/ParticleSimulation.cpp")
    assert "Fluid::fluidForcesNeedColumns(force_buffer, global_wind)" in forces
    assert "constants.column_nx = surface_columns ? state.grid.nx : 0;" in forces
    assert "constants.column_nz = surface_columns ? state.grid.nz : 0;" in forces
    assert "if (vulkan_force_kernel && surface_columns)" in forces
    assert "Fluid::packedForcesNeedFluidColumns(packed)" in read("src/Physics/SimulationWorld.cpp")
    force_shader = read("shaders/sim_fluid_particle_forces.comp")
    assert "f.enabled==0||f.type!=0||f.fluid_surface_drag==0" in force_shader
    assert "pc.column_nx>0&&pc.column_nz>0" in force_shader
    print("PASS: 13 GPU ABIs, indexed stages, contact ordering, budget, canonical commit, API wiring")


if __name__ == "__main__":
    main()
