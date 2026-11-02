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
        "p2g": (10, 40), "g2p": (12, 72), "stress_update": (18, 116),
        "stress_p2g": (9, 52), "settle": (5, 36), "advect": (11, 84),
        "occupancy": (6, 40), "contact": (19, 16), "copy": (2, 4),
        "clear": (1, 4), "zero_faces": (4, 36), "pores": (13, 40),
    }
    def expand(filename, defines=None):
        defines = set() if defines is None else defines
        active = [True]
        output = []
        for line in read("shaders/" + filename).splitlines():
            stripped = line.strip()
            if stripped.startswith("#ifdef "):
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
    assert driver.index("runGpuFluidMGPCGPressure(") < driver.index('contact.kernel = Fluid::macKernel(liquid_mac, "sim_matter_contact"')
    assert driver.index('contact.kernel = Fluid::macKernel(liquid_mac, "sim_matter_contact"') < driver.index("runGpuFluidG2P(")
    assert driver.index("runGpuFluidG2P(") < driver.index("auto liquid_result = particles;")
    assert driver.index("mixed GPU produced nonfinite") < driver.index("particles = std::move(liquid_result)")
    assert "copyParticleFrom(i, granular_result, i)" in driver
    assert "liquid_result.advanceMaterialCoordinates();" in driver
    assert "CPU fallback" in driver
    assert "Fluid::MatterGpuParticleLease shared_particles;" in driver
    assert "shared_particles.bind(*lanes[1], primary, count, error)" in driver
    assert "downloadGpuGranularParticles(granular_result, compute, *lanes[1], false)" in driver
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
    assert driver.count("solid_faces(lane, transfer_dt)") == 2
    mpm_wall = read("shaders/sim_matter_mpm_boundary.comp")
    assert '"sim_matter_mpm_boundary", "sim_matter_mpm_boundary.spv", 13, 32' in read(
        "src/Device/SimulationComputeVulkan.cpp")
    # The GRID carries contact: grid-SDF collision per face (phi_face < 0,
    # n = grad phi, Coulomb on approach, adhesion while leaving), the old
    # moving/separating rule only for walls without a distance. A parcel-only
    # contact piled a jelly into the h/2 band (2026-11-02 live).
    assert "float phiContact(float v, ivec3 low, ivec3 high, int component)" in mpm_wall
    assert "vt *= max(slip - mu * (-vn), 0.0) / slip;" in mpm_wall
    assert "rel = vt + n * max(vn - hold, 0.0);" in mpm_wall
    assert "(plainWall(low) || plainWall(f))" in mpm_wall
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
    assert "if (has_fluid && !runGpuFluidMGPCGPressure(" in driver
    assert "if (has_fluid && !compute->dispatch(contact))" in driver
    # Separate transfer, gather and advection loops each skip the empty lane.
    assert driver.count("if (lane == 0 && !has_fluid)") == 3
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
