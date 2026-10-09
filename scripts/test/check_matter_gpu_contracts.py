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
    assert "atomicAdd(count[resolved == 3u ? 2 : 3], 1u)" in shader
    specs = {
        "p2g": (10, 40), "g2p": (12, 72), "stress_update": (18, 68),
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
    assert "Fluid::resolveSingleMatterModel(particles, legacy_granular)" in driver
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
    print("PASS: 13 GPU ABIs, indexed stages, contact ordering, budget, canonical commit, API wiring")


if __name__ == "__main__":
    main()
