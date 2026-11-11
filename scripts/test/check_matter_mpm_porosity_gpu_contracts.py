"""Static contract: the MPM lane's porosity is built on the device (2026-11-11).

The host round trip it replaced read ~9.5 MB of MPM faces back per liquid step
(~170 MB a frame while an emitter poured at 17 liquid steps) and uploaded the
result. This keeps it from coming back.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "RayTrophiStudio" / "source"


def read(rel):
    return (SRC / rel).read_text(encoding="utf-8")


def bindings(glsl, sparse):
    text = glsl
    if not sparse:
        text = re.sub(r"#ifdef RT_SPARSE_MAC.*?#endif", "", text, flags=re.S)
    return sorted(int(b) for b in re.findall(r"binding\s*=\s*(\d+)", text))


def main():
    glsl = read("shaders/sim_matter_mpm_porosity.glsl")
    assert bindings(glsl, False) == list(range(14))
    assert bindings(glsl, True) == list(range(16))
    assert '#define RT_SPARSE_MAC' in read("shaders/sim_sparse_mac_matter_mpm_porosity.comp")
    vk = read("src/Device/SimulationComputeVulkan.cpp")
    assert '{ "sim_matter_mpm_porosity", "sim_matter_mpm_porosity.spv", 14, 32 }' in vk
    assert '{ "sim_sparse_mac_matter_mpm_porosity", "sim_sparse_mac_matter_mpm_porosity.spv", 16, 32 }' in vk
    bat = read("shaders/compile_sim_shaders.bat")
    assert "sim_matter_mpm_porosity" in bat and "sim_sparse_mac_matter_mpm_porosity" in bat
    for name in ("sim_matter_mpm_porosity.spv", "sim_sparse_mac_matter_mpm_porosity.spv"):
        assert (SRC / "shaders" / name).exists() and (ROOT / "x64/Release/shaders" / name).exists(), name

    pressure = read("src/Physics/Fluid/FluidGpuPressure.inl")
    body = pressure[pressure.index("bool runGpuFluidMGPCGPressure("):]
    run = body.index("!runMatterMpmPorosityGpu(grid, compute, gpu_buffers, mac)")
    assert body.index("uploadSparseMacSolidWeights(") < run, "after the compact weight pack"
    assert body.index("runGpuMatterLiquidWall(grid, compute, gpu_buffers, 1, dt)") < run, "after the collider wall"
    assert run < body.index("sim_fluid_divergence"), "before the divergence reads the weights"

    driver = read("src/Physics/Fluid/MatterGpuStep.inl")
    block = driver[driver.index("const bool mpm_porous = liquid_step"):
                   driver.index("if (liquid_step && !runGpuFluidMGPCGPressure(")]
    assert "downloadBuffer" not in block and "uploadBuffer" not in block, "no host round trip"
    assert "s_matter_porosity_gpu.active = true;" in block
    assert "buffers.matter_model.pressure_statics_uploaded = false;" in block
    assert not (SRC / "src/Physics/Fluid/MatterMpmPorosity.cpp").exists()
    assert "MatterMpmPorosity." not in (ROOT / "RayTrophiStudio/RayTrophiStudio.vcxproj").read_text(encoding="utf-8-sig")
    print("PASS MPM porosity on the device: 14/16 bindings, order pack -> wall -> porosity -> divergence, no host round trip")


if __name__ == "__main__":
    main()
