"""Non-build audit of Matter phase routing and project registration."""

from pathlib import Path
import ast
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[2] / "RayTrophiStudio"

    def source(path):
        return (root / "source" / path).read_text(encoding="utf-8-sig")

    simulation = source("src/Physics/ParticleSimulation.cpp")
    simulation += source("src/Physics/Fluid/MatterDomainSynchronization.inl")
    simulation += source("src/Physics/Fluid/MatterDomainRestore.inl")
    simulation += source("src/Physics/Fluid/MatterDomainSources.inl")
    assert "std::swap(state.grid, state.matter_liquid_grid)" not in simulation
    assert "Fluid::MatterLiquidScope liquid_scope(" in simulation
    assert "liquid_scope.restore();" in simulation
    restore = source("src/Physics/Fluid/MatterDomainRestore.inl")
    assert "state.grid.origin += delta;" in restore
    assert "Fluid::phaseStorageMatches(state, domain)" not in restore
    assert "const bool try_gpu_g2p =\n                fluid_gpu_requested &&" in simulation
    assert "Fluid::translateLiquidParticles(state, delta_min);" in simulation
    assert "Fluid::seedBox(state.particles,\n                           Fluid::liquidGrid(state)" in simulation
    assert "computeFluidFillSeedAABB(Fluid::liquidGrid(state).origin," in simulation

    for path in ["src/Physics/ParticleRenderBridge.cpp",
                 "src/UI/ParticleBillboardBuilder.cpp"]:
        text = source(path)
        assert "liquidGrid(state)" in text, path
        assert "state.grid." not in text, path

    for name in ["FluidCombustionExchange", "FluidMistPhaseExchange"]:
        text = source("src/Physics/Fluid/" + name + ".cpp")
        assert "fluid_state.voxel_size" not in text, name
        assert "gas_state.grid" not in text, name
        assert "gridsOverlap(liquidGrid(fluid_state), gasGrid(gas_state))" in text, name

    stats = source("src/Api/RtApiParticle.cpp")
    liquid_stats = stats.split("Result getFluidStepStats(", 1)[1].split(
        "Result getGasStepStats(", 1)[0]
    assert "state.resolution_" not in liquid_stats
    assert "liquidGrid(state).nx" in liquid_stats
    uvw = source("src/Api/RtApiFluid.cpp").split(
        "void fillFluidMaterialCoords(", 1)[1].split("    const auto& parts", 1)[0]
    assert "state.grid." not in uvw
    assert "liquidGrid(state).voxel_size" in uvw

    namespace = {"ms": "http://schemas.microsoft.com/developer/msbuild/2003"}
    for name in ["RayTrophiStudio.vcxproj", "RayTrophiStudio.vcxproj.filters"]:
        entries = ET.parse(root / name).findall(".//ms:ClInclude", namespace)
        paths = [entry.attrib["Include"] for entry in entries]
        assert paths.count(r"source\include\Fluid\MatterPhaseGrid.h") == 1, name
        assert paths.count(r"source\include\Fluid\MatterPhaseConfig.h") == 1, name
        compiled = ET.parse(root / name).findall(".//ms:ClCompile", namespace)
        compiled_paths = [entry.attrib["Include"] for entry in compiled if "Include" in entry.attrib]
        for path in [r"source\src\Physics\Fluid\MatterPhaseConfig.cpp",
                     r"source\src\UI\MatterPhaseGridControls.cpp",
                     r"source\src\Api\RtApiPhaseGrid.cpp",
                     r"source\src\Api\RtIpcPhaseGrid.cpp",
                     r"source\src\Api\RtPythonPhaseGrid.cpp"]:
            assert compiled_paths.count(path) == 1, (name, path)
    for path in ["src/Utils/SceneSerializer.cpp", "src/Core/ProjectManager.cpp"]:
        text = source(path)
        assert "phaseSettingsJson(" in text and "loadPhaseSettings(" in text, path
    render = source("include/scene_data.h")
    assert "state.grid" not in render
    assert "hashPhaseSettings(h, d)" in render
    assert "state.particles, render_grid" in render
    assert "phaseStorageMatches(" in source("src/Physics/Fluid/MatterPhaseConfig.cpp")
    assert "Fluid::gridContains(Fluid::liquidGrid(state), spawn_pos)" in simulation
    assert source("src/Scene/SceneDataParticlePresets.cpp").count("adoptPresetPhaseGrids(") == 2
    ipc = source("src/Api/RtIpcPhaseGrid.cpp")
    python = source("src/Api/RtPythonPhaseGrid.cpp")
    for method in ["get_phase_grids", "set_phase_grid"]:
        assert '"fluid.' + method + '"' in ipc
        assert '"' + method + '"' in python
    probe = root.parent / "scripts/test/rt_test_matter_phase_grids_ipc.py"
    ast.parse(probe.read_text(encoding="utf-8"))
    print("PASS: phase routing, seed, cache translation, API telemetry, project XML")


if __name__ == "__main__":
    main()
